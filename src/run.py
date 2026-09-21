import os
import sys
import datetime
import pprint
import shutil
import tqdm
import threading
import psutil
# from os.path import dirname, abspath
from types import SimpleNamespace as SN
from pathlib import Path

import torch
import numpy

from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.general_reward_support import test_alg_config_supports_reward
from utils.custom_logging import PyMARLLogger
from utils.maker import MACMaker, LearnerMaker, RunnerMaker


def run(_run, _config, _log):
    # setup loggers
    logger = PyMARLLogger("main").get_child_logger("run")

    # check args sanity
    _config = args_sanity_check(_config, logger)

    args = SN(**_config)
    args.device = torch.device(args.device)
    logger.args_storage = args  # Temporary solution for args sharing. Will be removed after Config is implemented.

    logger.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, underscore_numbers=True)
    logger.info("\n" + experiment_params + "\n")

    # configure tensorboard logger
    if args.use_tensorboard:
        tb_logs_dir = Path(__file__).resolve().parents[1] / "results" / "tensorboard_logs"
        tb_exp_dir = tb_logs_dir / f"{_config['unique_token']}"
        logger.setup_tb(tb_exp_dir)

    if args.use_wandb:
        logger.setup_wandb(
            _config, args.wandb_team, args.wandb_project, args.wandb_mode
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Finish logging
    logger.finish(args)

    # Clean up after finishing
    logger.info("Train process finished, exiting main.")

    logger.info("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            logger.info("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            logger.info("Thread joined")

    logger.info("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    runner.mac.eval()  # Set MAC to evaluation mode

    state_record_data = []
    obs_record_data = []
    available_actions_data = []
    n_eval_runs = max(1, (args.test_nepisode + runner.batch_size - 1) // runner.batch_size)
    for runs in tqdm.trange(
        n_eval_runs,
        mininterval=1,
        unit="batch",
        bar_format="{desc}{bar:12} | {n_fmt}/{total_fmt} batches{percentage:3.0f}% [{elapsed}<{remaining} {rate_fmt}]{postfix}",
        desc=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} | EVALUATE | ",
    ):
        batch = runner.run(test_mode=True)
        if getattr(args, "save_evaluate_state", False) is True:
            batch_max_sequence_len = batch.max_t_filled()
            batch = batch[:, :batch_max_sequence_len]

            for i in range(batch.batch_size):
                state_record_data.append(batch["state"][i].cpu().numpy())
                obs_record_data.append(batch["obs"][i].cpu().numpy())
                available_actions_data.append(batch["avail_actions"][i].cpu().numpy())

    trace_data_save_dir = Path(args.local_results_path) / "traces" / args.unique_token
    trace_data_save_dir.mkdir(parents=True, exist_ok=True)

    state_record_data_path = trace_data_save_dir / "state_data.npy"
    obs_record_data_path = trace_data_save_dir / "obs_data.npy"
    available_actions_data_path = trace_data_save_dir / "available_actions_data.npy"

    numpy.save(state_record_data_path, numpy.array(state_record_data, dtype=object))
    numpy.save(obs_record_data_path, numpy.array(obs_record_data, dtype=object))
    numpy.save(available_actions_data_path, numpy.array(available_actions_data, dtype=object))

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    # TODO: Environment is initialized in runner. It's difficult for other parts to get env_info.
    #       Now, we are using args_env_info for communicating between modules. It's not a good design.
    runner = RunnerMaker.make(args.runner, args=args, logger=logger)
    logger.debug(f"Running with {runner.__class__.__name__}.")

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.env_info = env_info

    # Default/Base scheme
    scheme = parse_buffer_scheme(env_info, args.common_reward)
    groups = {"agents": args.n_agents}
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    # mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    mac = MACMaker.make(args.mac, buffer.scheme, groups, args)
    logger.debug(f"Running with {mac.__class__.__name__}.")

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    # learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    learner = LearnerMaker.make(args.learner, mac, buffer.scheme, logger, args)
    logger.debug(f"Running with {learner.__class__.__name__}.")

    if args.use_cuda:
        learner.to(args.device)

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("running/episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_t = -args.test_interval - 1
    last_log_t = 0
    model_save_time = 0

    logger.console_logger.info(f"Beginning training for {args.t_max} timesteps")
    logger.console_logger.info("-" * 30 + "TRAINING_START" + "-" * 30)

    # Delay init tqdm bar
    tqdm_output = sys.stdout
    if sys.platform.startswith("linux"):
        try:
            tqdm_output = open("/dev/tty", "w")
        except OSError:
            tqdm_output = sys.stdout
    logger.info("Train process started")
    progress_bar = tqdm.tqdm(
        total=(args.t_max),
        mininterval=2,
        unit="step",
        ncols=120,
        dynamic_ncols=False,
        leave=False,
        bar_format="{desc}{bar:18} {n_fmt}/{total_fmt} {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        desc="TRAINING",
        postfix={"ep": 0},
        file=tqdm_output,
    )
    max_winrate = 0
    last_improvement_step = 0

    while runner.t_env <= args.t_max:
        with torch.no_grad():
            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            # TODO: Bigger batch_run should use bigger batch_size. Repeat training is not what parallelization is for.
            for _ in range(args.sample_times_per_run):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled time steps.
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

                del episode_sample

            # Reuse allocations across updates; release unused cache once per round.
            torch.cuda.empty_cache()

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_t) / args.test_interval >= 1.0:
            last_test_t = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
            torch.cuda.empty_cache()

        battle_stats = logger.stats.get("running/test_battle_won_mean") or []
        if battle_stats:
            new_winrate = battle_stats[-1][1]
            best_model = (new_winrate > max_winrate) or (episode == 0)
        else:
            # MPE and other non-SMAC envs do not log battle_won.
            new_winrate = max_winrate
            best_model = episode == 0
        
        if best_model is True:
            model_save_dir = Path(args.local_results_path) / "models" / args.unique_token
            best_model_path = model_save_dir / "best_model"
            best_model_path.mkdir(parents=True, exist_ok=True)
            learner.save_models(best_model_path)
            progress_bar.clear()
            logger.console_logger.info(
                f"Best model updated, winrate: {max_winrate} -> {new_winrate}"
            )
            last_improvement_step = runner.t_env
            max_winrate = new_winrate

            best_model_full_model_path = model_save_dir / "best_model_full_model"
            best_model_full_model_path.mkdir(parents=True, exist_ok=True)
            # torch.jit.save(torch.jit.script(learner.mac), best_model_full_model_path / "mac.pth")
            torch.save(learner.mac, best_model_full_model_path / "mac.th")

        # Save models to unique token directory
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            model_save_dir = Path(args.local_results_path) / "models" / args.unique_token
            model_token = str(model_save_time)
            save_path = model_save_dir / model_token
            save_path.mkdir(parents=True, exist_ok=True)

            progress_bar.clear()
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            if args.use_wandb and args.wandb_save_model:
                wandb_save_dir = os.path.join(
                    logger.wandb.dir, "models", args.unique_token, str(runner.t_env)
                )
                os.makedirs(wandb_save_dir, exist_ok=True)
                for f in os.listdir(save_path):
                    shutil.copyfile(
                        os.path.join(save_path, f), os.path.join(wandb_save_dir, f)
                    )

        episode += args.batch_size_run
        if getattr(args, "anneal_training", False):
            if runner.t_env - last_improvement_step > args.patience:
                progress_bar.clear()
                logger.console_logger.info(
                    f"No improvement over {args.patience} steps. Pushing back."
                )

                def anneal_weights(best_model_path: Path):
                    saved_agent = torch.load(best_model_path / "agent.th", weights_only=True)
                    saved_mixer = torch.load(best_model_path / "mixer.th", weights_only=True)

                    for saved_param, current_param in zip(
                        learner.mac.parameters(), saved_agent.values()
                    ):
                        current_param.data.copy_(saved_param * (1 - args.anneal_rate) + current_param * args.anneal_rate)

                    for saved_param, current_param in zip(
                        learner.mixer.parameters(), saved_mixer.values()
                    ):
                        current_param.data.copy_(saved_param * (1 - args.anneal_rate) + current_param * args.anneal_rate)

                if best_model_path.exists():
                    anneal_weights(best_model_path)

                last_improvement_step = runner.t_env

        if (runner.t_env - last_log_t) >= args.log_interval:
            logger.log_stat("running/episode", episode, runner.t_env)
            progress_bar.clear()
            logger.print_recent_stats()
            last_log_t = runner.t_env

        # Watch CPU usage.
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total / 1024 ** 3  # total memory in GB
        free_memory = memory_info.free / 1024 ** 3  # free memory in GB
        used_memory = memory_info.used / 1024 ** 3  # used memory in GB
        progress_bar_postfix = {
            "ep": episode,
            "mem": f"{used_memory:2.1f}G"
        }

        # Watch GPU usage.
        if args.use_cuda:
            gpu_available_memory, gpu_total_memory = torch.cuda.mem_get_info()
            gpu_available_memory = gpu_available_memory / 1024 ** 3
            gpu_total_memory = gpu_total_memory / 1024 ** 3
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3
            progress_bar_postfix.update(
                {
                    "gpu": f"{gpu_memory_allocated:2.1f}/{gpu_total_memory:2.1f}G"
                }
            )
        progress_bar.set_postfix(progress_bar_postfix)

        progress_bar.set_description_str("TRAINING")
        update_steps = runner.t_env - progress_bar.n
        progress_bar.update(min(update_steps, args.t_max - progress_bar.n))
        sys.stdout.flush()

    progress_bar.close()
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, logger):
    # set CUDA flags
    if config["use_cuda"] and not torch.cuda.is_available():
        config["use_cuda"] = False
        logger.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA device is available!"
        )

    # Preserve the requested MPE parallelism while selecting the runner variant
    # that matches whether the environment exposes delayed observations.
    env_name = config.get("env")
    runner_name = config.get("runner")
    if env_name == "mpe":
        runner_mapping = {
            "delayed_episode": "episode",
            "delayed_parallel": "parallel",
        }
        config["runner"] = runner_mapping.get(runner_name, runner_name)
    elif env_name == "delayed_mpe":
        runner_mapping = {
            "episode": "delayed_episode",
            "parallel": "delayed_parallel",
        }
        config["runner"] = runner_mapping.get(runner_name, runner_name)

    if config.get("runner") in {"episode", "delayed_episode"}:
        config["batch_size_run"] = 1

    # Adjust batch_size_run and test_nepisode to be divisible by batch_size_run.
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    # Check entity scheme availability.
    entity_env_implemented_list = ["sc2v2", "emulate_sc2v2"]
    if config.get("entity_scheme", False) and config["env"] not in entity_env_implemented_list:
        logger.critical(f"Entity scheme is not available in selected env: {config['env']}")
        raise NotImplementedError(f"Entity scheme is only available in {entity_env_implemented_list}. Selected env: {config['env']}")

    # Separate reward check.
    assert test_alg_config_supports_reward(
        config
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # Check sample_times_per_run
    if sample_times_per_run := config.get("sample_times_per_run", None) is not None:
        # sample_times_per_run is defined in config.
        if sample_times_per_run < 1:
            logger.error(
                "sample_times_per_run should be greater than or equal to 1. Setting it to 1."
            )
            config["sample_times_per_run"] = 1
        elif sample_times_per_run > config["batch_size_run"]:
            logger.warning(
                "sample_times_per_run should be less than or equal to batch_size_run. "
                "Consider enlarging batch_size instead of repeat training."
            )
    else:
        # sample_times_per_run is not defined in config.
        config["sample_times_per_run"] = config["batch_size_run"]   # Use batch_size_run as default.

    return config


def parse_buffer_scheme(env_info: dict, common_reward: bool = True):
    """Parse buffer scheme from env_info."""
    scheme = {
        "state": {"vshape": env_info["state_shape"], "dtype": torch.float32},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents", "dtype": torch.float32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "obs_delay": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "obs_gen_t": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "obs_fresh_mask": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    # For individual rewards in gymmai reward is of shape (1, n_agents)
    if common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (env_info["n_agents"],)}
    return scheme

