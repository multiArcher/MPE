import os
import sys

import logging
import yaml
import datetime
import random
from copy import deepcopy
from pathlib import Path

import numpy
import torch
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
# from sacred.utils import apply_backspaces_and_linefeeds

try:
    from collections import Mapping    # until python 3.10
except ImportError:
    from collections.abc import Mapping    # from python 3.10

from run import run
from utils.custom_logging import PyMARLLogger

from utils.torch_optimizer import optimize_tensor_display


# Optimize tensor display during debugging. This is only useful for debugging.
optimize_tensor_display(torch)

current_file_path = Path(__file__).resolve()
work_dir = current_file_path.parents[1]
results_path = work_dir / "results"

ex = Experiment("pymarl", save_git_info=False)
# set to "no" if you want to see stdout/stderr in console "sys" / "no" / "fd"
SETTINGS["CAPTURE_MODE"] = "no"
# ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    
    # Set the seed for all randomness sources
    # Python seed
    random.seed(config["seed"])
    os.environ["PYTHONHASHSEED"] = str(config["seed"])
    # Numpy seed
    numpy.random.seed(config["seed"])
    # Torch seed
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # SMAC seed.
    config["env_args"]["seed"] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
            encoding="utf-8",
        ) as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def _parse_cli_scalar(value):
    value = value.strip()
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _apply_flat_cli_overrides(config, params, keys):
    for param in params:
        if "=" not in param:
            continue
        key, value = param.split("=", 1)
        if key in keys:
            config[key] = _parse_cli_scalar(value)


if __name__ == "__main__":
    # region read_parameters
    # Params order is command line -> algs -> envs -> defaults
    params = deepcopy(sys.argv)
    # th.set_num_threads(1)
    # params = ["src/main.py", "--config=refil", "--env-config=sc2v2", "with", "env_args.map_name='protoss_5_vs_5'"]

    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # Scenario kwargs differ across MPE tasks. Register explicitly supplied keys
    # before Sacred validates updates; Sacred still parses their actual values.
    if config_dict["env"] in ("mpe", "delayed_mpe"):
        prefix = "env_args.scenario_args."
        for param in params:
            key, separator, _ = param.partition("=")
            if separator and key.startswith(prefix):
                scenario_key = key[len(prefix):]
                if not scenario_key.isidentifier():
                    raise ValueError(f"Invalid MPE scenario parameter: {scenario_key}")
                config_dict["env_args"].setdefault("scenario_args", {})
                config_dict["env_args"]["scenario_args"].setdefault(scenario_key, None)

    # endregion

    # Generate unique token.
    env_args = config_dict["env_args"]
    if env_args.get("map_name"):
        map_name = env_args["map_name"]
    elif env_args.get("scenario"):
        map_name = env_args["scenario"]
    elif env_args.get("key"):
        map_name = env_args["key"]
    else:
        map_name = config_dict.get("env", "env")

    # Get experiment name from yaml
    experiment_name = config_dict["name"]

    # Update map name and experiment name from command line parameters.
    for param in params:
        if param.startswith("env_args.map_name="):
            map_name = param.split("=", 1)[1]
        elif param.startswith("env_args.scenario="):
            map_name = param.split("=", 1)[1]
        elif param.startswith("env_args.key="):
            map_name = param.split("=", 1)[1]
        elif param.startswith("name"):
            experiment_name = param.split("=")[1]

    loss_weight_keys = [
        "td_loss_weight",
        "action_loss_weight",
        "continue_loss_weight",
        "aux_loss_weight",
        "entropy_loss_weight",
    ]
    _apply_flat_cli_overrides(config_dict, params, loss_weight_keys)
    loss_weight_suffix_names = {
        "td_loss_weight": "td_",
        "action_loss_weight": "action_",
        "continue_loss_weight": "continue_",
        "aux_loss_weight": "aux_",
        "entropy_loss_weight": "entropy_",
    }
    loss_weight_suffix = "".join(
        f"-{loss_weight_suffix_names[key]}={config_dict[key]}"
        for key in loss_weight_keys
        if key in config_dict
    )
    unique_token = (
        f"{experiment_name}__{map_name}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        f"{loss_weight_suffix}"
    )

    config_dict.update({"unique_token": unique_token})

    logger = PyMARLLogger(
        name="main",
        level=logging.DEBUG,
        file_log=True,
        log_dir=results_path / "logs",
        log_file_name=unique_token + ".log",
    )
    ex.logger = logger.logger

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in \"./results/sacred\".")
    file_obs_path = results_path / f"sacred/{unique_token}"

    ex.observers.append(
        FileStorageObserver(file_obs_path, copy_artifacts=False, copy_sources=False)
    )
    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    # ex.observers.append(MongoObserver())

    # Run experiment.
    ex.add_config(config_dict)
    ex.run_commandline(params)
