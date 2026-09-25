"""Import in a parameter-only .local.py entrypoint; set module values and call main."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = "delay_robustness"
# Each entry: id, config (saved training JSON), checkpoint (directory).
# Entries are independent training runs of ONE algorithm.
# Optional tensorboard directory supplements metrics missing from Sacred.
# For a legacy config missing bcrbc_time_block_every, explicitly supply
# legacy_time_block_every=4 in that model entry; never change the method.
MODELS = []
# SMAC grid. MPE uses the preset below and does not read these values.
MEANS = [-2, -1, 0, 1, 2]
STDS = [0, 0.5, 1, 1.5, 2]
CAP = 8
MPE_MEANS = [-2, 0, 2, 4, 6, 8]
MPE_STDS = [0, 2, 4, 6, 8, 10]
MPE_CAP = 16
MPE_FIXED = [0, 1, 2, 4, 6, 8, 10]
EPISODES = 64
PARALLEL = 8
SEED = 101
MASK_INTERVENTION = False
# Optional raw-observation feature slices, specified per map from its actual layout.
FEATURE_GROUPS = {}


def conditions(profile="smac"):
    if profile == "smac":
        means, stds, cap, extra_fixed = MEANS, STDS, CAP, (4, 8)
        specials = [
            dict(id="mixture_balanced", kind="mixture", means=[0, 2], stds=[1, 1],
                 high_probability=0.5, cap=cap),
            dict(id="mixture_rare_severe", kind="mixture", means=[0, 4], stds=[1, 1],
                 high_probability=0.1, cap=cap),
            dict(id="periodic_16", kind="periodic", means=[0, 2], stds=[1, 1], period=16, cap=cap),
            dict(id="markov_09", kind="markov", means=[0, 2], stds=[1, 1], stay_probability=0.9, cap=cap),
        ]
    elif profile == "mpe":
        means, stds, cap, extra_fixed = MPE_MEANS, MPE_STDS, MPE_CAP, MPE_FIXED
        specials = [
            dict(id="mixture_balanced", kind="mixture", means=[0, 4], stds=[4, 4],
                 high_probability=0.5, cap=cap),
            dict(id="mixture_rare_severe", kind="mixture", means=[0, 8], stds=[4, 4],
                 high_probability=0.1, cap=cap),
            dict(id="periodic_16", kind="periodic", means=[0, 4], stds=[4, 4], period=16, cap=cap),
            dict(id="markov_09", kind="markov", means=[0, 4], stds=[2, 4], stay_probability=0.9, cap=cap),
        ]
    else:
        raise ValueError(f"Unknown delay profile: {profile}")
    return _grid(means, stds, cap, extra_fixed, specials)


def _grid(means, stds, cap, extra_fixed, specials):
    cases = [{"id": "fixed_0", "kind": "fixed", "value": 0, "cap": cap}]
    cells = {"gaussian": [], "uniform": []}
    for family in cells:
        for mean in means:
            for std in stds:
                if std == 0:
                    value = min(cap, math.ceil(max(0, mean)))
                    name = f"fixed_{value}"
                    condition = dict(id=name, kind="fixed", value=value, cap=cap)
                else:
                    name = f"{family}_{mean:g}_{std:g}"
                    condition = dict(id=name, kind=family, mean=mean, std=std, cap=cap)
                    if family == "uniform":
                        # Match the Gaussian's raw standard deviation.
                        width = math.sqrt(12) * std
                        condition.update(width=width, low=mean - width / 2,
                                         high=mean + width / 2)
                cells[family].append(dict(mean=mean, std=std, condition_id=name))
                if not any(case["id"] == name for case in cases):
                    cases.append(condition)
    for value in extra_fixed:
        name = f"fixed_{value}"
        if not any(case["id"] == name for case in cases):
            cases.append(dict(id=name, kind="fixed", value=value, cap=cap))
    cases.extend(specials)
    return cases, cells


def delay_profile(config):
    env = config.get("env")
    if env in ("mpe", "delayed_mpe"):
        return "mpe"
    if env in ("sc2", "delayed_sc2"):
        return "smac"
    raise ValueError(f"Delay study supports sc2 and mpe, got env={env}")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert MODELS, "Set study.MODELS in a .local.py entrypoint before evaluating."
    assert len({m["id"] for m in MODELS}) == len(MODELS), "Model IDs must be unique."
    output = ROOT / "results/evaluate" / STUDY
    output.mkdir(parents=True, exist_ok=True)
    profiles = {delay_profile(json.loads((ROOT / model["config"]).read_text())) for model in MODELS}
    if len(profiles) != 1:
        raise ValueError("One study must be entirely SMAC or entirely MPE")
    profile = profiles.pop()
    print(f"Delay profile: {profile}", flush=True)
    cases, cells = conditions(profile)
    source_files = sorted((ROOT / "src/modules/bcrbc").rglob("*.py"))
    source_files += sorted((ROOT / "scripts/eval_scripts").glob("*.py"))
    source_files += [ROOT / p for p in (
        "src/components/evaluation_delay.py", "src/components/delay_model.py",
        "src/controllers/bcrbc_mac.py", "src/runners/delayed_parallel_runner.py",
        "src/envs/wrappers/delayed_wrapper.py")]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    manifest = dict(protocol="delay_study_v4_matched_grids", models=MODELS, conditions=cases,
                    gaussian_cells=cells["gaussian"], uniform_cells=cells["uniform"],
                    episodes=EPISODES, parallel=PARALLEL,
                    seed=SEED, mask_intervention=MASK_INTERVENTION,
                    feature_groups=FEATURE_GROUPS, sources=sources,
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    path = output / "manifest.json"
    # A study directory must not silently mix incompatible data.
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous == manifest, "Study definition changed; use a new STUDY name."
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pending_jobs = []
    for model in MODELS:
        config_path = ROOT / model["config"]
        checkpoint = ROOT / model["checkpoint"]
        config = json.loads(config_path.read_text())
        for condition in cases:
            for offset in range(0, EPISODES, PARALLEL):
                directory = output / "runs" / model["id"] / condition["id"] / f"batch_{offset:04d}"
                directory.mkdir(parents=True, exist_ok=True)
                job = dict(model=model, config=str(config_path), checkpoint=str(checkpoint),
                           config_sha256=digest(config_path), checkpoint_sha256=digest(checkpoint / "agent.th"),
                           condition=condition, parallel=min(PARALLEL, EPISODES - offset),
                           seed=SEED + offset, episode_offset=offset,
                           mask_intervention=MASK_INTERVENTION,
                           feature_groups=FEATURE_GROUPS.get(config["env_args"]["map_name"], {}))
                job_path = directory / "job.json"
                if (directory / "result.json").exists():
                    assert json.loads(job_path.read_text()) == job, "Checkpoint/config changed; use a new STUDY."
                    continue
                job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
                pending_jobs.append(str(job_path))
    if pending_jobs:
        print(f"Evaluation: {len(pending_jobs)} pending batches; results: {output}", flush=True)
        session_path = output / "session.json"
        session_path.write_text(json.dumps(dict(jobs=pending_jobs), indent=2))
        with (output / "session.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "scripts/eval_scripts/evaluate_study.py"),
                 str(session_path)], cwd=ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, process.args)
    print(f"Summarizing results: {output}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/eval_scripts/summarize_study.py"), str(output)], check=True)


if __name__ == "__main__":
    main()
