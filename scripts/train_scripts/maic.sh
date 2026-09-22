#!/usr/bin/env bash
# Run from the project root with the experiment conda env activated; inherit SC2PATH.
# Official MAIC (mansicer/MAIC) SC2 evaluation protocol: QMIX mixer, episode runner,
# 2M env steps, 32 test episodes every 10k steps.

EXPERIMENT_NAME=maic_3m
MAP_NAME=3m
SEED=1
T_MAX=1050000

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u src/main.py --config=maic --env-config=sc2 with \
    name="$EXPERIMENT_NAME" env_args.map_name="$MAP_NAME" seed="$SEED" \
    runner=episode batch_size_run=1 \
    batch_size=32 buffer_size=5000 buffer_cpu_only=True \
    use_cuda=True \
    test_nepisode=32 test_interval=10000 log_interval=10000 \
    runner_log_interval=10000 learner_log_interval=10000 \
    save_model_interval=100000 t_max="$T_MAX" "$@"
