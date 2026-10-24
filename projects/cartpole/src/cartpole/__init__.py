from mjlab.tasks.registry import register_mjlab_task

from cartpole.cartpole_env_cfg import (
    cartpole_balance_env_cfg,
    cartpole_ppo_runner_cfg,
    cartpole_swingup_env_cfg,
)

register_mjlab_task(
    task_id="Cartpole-Balance",
    env_cfg=cartpole_balance_env_cfg(),
    play_env_cfg=cartpole_balance_env_cfg(play=True),
    rl_cfg=cartpole_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Cartpole-Swingup",
    env_cfg=cartpole_swingup_env_cfg(),
    play_env_cfg=cartpole_swingup_env_cfg(play=True),
    rl_cfg=cartpole_ppo_runner_cfg(),
)

# Three poles need more force: with 10 N even a delay-aware LQR holds only 38-58%
# of the upright reset states (RESEARCH.md, Experiment 7).
_TRIPLE = {"n_poles": 3, "max_force_n": 30.0}

register_mjlab_task(
    task_id="Cartpole3-Balance",
    env_cfg=cartpole_balance_env_cfg(**_TRIPLE),
    play_env_cfg=cartpole_balance_env_cfg(play=True, **_TRIPLE),
    rl_cfg=cartpole_ppo_runner_cfg(experiment_name="cartpole3"),
)

# No arm delay for 3 poles: with any delay, the policy learned to drop upright
# starts before the termination armed, escaping it (Experiment 7, runs 1-2).
_TRIPLE_SWINGUP = {**_TRIPLE, "fall_arm_after_s": 0.0}

register_mjlab_task(
    task_id="Cartpole3-Swingup",
    env_cfg=cartpole_swingup_env_cfg(**_TRIPLE_SWINGUP),
    play_env_cfg=cartpole_swingup_env_cfg(play=True, **_TRIPLE_SWINGUP),
    rl_cfg=cartpole_ppo_runner_cfg(experiment_name="cartpole3"),
)
