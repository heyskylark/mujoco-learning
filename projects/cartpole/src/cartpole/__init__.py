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

register_mjlab_task(
    task_id="Cartpole3-Balance",
    env_cfg=cartpole_balance_env_cfg(n_poles=3),
    play_env_cfg=cartpole_balance_env_cfg(play=True, n_poles=3),
    rl_cfg=cartpole_ppo_runner_cfg(experiment_name="cartpole3"),
)

register_mjlab_task(
    task_id="Cartpole3-Swingup",
    env_cfg=cartpole_swingup_env_cfg(n_poles=3),
    play_env_cfg=cartpole_swingup_env_cfg(play=True, n_poles=3),
    rl_cfg=cartpole_ppo_runner_cfg(experiment_name="cartpole3"),
)
