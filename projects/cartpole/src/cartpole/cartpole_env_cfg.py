"""Cartpole balance and swingup environment configuration."""

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
    JointEffortActionCfg,
    RewardTermCfg,
    apply_body_impulse,
    dr,
    joint_pos_rel,
    joint_vel_rel,
    reset_joints_by_offset,
    time_out,
)
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.managers.event_manager import requires_model_fields
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer.viewer_config import ViewerConfig

_CARTPOLE_XML: Path = Path(__file__).parent / "cartpole.xml"
_CART_CFG = SceneEntityCfg("cartpole", joint_names=("slider",))
# Matches hinge_1..hinge_n in model order (base to tip), for any pole count.
_HINGE_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_.*",))


@dataclass
class Rk4MujocoCfg(MujocoCfg):
    """MujocoCfg that integrates with RK4, matching dm_control's cartpole.

    The semi-implicit Euler integrators mjlab exposes (``euler``, ``implicitfast``)
    inject energy into a fast-spinning double pendulum; the gain grows with spin
    rate until qvel overflows to NaN. RK4 conserves energy for this model at
    dt=0.01. Overrides the inherited ``integrator`` field.
    """

    def apply(self, model: mujoco.MjModel) -> None:
        super().apply(model)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4


# Entities


def _get_spec(n_poles: int, max_force_n: float) -> mujoco.MjSpec:
    """Load the 2-pole XML and chain on poles 3..n_poles at each previous pole's tip.

    The cart and rails sit at z = n_poles m so all poles hanging down clear the floor.
    The motor's gear is set to `max_force_n`; with ctrlrange ±1 that is its peak force.
    """
    spec = mujoco.MjSpec.from_file(str(_CARTPOLE_XML))
    pole_default = spec.find_default("pole")
    for i in range(3, n_poles + 1):
        pole = spec.body(f"pole_{i - 1}").add_body(name=f"pole_{i}", pos=[0, 0, 1])
        pole.add_joint(pole_default, name=f"hinge_{i}")
        pole.add_geom(pole_default, name=f"pole_{i}")
    spec.body("cart").pos[2] = n_poles
    spec.actuator("slide").gear[0] = max_force_n
    for rail in ("rail1", "rail2"):
        spec.geom(rail).pos[2] = n_poles
    return spec


_CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=("slider",),
            # Motor latency: commands arrive 0-2 physics steps (0-20 ms) late. Each
            # env holds its lag for 5 s (500 physics steps) before resampling, like
            # a roughly constant control-loop delay rather than per-step jitter.
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=500,
        ),
    ),
)


def _get_cartpole_cfg(swing_up: bool, n_poles: int, max_force_n: float) -> EntityCfg:
    return EntityCfg(
        spec_fn=functools.partial(_get_spec, n_poles, max_force_n),
        articulation=_CARTPOLE_ARTICULATION,
        # Swing-up hangs pole 1 down; later hinges are relative, so 0 keeps them in line.
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "slider": 0.0,
                "hinge_1": math.pi if swing_up else 0.0,
                "hinge_[2-9]": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )


# Observations


def pole_angle_cos_sin(
    env: ManagerBasedRlEnv,
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
    """Cosine and sine of each pole hinge angle. Shape: [num_envs, 2 * num_hinges].

    ``biased`` reads ``joint_pos_biased``, which includes ``dr.encoder_bias``.
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    angle = joint_pos[:, asset_cfg.joint_ids]
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


# Rewards.

# dm_control's tolerance() defaults to value_at_margin=0.1, used here by the gaussian
# terms. Its small_control term passes value_at_margin=0, so the quadratic sigmoid's
# scale sqrt(1 - value_at_margin) is 1 and needs no constant.
_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))


def _gaussian_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
    """Gaussian sigmoid tolerance: 1 at x=0, value_at_margin=0.1 at |x|=margin."""
    if margin == 0:
        return (x == 0).float()
    scaled = x / margin * _GAUSSIAN_SCALE
    return torch.exp(-0.5 * scaled**2)


def _quadratic_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
    """Quadratic sigmoid tolerance: 1 at x=0, 0 at |x|>=margin."""
    if margin == 0:
        return (x == 0).float()
    return torch.clamp(1 - (x / margin) ** 2, min=0.0)


def cartpole_smooth_reward(
    env: ManagerBasedRlEnv,
    cart_cfg: SceneEntityCfg = _CART_CFG,
    hinge_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
    """dm_control smooth n-pole cartpole reward: upright * centered * small_control * small_vel.

    Args:
      env: The environment.
      cart_cfg: Entity config selecting the slider joint.
      hinge_cfg: Entity config selecting the hinge joints, ordered base to tip.
    """
    asset: Entity = env.scene[cart_cfg.name]

    # Pole uprightness, averaged over poles. Each hinge is relative to the pole below
    # it, so a pole's world angle is the cumulative sum of the hinge angles.
    hinge_angles = asset.data.joint_pos[:, hinge_cfg.joint_ids]
    pole_cos = torch.cos(torch.cumsum(hinge_angles, dim=-1))
    upright = ((pole_cos + 1) / 2).mean(dim=-1)

    # Cart position.
    cart_pos = asset.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
    centered = (1 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2

    # Control effort (raw action from the policy).
    control = env.action_manager.action.squeeze(-1)
    small_control = (4 + _quadratic_tolerance(control, margin=1.0)) / 5

    # Pole angular velocity; the fastest-spinning hinge sets the term.
    hinge_vel = asset.data.joint_vel[:, hinge_cfg.joint_ids]
    small_velocity = (1 + _gaussian_tolerance(hinge_vel, margin=5.0).amin(dim=-1)) / 2

    return upright * centered * small_control * small_velocity


# Events.


def reset_fraction_upright(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    fraction: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Flip pole 1 from hanging to upright (hinge_1 - π) in a random `fraction` of envs.

    Must run after the hinge reset event so flipped envs keep its reset noise.
    Pole 2 is relative to pole 1, so it flips upright with it.
    """
    env_ids = resolve_env_ids(env, env_ids)
    flip = env_ids[torch.rand(len(env_ids), device=env.device) < fraction]
    if len(flip) == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = torch.tensor(asset_cfg.joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        asset.data.joint_pos[flip][:, joint_ids] - math.pi,
        asset.data.joint_vel[flip][:, joint_ids],
        joint_ids=joint_ids,
        env_ids=flip,
    )


@requires_model_fields("actuator_gear")
def actuator_gear_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Scale each env's motor gear (force per unit ctrl) by a factor in `scale_range`.

    ``dr.effort_limits`` rejects effort-commanded XML actuators, so motor strength is
    randomized through the gear instead. Max force = gear x ctrlrange.
    """
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    ctrl_ids = asset.indexing.ctrl_ids[asset_cfg.actuator_ids]
    default_gear = env.sim.get_default_field("actuator_gear")[ctrl_ids, 0]
    scale = sample_uniform(*scale_range, (len(env_ids), len(ctrl_ids)), env.device)
    env.sim.model.actuator_gear[env_ids[:, None], ctrl_ids, 0] = default_gear * scale


# Terminations.


class PoleFallenAfterUprightStart:
    """Terminate upright-start episodes that fall after holding balance for a while.

    The term arms once both poles' cosines have stayed above `upright_cos` for
    `arm_after_s` straight. An armed episode terminates when any pole's cosine drops
    below `min_cos`. Upright starts that fall before arming continue as swing-ups.
    Episodes that start hanging never terminate here.

    The start is classified in `reset()`, which the termination manager calls after
    the reset events have written the new episode's initial state.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._hinge_cfg: SceneEntityCfg = cfg.params["hinge_cfg"]
        self._arm_steps = round(cfg.params["arm_after_s"] / env.step_dt)
        shape, device = (env.num_envs,), env.device
        self._started_upright = torch.zeros(shape, dtype=torch.bool, device=device)
        self._armed = torch.zeros(shape, dtype=torch.bool, device=device)
        self._upright_steps = torch.zeros(shape, dtype=torch.long, device=device)

    def _pole_cos(self) -> torch.Tensor:
        """World-frame cosine of each pole angle. Shape: [num_envs, num_hinges]."""
        asset: Entity = self._env.scene[self._hinge_cfg.name]
        hinge_angles = asset.data.joint_pos[:, self._hinge_cfg.joint_ids]
        return torch.cos(torch.cumsum(hinge_angles, dim=-1))

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._started_upright[env_ids] = self._pole_cos()[env_ids, 0] > 0.0
        self._armed[env_ids] = False
        self._upright_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        hinge_cfg: SceneEntityCfg,
        min_cos: float,
        upright_cos: float,
        arm_after_s: float,
    ) -> torch.Tensor:
        del env, hinge_cfg, arm_after_s  # Cached in __init__.
        pole_cos = self._pole_cos()
        upright = (pole_cos > upright_cos).all(dim=-1)
        self._upright_steps = torch.where(upright, self._upright_steps + 1, 0)
        self._armed |= self._started_upright & (self._upright_steps >= self._arm_steps)
        return self._armed & (pole_cos < min_cos).any(dim=-1)


# Environment config.


def _make_env_cfg(
    swing_up: bool,
    n_poles: int,
    max_force_n: float,
    fall_arm_after_s: float = 1.0,
) -> ManagerBasedRlEnvCfg:
    cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))
    hinge_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_.*",))

    # The actor sees encoder-biased, noisy measurements; the critic sees the true state.
    actor_terms = {
        "cart_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": cart_cfg, "biased": True},
            noise=UniformNoiseCfg(n_min=-0.005, n_max=0.005),
        ),
        "pole_angle": ObservationTermCfg(
            func=pole_angle_cos_sin,
            params={"asset_cfg": hinge_cfg, "biased": True},
            # ~±0.17 deg per step, on the order of a 4096-count encoder's resolution.
            noise=UniformNoiseCfg(n_min=-0.003, n_max=0.003),
        ),
        "cart_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": cart_cfg},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),
        "pole_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": hinge_cfg},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        ),
    }
    critic_terms = {
        "cart_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": cart_cfg},
        ),
        "pole_angle": ObservationTermCfg(
            func=pole_angle_cos_sin,
            params={"asset_cfg": hinge_cfg},
        ),
        "cart_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": cart_cfg},
        ),
        "pole_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": hinge_cfg},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms),
    }

    actions: dict[str, ActionTermCfg] = {
        "effort": JointEffortActionCfg(
            entity_name="cartpole",
            actuator_names=("slider",),
            scale=1.0,
        ),
    }

    slider_range = (-0.1, 0.1) if not swing_up else (0.0, 0.0)
    events = {
        "reset_slider": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": slider_range,
                "velocity_range": (-0.01, 0.01),
                "asset_cfg": SceneEntityCfg("cartpole", joint_names=("slider",)),
            },
        ),
        "reset_hinge": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.034, 0.034),
                "velocity_range": (-0.01, 0.01),
                "asset_cfg": hinge_cfg,
            },
        ),
        # Domain randomization, sampled once per env at startup.
        "pole_inertia": EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                # Mass and inertia scale by e^(2 alpha): x0.95-1.05 (poles get weighed).
                "alpha_range": (-0.026, 0.024),
                # COM shift along the pole axis (body z).
                "t3_range": (-0.05, 0.05),
                "asset_cfg": SceneEntityCfg("cartpole", body_names=("pole_.*",)),
            },
        ),
        "cart_inertia": EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "alpha_range": (-0.11, 0.09),
                "asset_cfg": SceneEntityCfg("cartpole", body_names=("cart",)),
            },
        ),
        "hinge_damping": EventTermCfg(
            func=dr.joint_damping,
            mode="startup",
            params={"ranges": (0.0, 0.005), "operation": "abs", "asset_cfg": hinge_cfg},
        ),
        "slider_friction": EventTermCfg(
            func=dr.joint_friction,
            mode="startup",
            params={"ranges": (0.0, 1.0), "operation": "abs", "asset_cfg": cart_cfg},
        ),
        "slider_damping": EventTermCfg(
            func=dr.joint_damping,
            mode="startup",
            params={"ranges": (0.0, 0.5), "operation": "abs", "asset_cfg": cart_cfg},
        ),
        "motor_gear": EventTermCfg(
            func=actuator_gear_scale,
            mode="startup",
            params={
                "scale_range": (0.9, 1.2),
                "asset_cfg": SceneEntityCfg("cartpole", actuator_names=("slide",)),
            },
        ),
        "slider_encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={"bias_range": (-0.005, 0.005), "asset_cfg": cart_cfg},
        ),
        "hinge_encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            # ±0.17 deg: zeroing the encoders with the pendulum hanging at rest.
            params={"bias_range": (-0.003, 0.003), "asset_cfg": hinge_cfg},
        ),
        # External pushes on the cart (covers cartpole-play's 10 N, 0.15 s A/D push).
        "push_cart": EventTermCfg(
            func=apply_body_impulse,
            mode="step",
            params={
                "force_range": (-10.0, 10.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (0.05, 0.15),
                "cooldown_s": (3.0, 6.0),
                "asset_cfg": SceneEntityCfg("cartpole", body_names=("cart",)),
            },
        ),
    }
    if swing_up:
        # Start a fraction of episodes balanced so the policy practices holding the
        # poles up. Runs after reset_hinge, so those envs keep its noise.
        events["reset_upright_fraction"] = EventTermCfg(
            func=reset_fraction_upright,
            mode="reset",
            params={
                "fraction": 0.25,
                "asset_cfg": SceneEntityCfg("cartpole", joint_names=("hinge_1",)),
            },
        )

    rewards = {
        "smooth_reward": RewardTermCfg(
            func=cartpole_smooth_reward,
            weight=1.0,
            params={"cart_cfg": cart_cfg, "hinge_cfg": hinge_cfg},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }
    if swing_up:
        # Upright-start episodes end when a pole tips past ~37 deg (cos < 0.8), once
        # they have held all poles within ~18 deg (cos > 0.95) for `fall_arm_after_s`.
        # A delay is an escape hatch: a policy can drop the poles before the term
        # arms and keep collecting swing-up reward. 0 arms on the first step.
        terminations["fallen"] = TerminationTermCfg(
            func=PoleFallenAfterUprightStart,
            params={
                "hinge_cfg": hinge_cfg,
                "min_cos": 0.8,
                "upright_cos": 0.95,
                "arm_after_s": fall_arm_after_s,
            },
        )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"cartpole": _get_cartpole_cfg(swing_up, n_poles, max_force_n)},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="cartpole",
            body_name="cart",
            # Far enough to frame all poles (1 m each) above or below the cart.
            distance=3.0 * n_poles,
            elevation=-15.0,
            # Side view: the slider's +x axis points screen-right (D pushes right).
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            mujoco=Rk4MujocoCfg(timestep=0.01, disableflags=("contact",)),
        ),
        decimation=5,
        episode_length_s=50.0,
    )


def cartpole_balance_env_cfg(
    play: bool = False,
    n_poles: int = 2,
    max_force_n: float = 10.0,
) -> ManagerBasedRlEnvCfg:
    cfg = _make_env_cfg(swing_up=False, n_poles=n_poles, max_force_n=max_force_n)
    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
        # cartpole-play applies pushes from the keyboard instead.
        del cfg.events["push_cart"]
    return cfg


def cartpole_swingup_env_cfg(
    play: bool = False,
    n_poles: int = 2,
    max_force_n: float = 10.0,
    fall_arm_after_s: float = 1.0,
) -> ManagerBasedRlEnvCfg:
    cfg = _make_env_cfg(
        swing_up=True,
        n_poles=n_poles,
        max_force_n=max_force_n,
        fall_arm_after_s=fall_arm_after_s,
    )
    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
        del cfg.events["push_cart"]
        # Always start hanging in play; this also leaves "fallen" inert.
        cfg.events["reset_upright_fraction"].params["fraction"] = 0.0
    return cfg


# RL config.


def cartpole_ppo_runner_cfg(experiment_name: str = "cartpole") -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        save_interval=50,
        num_steps_per_env=32,
        max_iterations=500,
        # Log locally by default; opt into W&B metrics with `--agent.logger wandb`.
        logger="tensorboard",
        # Checkpoints stay in logs/ even when W&B is enabled.
        upload_model=False,
    )
