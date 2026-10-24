# Cartpole research log

This log records changes to the cartpole task in the order they happened, with
the reasoning behind each change and the results it produced.

Baseline: mjlab's cartpole task,
[`mjlab/tasks/cartpole`](https://github.com/mujocolab/mjlab/tree/main/src/mjlab/tasks/cartpole),
which ports dm_control's cartpole to mjlab.

## Experiments

### 1. Baseline: mjlab single-pole cartpole

**Goal:** check that the mjlab and RSL-RL training pipeline works end to end
on the stock example.

**Setup (upstream, unchanged):**

- **Model (`cartpole.xml`):**
  - The cart is a 1 kg box on a slide joint at z = 1 m. The slide range is ±1.8 m and its damping is 5e-4.
  - The single pole is a 1 m capsule (0.1 kg) on `hinge_1`, with damping 2e-6.
  - The actuator is a motor on `slider` with gear 10 and ctrlrange ±1, giving ±10 N.
- **Simulation:** `timestep=0.01` with contacts disabled. The integrator is
  mjlab's default, `implicitfast`. `decimation=5` gives a 20 Hz policy, and
  `episode_length_s=50` gives 1000 policy steps per episode.
- **Tasks:**
  - `Mjlab-Cartpole-Balance` starts with the pole upright (`hinge_1 = 0`) and adds reset noise of ±0.1 m to the slider.
  - `Mjlab-Cartpole-Swingup` starts with the pole hanging (`hinge_1 = π`) and no slider noise.
  - Both tasks add ±0.034 rad of noise to the hinge and ±0.01 to all velocities at reset.
- **Observations (5-D):** cart position, `cos`/`sin` of the hinge angle, cart
  velocity and hinge velocity. The actor gets noise corruption during training.
- **Reward:** dm_control's smooth cartpole reward, which multiplies four terms:
  - `upright = (cos θ + 1) / 2`
  - `centered = (1 + gaussian(cart_pos, margin=2)) / 2`
  - `small_control = (4 + quadratic(action, margin=1)) / 5`
  - `small_velocity = (1 + gaussian(θ̇, margin=5)) / 2`
- **Termination:** time-out only.
- **PPO (RSL-RL):**
  - Actor and critic are both 64×64 ELU MLPs. Gaussian policy with `init_std=1.0`.
  - `lr=1e-3` with an adaptive schedule (`desired_kl=0.01`), `gamma=0.99`, `lam=0.95`.
  - 5 epochs × 4 mini-batches, 32 steps per env, 500 iterations.

**Result:** trained successfully.

### 2. Double pendulum: second pole added, NaNs during training

**Goal:** make the task harder by turning the pole into a two-link pendulum.

**Changes:**

- **Model:**
  - Added `pole_2` with `hinge_2` as a child of `pole_1`, mounted at its tip (`pos="0 0 1"`). It uses the same `pole` default class: 1 m, 0.1 kg, damping 2e-6.
  - `hinge_2` is relative to `pole_1`, so pole 2's world angle is `hinge_1 + hinge_2`.
  - Raised the cart and rails to z = 2 m so both poles can hang down (2 m total) without reaching the floor.
  - Moved the cameras to match.
- **Entity and tasks:**
  - `_HINGE_CFG` now selects `("hinge_1", "hinge_2")`.
  - Both initial states set `hinge_2 = 0`: Balance starts with both poles up, and Swing-up starts with both hanging (`hinge_1 = π`).
  - Hinge reset noise now applies to both hinges.
- **Observations:** grew to 8-D. The `cos`/`sin` features now cover both
  relative hinge angles (4 values), and there are two hinge velocities.
- **Reward:** generalized to n poles, following dm_control's
  `two_poles` / `three_poles` reward:
  - `upright` is the mean over poles of `(cos φᵢ + 1) / 2`. Each `φᵢ` is a world-frame pole angle, taken as the cumulative sum of the hinge angles.
  - `small_velocity` uses the smallest Gaussian tolerance across hinges, so the fastest-spinning hinge sets the term.
- **Viewer:** distance 4 → 6 so both poles stay in frame. Azimuth 0 → 90 gives a
  side view where +x is screen-right.

**Result:** training produced NaNs.

**Cause:** the integrator. mjlab's `MujocoCfg.integrator` only allows `euler`
or `implicitfast` (default `implicitfast`). Both are semi-implicit Euler
schemes, and on a fast-spinning double pendulum they add energy every step
instead of conserving it. The added energy grows with spin rate, so pole
swings keep accelerating until `qvel` overflows to NaN. The single pole in
Experiment 1 doesn't spin fast enough for this to matter. dm_control's
cartpole model integrates with RK4, and mjlab's port doesn't.

### 3. RK4 integrator: stable double-pendulum training, poles don't stay up

**Goal:** remove the NaNs by fixing the physics rather than masking them.

**Changes:**

- Added `Rk4MujocoCfg`, a `MujocoCfg` subclass whose `apply()` sets
  `model.opt.integrator = mjINT_RK4` after the parent applies its settings.
  Kept `timestep=0.01` and contacts disabled. At this timestep, RK4 conserves
  energy for this model.
- Set the default logger to TensorBoard, with `upload_model=False` so
  checkpoints stay in `logs/` even when W&B is enabled.

> Gotcha: the `integrator` field still reads `implicitfast`, so
> `params/env.yaml` in a run directory shows `integrator: implicitfast` even
> though the sim uses RK4.

**Run:** `logs/rsl_rl/cartpole/2026-10-22_17-47-04`. Task `Cartpole-Swingup`,
16384 envs, 500 iterations, ~8.3e5 steps/s.

| Iteration | Mean episode reward | Mean per-step reward | Episode length | Policy std |
| --------: | ------------------: | -------------------: | -------------: | ---------: |
|         0 |                0.02 |               0.0001 |             30 |      0.990 |
|       125 |                20.8 |                0.417 |           1000 |      0.174 |
|       250 |                29.4 |                0.580 |           1000 |      0.092 |
|       375 |                31.7 |                0.631 |           1000 |      0.050 |
|       499 |                32.0 |                0.642 |           1000 |      0.043 |

How to read the table:

- **Episode reward:** per-step rewards are scaled by the policy step
  (0.05 s), so the maximum episode reward is 1000 × 0.05 = 50. "Mean per-step
  reward" is `Episode_Reward/smooth_reward`, where 1.0 is the maximum.
- **Episode length:** stays at 1000 because time-out is the only termination,
  so it says nothing about whether the policy succeeds.

**Result:**
- **No NaNs:** training completes without any.
- **Plateau:** the per-step reward levels off around 0.64.
- **Std collapse:** policy std shrinks to ~0.04, so the policy has largely
  stopped exploring.
- **Behaviour:** in play, the policy can't keep both poles upright for long.

**Diagnosis:** the current reward has gaps that don't push the policy toward
sustained balance of both links. A reward ablation is in progress.

### 4. Mixed start states: 25% of swing-up episodes start upright

**Goal:** give the swing-up policy direct practice at holding both poles up,
and make falling from upright expensive.

**Motivation (diagnostic runs, not kept):**

- **Physics is balanceable:** an LQR controller (10 N, 20 Hz) held both poles
  up for the full 50 s in 100% of 2048 `Cartpole-Balance` envs, and still did
  with Gaussian action noise of std 0.2.
- **More training doesn't help:** resuming Experiment 3 from `model_499` for
  1000 more iterations left the reward flat (~33) at policy std 0.03. Both
  poles were up (cos > 0.95) ~35–38% of the last 30 s, and no env stayed up
  for 90% of it.
- **PPO can't learn to balance without a fall termination:** 300 iterations
  on `Cartpole-Balance` reached 39% both-up. The same task plus a termination
  when any pole's cos < 0.8 reached episode length 1000 within 100 iterations,
  and held 100% of 2048 envs up for 50 s when evaluated without the termination.
- **Why:** with time-out as the only termination, a fall costs a few seconds
  of partial reward before the policy swings back up. Wobbling near the top
  already earns ~0.8 per step against 1.0 for true balance.

**Changes (`Cartpole-Swingup` only):**

- **Event `reset_upright_fraction`:** runs after `reset_hinge` and flips
  `hinge_1` by −π in a random 25% of resetting envs. Those envs start upright
  with the same ±0.034 rad hinge noise. The rest start hanging as before.
- **Termination `fallen` (`PoleFallenAfterUprightStart`):** ends an episode
  that started upright once any pole's world-frame cos < 0.8 (~37°). Hanging
  starts are never terminated by it, so they can swing up freely. The start is
  classified in the term's `reset()`, which the termination manager calls
  after the reset events have run.
- **Play:** `fraction = 0`, so play always starts hanging and `fallen` never
  fires.

How to read the training metrics now:

- **Episode length** is no longer always 1000: upright starts that fall end
  early.
- **`Episode_Termination/fallen`** counts falls from upright starts.
- **Mean reward** mixes upright and hanging starts, so it isn't comparable to
  Experiment 3's table.

**Runs:** `logs/rsl_rl/cartpole/2026-10-23_13-50-57` (iterations 0–499), then
resumed from its `model_499` as `logs/rsl_rl/cartpole/2026-10-23_14-04-55`
(iterations 500–1498). 16384 envs, same PPO settings as Experiment 3.

Each checkpoint was evaluated on 2048 envs for one 50 s episode, with
corruption off and `fallen` removed. "Hanging" starts every env hanging;
"upright" starts every env upright. "Held" means both poles had cos > 0.95 for
at least 90% of the last 30 s.

| Checkpoint                  | Hanging: both up, last 30 s | Hanging: held | Upright: held | Upright: mean time to first fall |
| --------------------------- | --------------------------: | ------------: | ------------: | -------------------------------: |
| Experiment 3 `model_499`    |                       36.8% |          0.0% |          0.0% |                            1.1 s |
| 100                         |                        3.2% |          0.0% |          0.0% |                            1.3 s |
| 250                         |                        3.1% |          0.0% |          0.0% |                            1.7 s |
| 400                         |                        2.9% |          0.0% |          0.0% |                           13.2 s |
| 450                         |                        4.0% |          0.0% |        100.0% |                             50 s |
| 499                         |                        6.0% |          0.3% |        100.0% |                             50 s |
| 750                         |                       11.5% |          1.1% |        100.0% |                             50 s |
| 1000                        |                       38.2% |         20.8% |        100.0% |                             50 s |
| 1250                        |                       72.4% |         65.2% |        100.0% |                             50 s |
| 1498                        |                       99.4% |         98.5% |        100.0% |                             50 s |

**Result:** solved by ~1500 iterations. The policy swings up from hanging and
holds both poles up in 98.5% of envs.

- **The two skills arrived in sequence:** balancing from upright starts was
  learned between iterations 400 and 450. Swing-up into a held balance only
  improved after that.
- **At 500 iterations it looks worse than Experiment 3:** hanging-start
  both-up time was 6% against 36.8%. Experiment 3's number came from wobbling
  near the top and it never held balance.
- **Hypothesis for the early swing-up slump (not tested):** the observation
  doesn't say whether an episode started upright. While upright starts kept
  falling and terminating, near-upright states looked low-value to the critic
  in every episode, which discouraged swinging up. Once balancing worked, those
  states became high-value and swing-up followed.

### 5. Domain randomization, latency, sensor noise and pushes

**Goal:** make the swing-up policy robust to the physical details that are
hard to control on real hardware. Rail and pole lengths stay fixed.

**Changes (both tasks; randomized values are drawn once per env at startup):**

| Item | Mechanism | Range |
| --- | --- | --- |
| Pole mass + inertia | `dr.pseudo_inertia` on `pole_1`, `pole_2` (independent), `alpha_range=(-0.11, 0.09)` | mass ×0.8–1.2 |
| Pole COM shift | same event, `t3_range` | ±5 cm along the pole |
| Hinge damping | `dr.joint_damping` (abs) on both hinges | 0–0.005 N·m·s/rad |
| Cart mass | `dr.pseudo_inertia` on `cart` | ×0.8–1.2 |
| Slider dry friction | `dr.joint_friction` (abs) | 0–1 N |
| Slider damping | `dr.joint_damping` (abs) | 0–0.5 N·s/m |
| Motor strength | custom `actuator_gear_scale` event | gear ×0.9–1.2 (9–12 N max) |
| Motor latency | `XmlActuatorCfg(delay_min_lag=0, delay_max_lag=2)`, resampled every physics step | 0–20 ms |
| Encoder offset | `dr.encoder_bias`; actor position terms read `joint_pos_biased` | slider ±5 mm, hinges ±0.01 rad |
| Sensor noise | `UniformNoiseCfg` per actor term, every step | cart pos ±5 mm, pole cos/sin ±0.01, cart vel ±0.05 m/s, hinge vel ±0.2 rad/s |
| Cart pushes | `apply_body_impulse` (`mode="step"`) on `cart` | ±10 N per component, 0.05–0.15 s, every 3–6 s |

- **Observations:** split into actor and critic groups. The actor gets offset,
  noisy measurements; the critic keeps the true state.
- **`fallen` arm delay:** the termination now arms only after both poles hold
  cos > 0.95 for 1.0 s, then terminates on cos < 0.8 as before. Upright starts
  that fall before arming continue as swing-ups. 1.0 s sits below the 1.1–1.7 s
  mean time to first fall of early policies in Experiments 3–4, so early falls
  still terminate. Hanging starts are unchanged.
- **Play:** pushes are removed (`cartpole-play` pushes from the keyboard) and
  noise is off, but startup randomization and encoder offsets still apply.

> Gotchas found while implementing:
> - `dr.effort_limits` raises `TypeError` on effort-commanded XML actuators,
>   hence the custom gear event.
> - `dr.encoder_bias` only shows up in observations that read
>   `joint_pos_biased`. Neither `pole_angle_cos_sin` nor `joint_pos_rel`
>   (`biased=False`) did by default.

**Run:** `logs/rsl_rl/cartpole/2026-10-23_15-07-22_exp5_dr`. 16384 envs,
3000 iterations, ~5.9e5 steps/s (Experiment 4: ~7.6e5), 45 min.

**Evaluation:** same method as Experiment 4, 2048 envs, one 50 s episode.

- **nominal:** no randomization, latency, noise or pushes.
- **randomized:** the training distribution without pushes.
- **+pushes:** training pushes on.
- **+A/D:** a +10 N, 0.15 s push every 4 s, matching `cartpole-play`'s D key.

"Held" means both poles had cos > 0.95 for at least 90% of the last 30 s.

| Policy | Condition | Hanging: both up | Hanging: held | Upright: held | Upright: mean first fall |
| --- | --- | ---: | ---: | ---: | ---: |
| Exp. 4 `model_1498` | nominal | 99.2% | 97.9% | 100% | 50 s |
| Exp. 4 `model_1498` | randomized | 21.6% | 0.0% | 0.0% | 3.1 s |
| Exp. 4 `model_1498` | +pushes | 19.2% | 0.0% | 0.0% | 2.8 s |
| Exp. 4 `model_1498` | +A/D | 12.7% | 0.0% | 0.0% | 2.5 s |
| Exp. 5 `model_2999` | nominal | 99.8% | 99.8% | 100% | 50 s |
| Exp. 5 `model_2999` | randomized | 91.0% | 69.6% | 67.3% | 28.9 s |
| Exp. 5 `model_2999` | +pushes | 73.0% | 15.4% | 15.8% | 12.5 s |
| Exp. 5 `model_2999` | +A/D | 24.6% | 0.0% | 0.0% | 4.3 s |

Training progress, randomized condition, hanging starts:

| Iteration | 500 | 1000 | 1500 | 2000 | 2500 | 2999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Both up | 16.7% | 25.3% | 37.6% | 79.1% | 88.5% | 91.0% |
| Held | 0.0% | 0.0% | 0.0% | 30.5% | 60.0% | 69.6% |

Held rate by quartile of each randomized parameter, `model_2999`, randomized
condition, 8192 hanging starts (overall 69.3%). Parameters with a spread under
10 points are omitted: slider and hinge damping, pole COM shift, slider and
hinge_1 encoder offsets, pole_1 mass.

| Parameter | Q1 (low) | Q2 | Q3 | Q4 (high) |
| --- | ---: | ---: | ---: | ---: |
| hinge_2 encoder offset | 66.2% | 85.4% | 79.1% | 46.4% |
| motor gear | 55.4% | 69.5% | 76.2% | 76.0% |
| cart mass | 73.2% | 76.4% | 70.9% | 56.6% |
| slider friction | 68.9% | 76.3% | 71.4% | 60.5% |

**Result:**

- **Nominal performance is preserved:** 99.8% held, against 97.9% for
  Experiment 4.
- **Robustness improved a lot but isn't complete:** under randomization, held
  went from 0% to ~70%. Experiment 4's policy fails completely under
  randomization.
- **Still improving at 3000 iterations:** held went 60.0% → 69.6% over the
  last 500.
- **Biggest failure driver is the hinge_2 encoder offset:** envs whose offset
  is near either end of ±0.01 rad fail far more often. A memoryless MLP can't
  tell a constant angle offset from a real tilt, so it balances around a
  slightly wrong upright and the cart has to keep moving to hold it.
- **Low force-to-mass ratio also hurts:** weak motors, heavy carts and high
  slider friction all fail more often.
- **Pushes aren't handled yet:** random pushes cut held to ~15%, and the full
  A/D push every 4 s isn't recovered from (0% held).

**Resumed training:** `logs/rsl_rl/cartpole/2026-10-23_16-29-27_exp5_dr_resume`,
resumed from `model_2999` for 1500 more iterations (to 4498).

Randomized condition, hanging starts. All checkpoints were scored on the same
startup randomization draw:

| Iteration | 2999 | 3500 | 3600 | 3700 | 3800 | 3900 | 4000 | 4100 | 4150 | 4200 | 4250 | 4300 | 4350 | 4400 | 4450 | 4498 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Held | 68.3% | 75.7% | 79.7% | 80.1% | 80.4% | 79.1% | 71.7% | 72.2% | 79.2% | 78.7% | 80.4% | 75.1% | 70.8% | 79.9% | 67.2% | 67.1% |

The top four checkpoints were re-scored on a fresh randomization draw:

| Checkpoint | Randomized hanging: held | Randomized upright: held | Nominal hanging: held | +A/D hanging: both up |
| --- | ---: | ---: | ---: | ---: |
| 3700 | 79.2% | 81.1% | 100.0% | 25.8% |
| 3800 | 79.5% | 80.7% | 100.0% | 26.4% |
| 4250 | 81.3% | 81.8% | 99.9% | 25.8% |
| 4400 | 81.4% | 82.5% | 98.8% | 25.3% |

**Result:**

- **Plateau:** held under randomization levels off around 80% from iteration
  ~3600 on.
- **Checkpoint noise:** individual checkpoints swing between 67% and 80%, so
  pick checkpoints by evaluation, not by recency. `model_4250` is the pick:
  81% under randomization and 99.9% nominal.
- **A/D pushes:** still not recovered from. The first push, at 4 s, knocks the
  poles over (mean first fall 4.4 s), and the policy swings them back up
  afterwards.

### 6. Realistic randomization ranges

**Goal:** pull the ranges that exceeded plausible hardware back into a
realistic band. The biggest Experiment 5 failure driver was the hinge_2
encoder offset at the edges of ±0.01 rad. Envs whose offset fell within
±0.005 rad held 79–85%.

**Changes from Experiment 5:**

| Item | Experiment 5 | Experiment 6 | Reasoning |
| --- | --- | --- | --- |
| Hinge encoder offset | ±0.01 rad (±0.57°) | ±0.003 rad (±0.17°) | Zeroing encoders with the pendulum hanging at rest gets within ~0.1–0.2°. |
| Pole cos/sin noise | ±0.01 | ±0.003 | About the resolution of a 4096-count encoder (0.09°/count). |
| Motor latency | 0–20 ms, resampled every physics step | 0–20 ms, held per env for 5 s (`delay_update_period=500`) | A roughly constant control-loop delay instead of per-step jitter. |
| Pole mass + inertia | ×0.8–1.2 | ×0.95–1.05 (`alpha_range=(-0.026, 0.024)`) | Poles can be weighed. |

Everything else, including the pole COM shift and the pushes, is unchanged.

Checked: pole masses span 0.095–0.105 kg, hinge offsets reach ±0.003 rad, and
actor-critic differences on the pole sin features reach 0.006 (offset plus
noise). Over 2 s, 27% of envs changed lag, against an expected
40% × 2/3 ≈ 27% for a 5 s period.

**Run:** `logs/rsl_rl/cartpole/2026-10-23_17-29-39`. Trained from scratch,
16384 envs, 2000 iterations, ~5.7e5 steps/s, 31 min.

**Evaluation:** same method and conditions as Experiment 5, all under the
Experiment 6 distribution. Experiment 5's `model_4250`, trained on the wider
ranges, is included for comparison. "+pushes" uses the training pushes.

Randomized condition, hanging starts:

| Iteration | 500 | 1000 | 1200 | 1400 | 1500 | 1600 | 1700 | 1750 | 1800 | 1850 | 1900 | 1950 | 1999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Both up | 15.9% | 26.6% | 31.9% | 51.7% | 71.5% | 78.0% | 88.2% | 92.7% | 93.0% | 94.5% | 93.0% | 95.7% | 95.4% |
| Held | 0.0% | 0.0% | 0.0% | 0.0% | 4.5% | 19.3% | 55.5% | 75.3% | 78.5% | 82.8% | 74.4% | 86.2% | 86.5% |

| Policy | Condition | Hanging: both up | Hanging: held | Upright: held | Upright: mean first fall |
| --- | --- | ---: | ---: | ---: | ---: |
| Exp. 6 `model_1999` | nominal | 99.8% | 100.0% | 100.0% | 50 s |
| Exp. 6 `model_1999` | randomized | 95.5% | 86.2% | 85.4% | 38.8 s |
| Exp. 6 `model_1999` | +pushes | 77.1% | 24.5% | 22.7% | 14.5 s |
| Exp. 6 `model_1999` | +A/D | 27.2% | 0.0% | 0.0% | 4.4 s |
| Exp. 5 `model_4250` | nominal | 99.8% | 100.0% | 100.0% | 50 s |
| Exp. 5 `model_4250` | randomized | 97.7% | 93.6% | 93.2% | 45.8 s |
| Exp. 5 `model_4250` | +pushes | 80.9% | 34.0% | 35.2% | 17.8 s |
| Exp. 5 `model_4250` | +A/D | 26.3% | 0.0% | 0.0% | 4.4 s |

Held rate by quartile of each randomized parameter, randomized condition,
8192 hanging starts. Parameters with a spread under 5 points for both policies
are omitted: pole masses, pole COM shifts, hinge_1 and slider encoder offsets.

| Parameter | `model_1999` Q1 → Q4 | `model_4250` Q1 → Q4 |
| --- | --- | --- |
| motor gear (low → high) | 75.1 / 87.9 / 90.2 / 87.6% | 81.4 / 95.2 / 98.0 / 98.9% |
| slider friction | 90.6 / 92.4 / 87.0 / 70.7% | 91.7 / 95.5 / 94.3 / 91.9% |
| slider damping | 89.4 / 88.5 / 84.4 / 78.5% | 92.2 / 93.9 / 93.8 / 93.6% |
| cart mass | 83.9 / 88.1 / 90.5 / 78.3% | 99.3 / 99.1 / 95.7 / 79.5% |
| hinge_2 encoder offset | 86.2 / 88.3 / 84.6 / 81.6% | 94.6 / 93.3 / 92.8 / 92.9% |

**Result:**

- **Encoder offset is no longer a failure driver:** hinge_2 offset spread fell
  from 39 points (Experiment 5) to 7 points for `model_1999` and 2 points for
  `model_4250`.
- **Faster to train:** the fresh run passed Experiment 5's 2999-iteration
  level (69.6% held on its own distribution) by iteration 1700, and reached
  86.5% by 1999.
- **Still improving at 2000:** held went 75.3% → 86.5% over the last 250
  iterations.
- **Wider-range training transfers well:** on the Experiment 6 distribution,
  Experiment 5's `model_4250` (4250 iterations on wider ranges) scores 93.6%
  held, ahead of `model_1999`. Training on a superset of the target
  distribution plus more iterations didn't hurt.
- **Remaining failures are force-limited:** the weakest motor quartile
  (75.1%) and the highest slider friction quartile (70.7%) fail most for
  `model_1999`. The heaviest cart quartile fails most for `model_4250` (79.5%).
- **A/D pushes:** still knock both policies over (0% held); they swing back up
  afterwards.

### 7. Triple pendulum

**Goal:** add a third link and train a fresh policy on it.

**Changes:**

- **Model:**
  - `_get_spec(n_poles)` loads the 2-pole XML and chains `pole_3` with
    `hinge_3` onto `pole_2`'s tip, using the same `pole` default class
    (1 m, 0.1 kg, damping 2e-6).
  - The cart and rails are raised to z = n_poles m (3 m), so all three poles
    hanging clear the floor.
  - The 2-pole model is unchanged.
- **Tasks:** new `Cartpole3-Balance` and `Cartpole3-Swingup`, logging to
  `logs/rsl_rl/cartpole3/`. The 2-pole tasks keep their IDs and behaviour.
  Everything else carries over from Experiment 6 unchanged: reward, upright
  starts, fall termination, randomization, noise, latency and pushes
  (±10 N).
- **Motor: 30 N** (gear 30, randomized ×0.9–1.2 to 27–36 N), chosen from
  the feasibility check below. The 2-pole tasks keep 10 N. The task config
  functions take `max_force_n`, which `_get_spec` writes to the actuator gear.
- **Generalized config:**
  - hinges are selected by `hinge_.*` and pole bodies by `pole_.*`
  - initial states set `hinge_1` (0 or π) plus `hinge_[2-9] = 0`
  - viewer distance is `3 × n_poles`
- **Observations:** 11-D: cart position, cos/sin of 3 hinges, cart velocity,
  3 hinge velocities.
- **Training:** fresh policies, trained from scratch.

**Feasibility check** (CPU MuJoCo, linearized at upright, discrete LQR with
the motor saturating at the gear limit):

| Poles | Gear (max N) | Fastest unstable mode | Max tip tilt recovered, 20 Hz | 50 Hz |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 10 | 7.0 1/s (142 ms) | 5.8° | 5.8° |
| 3 | 10 | 9.9 1/s (101 ms) | 1.3° | 1.1° |
| 3 | 20 | 9.9 1/s (101 ms) | 2.4° | 2.1° |
| 3 | 40 | 9.9 1/s (101 ms) | 4.4° | 3.8° |

- **The catch window is tiny with 10 N:** ~1.3° of tip tilt, against 5.8° for
  two poles.
- **No NaNs with RK4:** 2000 steps of random actions on 16384 envs produced
  no non-finite observations (peak |qvel| 63.7).

Fraction of 100 upright reset states (±0.034 rad per hinge, ±0.01 velocity
noise, the training reset distribution) held for 10 s by a saturated LQR.
The "delay-aware" controller is designed for the latency it runs with,
using an augmented state `[x, u_prev]`:

| Motor | 20 Hz, 0 ms | 20 Hz, 10 ms | 20 Hz, 20 ms | 50 Hz, 0 ms | 50 Hz, 10 ms | 50 Hz, 20 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 N | 58% | 47% | 38% | 57% | 49% | 47% |
| 20 N | 99% | 93% | 84% | 99% | 92% | 94% |
| 30 N | 100% | 99% | 89% | 100% | 100% | 100% |

- **Force is the binding constraint:** at 10 N even an ideal linear
  controller, with no sensor noise, offsets or randomization, loses 42–62% of
  the reset states. At 20 N it holds 84–99%.
- **Control rate matters only with latency:** 20 Hz and 50 Hz tie at 0 ms.
  At 20 ms, 50 Hz recovers 9–11 more points.
- **Latency-unaware control fails outright:** a plain LQR, not designed for
  the delay, held 0% at every gear with 20 ms latency. The policy doesn't
  observe its previous action, which the delay-aware controller uses.

**Decision:** 30 N, 20 Hz, no previous-action observation. At 30 N the
delay-aware LQR holds 89–100% of upright resets.

**Evaluation** for all runs below uses Experiment 5's method (2048 envs, one
50 s episode, `fallen` removed), in the randomized condition unless noted.

#### Run 1: fall-termination arm delay 1.0 s

`logs/rsl_rl/cartpole3/2026-10-23_19-10-58`. Fresh, 16384 envs,
2000 iterations.

| Iteration | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 1999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Randomized hanging: both up | 9.7% | 12.4% | 16.2% | 17.3% | 17.0% | 18.4% | 18.9% | 19.2% |
| Randomized hanging: held | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| Randomized upright: mean first fall | 0.5 s | 0.6 s | 0.7 s | 0.7 s | 0.7 s | 0.8 s | 0.8 s | 0.9 s |

`model_1999` under nominal physics (no randomization, latency, noise or
pushes): both up 20.7%, held 0%, mean first fall from upright 1.0 s.

Training curves (50-iteration means):

| Iteration | 250 | 500 | 1000 | 1500 | 1999 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-step reward | 0.449 | 0.467 | 0.492 | 0.489 | 0.474 |
| Policy std | 0.064 | 0.052 | 0.056 | 0.059 | 0.059 |
| `fallen` terminations per log step | 0.01 | 0.02 | 0.28 | 0.79 | 1.78 |
| Episode length | 1000 | 999 | 986 | 961 | 914 |

**Result:** no balancing. The policy hovers near the top (~19% both up) and
never holds, even under nominal physics. The reward is flat from
~iteration 1000.

- **The fall termination barely fired:** upright starts fall within 0.5–0.9 s,
  but `fallen` only arms after 1.0 s upright. The 1.0 s delay came from
  2-pole policies' early fall times (1.1–1.7 s); 3 poles fall faster (101 ms
  e-fold against 142 ms). Training therefore ran close to Experiment 3's
  setup, with no strong fall signal, which also plateaued at hovering.
  Terminations are rising late (0.01 → 1.78 per iteration) as more upright
  starts survive the full second.
- **Exploration collapsed early:** policy std was 0.064 by iteration 250. In
  Experiment 4 it was 0.215 at 100 and 0.146 at 300.

**Change for run 2:**

- **3-pole fall arm delay: 1.0 s → 0.3 s.** `cartpole_swingup_env_cfg`
  takes `fall_arm_after_s`; `Cartpole3-Swingup` passes 0.3, and the 2-pole
  tasks keep 1.0.
- **Check:** with zero action from upright starts, `fallen` ended 100% of
  3-pole episodes within 3 s (median 0.65 s). At 1.0 s, the same check ended
  4.0% of 2-pole episodes.

#### Run 2: arm delay 0.3 s

`logs/rsl_rl/cartpole3/2026-10-23_20-08-03`. Fresh, 16384 envs,
3000 iterations.

| Iteration | 250 | 500 | 1000 | 1500 | 2000 | 2500 | 2750 | 2999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hanging: both up | 7.7% | 9.1% | 8.5% | 7.5% | 6.9% | 2.7% | 3.5% | 3.7% |
| Hanging: held / ever held 2 s | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |
| Upright: mean first fall | 0.2 s | 0.2 s | 0.2 s | 0.2 s | 0.2 s | 0.2 s | 0.2 s | 0.2 s |

Training curves (50-iteration means), `fallen` vs `time_out` episode ends:

| Iteration | 250 | 1000 | 2000 | 2999 |
| --- | ---: | ---: | ---: | ---: |
| `fallen` / `time_out` per log step | 0.63 / 16.2 | 0.58 / 16.2 | 1.26 / 15.5 | 2.80 / 15.1 |
| Per-step reward | 0.435 | 0.466 | 0.460 | 0.434 |
| Policy std | 0.106 | 0.066 | 0.074 | 0.080 |

**Result:** worse than run 1. Swing-up both-up time fell to ~4%, and upright
starts now leave cos > 0.95 within 0.2 s.

**Diagnosis:**

- **The arm delay is a loophole:** with zero action, upright starts leave
  cos > 0.95 after 0.48 s on average. The trained policy leaves in 0.2 s, so
  it actively knocks the poles down before the 0.3 s arm time.
- **Why that pays off:** an episode that never arms can't end on `fallen`. The
  policy keeps collecting swing-up reward instead of risking a termination it
  can't yet avoid by balancing. Only ~4% of episode ends were `fallen` early
  on, although 25% of episodes start upright.
- **Run 1 fits the same pattern:** its upright starts also fell (0.5–0.9 s)
  before its 1.0 s arm time.
- **Experiment 4 had no escape:** its rule fired as soon as an upright start
  fell, with no arm delay, and that setup learned to balance. The 2-pole runs
  of Experiments 5–6 learned despite the 1.0 s delay. Presumably the 2-pole
  policy could balance within the second before the loophole mattered (not
  tested).

**Change for run 3:**

- **3-pole fall arm delay: 0.3 s → 0.** `Cartpole3-Swingup` now ends an
  upright-start episode as soon as any pole's cos < 0.8, matching
  Experiment 4. The 2-pole tasks keep 1.0 s.
- **Check:** a policy that throws the poles down at full force from upright
  is now ended by `fallen` in 100% of episodes (median 0.20 s). Hanging
  starts under random actions never trigger `fallen`.

#### Run 3: no arm delay

`logs/rsl_rl/cartpole3/2026-10-23_23-18-09`. Fresh, 16384 envs,
1000 iterations. "Time to fall" is the time to the `fallen` threshold (any
pole cos < 0.8) from upright starts; zero action takes 0.65 s (median).

| Iteration | 100 | 250 | 400 | 500 | 600 | 700 | 800 | 900 | 999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Upright: mean time to fall | 0.7 s | 0.7 s | 0.8 s | 0.8 s | 0.9 s | 0.9 s | 0.9 s | 0.9 s | 0.9 s |

| Iteration | 250 | 500 | 750 | 999 |
| --- | ---: | ---: | ---: | ---: |
| Hanging: all poles up | 10.4% | 11.8% | 14.4% | 15.9% |

- **Held or ever held 2 s:** 0% at every checkpoint, from both start types.
- **Nominal physics:** `model_999` gives the same result (upright time to
  fall 0.9 s, hanging all up 17.0%), so randomization isn't the blocker.
- **All runs at iteration ~1000 (upright: mean time to fall):** run 1 0.9 s,
  run 2 0.3 s, run 3 0.9 s.

Training curves (50-iteration means):

| Iteration | 100 | 250 | 500 | 750 | 999 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-step reward | 0.280 | 0.333 | 0.356 | ~0.36 | 0.373 |
| Episode length | 748 | 753 | 762 | ~757 | 753 |
| `fallen` / `time_out` per log step | 5.8 / 17.4 | 5.1 / 15.8 | 4.8 / 15.0 | ~5.7 / 17.3 | 5.0 / 14.8 |
| Policy std | 0.156 | 0.069 | 0.050 | 0.052 | 0.052 |

**Result:** the loophole is closed, but there's still no balancing.

- **No more deliberate drops:** upright starts survive 0.9 s, a little longer
  than zero action's 0.65 s, against 0.2 s in run 2.
- **Falls aren't getting rarer:** the `fallen` share of episode ends stays at
  ~25% from iteration 100 to 999. That is essentially every upright start,
  and episode length is flat at ~750. In Experiment 4 (2 poles), upright
  time to first fall went 1.3 → 1.7 → 13.2 s at iterations 100/250/400.
- **Exploration collapsed by ~500:** policy std was 0.05, so more of the
  same training is unlikely to find the precise feedback balance needs.
- **Slow upward trends only:** swing-up all-up rose 10% → 16% and reward
  0.33 → 0.37 over 1000 iterations.

**Next:** add the previous action to the actor's observations. The
feasibility check shows that with motor latency, a controller needs its
previous command to balance: 0% held without it, 89–100% with it at 30 N.
Raising `entropy_coef` to slow the std collapse is queued after that.
