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

### 7. Triple pendulum, warm-started from the 2-pole policy

**Goal:** add a third link and see how the 2-pole policy adapts when trained
on it.

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
  Everything else carries over from Experiment 6 unchanged:
  - reward, upright starts, fall termination
  - randomization, noise, latency, pushes
  - motor (gear 10, ±10 N)
- **Generalized config:**
  - hinges are selected by `hinge_.*` and pole bodies by `pole_.*`
  - initial states set `hinge_1` (0 or π) plus `hinge_[2-9] = 0`
  - viewer distance is `3 × n_poles`
- **Observations:** 11-D: cart position, cos/sin of 3 hinges, cart velocity,
  3 hinge velocities.
- **Warm start (`cartpole-warm-start`):**
  - Each network's first layer gets new zero columns for `cos(hinge_3)`,
    `sin(hinge_3)` and `vel(hinge_3)`.
  - Existing weights and their Adam moments move to their new column
    positions.
  - The iteration counter resets to 0.
  - The converted policy outputs exactly the 2-pole policy's actions (max
    difference 0.0 on 1024 random inputs) and ignores pole 3 until training
    changes those weights.
  - Source: Experiment 5's `model_4250` →
    `logs/rsl_rl/cartpole3/warm_start_exp5_model_4250/model_0.pt`.
  - Policy std carries over at 0.10, so exploration starts narrow.

**Feasibility check** (CPU MuJoCo, linearized at upright, discrete LQR with
the motor saturating at the gear limit):

| Poles | Gear (max N) | Fastest unstable mode | Max tip tilt recovered, 20 Hz | 50 Hz |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 10 | 7.0 1/s (142 ms) | 5.8° | 5.8° |
| 3 | 10 | 9.9 1/s (101 ms) | 1.3° | 1.1° |
| 3 | 20 | 9.9 1/s (101 ms) | 2.4° | 2.1° |
| 3 | 40 | 9.9 1/s (101 ms) | 4.4° | 3.8° |

- **The catch window is tiny with 10 N:** ~1.3° of tip tilt, against 5.8° for
  two poles. A faster control rate doesn't help; more force does.
- **Some upright starts may be uncatchable:** reset noise is ±0.034 rad
  (±1.9°) on each hinge, larger than the 1.3° window.
- **No NaNs with RK4:** 2000 steps of random actions on 16384 envs produced
  no non-finite observations (peak |qvel| 63.7).

**Result:** not yet trained.
