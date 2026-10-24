# mujoco-learning

Reinforcement learning experiments in [MuJoCo](https://mujoco.org/), built on
[mjlab](https://github.com/mujocolab/mjlab) and RSL-RL (PPO).

Each experiment lives in its own package under `projects/`. The repo is a
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): every
project shares one lockfile (`uv.lock`) and one virtual environment (`.venv`).

```
mujoco-learning/
├── pyproject.toml          # workspace root; depends on every project
├── uv.lock
├── logs/                   # training runs (git-ignored)
└── projects/
    └── cartpole/
        ├── pyproject.toml  # registers tasks via the `mjlab.tasks` entry point
        └── src/cartpole/
```

## Setup

```sh
uv sync
uv run list-envs   # lists every registered task, including mjlab's built-ins
```

Run every command from the repo root. Training writes to
`logs/rsl_rl/<experiment>/<timestamp>/`.

## Projects

### cartpole

A pendulum of two (or three) links on a sliding cart. The policy pushes the
cart to keep every link upright. See
[`projects/cartpole/RESEARCH.md`](projects/cartpole/RESEARCH.md) for the
experiment log.

| Task ID             | Goal                                                    |
| ------------------- | ------------------------------------------------------- |
| `Cartpole-Balance`  | Keep a nearly upright 2-link pendulum balanced          |
| `Cartpole-Swingup`  | Swing the 2-link pendulum up from hanging, then balance |
| `Cartpole3-Balance` | Same as `Cartpole-Balance` with 3 links and a 30 N motor |
| `Cartpole3-Swingup` | Same as `Cartpole-Swingup` with 3 links and a 30 N motor |

2-link runs log to `logs/rsl_rl/cartpole/` and 3-link runs to
`logs/rsl_rl/cartpole3/`.

Train:

```sh
uv run train Cartpole-Balance
uv run train Cartpole-Swingup
uv run train Cartpole3-Swingup
```

Play a trained checkpoint. `cartpole-play` wraps mjlab's `play` command and
adds keyboard pushes: **A** pushes the cart left and **D** pushes it right.
To pull a body with the mouse, double-click it, then hold Ctrl and right-drag.

```sh
uv run cartpole-play Cartpole-Balance \
  --checkpoint-file logs/rsl_rl/cartpole/<timestamp>/model_499.pt
```

## Common options

Both `train` and `play` take `--help`. Some useful flags:

```sh
uv run train <Task> --env.scene.num-envs 1024 --agent.max-iterations 1000
uv run train <Task> --agent.logger wandb          # log metrics to W&B (default: tensorboard)
uv run play  <Task> --checkpoint-file <path.pt> --viewer viser   # browser viewer
uv run play  <Task> --agent random                # watch an untrained (random) policy
uv run tensorboard --logdir logs/rsl_rl
```

## Adding a project

1. Create `projects/<name>/pyproject.toml` and `projects/<name>/src/<name>/`.
   Use `projects/cartpole` as a template. The key part is the entry point
   that tells mjlab where to find your tasks:

   ```toml
   [project.entry-points."mjlab.tasks"]
   <name> = "<name>"
   ```

2. In `src/<name>/__init__.py`, call `mjlab.tasks.registry.register_mjlab_task(...)`
   for each task.
3. Add the project to the root `pyproject.toml`:

   ```toml
   dependencies = ["cartpole", "<name>"]

   [tool.uv.sources]
   <name> = { workspace = true }
   ```

4. Run `uv sync`. Your tasks then show up in `uv run list-envs`.
