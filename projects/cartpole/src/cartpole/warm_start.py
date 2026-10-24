"""Convert an n-pole cartpole checkpoint to a policy for more poles.

Actor and critic observations are laid out as
``[cart_pos, cos(hinge_1..n), sin(hinge_1..n), cart_vel, vel(hinge_1..n)]``,
so adding poles inserts columns into each network's first layer. Existing
weights move to their new columns and the new poles' columns start at zero,
so the converted policy initially ignores the added poles and otherwise acts
exactly like the source policy. Adam moments are expanded the same way. The
iteration counter is reset to 0 so training on the new task starts fresh.

Usage:
    uv run cartpole-warm-start SRC.pt DST.pt --src-poles 2 --dst-poles 3
"""

import argparse
from pathlib import Path

import torch


def _obs_index_map(src_poles: int, dst_poles: int) -> list[int]:
    """Destination column for each source observation column."""
    s, d = src_poles, dst_poles
    return (
        [0]
        + [1 + i for i in range(s)]
        + [1 + d + i for i in range(s)]
        + [1 + 2 * d]
        + [2 + 2 * d + i for i in range(s)]
    )


def _expand_columns(t: torch.Tensor, index_map: list[int], dst_dim: int) -> torch.Tensor:
    out = torch.zeros(t.shape[0], dst_dim, dtype=t.dtype, device=t.device)
    out[:, index_map] = t
    return out


def convert(src: Path, dst: Path, src_poles: int, dst_poles: int) -> None:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    src_dim, dst_dim = 3 * src_poles + 2, 3 * dst_poles + 2
    index_map = _obs_index_map(src_poles, dst_poles)

    for key in ("actor_state_dict", "critic_state_dict"):
        w = ck[key]["mlp.0.weight"]
        assert w.shape[1] == src_dim, f"{key}: expected {src_dim} inputs, got {w.shape[1]}"
        ck[key]["mlp.0.weight"] = _expand_columns(w, index_map, dst_dim)

    # Adam moments for the two first-layer weights (actor and critic).
    first_layer_shape = (ck["actor_state_dict"]["mlp.0.weight"].shape[0], src_dim)
    expanded = 0
    for state in ck["optimizer_state_dict"]["state"].values():
        if tuple(state["exp_avg"].shape) == first_layer_shape:
            state["exp_avg"] = _expand_columns(state["exp_avg"], index_map, dst_dim)
            state["exp_avg_sq"] = _expand_columns(state["exp_avg_sq"], index_map, dst_dim)
            expanded += 1
    assert expanded == 2, f"expected 2 first-layer optimizer states, found {expanded}"

    ck["iter"] = 0
    ck["infos"] = {"env_state": {"common_step_counter": 0}}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    print(f"Wrote {dst}: {src_poles}-pole ({src_dim} obs) -> {dst_poles}-pole ({dst_dim} obs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--src-poles", type=int, default=2)
    parser.add_argument("--dst-poles", type=int, default=3)
    args = parser.parse_args()
    convert(args.src, args.dst, args.src_poles, args.dst_poles)
