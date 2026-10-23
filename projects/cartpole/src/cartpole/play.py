"""mjlab ``play`` with keyboard pushes on the cart in the native viewer.

A pushes the cart left, D pushes it right. Each key event applies the push for
``PUSH_DURATION_S``; key-repeat events from holding the key extend it.

Mouse pulls are built into mjlab's native viewer: double-click a body to select
it, then Ctrl+right-drag to pull it (Ctrl+left-drag applies torque).

mjlab's ``run_play`` constructs ``NativeMujocoViewer`` directly, so ``main``
swaps in ``CartPushViewer`` before delegating to mjlab's CLI.
"""

import time

import mjlab.scripts.play as mjlab_play
import mujoco
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import KEY_A, KEY_D

# Matches the policy's maximum actuator force (motor gear 10, ctrlrange +-1).
PUSH_FORCE_N = 10.0
PUSH_DURATION_S = 0.15

_KEY_DIRECTION = {KEY_A: -1.0, KEY_D: 1.0}
# MuJoCo's viewer also binds A and D to visualization toggles; undone on press.
_KEY_VIS_FLAG = {
    KEY_A: mujoco.mjtVisFlag.mjVIS_AUTOCONNECT,
    KEY_D: mujoco.mjtVisFlag.mjVIS_STATIC,
}


class CartPushViewer(NativeMujocoViewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._push_direction = 0.0
        self._push_until = 0.0
        self._slider_dof = -1

    def setup(self) -> None:
        super().setup()
        cartpole = self.env.unwrapped.scene["cartpole"]
        joint_ids, _ = cartpole.find_joints("slider")
        self._slider_dof = int(cartpole.indexing.joint_v_adr[joint_ids[0]])

    def _safe_key_callback(self, key: int) -> None:
        """Runs on the viewer thread, after MuJoCo has handled its own shortcuts."""
        direction = _KEY_DIRECTION.get(key)
        if direction is None:
            super()._safe_key_callback(key)
            return
        if self.viewer is not None:
            flags = self.viewer.opt.flags
            flags[_KEY_VIS_FLAG[key]] = 1 - flags[_KEY_VIS_FLAG[key]]
        self._push_direction = direction
        self._push_until = time.monotonic() + PUSH_DURATION_S

    def sync_viewer_to_env(self) -> None:
        # The parent overwrites qfrc_applied for this env (mouse force or zero).
        super().sync_viewer_to_env()
        if time.monotonic() < self._push_until:
            qfrc = self.env.unwrapped.sim.data.qfrc_applied
            qfrc[self.env_idx, self._slider_dof] += self._push_direction * PUSH_FORCE_N


def main() -> None:
    mjlab_play.NativeMujocoViewer = CartPushViewer
    mjlab_play.main()
