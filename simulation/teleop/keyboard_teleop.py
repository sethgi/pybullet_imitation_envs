import pybullet as p
import numpy as np
from ..core.teleop import TeleopBase


class KeyboardTeleop(TeleopBase):
    def __init__(self, lin_vel=0.05, ang_vel=0.2):
        """
        Keyboard teleoperation that outputs end-effector velocities (6D twist).
        W/S -> +X/-X linear velocity
        A/D -> +Y/-Y linear velocity
        R/F -> +Z/-Z linear velocity
        Q/E -> +roll/-roll angular velocity
        Z/C -> +pitch/-pitch angular velocity
        T/G -> +yaw/-yaw angular velocity
        """
        super().__init__()
        self.lin_vel = lin_vel   # m/s equivalent
        self.ang_vel = ang_vel   # rad/s equivalent

    def get_action(self, obs=None):
        """
        Reads the keyboard events and returns a 6D twist:
        [vx, vy, vz, wx, wy, wz]
        """
        keys = p.getKeyboardEvents()
        vx, vy, vz, wx, wy, wz = 0, 0, 0, 0, 0, 0

        # Translation velocities
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            vx += self.lin_vel
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            vx -= self.lin_vel
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            vy += self.lin_vel
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            vy -= self.lin_vel
        if ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
            vz += self.lin_vel
        if ord('f') in keys and keys[ord('f')] & p.KEY_IS_DOWN:
            vz -= self.lin_vel

        # Angular velocities
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            wx += self.ang_vel
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            wx -= self.ang_vel
        if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN:
            wy += self.ang_vel
        if ord('c') in keys and keys[ord('c')] & p.KEY_IS_DOWN:
            wy -= self.ang_vel
        if ord('t') in keys and keys[ord('t')] & p.KEY_IS_DOWN:
            wz += self.ang_vel
        if ord('g') in keys and keys[ord('g')] & p.KEY_IS_DOWN:
            wz -= self.ang_vel

        return np.array([vx, vy, vz, wx, wy, wz])
