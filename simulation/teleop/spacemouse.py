import numpy as np
import pyspacemouse
from ..core.teleop import TeleopBase
from ..core.registry import TELEOP_REGISTRY


@TELEOP_REGISTRY.register("spacemouse")
class SpaceMouseTeleop(TeleopBase):
    def __init__(self, lin_vel_scale=0.001, rot_vel_scale=0.001):
        """
        SpaceMouse teleoperation that outputs end-effector velocities (6D twist).
        Uses pyspacemouse to read 6DOF motion data.

        - Translation: move the cap (X/Y/Z)
        - Rotation: twist the cap (roll/pitch/yaw)
        - Buttons: can be mapped if needed
        """
        super().__init__()
        self.lin_vel_scale = lin_vel_scale   # scale factor for translations
        self.rot_vel_scale = rot_vel_scale   # scale factor for rotations

        # Try to initialize the spacemouse
        if not pyspacemouse.open():
            raise RuntimeError("Could not open 3Dconnexion SpaceMouse. Is it connected?")

    def get_action(self, obs=None):
        """
        Reads the SpaceMouse events and returns a 6D twist:
        [vx, vy, vz, wx, wy, wz]
        """
        
        try:
            event = pyspacemouse.read()
        except:
            event = None

        if event is None:
            return np.zeros(6)

        # event.trans -> (x, y, z) integers
        # event.rot -> (rx, ry, rz) integers
        tx, ty, tz = event.x, event.y, event.z
        rx, ry, rz = event.roll, event.pitch, event.yaw

        vx = tx * self.lin_vel_scale
        vy = ty * self.lin_vel_scale
        vz = tz * self.lin_vel_scale

        wx = rx * self.rot_vel_scale
        wy = ry * self.rot_vel_scale
        wz = rz * self.rot_vel_scale

        return np.array([vx, vy, vz, wx, wy, wz])
