# imitation_framework/core/env_manager.py

import pybullet as p
import pybullet_data
import time
import cv2
import numpy as np
from enum import Enum

from .robot import RobotBase
from .task import TaskBase
from .teleop import TeleopBase


class GUIMode(Enum):
    FULL = "full"          # PyBullet GUI window (interactive)
    IMAGE_ONLY = "image"   # Custom image-based renderer
    DISABLED = "disabled"  # Headless, no rendering


class EnvManager:
    def __init__(
        self,
        robot: RobotBase,
        task: TaskBase,
        teleop: TeleopBase = None,
        gui_mode: GUIMode = GUIMode.FULL,
        timestep: float = 1.0 / 240.0,
    ):
        self.robot = robot
        self.task = task
        self.timestep = timestep
        self.teleop = teleop
        self.gui_mode = gui_mode

        # Start simulation
        if gui_mode == GUIMode.FULL:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -9.8)

        # Camera for IMAGE_ONLY mode
        if gui_mode == GUIMode.IMAGE_ONLY:
            self._setup_camera()

        self.reset()

    def _setup_camera(self):
        """Define a fixed camera for IMAGE_ONLY rendering"""
        self.cam_target = [0.4, 0.0, 0.0]   # look at table center
        self.cam_distance = 1.2
        self.cam_yaw = 45
        self.cam_pitch = -30
        self.cam_up_axis = 2

        self.width = 640
        self.height = 480
        self.fov = 60
        self.near = 0.01
        self.far = 5.0

    def reset(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timestep)

        p.loadURDF("plane.urdf")
        self.robot.reset(self.client_id)
        self.task.reset(self.client_id, self.robot)

        return self.task.get_obs(self.robot)

    def step(self, action=None):
        if action is None and self.teleop is not None:
            action = self.teleop.get_action()

        self.robot.apply_action(action)
        p.stepSimulation(physicsClientId=self.client_id)

        obs = self.task.get_obs(self.robot)
        reward = self.task.reward(self.robot)
        done = self.task.check_success(self.robot)

        return obs, reward, done

    def close(self):
        p.disconnect(self.client_id)
        if self.gui_mode == GUIMode.IMAGE_ONLY:
            cv2.destroyAllWindows()

    def render(self, sleep=True):
        if self.gui_mode == GUIMode.DISABLED:
            return
        elif self.gui_mode == GUIMode.FULL:
            if sleep:
                time.sleep(self.timestep)
            return
        elif self.gui_mode == GUIMode.IMAGE_ONLY:
            # Camera matrices
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.cam_target,
                distance=self.cam_distance,
                yaw=self.cam_yaw,
                pitch=self.cam_pitch,
                roll=0,
                upAxisIndex=self.cam_up_axis,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.fov,
                aspect=float(self.width) / self.height,
                nearVal=self.near,
                farVal=self.far,
            )

            _, _, px, _, _ = p.getCameraImage(
                self.width,
                self.height,
                view_matrix,
                proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            rgb = np.array(px, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]

            cv2.imshow("PyBullet Viewer", rgb)
            cv2.waitKey(1)

            if sleep:
                time.sleep(self.timestep)
