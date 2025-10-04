# imitation_framework/core/env_manager.py

import pybullet as p
import pybullet_data
import time
from .robot import RobotBase
from .task import TaskBase
from .teleop import TeleopBase

class EnvManager:
    def __init__(self, robot: RobotBase, task: TaskBase, gui=True, timestep=1. / 240.):
        self.robot = robot
        self.task = task
        self.timestep = timestep

        # Start simulation
        if gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -9.8)

        self.reset()

    def reset(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timestep)

        plane_id = p.loadURDF("plane.urdf")
        self.robot.reset(self.client_id)
        self.task.reset(self.client_id, self.robot)

        obs = self.task.get_obs(self.robot)
        return obs

    def step(self, action):
        self.robot.apply_action(action)
        p.stepSimulation(physicsClientId=self.client_id)

        obs = self.task.get_obs(self.robot)
        reward = self.task.reward(self.robot)
        done = self.task.check_success(self.robot)

        return obs, reward, done

    def close(self):
        p.disconnect(self.client_id)

    def render(self, sleep=True):
        if sleep:
            time.sleep(self.timestep)
