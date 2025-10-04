# imitation_framework/core/urdf_robot.py

from collections import defaultdict
import os
import pybullet as p
from ..core.robot import RobotBase
from ..core.registry import ROBOT_REGISTRY
from enum import Enum
import numpy as np

class RobotURDFs(Enum):
    KINOVA = "kinova/gen3.urdf"
    GANTRY_2DOF = "gantry_2dof/gantry.urdf"
    
DEFAULT_CONFIGS = defaultdict(lambda rob_name: None)
DEFAULT_CONFIGS[RobotURDFs.KINOVA] = [0.0, -1.0, 1.2, 1.5, 0.0, 1.2, 0.0]
    
@ROBOT_REGISTRY.register("urdf_robot")
class URDFRobot(RobotBase):
    def __init__(self, urdf_path, base_pos=(0, 0, 0), base_orn=(0, 0, 0, 1), fixed_base=True):
        super().__init__()
        self.urdf_path = urdf_path
        self.base_pos = base_pos
        self.base_orn = base_orn
        self.fixed_base = fixed_base
        self.joint_indices = []

    @staticmethod
    def create(robot: RobotURDFs, base_pos=(0, 0, 0), base_orn=(0, 0, 0, 1), fixed_base=True):
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
        robots_path = os.path.join(base_path, "assets", "robots")
        return URDFRobot(
            os.path.join(robots_path, robot.value),
            base_pos=base_pos,
            base_orn=base_orn,
            fixed_base=fixed_base
        )

    def reset(self, client_id):
        self.client_id = client_id
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            useFixedBase=self.fixed_base,
            physicsClientId=self.client_id
        )

        # Collect controllable joints (revolute/prismatic)
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        self.joint_indices = [
            j for j in range(num_joints)
            if p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
        ]

        # Look up default config if available
        # Figure out which enum this robot corresponds to
        default_q = None
        for rob_enum in RobotURDFs:
            if rob_enum.value in self.urdf_path:
                default_q = DEFAULT_CONFIGS[rob_enum]
                break

        if default_q is not None and len(default_q) == len(self.joint_indices):
            for j, q in zip(self.joint_indices, default_q):
                p.resetJointState(self.robot_id, j, q, 0.0, physicsClientId=self.client_id)
        else:
            # Fall back to all zeros
            for j in self.joint_indices:
                p.resetJointState(self.robot_id, j, 0.0, 0.0, physicsClientId=self.client_id)
                