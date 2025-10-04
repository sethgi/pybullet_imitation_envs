
from abc import ABC, abstractmethod
import pybullet as p
import numpy as np

class RobotBase(ABC):
    def __init__(self):
        self.robot_id = None
        self._max_force = 200

    @abstractmethod
    def reset(self, client_id):
        """Load URDF and reset robot state."""
        pass


    def apply_action(self, end_eff_velocity, damping=1e-4, joint_name=None):
        """
        Cartesian control via Jacobian pseudoinverse.

        delta_pos: [dx, dy, dz] in world frame
        delta_rot_euler: [droll, dpitch, dyaw] (small orientation change, radians)
        step_size: scaling factor for applying the delta
        damping: damping factor for stability (damped least squares)
        joint_name: optional string, if provided selects the link by name,
                    otherwise uses the last joint (end-effector)
        """
        # Get joint states
        joint_states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.client_id)
        q = [s[0] for s in joint_states]
        dq = [s[1] for s in joint_states]

        # Pick which link to control
        if joint_name is not None:
            # Find link index by name
            ee_link = None
            num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
            for j in range(num_joints):
                info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
                if info[12].decode("utf-8") == joint_name:  # joint/link name
                    ee_link = j
                    break
            if ee_link is None:
                raise ValueError(f"Link with name '{joint_name}' not found in URDF")
        else:
            # Default to last controllable joint
            ee_link = self.joint_indices[-1]

        # Compute Jacobian
        zero_vec = [0.0] * len(self.joint_indices)
        jac_t, jac_r = p.calculateJacobian(
            self.robot_id,
            ee_link,
            localPosition=[0, 0, 0],
            objPositions=q,
            objVelocities=dq,
            objAccelerations=zero_vec,
            physicsClientId=self.client_id
        )

        J_t = np.asarray(jac_t)
        J_r = np.asarray(jac_r)
        J = np.concatenate((J_t, J_r), axis=0)
        JJt = J @ J.T
        lambda_2I = damping * np.eye(JJt.shape[0])
        if len(self.joint_indices) > 1:
            joint_vel = J.T @ np.linalg.pinv(JJt + lambda_2I) @ end_eff_velocity
        else:
            joint_vel = J.T @ end_eff_velocity

            
        p.setJointMotorControlArray(self.robot_id,
                                    self.controllable_joints,
                                    p.VELOCITY_CONTROL,
                                    targetVelocities=joint_vel,
                                    forces = [self._max_force] * (len(self.controllable_joints)))
        
    def get_state(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.client_id)
        q = [s[0] for s in joint_states]
        dq = [s[1] for s in joint_states]
        return {"q": q, "dq": dq}

    
