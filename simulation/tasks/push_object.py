import time
import pybullet as p
import pybullet_data
import numpy as np
from ..core.task import TaskBase
from ..core.registry import TASK_REGISTRY
import os
from enum import Enum
from scipy.spatial import cKDTree

class PushableObject(Enum):
    T = "t_object.urdf"

def get_body_aabb_points(body_id, client_id=0):
    pts = []
    visuals = p.getVisualShapeData(body_id, physicsClientId=client_id)
    for vis in visuals:
        link_index = vis[1]
        local_pos = np.array(vis[5])  # local offset
        local_orn = np.array(vis[6])  # local orientation (quat)
        dims = vis[3]                 # extents for box/sphere/cylinder, etc.
        
        # Get world transform of this link
        if link_index == -1:
            link_state = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
        else:
            link_state = p.getLinkState(body_id, link_index, physicsClientId=client_id)[:2]
        link_pos, link_orn = link_state

        # Combine transforms: link ∘ local
        world_pos, world_orn = p.multiplyTransforms(
            link_pos, link_orn,
            local_pos, local_orn
        )

        # For a box, dims = [half_x, half_y, half_z]
        if vis[2] == p.GEOM_BOX:
            half_extents = np.array(dims)
            for i in [0,1]:
                for j in [0,1]:
                    for k in [0,1]:
                        corner_local = [
                            -half_extents[0] if i==0 else half_extents[0],
                            -half_extents[1] if j==0 else half_extents[1],
                            -half_extents[2] if k==0 else half_extents[2],
                        ]
                        rot = np.array(p.getMatrixFromQuaternion(world_orn)).reshape(3,3)
                        pts.append(rot @ corner_local + np.array(world_pos))
        # For meshes, dims[0] is scale. You could load the mesh offline to compute exact AABB if needed.
    return np.array(pts)

def transform_vertices(verts, pos, orn):
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    return (verts @ rot.T) + np.array(pos)

def max_point_distance(pts_a, pts_b):
    tree = cKDTree(pts_b)
    dists, _ = tree.query(pts_a)
    return np.max(dists)


@TASK_REGISTRY.register("push_object")
class PushObjectTask(TaskBase):
    def __init__(self, urdf_path, 
                 table=True,
                 target_range=((0.3, 0.7), (-0.2, 0.2)), 
                 tol_pos=0.05,
                 **kwargs):
        """
        urdf_path: path to the object URDF to push
        table: whether to spawn a default table
        target_range: ((x_min, x_max), (y_min, y_max)) for random target positions
        tol_pos: success threshold (meters, Chamfer distance)
        """
        super().__init__()
        self.urdf_path = urdf_path
        self.table = table
        self.target_range = target_range
        self.tol_pos = tol_pos

        self.client_id = None
        self.object_id = None
        self.target_id = None
        self.target_pos = None
        self.target_orn = None

        self.target_vertices_world = None  # cache target mesh in world coords
    
    def _get_points_world(self, body_id):
        return get_body_aabb_points(body_id, self.client_id)

    def check_success(self, robot):
        pts_obj = self._get_points_world(self.object_id)
        dist = max_point_distance(pts_obj, self.target_points_world)
        return dist < self.tol_pos
    
    def create(object: PushableObject, table: bool = True,
               target_range=((0.3, 0.7), (-0.2, 0.2)),
               tol_pos=0.05, **_):
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
        objs_path = os.path.join(base_path, "assets", "objects")
        return PushObjectTask(
            os.path.join(objs_path, object.value),
            table, target_range, tol_pos
        )
        
    def _settle_body(self, body_id, steps=120):
        """Step physics a bit to let body settle on table/ground."""
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

    def reset(self, client_id, robot):
        self.client_id = client_id
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

        # Plane
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Optional table
        if self.table:
            p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65], physicsClientId=self.client_id)

        # Random target pose
        x = np.random.uniform(*self.target_range[0])
        y = np.random.uniform(*self.target_range[1])
        self.target_pos = np.array([x, y, 0.05])
        yaw = np.random.uniform(-np.pi, np.pi)
        self.target_orn = p.getQuaternionFromEuler([0, 0, yaw])

        # Target marker (duplicate object, transparent green)
        self.target_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.target_pos.tolist(),
            baseOrientation=self.target_orn,
            useFixedBase=False,  # let it fall
            physicsClientId=self.client_id
        )

        # temporarily make it semi-transparent
        p.changeVisualShape(self.target_id, -1, rgbaColor=[0, 1, 0, 0.3], physicsClientId=self.client_id)

        # let physics settle
        self._settle_body(self.target_id, steps=120)

        # now freeze it in place (so robot doesn't bump it)
        p.changeDynamics(self.target_id, -1, mass=0, lateralFriction=0, physicsClientId=self.client_id)
        
        num_joints = p.getNumJoints(self.target_id, physicsClientId=self.client_id)
        for link_idx in range(-1, num_joints):
            p.setCollisionFilterGroupMask(
                self.target_id, link_idx,
                collisionFilterGroup=0,  # no group
                collisionFilterMask=0,   # collides with nothing
                physicsClientId=self.client_id
            )
            
            
        
        # Object to push (orange)
        start_pos = [0.4, 0.0, 0.05]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.object_id = p.loadURDF(
            self.urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn,
            physicsClientId=self.client_id
        )
        p.changeVisualShape(self.object_id, -1, rgbaColor=[1, 0.5, 0, 1], physicsClientId=self.client_id)


        # Cache target mesh vertices in world coords
        self.target_points_world = self._get_points_world(self.target_id)
        
    def get_obs(self, robot):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        return {
            "robot": robot.get_state(),
            "object_pos": np.array(obj_pos),
            "object_orn": np.array(obj_orn),
            "target_pos": self.target_pos,
            "target_orn": np.array(self.target_orn),
        }

    def reward(self, robot):
        pts_obj = self._get_points_world(self.object_id)
        dist = max_point_distance(pts_obj, self.target_points_world)
        return -dist