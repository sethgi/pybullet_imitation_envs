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

def aabb_corners(aabb_min, aabb_max):
    """Return the 8 corner points of an AABB as (8,3) np.array."""
    mn = np.array(aabb_min, dtype=float)
    mx = np.array(aabb_max, dtype=float)
    xs = [mn[0], mx[0]]
    ys = [mn[1], mx[1]]
    zs = [mn[2], mx[2]]
    pts = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                pts.append([xs[i], ys[j], zs[k]])
    return np.asarray(pts)


def get_body_aabb_points(body_id, client_id=0):
    """
    Collect AABB corners from base + all links, regardless of actuation.
    Works for rigid multi-link objects and articulated ones.
    """
    pts = []
    try:
        num_links = p.getNumJoints(body_id, physicsClientId=client_id)
    except p.error:
        return np.empty((0, 3), dtype=float)

    for link_idx in range(-1, num_links):  # -1 = base
        try:
            aabb_min, aabb_max = p.getAABB(body_id, linkIndex=link_idx, physicsClientId=client_id)
            if aabb_min is not None and aabb_max is not None:
                pts.append(aabb_corners(aabb_min, aabb_max))
        except p.error:
            continue

    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)
    return np.vstack(pts)

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