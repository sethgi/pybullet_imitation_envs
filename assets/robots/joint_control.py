# kinova_joint_control.py
import pybullet as p
import pybullet_data
import time
import os

def main():
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load plane and Kinova URDF
    p.loadURDF("plane.urdf")
    urdf_path = os.path.join("kinova", "gen3.urdf")  # adjust path if needed
    robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0], useFixedBase=True)

    # Gather movable joints
    num_joints = p.getNumJoints(robot_id)
    movable_joints = []
    sliders = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_type = info[2]
        if joint_type != p.JOINT_FIXED:
            movable_joints.append(j)
            lower, upper = info[8], info[9]
            if lower < upper:  # URDF specified limits
                slider = p.addUserDebugParameter(info[1].decode("utf-8"), lower, upper, 0)
            else:  # no limits, use a default range
                slider = p.addUserDebugParameter(info[1].decode("utf-8"), -3.14, 3.14, 0)
            sliders.append(slider)

    print("Controllable joints:", [p.getJointInfo(robot_id, j)[1].decode("utf-8") for j in movable_joints])

    # Simulation loop
    while True:
        # Read slider values and apply to robot
        for j, slider in zip(movable_joints, sliders):
            target = p.readUserDebugParameter(slider)
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=200
            )
        p.stepSimulation()
        time.sleep(1. / 240.)


if __name__ == "__main__":
    main()
