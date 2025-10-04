# imitation_framework/scripts/run_demo.py

import argparse

import numpy as np
from .robots.urdf_robot import URDFRobot, RobotURDFs
from .core.registry import ROBOT_REGISTRY, TASK_REGISTRY, TELEOP_REGISTRY
from .core.env_manager import EnvManager
from .tasks.push_object import PushableObject
from .teleop.keyboard_teleop import KeyboardTeleop

def parse_args():
    parser = argparse.ArgumentParser(description="Imitation Learning Framework")

    # Robot selection
    parser.add_argument(
        "--robot",
        type=str,
        default="KINOVA",
        choices=[e.name for e in RobotURDFs],
        help="Robot model to load"
    )

    parser.add_argument(
        "--teleop",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
        help="Type of teleoperator"
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default="push_object",
        choices=list(TASK_REGISTRY._map.keys()),
        help="Task to solve"
    )

    # Common PushObject args (but harmless if unused)
    parser.add_argument(
        "--target_object",
        type=str,
        default="T",
        choices=[e.name for e in PushableObject],
        help="Target object URDF enum"
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Spawn a table under the object"
    )
    parser.add_argument(
        "--no-table",
        dest="table",
        action="store_false",
        help="Disable table spawn"
    )
    parser.set_defaults(table=True)

    parser.add_argument(
        "--target_range",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=(0.3, 0.7, -0.2, 0.2),
        help="Target XY position sampling range"
    )
    parser.add_argument(
        "--tol_pos",
        type=float,
        default=0.02,
        help="Position success tolerance (meters)"
    )

    parser.add_argument(
        "--linear_velocity_scale",
        type=float,
        default=0.1,
        help="Multiplier for linear velocity for teleop"
    )

    parser.add_argument(
        "--rot_velocity_scale",
        type=float,
        default=0.1,
        help="Multiplier for rotational velocity for teleop"
    )


    return parser.parse_args()



def main():
    args = parse_args()

    if args.teleop == "spacemouse":
        from .teleop.spacemouse import SpaceMouseTeleop

    # Robot from registry + enum
    robot_enum = RobotURDFs[args.robot]
    robot = ROBOT_REGISTRY.build("urdf_robot", robot_enum)

    # Normalize args for task creation
    task_kwargs = vars(args)
    if "target_range" in task_kwargs:
        task_kwargs["target_range"] = (
            (args.target_range[0], args.target_range[1]),
            (args.target_range[2], args.target_range[3])
        )
    if "target_object" in task_kwargs:
        task_kwargs["object"] = PushableObject[args.target_object]

    # Generic task build: tasks decide what to use
    teleop_kwargs = {
        "lin_vel_scale": args.linear_velocity_scale,
        "rot_vel_scale": args.rot_velocity_scale
    }
    task = TASK_REGISTRY.build(args.task, **task_kwargs)

    teleop = TELEOP_REGISTRY.build(args.teleop, **teleop_kwargs)

    env = EnvManager(robot, task, teleop=teleop)

    print(f"Robot initialized: {robot.urdf_path}")
    print(f"Task initialized: {args.task}")
    print(f"Task kwargs passed: {task_kwargs}")
    
    # obs = env.reset()
    done=False
    
    while not done:
        obs, reward, done = env.step()
        env.render()

        print(f"Reward: {reward:.3f}, Done: {done}")

    env.close()

if __name__ == "__main__":
    main()
