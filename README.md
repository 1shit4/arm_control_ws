# Aerial-Manipulator Modeling and Arm Control

**ROS 2 and Gazebo workspace for simulating and controlling a robotic arm mounted on a quadrotor.**

This repository supports the [Robust Goal-Directed Aerial Manipulation project](https://github.com/ishita-chaudhary/aerial-manipulation-rl-control), which investigates coordinated vehicle–arm motion and disturbance compensation. Its focus is the **manipulator subsystem**: the robot description, simulated joint-control interfaces, joint-feedback controllers, and reference trajectories used to move the arm during aerial-manipulation experiments.

## Robot Model and Joint Interfaces

The workspace includes X500-based aerial-manipulator descriptions in **URDF/Xacro and SDF**, together with ROS 2 launch files and Gazebo model resources. The three-joint configuration exposes the actuated joints `L1_to_L2`, `L2_to_L3`, and `L3_to_L4` through `ros2_control`/`gz_ros2_control`.

Two control interfaces are configured:

- **Joint-position control:** desired joint angles are sent to a `JointGroupPositionController`, which commands simulated joint positions.
- **Joint-effort control:** desired joint torques/efforts are sent to a `JointGroupEffortController`; custom ROS 2 feedback controllers can calculate those efforts from joint-state measurements.

The arm can therefore be commanded independently of the quadrotor's flight controller, allowing its motion to be used in coupled-dynamics and disturbance-rejection experiments.

## Arm Controllers

### Cascaded position–velocity effort control

**Source:** [`arm_effort.py`](src/darm/darm/arm_effort.py)

A two-loop feedback controller tracks desired positions of the three arm joints:

```text
Desired joint angles
    → Position PD controller → desired joint velocities
    → Velocity PID controller → desired joint efforts
    → ros2_control → Gazebo arm joints
```

Joint position and velocity measurements are read from the simulation. The outer position loop runs at a configured interval of 20 ms; the inner velocity/effort loop runs at 5 ms. The output is a three-element joint-effort command.

### Alternative joint-feedback controllers

**Sources:** [`manipulator_controller.py`](src/darm/darm/manipulator_controller.py) and [`manipulator_controller_mod.py`](src/darm/darm/manipulator_controller_mod.py)

These files explore sliding-surface-based joint control and additional feedback terms, including effort saturation and an approximate gravity-compensation variant. They provide alternative experimental formulations for commanding the arm's joint efforts.

## Arm Motion Generation

Several reference generators are included for producing repeatable or varied arm motion:

- [`joint_updater.py`](src/darm/darm/joint_updater.py): periodic joint-reference updates.
- [`arm_loop.py`](src/darm/darm/arm_loop.py): repeated movement between configured joint targets.
- [`arm_position.py`](src/darm/darm/arm_position.py): randomized joint targets with varying transition speeds.

These motions can be used to exercise the arm or introduce time-varying manipulator configurations while evaluating the aerial vehicle's response.

## Repository Structure

| Path | Purpose |
| --- | --- |
| [`src/darm/urdf/`](src/darm/urdf/) | Aerial-manipulator URDF/Xacro models and simulated joint interfaces |
| [`src/darm/models/`](src/darm/models/) | Gazebo model assets, including adapted PX4/Gazebo resources |
| [`src/darm/config/`](src/darm/config/) | Position- and effort-control configurations |
| [`src/darm/launch/`](src/darm/launch/) | Robot-description and joint-controller launch files |
| [`src/darm/darm/arm_effort.py`](src/darm/darm/arm_effort.py) | Cascaded joint position/velocity feedback controller |
| [`src/darm/darm/manipulator_controller.py`](src/darm/darm/manipulator_controller.py) and [`manipulator_controller_mod.py`](src/darm/darm/manipulator_controller_mod.py) | Alternative joint-effort control experiments |
| [`src/darm/darm/joint_updater.py`](src/darm/darm/joint_updater.py), [`arm_loop.py`](src/darm/darm/arm_loop.py), [`arm_position.py`](src/darm/darm/arm_position.py) | Joint-reference generators |

## Build and Simulation

The workspace uses **ROS 2 Humble, Gazebo Harmonic, `ros2_control`, and `gz_ros2_control`**, alongside a compatible PX4/Gazebo vehicle simulation. From the workspace root:

```bash
colcon build --packages-select darm
source install/setup.bash
```

The included launch configurations support different arm-control interfaces:

```bash
# Position-controlled three-joint arm
ros2 launch darm trial_arm.launch.py

# Effort-controlled arm configuration
ros2 launch darm trial.launch.py
```

A compatible PX4/Gazebo simulation and the required ROS/Gazebo bridges must be running separately. The selected robot model, controller type, and joint interfaces must match.

## Related Repositories

- [Aerial Manipulation — Project Overview](https://github.com/ishita-chaudhary/aerial-manipulation-rl-control): overall research objective and linked implementations.
- [Quadrotor Inner-Loop Control](https://github.com/ishita-chaudhary/quadcopter_inner_loop_control): flight-control and disturbance-rejection experiments associated with manipulator motion.
- [Aerial Gym Simulator](https://github.com/ishita-chaudhary/aerial_gym_simulator): aerial-manipulator learning environments and controller-aware RL experiments.

*Research conducted collaboratively at the Robotics Research Center, IIIT Hyderabad. PX4, Gazebo, and ROS 2 components are credited to their respective upstream projects.*
