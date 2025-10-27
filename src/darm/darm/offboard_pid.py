#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleAttitude, VehicleThrustSetpoint, VehicleTorqueSetpoint, VehicleOdometry

import numpy as np
from scipy.spatial.transform import Rotation as R

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.vehicle_thrust_setpoint_publisher = self.create_publisher(
            VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.vehicle_torque_setpoint_publisher = self.create_publisher(
            VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile)
        self.vehicle_rates_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -5.0
        self.desired_pos = np.array([0.0, 0.0, self.takeoff_height])
        self.desired_yaw = 0.0  # Desired yaw angle (radians)
        
        #pid vars
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_attitude = np.array([0.0, 0.0, 0.0, 0.0])
        self.vehicle_rates = np.array([0.0, 0.0, 0.0])  
        # for z -> thrust (direct mapping)
        self.hover_thrust = 1  # Assumed hover thrust (tune this)
        self.z_gains_pid = np.array([0.8, 0.1, 0.4])  # P, I, D
        self.z_integral = 0.0
        self.z_last_error = 0.0

        self.dt = 0.1  # Time step for PID calculations
        # for x,y -> roll, pitch
        self.pos_xy_p_gain = np.array([0.1, 0.1])
        self.att_p_gain = np.array([8.0, 8.0, 1.0])
        self.rate_pid_gains = np.array([
            [0.15, 0.05, 0.01],  # Roll gains
            [0.15, 0.05, 0.01],  # Pitch gains
            [0.1, 0.03, 0.005]   # Yaw gains
        ])
        self.rate_integral = np.zeros(3)
        self.rate_last_error = np.zeros(3)
   
        self.vehicle_thrust_setpoint = VehicleThrustSetpoint()
        self.vehicle_torque_setpoint = VehicleTorqueSetpoint()

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)
    
    def vehicle_odometry_callback(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber."""
        self.vehicle_rates = np.array([vehicle_odometry.angular_velocity[0], vehicle_odometry.angular_velocity[1], vehicle_odometry.angular_velocity[2]])
    def vehicle_attitude_callback(self, vehicle_attitude):
        """Callback function for vehicle_attitude topic subscriber."""
        self.vehicle_attitude = np.array([vehicle_attitude.q[0], vehicle_attitude.q[1], vehicle_attitude.q[2], vehicle_attitude.q[3]])

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")
    
    def publish_torque_setpoint(self, torques: np.ndarray):
        """Publish the vehicle torque setpoint."""
        msg = VehicleTorqueSetpoint()
        msg.xyz = [0.0, 0.0, 0.0] #[float(torques[0]), float(torques[1]), float(torques[2])]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_torque_setpoint_publisher.publish(msg)

    def publish_thrust_setpoint(self, thrust: float):
        """Publish the vehicle thrust setpoint."""
        msg = VehicleThrustSetpoint()
        # PX4 thrust is normalized [0, 1] and applied on the Z-axis (NED)
        # We send a negative thrust command because the Z-axis is "down"
        msg.xyz = [0.0, 0.0, -np.clip(-thrust, 0.0, 1.0)]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_thrust_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing thrust setpoint: {thrust}")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = True
        msg.direct_actuator = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        # Heartbeat is always sent, once per tick
        self.publish_offboard_control_heartbeat_signal()

        # Arm and engage offboard mode just once
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        # Run the PID controller only if we are in offboard mode
        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            
            # --- 1. Check for landing condition FIRST ---
            # If we've reached the target height, land
            # (Added a 0.5m buffer so it doesn't trigger immediately)
            if self.vehicle_local_position.z <= self.takeoff_height + 0.5: 
                self.land()
                self.get_logger().info("Target altitude reached, landing and shutting down.")
                rclpy.shutdown()  # Use rclpy.shutdown() for a clean exit
                return # Exit the callback

            # --- 2. Z-axis Thrust PID Controller ---
            current_z = self.vehicle_local_position.z
            z_err = self.desired_pos[2] - current_z
            
            self.z_integral += z_err * self.dt
            self.z_integral = np.clip(self.z_integral, -10.0, 10.0) # Anti-windup
            
            z_deriv = (z_err - self.z_last_error) / self.dt
            self.z_last_error = z_err
            
            p_term_z = self.z_gains_pid[0] * z_err
            i_term_z = self.z_gains_pid[1] * self.z_integral
            d_term_z = self.z_gains_pid[2] * z_deriv
            
            z_thrust_cmd = p_term_z + i_term_z + d_term_z
            thrust_cmd = z_thrust_cmd + self.hover_thrust
           
            print(f"Z Error: {z_err:.2f}, Thrust Command: {thrust_cmd:.2f}")

            # --- 3. XY Position P-Controller -> Desired Attitude ---
            current_x = self.vehicle_local_position.x
            current_y = self.vehicle_local_position.y
            x_err = self.desired_pos[0] - current_x
            y_err = self.desired_pos[1] - current_y

            desired_roll = np.clip(self.pos_xy_p_gain[1] * y_err, -0.25, 0.25)
            desired_pitch = np.clip(self.pos_xy_p_gain[0] * x_err, -0.25, 0.25)

            q_d_scipy = R.from_euler('xyz', [desired_roll, desired_pitch, self.desired_yaw])
            q_d = q_d_scipy.as_quat()  # Returns [x, y, z, w]
            q_d = np.array([q_d[3], q_d[0], q_d[1], q_d[2]])  # Reorder to [w, x, y, z]

            # --- 4. Attitude P-Controller -> Desired Rates ---
            q_current = self.vehicle_attitude
            
            q_current_scipy = R.from_quat([q_current[1], q_current[2], q_current[3], q_current[0]])
            q_d_scipy = R.from_quat([q_d[1], q_d[2], q_d[3], q_d[0]])

            q_err_scipy = q_d_scipy * q_current_scipy.inv()
            att_err_vect = q_err_scipy.as_rotvec()
            desired_rates = self.att_p_gain * att_err_vect

            # --- 5. Rate PID-Controller -> Desired Torques ---
            current_rates = self.vehicle_rates
            rate_err = desired_rates - current_rates

            self.rate_integral += rate_err * self.dt
            self.rate_integral = np.clip(self.rate_integral, -1.0, 1.0) # Anti-windup
            
            rate_deriv = (rate_err - self.rate_last_error) / self.dt
            self.rate_last_error = rate_err

            p_cmd = (self.rate_pid_gains[0,0] * rate_err[0]) + (self.rate_pid_gains[0,1] * self.rate_integral[0]) + (self.rate_pid_gains[0,2] * rate_deriv[0])
            q_cmd = (self.rate_pid_gains[1,0] * rate_err[1]) + (self.rate_pid_gains[1,1] * self.rate_integral[1]) + (self.rate_pid_gains[1,2] * rate_deriv[1])
            r_cmd = (self.rate_pid_gains[2,0] * rate_err[2]) + (self.rate_pid_gains[2,1] * self.rate_integral[2]) + (self.rate_pid_gains[2,2] * rate_deriv[2])
            
            torques = np.array([p_cmd, q_cmd, r_cmd])
            torques = np.clip(torques, -1.0, 1.0) # Clamp torques

            # --- 6. Publish Commands ---
            # These commands reset the 0.5s offboard timeout
            self.publish_thrust_setpoint(thrust_cmd)
            self.publish_torque_setpoint(torques)

        # This is the ONLY place the counter should be incremented
        # It increments unconditionally until the PID loop takes over
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1



def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)