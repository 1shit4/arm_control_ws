#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, 
    VehicleCommand, 
    VehicleLocalPosition, 
    VehicleStatus, 
    VehicleAttitude, 
    VehicleThrustSetpoint, 
    VehicleTorqueSetpoint, 
    VehicleOdometry
)
import numpy as np
from scipy.spatial.transform import Rotation as R


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode with cascaded PID."""

    def __init__(self) -> None:
        super().__init__('offboard_control_cascaded_pid')

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
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.vehicle_thrust_setpoint_publisher = self.create_publisher(
            VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.vehicle_torque_setpoint_publisher = self.create_publisher(
            VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', 
            self.vehicle_local_position_callback, qos_profile)
        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', 
            self.vehicle_attitude_callback, qos_profile)
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', 
            self.vehicle_odometry_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', 
            self.vehicle_status_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.vehicle_rates = np.array([0.0, 0.0, 0.0])  # [roll_rate, pitch_rate, yaw_rate]
        
        # Desired setpoints
        self.takeoff_height = -5.0
        self.desired_pos = np.array([0.0, 0.0, self.takeoff_height])
        self.desired_yaw = 0.0  # Desired yaw angle (radians)
        
        # Control gains - TUNED for stability
        # Position controller (outputs desired angles)
        self.pos_xy_p_gain = 0.15  # Very conservative for stability
        self.pos_xy_d_gain = 0.2
        self.pos_xy_last_error = np.array([0.0, 0.0])
        
        # Altitude controller (outputs thrust)
        self.hover_thrust = 0.6  # Baseline thrust to hover (adjusted)
        self.z_gains_pid = np.array([0.1, 0.017, 0.05])  # [P, I, D] - reduced for stability
        self.z_integral = 0.0
        self.z_last_error = 0.0
        
        # Attitude controller (outputs desired body rates)
        self.att_p_gain = np.array([3.5, 3.5, 1.5])  # [roll, pitch, yaw] - reduced for stability
        
        # Rate controller (outputs torques)
        self.rate_pid_gains = np.array([
            [0.08, 0.01, 0.002],  # Roll gains [P, I, D] - reduced significantly
            [0.08, 0.01, 0.002],  # Pitch gains
            [0.1, 0.008, 0.003]   # Yaw gains
        ])
        self.rate_integral = np.zeros(3)
        self.rate_last_error = np.zeros(3)
        
        self.dt = 0.1  # Control loop period
        
        # Mission state tracking
        self.mission_complete = False

        # Create a timer to publish control commands
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info('Cascaded PID Control Node initialized')
    
    def vehicle_odometry_callback(self, msg):
        """Callback function for vehicle_odometry topic subscriber."""
        self.vehicle_rates = np.array([
            msg.angular_velocity[0], 
            msg.angular_velocity[1], 
            msg.angular_velocity[2]
        ])
    
    def vehicle_attitude_callback(self, msg):
        """Callback function for vehicle_attitude topic subscriber."""
        self.vehicle_attitude = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])

    def vehicle_local_position_callback(self, msg):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = msg

    def vehicle_status_callback(self, msg):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = msg

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
        msg.xyz = [float(torques[0]), float(torques[1]), float(torques[2])]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_torque_setpoint_publisher.publish(msg)

    def publish_thrust_setpoint(self, thrust: float):
        """Publish the vehicle thrust setpoint."""
        msg = VehicleThrustSetpoint()
        # PX4 expects negative thrust in NED frame (Z-axis points down)
        normalized_thrust = np.clip(thrust, 0.0, 1.0)
        msg.xyz = [0.0, 0.0, -normalized_thrust]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_thrust_setpoint_publisher.publish(msg)

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

    def quaternion_to_euler(self, q):
        """Convert quaternion [w,x,y,z] to euler angles [roll, pitch, yaw]."""
        rotation = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
        euler = rotation.as_euler('xyz', degrees=False)
        return euler  # [roll, pitch, yaw]

    def timer_callback(self) -> None:
        """Callback function for the timer - main control loop."""
        
        # Always publish heartbeat
        self.publish_offboard_control_heartbeat_signal()

        # Publish initial setpoints before arming (required by PX4)
        if self.offboard_setpoint_counter < 10:
            self.publish_thrust_setpoint(0.0)
            self.publish_torque_setpoint(np.zeros(3))
            self.offboard_setpoint_counter += 1
            return

        # Engage offboard mode and arm
        if self.offboard_setpoint_counter == 10:
            self.offboard_setpoint_counter += 1
            # Give a small delay for mode switch
            
        if self.offboard_setpoint_counter == 11:
            self.engage_offboard_mode()
            self.offboard_setpoint_counter += 1
            return
            
        if self.offboard_setpoint_counter == 12:
            self.arm()
            self.offboard_setpoint_counter += 1
            return

        # Check if mission is complete
        if self.mission_complete:
            return

        # Continue publishing setpoints even before offboard mode is active
        # This is critical - PX4 needs continuous setpoint stream
        if self.offboard_setpoint_counter < 20:
            self.publish_thrust_setpoint(self.hover_thrust)
            self.publish_torque_setpoint(np.zeros(3))
            self.offboard_setpoint_counter += 1
            
            if self.offboard_setpoint_counter % 10 == 0:
                self.get_logger().info(
                    f"Streaming setpoints... nav_state: {self.vehicle_status.nav_state}, "
                    f"arming_state: {self.vehicle_status.arming_state}"
                )
            return

        # Wait for offboard mode to be active
        if self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            # Keep streaming setpoints while waiting
            self.publish_thrust_setpoint(self.hover_thrust)
            self.publish_torque_setpoint(np.zeros(3))
            
            if self.offboard_setpoint_counter % 20 == 0:
                self.get_logger().info(
                    f"Waiting for offboard mode... nav_state: {self.vehicle_status.nav_state}, "
                    f"arming_state: {self.vehicle_status.arming_state}"
                )
            self.offboard_setpoint_counter += 1
            return

        # Get current state
        current_pos = np.array([
            self.vehicle_local_position.x,
            self.vehicle_local_position.y,
            self.vehicle_local_position.z
        ])
        
        current_vel = np.array([
            self.vehicle_local_position.vx,
            self.vehicle_local_position.vy,
            self.vehicle_local_position.vz
        ])

        # ========== CASCADED PID CONTROL ==========
        
        # --- 1. ALTITUDE CONTROLLER (Z-axis) -> Thrust ---
        z_err = self.desired_pos[2] - current_pos[2]
        
        # PID terms
        p_term_z = self.z_gains_pid[0] * z_err
        
        self.z_integral += z_err * self.dt
        self.z_integral = np.clip(self.z_integral, -5.0, 5.0)  # Anti-windup
        i_term_z = self.z_gains_pid[1] * self.z_integral
        
        z_deriv = (z_err - self.z_last_error) / self.dt
        d_term_z = self.z_gains_pid[2] * z_deriv
        self.z_last_error = z_err
        
        # Total thrust command
        thrust_correction = p_term_z + i_term_z + d_term_z
        thrust_cmd = self.hover_thrust - thrust_correction
        # print(thrust_correction)
        thrust_cmd = np.clip(thrust_cmd, 0.1, 1.0)
        # print(thrust_cmd)

        # --- 2. POSITION CONTROLLER (XY-plane) -> Desired Roll/Pitch ---
        xy_err = self.desired_pos[0:2] - current_pos[0:2]
        
        # PD control for position
        xy_p_term = self.pos_xy_p_gain * xy_err
        xy_d_term = self.pos_xy_d_gain * (xy_err - self.pos_xy_last_error) / self.dt
        self.pos_xy_last_error = xy_err.copy()
        
        xy_correction = xy_p_term + xy_d_term
        
        # Desired angles (small angle approximation)
        # In NED: positive pitch -> forward, positive roll -> right
        # To go forward (+X), need negative pitch
        # To go right (+Y), need positive roll
        desired_pitch = -np.clip(xy_correction[0], -0.3, 0.3)  # X error -> pitch
        desired_roll = np.clip(xy_correction[1], -0.3, 0.3)    # Y error -> roll

        # --- 3. ATTITUDE CONTROLLER -> Desired Body Rates ---
        # Get current euler angles
        current_euler = self.quaternion_to_euler(self.vehicle_attitude)
        current_roll = current_euler[0]
        current_pitch = current_euler[1]
        current_yaw = current_euler[2]
        
        # Attitude errors
        roll_err = desired_roll - current_roll
        pitch_err = desired_pitch - current_pitch
        yaw_err = self.desired_yaw - current_yaw
        
        # Normalize yaw error to [-pi, pi]
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
        
        # P controller: attitude error -> desired rates
        desired_rates = np.array([
            self.att_p_gain[0] * roll_err,
            self.att_p_gain[1] * pitch_err,
            self.att_p_gain[2] * yaw_err
        ])
        
        # Limit desired rates
        desired_rates = np.clip(desired_rates, -2.0, 2.0)

        # --- 4. RATE CONTROLLER -> Torques ---
        rate_err = desired_rates - self.vehicle_rates
        
        # PID for each axis
        torques = np.zeros(3)
        for i in range(3):
            # P term
            p_term = self.rate_pid_gains[i, 0] * rate_err[i]
            
            # I term with anti-windup
            self.rate_integral[i] += rate_err[i] * self.dt
            self.rate_integral[i] = np.clip(self.rate_integral[i], -0.5, 0.5)
            i_term = self.rate_pid_gains[i, 1] * self.rate_integral[i]
            
            # D term
            rate_deriv = (rate_err[i] - self.rate_last_error[i]) / self.dt
            d_term = self.rate_pid_gains[i, 2] * rate_deriv
            
            torques[i] = p_term + i_term + d_term
            self.rate_last_error[i] = rate_err[i]
        
        # Limit torques
        torques = np.clip(torques, -1.0, 1.0)

        # --- 5. PUBLISH COMMANDS ---
        self.publish_thrust_setpoint(thrust_cmd)
        self.publish_torque_setpoint(torques)

        # --- 6. CHECK LANDING CONDITION ---
        position_error = np.linalg.norm(current_pos - self.desired_pos)
        
        # Log every 10 iterations
        if self.offboard_setpoint_counter % 10 == 0:
            self.get_logger().info(
                f"Pos: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] | "
                f"Target: [{self.desired_pos[0]:.2f}, {self.desired_pos[1]:.2f}, {self.desired_pos[2]:.2f}] | "
                f"Error: {position_error:.2f}m | "
                f"Thrust: {thrust_cmd:.2f}"
            )
        
        # If reached target (within 0.5m), initiate landing
        if position_error < 0.5 and not self.mission_complete:
            self.get_logger().info("Target reached! Landing...")
            # self.land()
            self.mission_complete = True
            # Schedule shutdown after 2 seconds
            self.create_timer(2.0, lambda: rclpy.shutdown())

        self.offboard_setpoint_counter += 1


def main(args=None) -> None:
    print('Starting offboard cascaded PID control node...')
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