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
    """Node for controlling a vehicle in offboard mode with cascaded PID at multiple rates."""

    def __init__(self) -> None:
        super().__init__('offboard_control_cascaded_pid')

        # Configure QoS profile for communicating with PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.vehicle_thrust_setpoint_publisher = self.create_publisher(
            VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.vehicle_torque_setpoint_publisher = self.create_publisher(
            VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)

        # Subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback, qos_profile)
        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback, qos_profile)
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback, qos_profile)

        # State variables
        self.offboard_setpoint_counter = 0
        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.vehicle_rates = np.array([0.0, 0.0, 0.0])  # Body frame rates [p, q, r]
        
        # Mission parameters (NED frame: negative Z is up)
        self.takeoff_height = -5.0  # 5 meters above ground
        self.desired_pos = np.array([1.0, 2.0, self.takeoff_height])
        self.desired_yaw = 0.0
        self.desired_pitch = 0.0
        self.desired_roll = 0.0
        self.desired_rates = np.array([0.0, 0.0, 0.0])

        # Current state
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0

        # Outer loop (Position) PID gains
        self.pos_xy_p_gain = 0.18
        self.pos_xy_d_gain = 0.25
        self.pos_xy_last_error = np.array([0.0, 0.0])
        
        # Z-axis (altitude) PID gains [P, I, D]
        self.hover_thrust = 0.59  # Nominal hover thrust (tune for your vehicle)
        self.z_gains_pid = np.array([0.35, 0.08, 0.18])
        self.z_integral = 0.0
        self.z_last_error = 0.0
        self.z_err = 0.0

        # Middle loop (Attitude) P gains
        self.att_p_gain = np.array([4.0, 4.0, 1.8])  # [roll, pitch, yaw]

        # Inner loop (Rate) PID gains [P, I, D] for [roll_rate, pitch_rate, yaw_rate]
        self.rate_pid_gains = np.array([
            [0.10, 0.015, 0.003],  # Roll rate
            [0.10, 0.015, 0.003],  # Pitch rate
            [0.12, 0.012, 0.004]   # Yaw rate
        ])
        self.rate_integral = np.zeros(3)
        self.rate_last_error = np.zeros(3)

        # Control outputs
        self.thrust_cmd = self.hover_thrust
        self.torques = np.zeros(3)

        # Multi-rate timing configuration
        # Outer loop: Position control at 20 Hz (dt = 0.05s)
        # Middle loop: Attitude control at 80 Hz (dt/4 = 0.0125s) - 4x faster
        # Inner loop: Rate control at 1000 Hz (dt/20 = 0.0025s) - 20x faster than outer
        self.dt_outer = 0.05  # 20 Hz - Position control
        self.dt_mid = self.dt_outer / 4  # 80 Hz - Attitude control
        self.dt_inner = self.dt_outer / 20  # 1000 Hz - Rate control
        
        self.mission_complete = False
        self.control_active = False  # Flag to enable main control loops
        
        # Create timers for cascaded control at different rates
        # Inner loop (Rate control) - Highest frequency
        self.timer_inner = self.create_timer(self.dt_inner, self.loop_inner_callback)
        
        # Middle loop (Attitude control) - Medium frequency
        self.timer_mid = self.create_timer(self.dt_mid, self.loop_mid_callback)
        
        # Outer loop (Position control) - Lowest frequency
        self.timer_outer = self.create_timer(self.dt_outer, self.loop_outer_callback)
        
        self.get_logger().info('Cascaded PID Control Node initialized')
        self.get_logger().info(f'Outer loop (Position): {1.0/self.dt_outer:.0f} Hz')
        self.get_logger().info(f'Middle loop (Attitude): {1.0/self.dt_mid:.0f} Hz')
        self.get_logger().info(f'Inner loop (Rate): {1.0/self.dt_inner:.0f} Hz')

    def vehicle_odometry_callback(self, msg):
        """Callback function for vehicle_odometry topic subscriber."""
        self.vehicle_rates = np.array([
            msg.angular_velocity[0],  # Roll rate (p)
            msg.angular_velocity[1],  # Pitch rate (q)
            msg.angular_velocity[2]   # Yaw rate (r)
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
        """Publish the vehicle torque setpoint in body frame (FRD)."""
        msg = VehicleTorqueSetpoint()
        msg.xyz = [float(torques[0]), float(torques[1]), float(torques[2])]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_torque_setpoint_publisher.publish(msg)

    def publish_thrust_setpoint(self, thrust: float):
        """Publish the vehicle thrust setpoint in NED frame (negative Z is up)."""
        msg = VehicleThrustSetpoint()
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
        """Convert quaternion [w,x,y,z] to euler angles [roll, pitch, yaw] in radians."""
        rotation = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
        euler = rotation.as_euler('xyz', degrees=False)
        return euler

    def loop_outer_callback(self) -> None:
        """Outer control loop callback: Position control at 20 Hz -> desired attitude."""
        # Always publish heartbeat
        self.publish_offboard_control_heartbeat_signal()

        # Startup sequence
        if self.offboard_setpoint_counter < 10:
            self.publish_thrust_setpoint(0.0)
            self.publish_torque_setpoint(np.zeros(3))
            self.offboard_setpoint_counter += 1
            return

        if self.offboard_setpoint_counter == 10:
            self.offboard_setpoint_counter += 1

        if self.offboard_setpoint_counter == 11:
            self.engage_offboard_mode()
            self.offboard_setpoint_counter += 1
            self.publish_thrust_setpoint(self.hover_thrust)
            self.publish_torque_setpoint(np.zeros(3))
            return

        if self.offboard_setpoint_counter == 12:
            self.arm()
            self.offboard_setpoint_counter += 1
            self.publish_thrust_setpoint(self.hover_thrust)
            self.publish_torque_setpoint(np.zeros(3))
            return

        if self.offboard_setpoint_counter < 20:
            self.publish_thrust_setpoint(self.hover_thrust)
            self.publish_torque_setpoint(np.zeros(3))
            self.offboard_setpoint_counter += 1
            if self.offboard_setpoint_counter % 10 == 0:
                self.get_logger().info(
                    f"Stabilizing... nav_state: {self.vehicle_status.nav_state}, "
                    f"arming_state: {self.vehicle_status.arming_state}"
                )
            return

        if self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_thrust_setpoint(self.hover_thrust)
            self.publish_torque_setpoint(np.zeros(3))
            if self.offboard_setpoint_counter % 20 == 0:
                self.get_logger().info(
                    f"Waiting for offboard mode... nav_state: {self.vehicle_status.nav_state}"
                )
            self.offboard_setpoint_counter += 1
            return

        # Enable cascaded control
        self.control_active = True

        # OUTER LOOP: Position control -> desired attitude and thrust
        self.current_pos = np.array([
            self.vehicle_local_position.x,
            self.vehicle_local_position.y,
            self.vehicle_local_position.z
        ])

        # Z-axis (altitude) PID control
        z_err = self.current_pos[2] - self.desired_pos[2]
        self.z_err = z_err
        
        p_term_z = self.z_gains_pid[0] * z_err
        
        self.z_integral += z_err * self.dt_outer
        self.z_integral = np.clip(self.z_integral, -2.5, 2.5)
        i_term_z = self.z_gains_pid[1] * self.z_integral
        
        z_deriv = (z_err - self.z_last_error) / self.dt_outer
        d_term_z = self.z_gains_pid[2] * z_deriv
        self.z_last_error = z_err
        
        thrust_correction = p_term_z + i_term_z + d_term_z
        thrust_cmd = self.hover_thrust + thrust_correction
        self.thrust_cmd = np.clip(thrust_cmd, 0.35, 0.85)

        # XY position control -> desired pitch/roll
        xy_err = self.desired_pos[0:2] - self.current_pos[0:2]
        xy_p_term = self.pos_xy_p_gain * xy_err
        xy_d_term = self.pos_xy_d_gain * (xy_err - self.pos_xy_last_error) / self.dt_outer
        self.pos_xy_last_error = xy_err.copy()
        
        xy_correction = xy_p_term + xy_d_term
        
        self.desired_pitch = -np.clip(xy_correction[0], -0.18, 0.18)
        self.desired_roll = np.clip(xy_correction[1], -0.18, 0.18)

        # Logging
        position_error = np.linalg.norm(self.current_pos - self.desired_pos)
        
        if self.offboard_setpoint_counter % 20 == 0:
            self.get_logger().info(
                f"Pos: [{self.current_pos[0]:.2f}, {self.current_pos[1]:.2f}, {self.current_pos[2]:.2f}] | "
                f"Target: [{self.desired_pos[0]:.2f}, {self.desired_pos[1]:.2f}, {self.desired_pos[2]:.2f}] | "
                f"Err: {position_error:.2f}m | Z_err: {self.z_err:.2f} | "
                f"Thrust: {self.thrust_cmd:.3f} | "
                f"Roll: {np.rad2deg(self.current_roll):.1f}° | "
                f"Pitch: {np.rad2deg(self.current_pitch):.1f}° | "
                f"Yaw: {np.rad2deg(self.current_yaw):.1f}°"
            )

        if position_error < 0.5 and not self.mission_complete:
            self.get_logger().info("✓ Target position reached! Hovering...")
            self.mission_complete = True

        self.offboard_setpoint_counter += 1

    def loop_mid_callback(self) -> None:
        """Middle control loop callback: Attitude control at 80 Hz -> desired body rates."""
        if not self.control_active:
            return

        # MIDDLE LOOP: Attitude control -> desired body rates
        current_euler = self.quaternion_to_euler(self.vehicle_attitude)
        self.current_roll = current_euler[0]
        self.current_pitch = current_euler[1]
        self.current_yaw = current_euler[2]

        roll_err = self.desired_roll - self.current_roll
        pitch_err = self.desired_pitch - self.current_pitch
        yaw_err = self.desired_yaw - self.current_yaw
        
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))

        desired_rates = np.array([
            self.att_p_gain[0] * roll_err,
            self.att_p_gain[1] * pitch_err,
            self.att_p_gain[2] * yaw_err
        ])
        
        self.desired_rates = np.clip(desired_rates, -1.2, 1.2)

    def loop_inner_callback(self) -> None:
        """Inner control loop callback: Rate control at 1000 Hz -> torque commands."""
        if not self.control_active:
            # Publish safe setpoints during startup
            self.publish_thrust_setpoint(self.thrust_cmd)
            self.publish_torque_setpoint(np.zeros(3))
            return

        # INNER LOOP: Rate control -> torques
        rate_err = self.desired_rates - self.vehicle_rates
        
        torques = np.zeros(3)
        for i in range(3):
            p_term = self.rate_pid_gains[i, 0] * rate_err[i]
            
            self.rate_integral[i] += rate_err[i] * self.dt_inner
            self.rate_integral[i] = np.clip(self.rate_integral[i], -0.4, 0.4)
            i_term = self.rate_pid_gains[i, 1] * self.rate_integral[i]
            
            rate_deriv = (rate_err[i] - self.rate_last_error[i]) / self.dt_inner
            d_term = self.rate_pid_gains[i, 2] * rate_deriv
            
            torques[i] = p_term + i_term + d_term
            self.rate_last_error[i] = rate_err[i]
        
        self.torques = np.clip(torques, -0.6, 0.6)

        # Publish control commands (runs at highest frequency)
        self.publish_thrust_setpoint(self.thrust_cmd)
        self.publish_torque_setpoint(self.torques)


def main(args=None) -> None:
    print('Starting offboard cascaded PID control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    
    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        offboard_control.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        offboard_control.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
