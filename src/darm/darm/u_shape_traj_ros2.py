import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
from scipy.interpolate import CubicSpline

# ROS 2 Messages (ROS 1 names were the same)
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist
from mavros_msgs.msg import Thrust

class UShapeTrajectoryPy(Node):
    """
    ROS 2 version of the u_shape_trajectory_py node.
    - Uses standard ROS 2 class-based structure.
    - Preserves the exact function names, variable names, and execution
      logic of the original ROS 1 script.
    """
    def __init__(self, node_name):
        super().__init__(node_name)

        # This flag is the direct equivalent of the global `command_active`
        self.command_active = True

        # Publisher for the trajectory
        self.trajectory_pub = self.create_publisher(MultiDOFJointTrajectory, '/command/trajectory', 10)
        
        # Subscriber to trigger the trajectory
        self.command_sub = self.create_subscription(
            Thrust,
            '/mavros/setpoint_attitude/thrust',
            self.sp_callback, # Using original function name
            10
        )

    def sp_callback(self, msg):
        """
        Callback for the subscriber. Sets the command_active flag to True.
        """
        if not self.command_active:
            self.get_logger().info("Command received, trajectory will start.")
            self.command_active = True
            
            # To ensure one-shot behavior, we can destroy the subscriber after it's used.
            # This is a robust way to prevent the trajectory from re-triggering.
            self.destroy_subscription(self.command_sub)
            self.command_sub = None

    def generate_u_shape_waypoints(self, leg_length, width, height):
        """Generates the key waypoints for a U-shaped trajectory. (Identical to original)"""
        waypoints = []
        radius = width / 2.0
        center_x = radius
        center_y = leg_length

        waypoints.append([0.0, 0.0, height])
        waypoints.append([0.0, leg_length, height])
        waypoints.append([center_x - radius * 0.7071, center_y + radius * 0.7071, height])
        waypoints.append([center_x, center_y + radius, height])
        waypoints.append([center_x + radius * 0.7071, center_y + radius * 0.7071, height])
        waypoints.append([width, leg_length, height])
        waypoints.append([width, 0.0, height])

        return np.array(waypoints)

    def start_trajectory(self, waypoints):
        """
        Generates and publishes a smooth trajectory from a list of waypoints.
        (Identical logic and variable names to original)
        """
        v_max = 0.75
        sampling_hz = 50.0
        dt = 1.0 / sampling_hz # Replaces rospy.Rate

        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        segment_times = distances / v_max
        time_vector = np.concatenate(([0], np.cumsum(segment_times)))
        total_duration = time_vector[-1]

        cs_x = CubicSpline(time_vector, waypoints[:, 0])
        cs_y = CubicSpline(time_vector, waypoints[:, 1])
        cs_z = CubicSpline(time_vector, waypoints[:, 2])

        t_sample = np.arange(0, total_duration, dt)
        self.get_logger().info(f"Publishing {len(t_sample)} trajectory points over {total_duration:.2f} seconds.")

        for t in t_sample:
            if not rclpy.ok():
                break

            pt = MultiDOFJointTrajectoryPoint()

            transform = Transform()
            transform.translation.x = float(cs_x(t))
            transform.translation.y = float(cs_y(t))
            transform.translation.z = float(cs_z(t))
            transform.rotation.w = 1.0

            velocities = Twist()
            velocities.linear.x = float(cs_x(t, 1))
            velocities.linear.y = float(cs_y(t, 1))
            velocities.linear.z = float(cs_z(t, 1))

            accelerations = Twist()
            accelerations.linear.x = float(cs_x(t, 2))
            accelerations.linear.y = float(cs_y(t, 2))
            accelerations.linear.z = float(cs_z(t, 2))

            pt.transforms.append(transform)
            pt.velocities.append(velocities)
            pt.accelerations.append(accelerations)

            traj_msg = MultiDOFJointTrajectory()
            # Use node's clock instead of rospy.Time
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.header.frame_id = 'map'
            traj_msg.points.append(pt)

            self.trajectory_pub.publish(traj_msg)
            # Use standard time.sleep() instead of rate.sleep()
            time.sleep(dt)

        self.get_logger().info("All trajectory points published.")

def main(args=None):
    """
    Main function to initialize the node and execute the trajectory logic.
    This section directly mirrors the execution flow of the original ROS 1 main function.
    """
    rclpy.init(args=args)
    # Use the original node name from rospy.init_node
    node = UShapeTrajectoryPy('u_shape_trajectory_py')

    # --- Start of Original Logic Flow ---
    
    # 1. Wait for a subscriber to connect (Identical to original)
    node.get_logger().info("Waiting for a subscriber to connect to /command/trajectory...")
    # Using `get_num_connections()` equivalent: get_subscription_count()
    while node.trajectory_pub.get_subscription_count() == 0 and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=1.0)
    
    node.get_logger().info("Subscriber connected. Waiting for command on /mavros/setpoint_attitude/thrust...")

    # 2. Main wait loop (Identical to original)
    while rclpy.ok():
        # Check the `command_active` flag
        if node.command_active:
            start_time = node.get_clock().now()
            
            constant_height = 1.0
            leg_length = 10.0
            width = 5.0
            
            waypoints = node.generate_u_shape_waypoints(leg_length, width, constant_height)
            
            # Call the main trajectory publishing function
            node.start_trajectory(waypoints)
            
            duration = (node.get_clock().now() - start_time).nanoseconds / 1e9
            node.get_logger().info(f"Trajectory generation and publishing took {duration:.2f} seconds.")
            
            # Reset the flag and break, so it only runs once (Identical to original)
            node.command_active = False
            break 
        
        # This is the ROS 2 equivalent of rospy.sleep() in a main loop
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("Script finished.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()