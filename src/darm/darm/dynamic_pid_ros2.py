

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import math
import numpy as np
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class OffboardControlNode(Node):
    def __init__(self):
        super().__init__('offb_node_py')

        # Internal state
        self.current_state = State()
        self.current_target = PoseStamped()
        self.roll = self.pitch = self.yaw = 0.0
        self.allow_setpoints = False
        self.points_array = np.empty((0, 2))

        # Publishers and Subscribers
        self.create_subscription(State, "/mavros/state", self.state_cb, 10)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.local_pose_cb, qos_profile_sensor_data)
        self.create_subscription(Mdjt, "/command/trajectory", self.setpoint_cb, 10)

        self.local_pos_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)

        # Service clients
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")

        # Initialize pose
        self.current_target.header.frame_id = "map"
        self.current_target.pose.position.x = 0.0
        self.current_target.pose.position.y = 0.0
        self.current_target.pose.position.z = 1.2

        # Wait for service clients
        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for arming service...")
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for set_mode service...")

        self.rate_hz = 20
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)
        self.initialization_counter = 0
        self.last_req = self.get_clock().now()

    def state_cb(self, msg):
        self.current_state = msg

    def local_pose_cb(self, msg):
        q = msg.pose.orientation
        self.roll, self.pitch, self.yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))

    def setpoint_cb(self, msg):
        if self.allow_setpoints:
            pt = msg.points[0]
            self.current_target.pose.position.x = pt.transforms[0].translation.x
            self.current_target.pose.position.y = pt.transforms[0].translation.y
            self.current_target.pose.position.z = pt.transforms[0].translation.z

            new_point = np.array([[pt.transforms[0].translation.x, pt.transforms[0].translation.y]])
            self.points_array = np.vstack((self.points_array, new_point))

            if self.points_array.shape[0] > 15:
                self.points_array = self.points_array[-15:, :]

    def compute_last_tangent(self):
        if self.points_array.shape[0] < 2:
            raise ValueError("Need at least two points to compute a tangent.")
        p_prev = self.points_array[-2]
        p_last = self.points_array[-1]
        dx, dy = p_last - p_prev
        norm = np.hypot(dx, dy)
        if norm == 0:
            raise ValueError("Last two points are identical—tangent is undefined.")
        return math.atan2(dy, dx)

    def rpy_to_quaternion(self, roll, pitch, yaw):
        return quaternion_from_euler(roll, pitch, yaw)

    def control_loop(self):
        now = self.get_clock().now()
        self.allow_setpoints = (self.current_state.mode == "OFFBOARD" and self.current_state.armed)

        if not self.current_state.connected:
            return

        # Initial setpoints to stabilize
        if self.initialization_counter < 50:
            self.current_target.header.stamp = self.get_clock().now().to_msg()
            self.local_pos_pub.publish(self.current_target)
            self.initialization_counter += 1
            return

        # Mode switching
        if self.current_state.mode != "OFFBOARD" and (now - self.last_req).nanoseconds > 1e9:
            req = SetMode.Request()
            req.custom_mode = 'OFFBOARD'
            self.set_mode_client.call_async(req)
            self.get_logger().info("Sent OFFBOARD mode request")
            self.last_req = now

        # Arming
        elif not self.current_state.armed and (now - self.last_req).nanoseconds > 1e9:
            arm_req = CommandBool.Request()
            arm_req.value = True
            self.arming_client.call_async(arm_req)
            self.get_logger().info("Sent arming request")
            self.last_req = now

        # Optional: dynamic orientation from tangent
        # try:
        #     theta = self.compute_last_tangent()
        #     q = self.rpy_to_quaternion(self.roll, self.pitch, theta)
        #     self.current_target.pose.orientation.x = q[0]
        #     self.current_target.pose.orientation.y = q[1]
        #     self.current_target.pose.orientation.z = q[2]
        #     self.current_target.pose.orientation.w = q[3]
        # except ValueError:
        #     pass

        self.current_target.header.stamp = self.get_clock().now().to_msg()
        self.local_pos_pub.publish(self.current_target)


def main(args=None):
    rclpy.init(args=args)
    node = OffboardControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
