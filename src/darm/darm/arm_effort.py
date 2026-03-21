import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import math
import numpy as np
import signal

from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicJointState

from time import time

class ArmEffortController(Node):
    def __init__(self):
        super().__init__('arm_effort_controller')

        ''' joint values:'''

        self.joint_ref = [0.0 , 0.0, 0.0]
        self.joint_state = np.zeros((3, 3))
        self.vel_ref = np.array([0.0, 0.0, 0.0])

        ''' Controller parameters:'''

        self.dt_outer = 0.02
        self.dt_inner = 0.005
        self.error_p = np.array([0.0, 0.0, 0.0])
        self.prev_error_p = np.array([0.0, 0.0, 0.0])
    
        self.pd_p = np.array([
                            [15.0, 1.0],
                            [15.0, 1.0],
                            [10.0, 1.0]])*0.1
        
        self.error_v = np.array([0.0, 0.0, 0.0])
        self.prev_error_v = np.array([0.0, 0.0, 0.0])
        self.integral_error_v = np.array([0.0, 0.0, 0.0])
        self.pid_v = np.array([
                            [15.0, 15.0, 1.0],
                            [15.0, 15.0, 1.0],
                            [15.0, 15.0, 1.0]])*0.1
        
        

        self.command = [0.0, 0.0, 0.0]

        ''' Publishers and Subscribers:'''

        self.effort_pub = self.create_publisher(
            Float64MultiArray, 
            '/arm_effort_controller/commands', 
            10)
        self.joint_sen_ = self.create_subscription(
            DynamicJointState,
            '/dynamic_joint_states',
            self.joint_state_callback,
            qos_profile_sensor_data
        )
        self.joint_ref_ = self.create_subscription(
            Float64MultiArray,
            '/joint_reference',
            self.joint_reference_callback,
            qos_profile_sensor_data
        )

        self.outer_loop = self.create_timer(self.dt_outer, self.effort_loop_outer)
        self.inner_loop = self.create_timer(self.dt_inner, self.effort_loop_inner)
    
    def effort_loop_outer(self):
        for i in range(len(self.joint_ref)):
            #outer position loop (PD)
            self.error_p[i] = self.joint_ref[i] - self.joint_state[i, 0]
            self.vel_ref[i] = self.pd_p[i, 0] * self.error_p[i] + self.pd_p[i, 1] * (self.error_p[i] - self.prev_error_p[i]) / self.dt_outer
            self.prev_error_p[i] = self.error_p[i]
    
    def effort_loop_inner(self):
        for i in range(len(self.joint_ref)):
            #inner velocity loop (PID) _> todo , figure out how to increase inner frequency
            self.error_v[i] = self.vel_ref[i] - self.joint_state[i, 1]
            self.integral_error_v[i] += self.error_v[i] * self.dt_inner
            self.command[i] = (self.pid_v[i, 0] * self.error_v[i] + 
                               self.pid_v[i, 2] * self.integral_error_v[i] + 
                               self.pid_v[i, 1] * (self.error_v[i] - self.prev_error_v[i]) / self.dt_inner)
            self.prev_error_v[i] = self.error_v[i]
        self.send_effort_commands()
             

    def send_effort_commands(self):
        # Example effort command (replace with actual control logic)
        effort_command = Float64MultiArray()
        effort_command.data = [self.command[0], self.command[1], self.command[2]]  # Example efforts for 3 joints
        self.effort_pub.publish(effort_command)
        self.get_logger().info(f'Sent effort commands: {effort_command.data}')

    def joint_state_callback(self, msg):
        for i, interface in enumerate(msg.interface_values):
            self.joint_state[i, 0] = interface.values[0] 
            self.joint_state[i, 1] = interface.values[1]
            self.joint_state[i, 2] = interface.values[2]
    
    def joint_reference_callback(self, msg):
        self.joint_ref[0] = msg.data[0]
        self.joint_ref[1] = msg.data[1]
        self.joint_ref[2] = msg.data[2]
    
    def zero_effort_and_stop(self):
        """Publish zero effort commands and stop timers."""
        self.get_logger().info('Sending zero effort commands and stopping timers...')
        
        # Stop timers to prevent new commands being calculated
        self.outer_loop.cancel()
        self.inner_loop.cancel()
        
        # Publish zero effort three times to ensure it gets through
        self.command = [0.0, 0.0, 0.0]
        for _ in range(3):
            self.send_effort_commands()
        self.get_logger().info('Zero effort sent. Node cleanup complete.')


def main(args=None):
    rclpy.init(args=args)
    arm_effort_controller = ArmEffortController()
    
    # 1. Handle SIGINT (Ctrl+C)
    def sigint_handler(signal, frame):
        arm_effort_controller.get_logger().info('SIGINT received. Initiating graceful shutdown.')
        arm_effort_controller.zero_effort_and_stop()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        rclpy.spin(arm_effort_controller)
    except KeyboardInterrupt:
        pass # The signal handler now manages the shutdown
    finally:
        # 2. Handle normal shutdown (e.g., when rclpy.spin completes)
        if rclpy.ok():
            arm_effort_controller.zero_effort_and_stop()
            arm_effort_controller.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()