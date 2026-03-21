#!/usr/bin/env python3
import sys
# import jax.numpy as jnp

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import math
import numpy as np
import signal

from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicJointState

class ArmEffortController(Node):
    def __init__(self, n_joints=3):
        super().__init__('arm_effort_controller')

        self.n = n_joints
        self.max_effort = 10.0
        self.pre_time = self.get_clock().now().nanoseconds * 1e-9


        self.effort_pub = self.create_publisher(Float64MultiArray, '/arm_effort_controller/commands', 10)
        self.joint_sen_ = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, qos_profile_sensor_data)
        self.joint_ref_ = self.create_subscription(Float64MultiArray, '/joint_reference', self.joint_reference_callback, qos_profile_sensor_data)

        self.curr_pos = np.zeros(self.n)
        self.curr_vel = np.zeros(self.n)
        self.curr_eff = np.zeros(self.n)

        self.des_pos = np.zeros(self.n)
        self.des_vel = np.zeros(self.n)
        self.des_acc = np.zeros(self.n)

        self.gravity_comp = [1.0, 0.16, 0.0]
      
        # Gains
        self.Lam_m = np.array([0.1, 0.1, 2.0])
        self.Phi_m = np.array([20, 20, 0.2])
        self.M_bar = 0.05
        self.Kp0_m, self.alpha_m, self.v_m = (
            np.array([0.1, 0.1, 0.1]),
            np.array([1, 1, 1]),
            0.1,
        ) 

        self.control_rate = 0.01
        self.timer = self.create_timer(self.control_rate, self.control_loop)

        self.get_logger().info(f'ArmEffortController ready for {self.n} joints.')


    def joint_state_callback(self, msg):
        L = min(len(msg.interface_values), self.n)
        for i in range(L):
            vals = msg.interface_values[i].values
            if len(vals) >= 1:
                self.curr_pos[i] = vals[0]
            if len(vals) >= 2:
                self.curr_vel[i] = vals[1]
            if len(vals) >= 3:
                self.curr_eff[i] = vals[2]
    
    def joint_reference_callback(self, msg):
        num_inputs = min(len(msg.data), self.n)
        self.des_pos[:num_inputs] = np.array(msg.data[:num_inputs])

        # Trigger control computation for first two joints
        self.des_eff(active_joints=[0, 1])

    def control_loop(self):
        self.des_eff(active_joints=[0, 1])


    def des_eff(self, active_joints=[0, 1]):
        # _, accel = self.get_obs()

        dt = self.get_clock().now().nanoseconds * 1e-9 - self.pre_time
        self.pre_time = self.pre_time + dt

        if dt > 0.04:
            dt = 0.04

        errPos = self.curr_pos - self.des_pos
        errVel = self.curr_vel - self.des_vel
        print(errPos)

        sv = errVel + np.multiply(self.Phi_m, errPos)
        # print(sv)
        
        # if self.armed:
        # self.Kp0_m = (np.linalg.norm(sv) - np.multiply(self.alpha_m, self.Kp0_m)) * dt
        # self.Kp0 += (sv - np.multiply(self.alpha_0, self.Kp0)) * dt
        # self.Kp0_m = np.maximum(self.Kp0_m, 0.0001 * np.ones(3))

        # Rho = self.Kp0_m 
        Rho = np.zeros(3)

        delTau = np.zeros(3)
        delTau[0] = Rho[0] * self.sigmoid(sv[0], self.v_m)
        delTau[1] = Rho[1] * self.sigmoid(sv[1], self.v_m)
        delTau[2] = Rho[2] * self.sigmoid(sv[2], self.v_m)



        des_a = (
            -np.multiply(self.Lam_m, sv)
            # + self.M_bar * (self.des_acc - np.multiply(self.Phi_m, errVel))
            # + self.currN
            # - delTau
        )

        # if np.linalg.norm(des_a) > self.max_th:
        #     des_a = (self.max_th / np.linalg.norm(des_a)) * des_a

        efforts = np.zeros(self.n)
        for i in active_joints:
            efforts[i] = np.clip(des_a[i], -self.max_effort, self.max_effort)

        # print(efforts[0:2])

        # Publish efforts
        effort_msg = Float64MultiArray()
        effort_msg.data = efforts.tolist()
        self.effort_pub.publish(effort_msg)

        # self.get_logger().info(f'Sent effort cmd (active joints {active_joints}): {efforts}')


    def sigmoid(self, s, v):
        if np.absolute(s) > v:
            return s / np.absolute(s)
        else:
            return s / v
        


def main(args=None):
    rclpy.init(args=args)
    node = ArmEffortController(n_joints=3)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

