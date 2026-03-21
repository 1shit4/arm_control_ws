#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicJointState

class ArmEffortController(Node):
    def __init__(self, n_joints=3):
        super().__init__('arm_effort_controller')

        self.n = n_joints
        # 1. LIMIT: Ensure this matches your URDF limits
        self.max_effort = 100.0 

        self.effort_pub = self.create_publisher(Float64MultiArray, '/arm_effort_controller/commands', 10)
        self.joint_sen_ = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, qos_profile_sensor_data)
        self.joint_ref_ = self.create_subscription(Float64MultiArray, '/joint_reference', self.joint_reference_callback, qos_profile_sensor_data)

        self.curr_pos = np.zeros(self.n)
        self.curr_vel = np.zeros(self.n)
        self.des_pos = np.zeros(self.n)
        
        # --- TUNING PARAMETERS ---
        
        # 2. INERTIA SCALING: This is the critical fix.
        # Estimate the inertia of your links (L1, L2, L3). 
        # Start small (0.05) and increase if the robot is sluggish.
        self.inertia_estimates = np.array([0.1, 0.05, 0.02]) 

        # 3. GAINS: Now you can increase these because of the scaling above.
        # Try Lambda=2.0, Phi=5.0 now.
        self.Lam_m = np.array([0,0, 3.0]) 
        self.Phi_m = np.array([0,0, 5.0]) 

        # Adaptive term (SMC)
        self.Kp0_m = np.array([0.1, 0.1, 0.1]) 
        self.alpha_m = np.array([0.5, 0.5, 0.5])
        self.v_m = 0.5 

        self.control_rate = 0.01 
        self.timer = self.create_timer(self.control_rate, self.control_loop)
        self.pre_time = self.get_clock().now().nanoseconds * 1e-9

    def joint_state_callback(self, msg):
        L = min(len(msg.interface_values), self.n)
        for i in range(L):
            vals = msg.interface_values[i].values
            if len(vals) >= 1: self.curr_pos[i] = vals[0]
            if len(vals) >= 2: self.curr_vel[i] = vals[1]
    
    def joint_reference_callback(self, msg):
        num = min(len(msg.data), self.n)
        self.des_pos[:num] = np.array(msg.data[:num])

    def control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self.pre_time
        self.pre_time = now
        if dt > 0.1 or dt <= 0.0: dt = 0.01

        # 4. GRAVITY COMPENSATION (Simplified Model)
        # You need to calculate torque required to hold the arm up.
        # This is a basic Cosine approximation. You might need to adjust signs (+/-)
        # depending on your URDF orientation.
        g_force = 9.81
        tau_gravity = np.zeros(3)
        
        # Example: Joint 1 (Base) usually doesn't need gravity comp if it rotates around Z.
        # Joint 2 (Shoulder): Needs a lot.
        # Joint 3 (Elbow): Needs some.
        # Tune these 'mass_radius' values until the arm roughly holds position with 0 gains.
        tau_gravity[0] = 0.0 
        tau_gravity[1] = 2.0 * g_force * np.cos(self.curr_pos[1]) 
        tau_gravity[2] = 0.5 * g_force * np.cos(self.curr_pos[1] + self.curr_pos[2])

        # --- SMC CONTROL CALCULATION ---
        errPos = self.curr_pos - self.des_pos
        errVel = self.curr_vel # Assuming desired vel is 0
        
        sv = errVel + np.multiply(self.Phi_m, errPos)
        
        # Adaptive Update
        self.Kp0_m = (np.abs(sv) - np.multiply(self.alpha_m, self.Kp0_m)) * dt
        self.Kp0_m = np.maximum(self.Kp0_m, 0.01 * np.ones(3))
        
        # Switching Term
        delTau = np.zeros(3)
        for i in range(3):
            # Soft sigmoid
            delTau[i] = self.Kp0_m[i] * (sv[i] / (np.abs(sv[i]) + self.v_m))

        # Desired Acceleration (Virtual Control)
        des_accel = -np.multiply(self.Lam_m, sv) - delTau

        # 5. FINAL TORQUE CALCULATION
        # Torque = (Inertia * Accel) + Gravity + Friction(optional)
        efforts = np.zeros(self.n)
        for i in range(3):
            # Apply Inertia Scaling
            tau_ctrl = des_accel[i] * self.inertia_estimates[i]
            
            # Combine
            total_tau = tau_ctrl + tau_gravity[i]
            
            # Clip
            efforts[i] = np.clip(total_tau, -self.max_effort, self.max_effort)

        msg = Float64MultiArray()
        msg.data = efforts.tolist()
        self.effort_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArmEffortController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()