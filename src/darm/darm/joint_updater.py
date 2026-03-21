import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import math
import numpy as np
import signal
from time import time

from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicJointState

class JointUpdate(Node):
    def __init__(self):
        super().__init__('jointUpdater')

        self.dt = 1
        self.joint_ref = [0.0, 0.0, 0.0]
        self.time_init = time()
        self.joint_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_reference',
            10
        )
        self.joint_loop = self.outer_loop = self.create_timer(self.dt, self.traj_gen_)

    
    def traj_gen_(self):
        t = time() - self.time_init
        l1l2 = 1.57*( 1 - max(0, math.sin(t/10)))
        l2l3 = 0.0
        l3l4 = 0.0


        self.joint_ref = [l1l2, l2l3, l3l4]
        self.traj_send_()

    
    def traj_send_(self):
        joint_updates = Float64MultiArray()
        joint_updates.data = self.joint_ref
        self.joint_pub.publish(joint_updates)
        self.get_logger().info(f'Sent effort commands: {joint_updates.data}')

def main(args=None) -> None:
    print('Sending_joint_states...')
    rclpy.init(args=args)
    joint_update = JointUpdate()
    rclpy.spin(joint_update)
    joint_update.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)