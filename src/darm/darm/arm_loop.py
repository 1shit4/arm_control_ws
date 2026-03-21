import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np

class ArmPositionCommander(Node):
    def __init__(self):
        super().__init__('arm_position_commander')

        # Publisher
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_reference', 10)

        # Define min and max joint limits
        self.min_pos = np.array([-2.5, -1.8, 0.0])
        self.max_pos = np.array([-0.93, 1.8, 0.0])

        # Define fixed target sequence (min <-> max)
        self.targets = [self.min_pos, self.max_pos]
        self.target_index = 1  # start by moving toward max

        # Start at min position
        self.current_pos = np.copy(self.min_pos)
        self.target_pos = self.targets[self.target_index]

        # Speed control (fixed speed now)
        self.step_size = 0.2

        # Timer for loop updates
        self.timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(
            'ArmPositionCommander started: moving between min and max poses.'
        )

    def switch_target(self):
        """Switch between min and max positions."""
        self.target_index = 1 - self.target_index
        self.target_pos = self.targets[self.target_index]
        self.get_logger().info(f'Switching target to: {self.target_pos}')

    def timer_callback(self):
        direction = self.target_pos - self.current_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-3:
            self.switch_target()
            return

        step_size = min(self.step_size, distance)
        step = step_size * direction / (distance + 1e-9)
        self.current_pos += step

        # Publish command
        cmd_msg = Float64MultiArray()
        cmd_msg.data = self.current_pos.tolist()
        self.cmd_pub.publish(cmd_msg)

        self.get_logger().debug(f'Current pos: {self.current_pos}')

def main(args=None):
    rclpy.init(args=args)
    node = ArmPositionCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
