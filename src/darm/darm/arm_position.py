import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np

class ArmPositionCommander(Node):
    def __init__(self):
        super().__init__('arm_position_commander')

        # Publisher
        # self.cmd_pub = self.create_publisher(Float64MultiArray, '/arm_position_controller/commands', 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_reference', 10)

        # Define min and max joint limits
        self.min_pos = np.array([-2.5, -1.8, 0.0])
        self.max_pos = np.array([-0.93, 1.8, 0.0])

        # Current and target joint positions
        self.current_pos = np.copy(self.min_pos)
        self.target_pos = self.generate_random_target()

        # Speed control
        self.max_step_size = 0.5       # maximum speed per step
        self.min_step_size = 0.1       # minimum speed per step
        self.current_step_size = np.random.uniform(self.min_step_size, self.max_step_size)

        # Timer for loop updates
        self.timer_period = 0.05        # seconds (20 Hz update rate)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info('ArmPositionCommander started: moving joints randomly with variable speed.')

    def generate_random_target(self):
        """Generate a random target within the joint limits."""
        random_target = np.random.uniform(self.min_pos, self.max_pos)
        self.get_logger().info(f'New random target: {random_target}')
        return random_target

    def timer_callback(self):
        # Compute direction vector toward the target
        direction = self.target_pos - self.current_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-3:
            self.target_pos = self.generate_random_target()
            self.current_step_size = np.random.uniform(
                self.min_step_size, self.max_step_size
            )
        else:
            self.current_step_size += np.random.uniform(-0.005, 0.005)
            self.current_step_size = np.clip(
                self.current_step_size, self.min_step_size, self.max_step_size
            )

            step_size = min(self.current_step_size, distance)
            step = step_size * direction / (distance + 1e-9)
            self.current_pos += step

        # Publish command
        cmd_msg = Float64MultiArray()
        cmd_msg.data = self.current_pos.tolist()
        self.cmd_pub.publish(cmd_msg)

        # Debug logging (change to info if you want more frequent updates)
        self.get_logger().debug(
            f'Current pos: {self.current_pos}, step size: {self.current_step_size:.3f}'
        )

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
