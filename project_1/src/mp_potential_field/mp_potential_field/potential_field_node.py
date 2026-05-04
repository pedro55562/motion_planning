import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException


class PotentialFieldNode(Node):

    def __init__(self):
        super().__init__('potential_field_node')
        self.get_logger().info('Potential Field Node iniciado!')


def main(args=None):
    rclpy.init(args=args)

    node = PotentialFieldNode()

    try:
        node.get_logger().info('Nó iniciado.')
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('Shutdown recebido.')

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()