#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


class CmdVelPublisher(Node):
    def __init__(self) -> None:
        super().__init__('cmd_vel_publisher')

        self.publisher_ = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('CmdVelPublisher started. Publishing to /cmd_vel')

    def timer_callback(self) -> None:
        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Comando para frente
        msg.twist.linear.x = 1.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        # Sem rotação
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 1.0

        self.publisher_.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Publica comando zero ao sair
        stop_msg = TwistStamped()
        stop_msg.header.stamp = node.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        stop_msg.twist.linear.x = 0.0
        stop_msg.twist.angular.z = 0.0
        node.publisher_.publish(stop_msg)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()