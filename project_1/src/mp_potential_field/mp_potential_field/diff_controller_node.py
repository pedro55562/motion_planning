import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, Vector3Stamped

# Converter de quaternion para yaw, funcao feita com base em:
# https://gist.github.com/michaelwro/1450283a6a1226eaf707d9adde378798
def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

class DiffControllerNode(Node):

    def __init__(self):
        super().__init__('diff_controller_node')
        self.get_logger().info('Diff Controller Node iniciado!')

        self.cmd_pub = self.create_publisher(
            msg_type=TwistStamped,
            topic='/cmd_vel',
            qos_profile=10
        )
        
        self.odom_sub = self.create_subscription(
            msg_type=Odometry,
            topic='/odom',
            callback=self.odom_callback,
            qos_profile=10
        )
        
        self.desired_vel_sub = self.create_subscription(
            Vector3Stamped,
            '/desired_xy_vel',
            self.desired_vel_callback,
            10
        )
        
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.xdot_d = 0.0
        self.ydot_d = 0.0

        self.timer = self.create_timer(0.01, self.control_loop)

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = quaternion_to_yaw(msg.pose.pose.orientation)

    def publish_cmd(self, v, w):
        cmd = TwistStamped()

        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        cmd.twist.linear.x = float(v)
        cmd.twist.angular.z = float(w)

        self.cmd_pub.publish(cmd)
        
    def desired_vel_callback(self, msg):
        self.xdot_d = msg.vector.x
        self.ydot_d = msg.vector.y

    def control_loop(self):        
        d = .1
        theta = self.theta
        Ainv = (1.0 / d) * np.array([
            [d * np.cos(theta), d * np.sin(theta)],
            [-np.sin(theta),    np.cos(theta)]
        ])        
        
        uv = np.array([self.xdot_d, self.ydot_d])
                
        u = Ainv @ uv
    
        v=u[0]
        w=u[1]
        v = np.clip(v, -0.22, 0.22)
        w = np.clip(w, -2.84, 2.84)
        self.publish_cmd(v, w)
        self.get_logger().info(
            f"v={v:.2f}, w={w:.2f}",
            throttle_duration_sec=1.0
        )
        
        
        
        
        
        
def main(args=None):
    rclpy.init(args=args)

    node = DiffControllerNode()

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