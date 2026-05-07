#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64
from rclpy.executors import ExternalShutdownException


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def curva(t, a=1.5, omega=0.15):
    s = omega * t

    p = np.array([
        a * np.sin(2.0 * s),
        a * np.sin(3.0 * s)
    ])

    pdot = np.array([
        2.0 * a * omega * np.cos(2.0 * s),
        3.0 * a * omega * np.cos(3.0 * s)
    ])

    return p, pdot


class TwoRobotsCurveNode(Node):

    def __init__(self):
        super().__init__('two_robots_curve_node')

        # ================================
        # Publishers de velocidade
        # ================================
        self.cmd_pub_tb1 = self.create_publisher(
            TwistStamped,
            '/tb1/cmd_vel',
            10
        )

        self.cmd_pub_tb2 = self.create_publisher(
            TwistStamped,
            '/tb2/cmd_vel',
            10
        )

        # ================================
        # Subscribers de odometria
        # ================================
        self.odom_sub_tb1 = self.create_subscription(
            Odometry,
            '/tb1/odom',
            self.odom_callback_tb1,
            10
        )

        self.odom_sub_tb2 = self.create_subscription(
            Odometry,
            '/tb2/odom',
            self.odom_callback_tb2,
            10
        )

        # ================================
        # Publishers para debug
        # ================================
        self.error_pub_tb1 = self.create_publisher(Float64, '/tb1/controle/erro_norma', 10)
        self.v_pub_tb1 = self.create_publisher(Float64, '/tb1/controle/v', 10)
        self.w_pub_tb1 = self.create_publisher(Float64, '/tb1/controle/w', 10)

        self.error_pub_tb2 = self.create_publisher(Float64, '/tb2/controle/erro_norma', 10)
        self.v_pub_tb2 = self.create_publisher(Float64, '/tb2/controle/v', 10)
        self.w_pub_tb2 = self.create_publisher(Float64, '/tb2/controle/w', 10)

        # ================================
        # Estados do robô 1
        # ================================
        self.x1 = 0.0
        self.y1 = 0.0
        self.theta1 = 0.0
        self.odom1_received = False

        # ================================
        # Estados do robô 2
        # ================================
        self.x2 = 0.0
        self.y2 = 0.0
        self.theta2 = 0.0
        self.odom2_received = False

        # ================================
        # Tempo
        # ================================
        self.start_time = self.get_clock().now()
        self.t = 0.0

        # ================================
        # Parâmetros da curva
        # ================================
        self.a = 1.0
        self.omega = 0.08

        # atraso do segundo robô
        self.delay_tb2 = 2.0

        # parâmetro do robô diferencial
        self.d = 0.03

        # saturação
        self.max_v = 0.22
        self.max_w = 2.84

        # ================================
        # Publishers para RViz
        # ================================
        self.curve_pub = self.create_publisher(Path, '/curva_parametrica', 10)

        self.tracking_point_pub_tb1 = self.create_publisher(
            PointStamped,
            '/tb1/ponto_rastreio',
            10
        )

        self.tracking_point_pub_tb2 = self.create_publisher(
            PointStamped,
            '/tb2/ponto_rastreio',
            10
        )

        # ================================
        # Timers
        # ================================
        self.timer = self.create_timer(0.01, self.control_loop)
        self.curve_timer = self.create_timer(1.0, self.publish_curve)

        self.get_logger().info('Nó de controle para tb1 e tb2 iniciado.')

    def att_time(self):
        self.t = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def publish_curve(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'

        T = 2.0 * np.pi / self.omega
        ts = np.linspace(0.0, T, 500)

        for t in ts:
            p_ref, _ = curva(t, self.a, self.omega)

            pose = PoseStamped()
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = 'odom'

            pose.pose.position.x = float(p_ref[0])
            pose.pose.position.y = float(p_ref[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path.poses.append(pose)

        self.curve_pub.publish(path)

    def publish_tracking_point_tb1(self, pd):
        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = 'odom'

        point.point.x = float(pd[0])
        point.point.y = float(pd[1])
        point.point.z = 0.0

        self.tracking_point_pub_tb1.publish(point)

    def publish_tracking_point_tb2(self, pd):
        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = 'odom'

        point.point.x = float(pd[0])
        point.point.y = float(pd[1])
        point.point.z = 0.0

        self.tracking_point_pub_tb2.publish(point)

    def odom_callback_tb1(self, msg):
        self.x1 = msg.pose.pose.position.x
        self.y1 = msg.pose.pose.position.y
        self.theta1 = quaternion_to_yaw(msg.pose.pose.orientation)
        self.odom1_received = True

    def odom_callback_tb2(self, msg):
        self.x2 = msg.pose.pose.position.x
        self.y2 = msg.pose.pose.position.y
        self.theta2 = quaternion_to_yaw(msg.pose.pose.orientation)
        self.odom2_received = True

    def publish_cmd_tb1(self, v, w):
        cmd = TwistStamped()

        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        cmd.twist.linear.x = float(v)
        cmd.twist.angular.z = float(w)

        self.cmd_pub_tb1.publish(cmd)

    def publish_cmd_tb2(self, v, w):
        cmd = TwistStamped()

        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        cmd.twist.linear.x = float(v)
        cmd.twist.angular.z = float(w)

        self.cmd_pub_tb2.publish(cmd)

    def stop_robots(self):
        self.publish_cmd_tb1(0.0, 0.0)
        self.publish_cmd_tb2(0.0, 0.0)
        self.get_logger().info('Robôs parados.')

    def control_loop(self):
        self.att_time()

        if not self.odom1_received or not self.odom2_received:
            return

        # =====================================================
        # Controle do robô 1
        # =====================================================
        theta = self.theta1

        Ainv = (1.0 / self.d) * np.array([
            [self.d * np.cos(theta), self.d * np.sin(theta)],
            [-np.sin(theta),         np.cos(theta)]
        ])

        pd1, pddot1 = curva(self.t, self.a, self.omega)
        p1 = np.array([self.x1, self.y1])

        error1 = pd1 - p1
        uv1 = pddot1 + 2.0 * error1

        u1 = Ainv @ uv1

        v1 = u1[0]
        w1 = u1[1]

        v1 = np.clip(v1, -self.max_v, self.max_v)
        w1 = np.clip(w1, -self.max_w, self.max_w)

        self.publish_cmd_tb1(v1, w1)
        self.publish_tracking_point_tb1(pd1)

        self.error_pub_tb1.publish(Float64(data=float(np.linalg.norm(error1))))
        self.v_pub_tb1.publish(Float64(data=float(v1)))
        self.w_pub_tb1.publish(Float64(data=float(w1)))

        # =====================================================
        # Controle do robô 2
        # Mesmo ponto da curva, mas atrasado 2 segundos
        # =====================================================
        theta = self.theta2

        Ainv = (1.0 / self.d) * np.array([
            [self.d * np.cos(theta), self.d * np.sin(theta)],
            [-np.sin(theta),         np.cos(theta)]
        ])

        t2 = self.t - self.delay_tb2
        if t2 < 0.0:
            t2 = 0.0

        pd2, pddot2 = curva(t2, self.a, self.omega)
        p2 = np.array([self.x2, self.y2])

        error2 = pd2 - p2
        uv2 = pddot2 + 2.0 * error2

        u2 = Ainv @ uv2

        v2 = u2[0]
        w2 = u2[1]

        v2 = np.clip(v2, -self.max_v, self.max_v)
        w2 = np.clip(w2, -self.max_w, self.max_w)

        self.publish_cmd_tb2(v2, w2)
        self.publish_tracking_point_tb2(pd2)

        self.error_pub_tb2.publish(Float64(data=float(np.linalg.norm(error2))))
        self.v_pub_tb2.publish(Float64(data=float(v2)))
        self.w_pub_tb2.publish(Float64(data=float(w2)))

        self.get_logger().info(
            f"tb1: v={v1:.2f}, w={w1:.2f}, erro={np.linalg.norm(error1):.2f} | "
            f"tb2: v={v2:.2f}, w={w2:.2f}, erro={np.linalg.norm(error2):.2f}",
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)

    node = TwoRobotsCurveNode()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('Shutdown recebido.')

    finally:
        node.stop_robots()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
