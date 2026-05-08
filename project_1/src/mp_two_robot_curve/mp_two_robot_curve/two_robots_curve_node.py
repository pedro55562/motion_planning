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


def curva(t, a=1.0, omega=0.08):
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

        # =====================================================
        # Publishers de velocidade
        # =====================================================
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

        # =====================================================
        # Subscribers de odometria NORMAL
        #
        # Importante:
        # Não é true_odom.
        # Não é pose do Gazebo.
        # É a odometria normal de cada robô.
        # =====================================================
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

        # =====================================================
        # Publishers para debug
        # =====================================================
        self.error_pub_tb1 = self.create_publisher(
            Float64,
            '/tb1/controle/erro_norma',
            10
        )

        self.v_pub_tb1 = self.create_publisher(
            Float64,
            '/tb1/controle/v',
            10
        )

        self.w_pub_tb1 = self.create_publisher(
            Float64,
            '/tb1/controle/w',
            10
        )

        self.error_pub_tb2 = self.create_publisher(
            Float64,
            '/tb2/controle/erro_norma',
            10
        )

        self.v_pub_tb2 = self.create_publisher(
            Float64,
            '/tb2/controle/v',
            10
        )

        self.w_pub_tb2 = self.create_publisher(
            Float64,
            '/tb2/controle/w',
            10
        )

        # =====================================================
        # Tempo
        # =====================================================
        self.start_time = self.get_clock().now()
        self.t = 0.0

        # =====================================================
        # Parâmetros da curva
        # =====================================================
        self.a = 1.0
        self.omega = 0.08

        # Atraso do segundo robô na mesma curva
        self.delay_tb2 = 3.0

        # Parâmetro do robô diferencial
        self.d = 0.03

        # Saturação
        self.max_v = 0.5
        self.max_w = 4.5

        # Ganho de convergência para a curva
        self.k = 2.0

        # =====================================================
        # Poses iniciais desejadas no frame global da curva
        #
        # Como não vamos escolher yaw inicial, usamos somente x/y.
        #
        # tb1 começa em curva(0)
        # tb2 começa em curva(-delay_tb2)
        # =====================================================
        self.spawn_tb1, _ = curva(
            0.0,
            self.a,
            self.omega
        )

        self.spawn_tb2, _ = curva(
            -self.delay_tb2,
            self.a,
            self.omega
        )

        # =====================================================
        # Estados crus da odometria do robô 1
        # =====================================================
        self.odom1_origin_pos = None

        self.raw_x1 = 0.0
        self.raw_y1 = 0.0
        self.raw_theta1 = 0.0

        # Pose estimada no frame global da curva
        self.x1 = 0.0
        self.y1 = 0.0
        self.theta1 = 0.0

        self.odom1_received = False

        # =====================================================
        # Estados crus da odometria do robô 2
        # =====================================================
        self.odom2_origin_pos = None

        self.raw_x2 = 0.0
        self.raw_y2 = 0.0
        self.raw_theta2 = 0.0

        # Pose estimada no frame global da curva
        self.x2 = 0.0
        self.y2 = 0.0
        self.theta2 = 0.0

        self.odom2_received = False

        # =====================================================
        # Publishers para RViz
        #
        # Estou mantendo frame_id = 'odom' para facilitar no RViz.
        # Conceitualmente, este 'odom' está sendo usado como o
        # frame global da curva, com origem no tb1.
        # =====================================================
        self.curve_pub = self.create_publisher(
            Path,
            '/curva_parametrica',
            10
        )

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

        # =====================================================
        # Timers
        # =====================================================
        self.timer = self.create_timer(0.01, self.control_loop)
        self.curve_timer = self.create_timer(1.0, self.publish_curve)

        self.get_logger().info('Nó de controle para tb1 e tb2 iniciado.')

        self.get_logger().info(
            f'Pose recomendada tb1: '
            f'x={self.spawn_tb1[0]:.4f}, '
            f'y={self.spawn_tb1[1]:.4f}'
        )

        self.get_logger().info(
            f'Pose recomendada tb2: '
            f'x={self.spawn_tb2[0]:.4f}, '
            f'y={self.spawn_tb2[1]:.4f}'
        )

        self.get_logger().info(
            f'Parâmetros: a={self.a:.2f}, '
            f'omega={self.omega:.3f}, '
            f'delay_tb2={self.delay_tb2:.2f}, '
            f'max_v={self.max_v:.2f}, '
            f'max_w={self.max_w:.2f}'
        )

    def att_time(self):
        self.t = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def estimar_pose_global(self, raw_pos, raw_theta, odom_origin_pos, spawn_pos):
        """
        Converte odometria local para o frame global da curva.

        Não usa ground truth.
        Não usa pose do Gazebo.

        Usa somente:
            1. deslocamento medido pela odometria
            2. posição inicial conhecida do robô na curva

        Fórmula:
            p_global = p_spawn + (p_odom_atual - p_odom_inicial)

        Como não estamos escolhendo yaw inicial, assumimos que os robôs
        nascem com yaw padrão do launch, normalmente zero.
        """
        delta_odom = raw_pos - odom_origin_pos

        pos_global = spawn_pos + delta_odom

        # Mantemos o yaw vindo da odometria.
        # Isso funciona bem se os robôs nascem com yaw default zero,
        # que é o caso mais comum quando o launch não define yaw.
        theta_global = raw_theta

        return pos_global, theta_global

    def publish_curve(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()

        # Mantido como 'odom' para aparecer fácil no RViz.
        # Conceitualmente é o frame global da curva.
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
        raw_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])

        raw_theta = quaternion_to_yaw(msg.pose.pose.orientation)

        if self.odom1_origin_pos is None:
            self.odom1_origin_pos = raw_pos.copy()

            self.get_logger().info(
                f'tb1 origem odom capturada: '
                f'x={raw_pos[0]:.4f}, '
                f'y={raw_pos[1]:.4f}, '
                f'theta={raw_theta:.4f}'
            )

        pos_global, theta_global = self.estimar_pose_global(
            raw_pos,
            raw_theta,
            self.odom1_origin_pos,
            self.spawn_tb1
        )

        self.raw_x1 = float(raw_pos[0])
        self.raw_y1 = float(raw_pos[1])
        self.raw_theta1 = float(raw_theta)

        self.x1 = float(pos_global[0])
        self.y1 = float(pos_global[1])
        self.theta1 = float(theta_global)

        self.odom1_received = True

    def odom_callback_tb2(self, msg):
        raw_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])

        raw_theta = quaternion_to_yaw(msg.pose.pose.orientation)

        if self.odom2_origin_pos is None:
            self.odom2_origin_pos = raw_pos.copy()

            self.get_logger().info(
                f'tb2 origem odom capturada: '
                f'x={raw_pos[0]:.4f}, '
                f'y={raw_pos[1]:.4f}, '
                f'theta={raw_theta:.4f}'
            )

        pos_global, theta_global = self.estimar_pose_global(
            raw_pos,
            raw_theta,
            self.odom2_origin_pos,
            self.spawn_tb2
        )

        self.raw_x2 = float(raw_pos[0])
        self.raw_y2 = float(raw_pos[1])
        self.raw_theta2 = float(raw_theta)

        self.x2 = float(pos_global[0])
        self.y2 = float(pos_global[1])
        self.theta2 = float(theta_global)

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

    def calcular_controle(self, x, y, theta, t_ref):
        Ainv = (1.0 / self.d) * np.array([
            [self.d * np.cos(theta), self.d * np.sin(theta)],
            [-np.sin(theta),         np.cos(theta)]
        ])

        pd, pddot = curva(t_ref, self.a, self.omega)
        p = np.array([x, y])

        error = pd - p

        uv = pddot + self.k * error

        u = Ainv @ uv

        v = float(u[0])
        w = float(u[1])

        v = float(np.clip(v, -self.max_v, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))

        return v, w, pd, error

    def control_loop(self):
        self.att_time()

        if not self.odom1_received or not self.odom2_received:
            return

        # =====================================================
        # Controle do robô 1
        #
        # tb1 segue curva(t)
        # Como tb1 nasce em curva(0) = (0, 0), ele define o
        # frame global da curva.
        # =====================================================
        v1, w1, pd1, error1 = self.calcular_controle(
            self.x1,
            self.y1,
            self.theta1,
            self.t
        )

        self.publish_cmd_tb1(v1, w1)
        self.publish_tracking_point_tb1(pd1)

        self.error_pub_tb1.publish(
            Float64(data=float(np.linalg.norm(error1)))
        )

        self.v_pub_tb1.publish(Float64(data=float(v1)))
        self.w_pub_tb1.publish(Float64(data=float(w1)))

        # =====================================================
        # Controle do robô 2
        #
        # tb2 segue a mesma curva, mas atrasado no tempo.
        # Aqui NÃO fazemos clamp para zero.
        #
        # No instante inicial:
        #   t = 0
        #   t2 = -delay_tb2
        #
        # Então o alvo inicial do tb2 é curva(-delay_tb2),
        # que é exatamente onde ele deve nascer.
        # =====================================================
        t2 = self.t - self.delay_tb2

        v2, w2, pd2, error2 = self.calcular_controle(
            self.x2,
            self.y2,
            self.theta2,
            t2
        )

        self.publish_cmd_tb2(v2, w2)
        self.publish_tracking_point_tb2(pd2)

        self.error_pub_tb2.publish(
            Float64(data=float(np.linalg.norm(error2)))
        )

        self.v_pub_tb2.publish(Float64(data=float(v2)))
        self.w_pub_tb2.publish(Float64(data=float(w2)))

        self.get_logger().info(
            f"tb1: "
            f"pos_global=({self.x1:.2f}, {self.y1:.2f}), "
            f"v={v1:.2f}, w={w1:.2f}, "
            f"erro={np.linalg.norm(error1):.2f} | "
            f"tb2: "
            f"pos_global=({self.x2:.2f}, {self.y2:.2f}), "
            f"v={v2:.2f}, w={w2:.2f}, "
            f"erro={np.linalg.norm(error2):.2f}",
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