#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64


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


def saturate_vector(v, max_norm):
    norm = np.linalg.norm(v)

    if norm < 1e-9:
        return v

    if norm > max_norm:
        return (max_norm / norm) * v

    return v


class RobotInfo:
    def __init__(self, name, spawn):
        self.name = name
        self.spawn = np.array(spawn, dtype=float)

        self.odom_origin_pos = None

        self.raw_pos = np.array([0.0, 0.0])
        self.raw_theta = 0.0

        self.q = np.array([0.0, 0.0])
        self.theta = 0.0

        self.odom_received = False
        self.scan = None


class MultiRobotAvoidanceNode(Node):

    def __init__(self):
        super().__init__('multi_robot_avoidance_node')

        self.get_logger().info('Multi Robot Avoidance Node iniciado.')

        # =====================================================
        # Parâmetros da mesma curva usada pelo controlador
        #
        # Precisa bater com two_robots_curve_node.py.
        # =====================================================
        self.declare_parameter('a', 1.0)
        self.declare_parameter('omega', 0.08)
        self.declare_parameter('delay_tb2', 3.0)

        self.a = float(self.get_parameter('a').value)
        self.omega = float(self.get_parameter('omega').value)
        self.delay_tb2 = float(self.get_parameter('delay_tb2').value)

        # =====================================================
        # Parâmetros dos potenciais repulsivos
        # =====================================================
        self.declare_parameter('obstacle_Q_star', 0.60)
        self.declare_parameter('obstacle_eta', 0.001)

        self.declare_parameter('robot_Q_star', 0.80)
        self.declare_parameter('robot_eta', 0.005)
        self.declare_parameter('robot_radius', 0.105)


        self.obstacle_Q_star = float(self.get_parameter('obstacle_Q_star').value)
        self.obstacle_eta = float(self.get_parameter('obstacle_eta').value)

        self.robot_Q_star = float(self.get_parameter('robot_Q_star').value)
        self.robot_eta = float(self.get_parameter('robot_eta').value)
        self.robot_radius = float(self.get_parameter('robot_radius').value)



        # =====================================================
        # Poses iniciais globais
        #
        # tb1 define o frame global:
        #   tb1 nasce em curva(0) = (0, 0)
        #
        # tb2 nasce atrasado:
        #   tb2 nasce em curva(-delay_tb2)
        #
        # Isso precisa ser igual ao robots.yaml.
        # =====================================================
        spawn_tb1, _ = curva(0.0, self.a, self.omega)
        spawn_tb2, _ = curva(-self.delay_tb2, self.a, self.omega)

        self.tb1 = RobotInfo('tb1', spawn_tb1)
        self.tb2 = RobotInfo('tb2', spawn_tb2)

        # =====================================================
        # Subscribers
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

        self.scan_sub_tb1 = self.create_subscription(
            LaserScan,
            '/tb1/scan',
            self.scan_callback_tb1,
            10
        )

        self.scan_sub_tb2 = self.create_subscription(
            LaserScan,
            '/tb2/scan',
            self.scan_callback_tb2,
            10
        )

        # =====================================================
        # Publishers principais
        #
        # Estes são os vetores que o controlador da curva deve somar.
        # =====================================================
        self.avoid_pub_tb1 = self.create_publisher(
            Vector3Stamped,
            '/tb1/avoidance/qdot',
            10
        )

        self.avoid_pub_tb2 = self.create_publisher(
            Vector3Stamped,
            '/tb2/avoidance/qdot',
            10
        )

        # =====================================================
        # Publishers de debug
        # =====================================================
        self.obs_pub_tb1 = self.create_publisher(
            Vector3Stamped,
            '/tb1/avoidance/obstacles_qdot',
            10
        )

        self.obs_pub_tb2 = self.create_publisher(
            Vector3Stamped,
            '/tb2/avoidance/obstacles_qdot',
            10
        )

        self.robot_pub_tb1 = self.create_publisher(
            Vector3Stamped,
            '/tb1/avoidance/robots_qdot',
            10
        )

        self.robot_pub_tb2 = self.create_publisher(
            Vector3Stamped,
            '/tb2/avoidance/robots_qdot',
            10
        )

        self.dist_pub = self.create_publisher(
            Float64,
            '/avoidance/dist_tb1_tb2',
            10
        )

        # =====================================================
        # Timer
        # =====================================================
        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info(
            f'spawn tb1 = ({self.tb1.spawn[0]:.4f}, {self.tb1.spawn[1]:.4f})'
        )

        self.get_logger().info(
            f'spawn tb2 = ({self.tb2.spawn[0]:.4f}, {self.tb2.spawn[1]:.4f})'
        )

        self.get_logger().info(
            f'obstacle_Q_star={self.obstacle_Q_star:.2f}, '
            f'obstacle_eta={self.obstacle_eta:.5f}, '
            f'robot_Q_star={self.robot_Q_star:.2f}, '
            f'robot_eta={self.robot_eta:.5f}, '
            f'robot_radius={self.robot_radius:.3f}'
        )

    # =========================================================
    # Callbacks
    # =========================================================
    def update_robot_from_odom(self, robot, msg):
        raw_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])

        raw_theta = quaternion_to_yaw(msg.pose.pose.orientation)

        if robot.odom_origin_pos is None:
            robot.odom_origin_pos = raw_pos.copy()

            self.get_logger().info(
                f'{robot.name}: origem odom capturada '
                f'raw=({raw_pos[0]:.4f}, {raw_pos[1]:.4f}), '
                f'theta={raw_theta:.4f}'
            )

        # =====================================================
        # Transformação consistente com o controlador da curva:
        #
        # q_global = q_spawn + (q_odom_atual - q_odom_inicial)
        #
        # Não usa ground truth.
        # Usa apenas a posição inicial conhecida e odometria.
        # =====================================================
        delta_odom = raw_pos - robot.odom_origin_pos

        robot.raw_pos = raw_pos
        robot.raw_theta = raw_theta

        robot.q = robot.spawn + delta_odom
        robot.theta = raw_theta

        robot.odom_received = True

    def odom_callback_tb1(self, msg):
        self.update_robot_from_odom(self.tb1, msg)

    def odom_callback_tb2(self, msg):
        self.update_robot_from_odom(self.tb2, msg)

    def scan_callback_tb1(self, msg):
        self.tb1.scan = msg

    def scan_callback_tb2(self, msg):
        self.tb2.scan = msg

    # =========================================================
    # Publicação
    # =========================================================
    def publish_vector(self, pub, qdot):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        # Mesmo frame global usado no controlador da curva.
        msg.header.frame_id = 'odom'

        msg.vector.x = float(qdot[0])
        msg.vector.y = float(qdot[1])
        msg.vector.z = 0.0

        pub.publish(msg)

    # =========================================================
    # Obstáculos via laser
    # =========================================================
    def find_obstacle_minima(self, robot, Q_star):
        scan = robot.scan

        if scan is None:
            return []

        ranges = np.array(scan.ranges, dtype=float)
        obstacles = []
        cluster = []

        for i, r in enumerate(ranges):
            valid = (
                np.isfinite(r)
                and scan.range_min <= r <= scan.range_max
                and r <= Q_star
            )

            if valid:
                cluster.append(i)
            else:
                if len(cluster) > 0:
                    obs = self.extract_minimum_from_cluster(
                        robot,
                        scan,
                        cluster,
                        ranges
                    )
                    obstacles.append(obs)
                    cluster = []

        if len(cluster) > 0:
            obs = self.extract_minimum_from_cluster(
                robot,
                scan,
                cluster,
                ranges
            )
            obstacles.append(obs)

        return obstacles

    def extract_minimum_from_cluster(self, robot, scan, cluster, ranges):
        min_idx = cluster[int(np.argmin(ranges[cluster]))]
        d_i = float(ranges[min_idx])

        angle = scan.angle_min + min_idx * scan.angle_increment

        # Direção do feixe no frame do robô/base_scan.
        # Assumindo base_scan alinhado com base_link.
        dir_base = np.array([
            math.cos(angle),
            math.sin(angle)
        ])

        # grad d_i(q):
        # direção que aumenta distância ao obstáculo.
        #
        # Se o laser mede obstáculo à frente, dir_base aponta para o obstáculo.
        # Para se afastar do obstáculo, o gradiente é o oposto.
        grad_base = -dir_base

        # Converte para o frame global usado pela curva.
        c = math.cos(robot.theta)
        s = math.sin(robot.theta)

        R = np.array([
            [c, -s],
            [s,  c]
        ])

        grad_global = R @ grad_base

        return d_i, grad_global

    def obstacle_repulsive_gradient(self, robot):
        if robot.scan is None:
            return np.array([0.0, 0.0])

        obstacles = self.find_obstacle_minima(
            robot,
            self.obstacle_Q_star
        )

        qdot_rep = np.array([0.0, 0.0])

        for d_i, grad_d_i in obstacles:
            if d_i < 1e-6:
                continue

            qdot_i = (
                self.obstacle_eta
                * (1.0 / d_i - 1.0 / self.obstacle_Q_star)
                * (1.0 / (d_i ** 2))
                * grad_d_i
            )

            qdot_rep += qdot_i

        return qdot_rep

    # =========================================================
    # Repulsão robô-robô
    # =========================================================
    def robot_robot_repulsive_gradient(self, robot, other_robot):
        if not robot.odom_received or not other_robot.odom_received:
            return np.array([0.0, 0.0])

        delta = robot.q - other_robot.q
        d_center = float(np.linalg.norm(delta))

        if d_center < 1e-6:
            # Caso degenerado: evita divisão por zero.
            return np.array([0.0, 0.0])

        # Distância aproximada entre as "bordas" dos robôs.
        d_surface = d_center - 2.0 * self.robot_radius

        # Se as bordas já estão muito próximas ou sobrepostas,
        # força uma distância mínima para não explodir numericamente.
        d_surface = max(d_surface, 0.03)

        if d_surface > self.robot_Q_star:
            return np.array([0.0, 0.0])

        # Gradiente que aumenta distância ao outro robô.
        grad_d = delta / d_center

        qdot_rep = (
            self.robot_eta
            * (1.0 / d_surface - 1.0 / self.robot_Q_star)
            * (1.0 / (d_surface ** 2))
            * grad_d
        )

        return qdot_rep

    # =========================================================
    # Loop principal
    # =========================================================
    def control_loop(self):
        if not self.tb1.odom_received or not self.tb2.odom_received:
            return

        # Obstáculos estáticos detectados pelo laser.
        qdot_obs_tb1 = self.obstacle_repulsive_gradient(self.tb1)
        qdot_obs_tb2 = self.obstacle_repulsive_gradient(self.tb2)

        # Repulsão entre robôs.
        qdot_robot_tb1 = self.robot_robot_repulsive_gradient(
            self.tb1,
            self.tb2
        )

        qdot_robot_tb2 = self.robot_robot_repulsive_gradient(
            self.tb2,
            self.tb1
        )

        # Soma total.
        qdot_total_tb1 = qdot_obs_tb1 + qdot_robot_tb1
        qdot_total_tb2 = qdot_obs_tb2 + qdot_robot_tb2



        # Publica total.
        self.publish_vector(self.avoid_pub_tb1, qdot_total_tb1)
        self.publish_vector(self.avoid_pub_tb2, qdot_total_tb2)

        # Publica componentes para debug.
        self.publish_vector(self.obs_pub_tb1, qdot_obs_tb1)
        self.publish_vector(self.obs_pub_tb2, qdot_obs_tb2)

        self.publish_vector(self.robot_pub_tb1, qdot_robot_tb1)
        self.publish_vector(self.robot_pub_tb2, qdot_robot_tb2)

        dist = float(np.linalg.norm(self.tb1.q - self.tb2.q))
        self.dist_pub.publish(Float64(data=dist))

        self.get_logger().info(
            f'tb1 q=({self.tb1.q[0]:.2f}, {self.tb1.q[1]:.2f}) '
            f'avoid=({qdot_total_tb1[0]:.3f}, {qdot_total_tb1[1]:.3f}) | '
            f'tb2 q=({self.tb2.q[0]:.2f}, {self.tb2.q[1]:.2f}) '
            f'avoid=({qdot_total_tb2[0]:.3f}, {qdot_total_tb2[1]:.3f}) | '
            f'd12={dist:.2f}',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)

    node = MultiRobotAvoidanceNode()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('Shutdown recebido.')

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()