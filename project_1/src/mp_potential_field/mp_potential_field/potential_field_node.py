import math
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import numpy as np

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3Stamped
from collections import deque


# Converter de quaternion para yaw, funcao feita com base em:
# https://gist.github.com/michaelwro/1450283a6a1226eaf707d9adde378798
def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

# ros2 run mp_potential_field potential_field_node --ros-args -p goal_x:=1.0 -p goal_y:=0.0

class PotentialFieldNode(Node):

    def __init__(self):
        super().__init__('potential_field_node')
        self.get_logger().info('Potential Field Node iniciado!')

        self.desired_vel_pub = self.create_publisher(
            Vector3Stamped,
            '/desired_xy_vel',
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Estado do robô
        self.q = np.array([0.0, 0.0])
        self.theta = 0.0

        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)

        self.qdot_d = np.array([0, 0])
        
        goal_x = self.get_parameter('goal_x').value
        goal_y = self.get_parameter('goal_y').value

        self.q_goal = np.array([goal_x, goal_y])
           
           
        self.scan = None
        self.timer = self.create_timer(0.01, self.control_loop)

        self.dist_history = deque(maxlen=50)


    def odom_callback(self, msg):
        self.q[0] = msg.pose.pose.position.x
        self.q[1] = msg.pose.pose.position.y
        self.theta = quaternion_to_yaw(msg.pose.pose.orientation)



    def scan_callback(self, msg):
        self.scan = msg



    def publish_desired_velocity(self, qdot):
        msg = Vector3Stamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.vector.x = float(qdot[0])
        msg.vector.y = float(qdot[1])
        msg.vector.z = 0.0

        self.desired_vel_pub.publish(msg)


    # Combined attractie potential function
    # com base na do livro Principles of Robot Motion Theory
    def attractive_gradient(self, d_goal, zeta):
        d = np.linalg.norm(self.q - self.q_goal)
        if d <= d_goal:
            return - zeta*(self.q - self.q_goal)
        else:
            return - d_goal * zeta * (self.q - self.q_goal) / d    



    def find_obstacle_minima(self, Q_star):
        if self.scan is None:
            return []

        ranges = np.array(self.scan.ranges, dtype=float)
        obstacles = []

        cluster = []

        for i, r in enumerate(ranges):
            valid = (
                np.isfinite(r)
                and self.scan.range_min <= r <= self.scan.range_max
                and r <= Q_star
            )

            if valid:
                cluster.append(i)
            else:
                if len(cluster) > 0:
                    obs = self.extract_minimum(cluster, ranges)
                    obstacles.append(obs)
                    cluster = []

        if len(cluster) > 0:
            obs = self.extract_minimum(cluster, ranges)
            obstacles.append(obs)

        return obstacles



    def extract_minimum(self, cluster, ranges):
        min_idx = cluster[int(np.argmin(ranges[cluster]))]
        d_i = ranges[min_idx]

        angle = self.scan.angle_min + min_idx * self.scan.angle_increment

        # direção no frame do robô (base_scan)
        dir_base = np.array([
            np.cos(angle),
            np.sin(angle)
        ])

        # grad d_i(q) = direção que AUMENTA distância ao obstáculo
        # = do obstáculo para o robô → oposto do sensor
        grad_base = -dir_base

        # converte para odom
        c = np.cos(self.theta)
        s = np.sin(self.theta)

        R = np.array([
            [c, -s],
            [s,  c]
        ])

        grad_odom = R @ grad_base

        return d_i, grad_odom



    def repulsive_gradient(self, Q_star, eta):
        if self.scan is None:
            return np.array([0.0, 0.0])

        obstacles = self.find_obstacle_minima(Q_star)

        qdot_rep = np.array([0.0, 0.0])

        for d_i, grad_d_i in obstacles:

            if d_i < 1e-6:
                continue

            # fórmula do livro
            qdot_i = (
                eta
                * (1.0 / d_i - 1.0 / Q_star)
                * (1.0 / (d_i ** 2))
                * grad_d_i
            )

            qdot_rep += qdot_i

        return qdot_rep
    
    
        
    def control_loop(self):
        # distância ao objetivo
        error = self.q_goal - self.q
        dist = np.linalg.norm(error)

        # tolerância de parada
        goal_tol = 0.015

        if dist <= goal_tol:
            self.qdot_d = np.array([0.0, 0.0])

            self.publish_desired_velocity(self.qdot_d)

            self.get_logger().info(
                f"GOAL atingido | q=({self.q[0]:.2f}, {self.q[1]:.2f})",
                throttle_duration_sec=1.0
            )
            return


        # campos
        qdot_att = self.attractive_gradient(0.25, 0.9)
        qdot_rep = self.repulsive_gradient(Q_star=0.6, eta=0.0015)
        
        self.qdot_d = qdot_att + qdot_rep

        self.dist_history.append(dist)        
        self.publish_desired_velocity(self.qdot_d)
        self.get_logger().info(
            f"q=({self.q[0]:.2f}, {self.q[1]:.2f})",
            throttle_duration_sec=1.0
        )


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