import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3Stamped, PointStamped


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, low, high):
    return max(low, min(high, value))


class TangentBugNode(Node):

    def __init__(self):
        super().__init__('tangent_bug_node')

        self.get_logger().info('Tangent Bug Node iniciado.')

        # ==========================================================
        # Parameters
        # ==========================================================

        self.v_max = 0.5
        self.omega_max = 2.84
        self.goal_tolerance = 0.10

        self.safe_distance = 0.45
        self.wall_follow_distance = 0.60
        self.emergency_distance = 0.32
        self.obstacle_influence_distance = 0.90
        self.robot_radius=0.2
        self.path_margin = 0.15
        self.path_corridor_width = self.robot_radius + self.path_margin


        self.discontinuity_threshold = 0.50
        self.loop_closure_dist = 0.30
        self.loop_closure_min_travel = 1.50
        self.boundary_stagnation_timeout = 5.0

        # Anti-oscilação de estado.
        self.blocked_count_required = 3
        self.clear_count_required = 5
        self.min_boundary_time = 1.0
        self.min_boundary_travel = 0.30
        self.leave_margin = 0.05

        # ==========================================================
        # Publishers
        # ==========================================================

        self.desired_vel_pub = self.create_publisher(
            Vector3Stamped,
            '/desired_xy_vel',
            10
        )

        # ==========================================================
        # Subscribers
        # ==========================================================

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        scan_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            scan_qos
        )

        self.goal_sub = self.create_subscription(
            PointStamped,
            '/goal_point',
            self.goal_callback,
            10
        )

        # ==========================================================
        # Robot state
        # ==========================================================

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.odom_received = False

        # ==========================================================
        # Goal state
        # ==========================================================

        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_received = False

        # ==========================================================
        # Laser data
        # ==========================================================

        self.scan = None
        self.ranges = None
        self.angle_min = 0.0
        self.angle_max = 0.0
        self.angle_increment = 0.0
        self.range_min = 0.0
        self.range_max = 0.0
        self.scan_received = False

        # ==========================================================
        # Internal states / modes
        # ==========================================================

        self.mode = 'MOTION_TO_GOAL'

        self.d_reach = float('inf')
        self.d_followed = float('inf')

        self.path_clear_count = 0
        self.path_blocked_count = 0

        self.boundary_side = 1.0
        self.boundary_start_x = 0.0
        self.boundary_start_y = 0.0
        self.boundary_prev_x = 0.0
        self.boundary_prev_y = 0.0
        self.boundary_travel = 0.0
        self.boundary_start_time = 0.0
        self.boundary_best_dist = float('inf')
        self.boundary_best_time = 0.0

        # ==========================================================
        # Timer
        # ==========================================================

        self.timer = self.create_timer(
            0.05,
            self.control_loop
        )




    def robot_to_world_velocity(self, vx_robot, vy_robot):
        """Converte velocidade no frame do robô para frame odom."""
        vx_world = (
            vx_robot * math.cos(self.theta)
            - vy_robot * math.sin(self.theta)
        )

        vy_world = (
            vx_robot * math.sin(self.theta)
            + vy_robot * math.cos(self.theta)
        )

        return vx_world, vy_world
    # ==============================================================
    # Callbacks
    # ==============================================================

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.theta = quaternion_to_yaw(q)

        self.odom_received = True

    def scan_callback(self, msg):
        self.scan = msg
        self.ranges = np.array(msg.ranges, dtype=float)

        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment

        self.range_min = msg.range_min
        self.range_max = msg.range_max

        self.scan_received = True

    def goal_callback(self, msg):
        self.goal_x = msg.point.x
        self.goal_y = msg.point.y
        self.goal_received = True

        self.mode = 'MOTION_TO_GOAL'
        self.d_reach = float('inf')
        self.d_followed = float('inf')
        self.path_clear_count = 0
        self.path_blocked_count = 0
        self.boundary_travel = 0.0

        self.get_logger().info(
            f'Nova meta recebida: ({self.goal_x:.2f}, {self.goal_y:.2f})'
        )

    # ==============================================================
    # Helper methods
    # ==============================================================

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def distance_to_goal(self):
        return math.hypot(self.goal_x - self.x, self.goal_y - self.y)

    def angle_to_goal(self):
        return math.atan2(self.goal_y - self.y, self.goal_x - self.x)

    def beam_angle(self, i):
        return self.angle_min + i * self.angle_increment

    def valid_obstacle_range(self, r):
        r = float(r)
        return (
            math.isfinite(r)
            and r >= self.range_min
            and r <= self.range_max
        )

    def laser_to_world(self, r, angle):
        world_angle = self.theta + angle
        px = self.x + r * math.cos(world_angle)
        py = self.y + r * math.sin(world_angle)
        return px, py

    def normalize_velocity(self, vx, vy, max_speed=None):
        if max_speed is None:
            max_speed = self.v_max

        norm = math.hypot(vx, vy)
        if norm < 1e-9:
            return 0.0, 0.0

        if norm > max_speed:
            scale = max_speed / norm
            vx *= scale
            vy *= scale

        return vx, vy

    def publish_desired_velocity(self, xdot_d, ydot_d):
        msg = Vector3Stamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.vector.x = float(xdot_d)
        msg.vector.y = float(ydot_d)
        msg.vector.z = 0.0

        self.desired_vel_pub.publish(msg)

    # ==============================================================
    # Laser methods
    # ==============================================================

    def is_path_clear(self):
        """Retorna True se o corredor robô-meta está livre.

        Diferente de olhar só um cone angular, isso checa a distância lateral
        do feixe até a linha robô-meta. Isso reduz falso positivo do laser
        quando existe obstáculo perto, mas fora do caminho real.
        """
        if self.ranges is None or len(self.ranges) == 0:
            return False

        dist_goal = self.distance_to_goal()
        if dist_goal <= self.goal_tolerance:
            return True

        goal_angle_laser = wrap_to_pi(self.angle_to_goal() - self.theta)

        for i, r in enumerate(self.ranges):
            if not self.valid_obstacle_range(r):
                continue

            r = float(r)
            beam = self.beam_angle(i)
            diff = wrap_to_pi(beam - goal_angle_laser)

            forward_dist = r * math.cos(diff)
            lateral_dist = abs(r * math.sin(diff))

            obstacle_between_robot_and_goal = (
                forward_dist > self.range_min
                and forward_dist < dist_goal - self.goal_tolerance
            )

            obstacle_inside_corridor = lateral_dist < self.path_corridor_width

            if obstacle_between_robot_and_goal and obstacle_inside_corridor:
                return False

        return True

    def update_path_counters(self, path_clear):
        if path_clear:
            self.path_clear_count += 1
            self.path_blocked_count = 0
        else:
            self.path_blocked_count += 1
            self.path_clear_count = 0

    def closest_obstacle(self):
        if self.ranges is None or len(self.ranges) == 0:
            return None

        best_range = float('inf')
        best_angle = 0.0
        best_index = -1

        for i, r in enumerate(self.ranges):
            if not self.valid_obstacle_range(r):
                continue

            r = float(r)
            if r < best_range:
                best_range = r
                best_angle = self.beam_angle(i)
                best_index = i

        if best_index < 0:
            return None

        return best_range, best_angle, best_index

    def find_discontinuities(self):
        discontinuities = []

        if self.ranges is None or len(self.ranges) < 2:
            return discontinuities

        for i in range(len(self.ranges) - 1):
            r1 = float(self.ranges[i])
            r2 = float(self.ranges[i + 1])

            valid1 = self.valid_obstacle_range(r1)
            valid2 = self.valid_obstacle_range(r2)

            if valid1 and valid2:
                if abs(r2 - r1) < self.discontinuity_threshold:
                    continue

                if r1 <= r2:
                    r = r1
                    angle = self.beam_angle(i)
                else:
                    r = r2
                    angle = self.beam_angle(i + 1)

            elif valid1 and not valid2:
                r = r1
                angle = self.beam_angle(i)

            elif valid2 and not valid1:
                r = r2
                angle = self.beam_angle(i + 1)

            else:
                continue

            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            discontinuities.append({
                'angle': angle,
                'range': r,
                'point': (lx, ly),
            })

        return discontinuities

    def compute_d_reach(self, path_clear):
        dist_direct = self.distance_to_goal()

        if path_clear:
            return dist_direct

        candidates = self.find_discontinuities()

        # Igual à ideia do arquivo de referência: se as quinas não forem boas,
        # usa os pontos visíveis do laser como candidatos conservadores.
        if not candidates:
            step = max(1, len(self.ranges) // 120)

            for i in range(0, len(self.ranges), step):
                r = float(self.ranges[i])
                if not self.valid_obstacle_range(r):
                    continue

                angle = self.beam_angle(i)
                candidates.append({
                    'angle': angle,
                    'range': r,
                    'point': (r * math.cos(angle), r * math.sin(angle)),
                })

        d_reach = float('inf')

        for candidate in candidates:
            r = candidate['range']
            angle = candidate['angle']

            px, py = self.laser_to_world(r, angle)

            d_candidate = r + math.hypot(self.goal_x - px, self.goal_y - py)

            if d_candidate < d_reach:
                d_reach = d_candidate

        return d_reach

    # ==============================================================
    # Tangent Bug logic
    # ==============================================================

    def go_to_goal(self):
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return 0.0, 0.0

        speed = min(self.v_max, 0.8 * dist)

        xdot_d = speed * dx / dist
        ydot_d = speed * dy / dist

        return xdot_d, ydot_d

    def choose_boundary_side(self):
        obs = self.closest_obstacle()
        if obs is None:
            self.boundary_side = 1.0
            return

        obs_range, obs_angle, _ = obs
        ox, oy = self.laser_to_world(obs_range, obs_angle)

        to_obs_x = ox - self.x
        to_obs_y = oy - self.y

        to_goal_x = self.goal_x - self.x
        to_goal_y = self.goal_y - self.y

        cross = to_obs_x * to_goal_y - to_obs_y * to_goal_x

        self.boundary_side = 1.0 if cross > 0.0 else -1.0

    def start_boundary_following(self):
        self.mode = 'BOUNDARY_FOLLOWING'

        self.d_followed = self.distance_to_goal()

        self.choose_boundary_side()

        self.boundary_start_x = self.x
        self.boundary_start_y = self.y
        self.boundary_prev_x = self.x
        self.boundary_prev_y = self.y
        self.boundary_travel = 0.0

        now = self.now_seconds()
        self.boundary_start_time = now
        self.boundary_best_dist = self.d_followed
        self.boundary_best_time = now

        # Importante: zera contador de caminho livre para não sair do contorno
        # imediatamente por causa de uma única leitura oscilante do laser.
        self.path_clear_count = 0

        side_name = 'esq' if self.boundary_side > 0.0 else 'dir'
        self.get_logger().info(
            f'Obstáculo detectado. Entrando em BOUNDARY_FOLLOWING, '
            f'lado={side_name}, d_followed={self.d_followed:.2f}'
        )

    def twist_like_to_world_velocity(self, linear_x, angular_z):
        """Converte uma ação tipo Twist em vetor desejado no mundo.

        O arquivo de referência comanda /cmd_vel. Aqui mantemos sua estrutura,
        então transformamos a intenção de virar em um vetor XY apontando um pouco
        para a esquerda/direita do heading atual.
        """
        turn_ratio = clamp(angular_z / self.omega_max, -1.0, 1.0)
        desired_heading = self.theta + turn_ratio * math.radians(80.0)

        vx = linear_x * math.cos(desired_heading)
        vy = linear_x * math.sin(desired_heading)

        return vx, vy



    def follow_boundary(self):
        """Segue contorno com repulsão forte para evitar colisão.

        A ideia é:
        1. calcular repulsão usando vários feixes próximos do laser;
        2. calcular uma direção tangente ao obstáculo mais próximo;
        3. reduzir velocidade quando estiver perto;
        4. se estiver perto demais, fugir do obstáculo em vez de seguir tangente.
        """

        if self.ranges is None or len(self.ranges) == 0:
            return 0.0, 0.0

        closest = self.closest_obstacle()

        if closest is None:
            return self.go_to_goal()

        closest_range, closest_angle, _ = closest

        # ==========================================================
        # 1. Repulsão usando todos os feixes próximos
        # ==========================================================

        rep_x = 0.0
        rep_y = 0.0
        valid_close_points = 0

        front_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')
        all_min = float('inf')

        for i, r in enumerate(self.ranges):
            if not self.valid_obstacle_range(r):
                continue

            r = float(r)
            angle = self.beam_angle(i)

            all_min = min(all_min, r)

            if abs(angle) < math.radians(35.0):
                front_min = min(front_min, r)

            if math.radians(35.0) < angle < math.radians(135.0):
                left_min = min(left_min, r)

            if math.radians(-135.0) < angle < math.radians(-35.0):
                right_min = min(right_min, r)

            if r > self.obstacle_influence_distance:
                continue

            # Direção do obstáculo no frame do robô.
            obs_x = math.cos(angle)
            obs_y = math.sin(angle)

            # Peso cresce muito quando o obstáculo fica perto.
            weight = (
                1.0 / max(r, 1e-3)
                - 1.0 / self.obstacle_influence_distance
            )

            weight = max(0.0, weight)

            # Repulsão aponta para longe do obstáculo.
            rep_x += -weight * obs_x
            rep_y += -weight * obs_y

            valid_close_points += 1

        if valid_close_points > 0:
            rep_x /= valid_close_points
            rep_y /= valid_close_points

        # ==========================================================
        # 2. Emergência: se está perto demais, não segue tangente
        # ==========================================================

        if all_min < self.emergency_distance or front_min < self.emergency_distance:
            # Foge do obstáculo. Não tenta avançar.
            vx_robot = 0.0
            vy_robot = 0.0

            rep_norm = math.hypot(rep_x, rep_y)

            if rep_norm > 1e-6:
                vx_robot = 0.10 * rep_x / rep_norm
                vy_robot = 0.10 * rep_y / rep_norm
            else:
                # fallback: se não conseguiu calcular repulsão,
                # anda para trás lentamente.
                vx_robot = -0.05
                vy_robot = 0.0

            vx_world, vy_world = self.robot_to_world_velocity(vx_robot, vy_robot)

            self.get_logger().warn(
                f'EMERGENCIA BF: obstaculo muito perto. '
                f'all_min={all_min:.2f}, front_min={front_min:.2f}',
                throttle_duration_sec=0.5
            )

            return vx_world, vy_world

        # ==========================================================
        # 3. Vetor tangente ao obstáculo mais próximo
        # ==========================================================

        obs_x = math.cos(closest_angle)
        obs_y = math.sin(closest_angle)

        # Tangente no frame do robô.
        # boundary_side = +1 ou -1 define sentido de contorno.
        tangent_x = -self.boundary_side * obs_y
        tangent_y = self.boundary_side * obs_x

        # ==========================================================
        # 4. Controle da distância à parede
        # ==========================================================

        # Erro positivo: está longe da parede.
        # Erro negativo: está perto demais.
        dist_error = closest_range - self.wall_follow_distance

        # Componente radial:
        # se está longe, aproxima um pouco;
        # se está perto, afasta.
        radial_gain = 0.55

        radial_x = radial_gain * dist_error * obs_x
        radial_y = radial_gain * dist_error * obs_y

        # ==========================================================
        # 5. Atração pequena para a meta
        # ==========================================================

        goal_vx_world, goal_vy_world = self.go_to_goal()

        goal_vx_robot = (
            goal_vx_world * math.cos(self.theta)
            + goal_vy_world * math.sin(self.theta)
        )

        goal_vy_robot = (
            -goal_vx_world * math.sin(self.theta)
            + goal_vy_world * math.cos(self.theta)
        )

        # ==========================================================
        # 6. Composição final
        # ==========================================================

        tangent_gain = 0.16
        repulsion_gain = 0.65
        goal_gain = 0.08

        vx_robot = (
            tangent_gain * tangent_x
            + radial_x
            + repulsion_gain * rep_x
            + goal_gain * goal_vx_robot
        )

        vy_robot = (
            tangent_gain * tangent_y
            + radial_y
            + repulsion_gain * rep_y
            + goal_gain * goal_vy_robot
        )

        # ==========================================================
        # 7. Redução de velocidade perto do obstáculo
        # ==========================================================

        if all_min < self.safe_distance:
            max_speed = 0.07
        elif all_min < self.wall_follow_distance:
            max_speed = 0.11
        else:
            max_speed = 0.15

        norm = math.hypot(vx_robot, vy_robot)

        if norm > max_speed:
            vx_robot = max_speed * vx_robot / norm
            vy_robot = max_speed * vy_robot / norm

        vx_world, vy_world = self.robot_to_world_velocity(vx_robot, vy_robot)

        self.get_logger().info(
            f'BF: all_min={all_min:.2f}, '
            f'front_min={front_min:.2f}, '
            f'closest={closest_range:.2f}, '
            f'vx_r={vx_robot:.2f}, vy_r={vy_robot:.2f}',
            throttle_duration_sec=1.0
        )

        return vx_world, vy_world


    def update_boundary_progress(self, dist_goal):
        if dist_goal < self.d_followed:
            self.d_followed = dist_goal

        step = math.hypot(
            self.x - self.boundary_prev_x,
            self.y - self.boundary_prev_y
        )

        self.boundary_travel += step
        self.boundary_prev_x = self.x
        self.boundary_prev_y = self.y

        now = self.now_seconds()

        if dist_goal < self.boundary_best_dist - self.leave_margin:
            self.boundary_best_dist = dist_goal
            self.boundary_best_time = now

    def boundary_elapsed(self):
        return self.now_seconds() - self.boundary_start_time

    def can_leave_boundary(self):
        return (
            self.boundary_elapsed() >= self.min_boundary_time
            and self.boundary_travel >= self.min_boundary_travel
        )

    def completed_obstacle_loop(self):
        if self.boundary_travel <= self.loop_closure_min_travel:
            return False

        dist_to_start = math.hypot(
            self.x - self.boundary_start_x,
            self.y - self.boundary_start_y
        )

        return dist_to_start < self.loop_closure_dist

    # ==============================================================
    # Main loop
    # ==============================================================

    def control_loop(self):
        xdot_d = 0.0
        ydot_d = 0.0

        if not self.odom_received or not self.scan_received or not self.goal_received:
            self.publish_desired_velocity(0.0, 0.0)
            return

        dist_goal = self.distance_to_goal()

        if dist_goal <= self.goal_tolerance:
            if self.mode != 'GOAL_REACHED':
                self.get_logger().info(f'Meta alcançada. d={dist_goal:.3f}')

            self.mode = 'GOAL_REACHED'
            self.publish_desired_velocity(0.0, 0.0)
            return

        if self.mode == 'GOAL_REACHED':
            self.mode = 'MOTION_TO_GOAL'

        if self.mode == 'NO_PATH':
            self.publish_desired_velocity(0.0, 0.0)
            self.get_logger().warn('NO_PATH: robô parado.', throttle_duration_sec=2.0)
            return

        path_clear = self.is_path_clear()
        self.update_path_counters(path_clear)
        self.d_reach = self.compute_d_reach(path_clear)

        # ==========================================================
        # MOTION TO GOAL
        # ==========================================================

        if self.mode == 'MOTION_TO_GOAL':

            if self.path_blocked_count >= self.blocked_count_required:
                self.start_boundary_following()
                xdot_d, ydot_d = self.follow_boundary()

            else:
                xdot_d, ydot_d = self.go_to_goal()

        # ==========================================================
        # BOUNDARY FOLLOWING
        # ==========================================================

        elif self.mode == 'BOUNDARY_FOLLOWING':

            self.update_boundary_progress(dist_goal)

            can_leave = self.can_leave_boundary()

            if (
                can_leave
                and path_clear
                and self.path_clear_count >= self.clear_count_required
            ):
                self.mode = 'MOTION_TO_GOAL'
                self.d_followed = float('inf')
                xdot_d, ydot_d = self.go_to_goal()

                self.get_logger().info(
                    'Caminho direto livre. Voltando para MOTION_TO_GOAL.'
                )

            elif (
                can_leave
                and path_clear
                and self.path_clear_count >= self.clear_count_required
                and self.d_reach < self.d_followed - self.leave_margin
            ):
                old_d_followed = self.d_followed

                self.mode = 'MOTION_TO_GOAL'
                self.d_followed = float('inf')
                xdot_d, ydot_d = self.go_to_goal()

                self.get_logger().info(
                    f'Atalho Tangent Bug: d_reach={self.d_reach:.2f} '
                    f'< d_followed={old_d_followed:.2f}'
                )

            elif self.completed_obstacle_loop():
                self.mode = 'NO_PATH'
                xdot_d = 0.0
                ydot_d = 0.0

                self.get_logger().warn(
                    f'Sem caminho detectado. Volta completa: '
                    f'{self.boundary_travel:.2f} m.'
                )

            elif (
                self.now_seconds() - self.boundary_best_time
                > self.boundary_stagnation_timeout
            ):
                self.mode = 'MOTION_TO_GOAL'
                self.d_followed = float('inf')
                self.path_clear_count = 0
                self.path_blocked_count = 0
                xdot_d, ydot_d = self.go_to_goal()

                self.get_logger().warn(
                    'Estagnação no contorno. Tentando MOTION_TO_GOAL.'
                )

            else:
                xdot_d, ydot_d = self.follow_boundary()

        else:
            self.mode = 'MOTION_TO_GOAL'
            xdot_d, ydot_d = self.go_to_goal()

        self.publish_desired_velocity(xdot_d, ydot_d)

        self.get_logger().info(
            f'mode={self.mode}, '
            f'd_goal={dist_goal:.2f}, '
            f'd_reach={self.d_reach:.2f}, '
            f'd_followed={self.d_followed:.2f}, '
            f'clear_count={self.path_clear_count}, '
            f'blocked_count={self.path_blocked_count}, '
            f'vx={xdot_d:.2f}, '
            f'vy={ydot_d:.2f}',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)

    node = TangentBugNode()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()