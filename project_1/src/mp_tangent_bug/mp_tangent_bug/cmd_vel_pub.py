import math
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
import matplotlib.pyplot as plt
from std_msgs.msg import Float64
from rclpy.executors import ExternalShutdownException

# https://wiki.ros.org/turtlebot3
# https://wiki.ros.org/turtlebot3_gazebo
# https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/TwistStamped.html


# ros2 topic pub -1 /cmd_vel geometry_msgs/msg/TwistStamped "{
#   header: {frame_id: 'base_link'},
#   twist: {
#     linear: {x: 0.0, y: 0.0, z: 0.0},
#     angular: {x: 0.0, y: 0.0, z: 0.0}
#   }
# }"

# ros2 launch turtlebot3_gazebo empty_world.launch.py



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

class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller_node')

        # Publisher de velocidade
        self.cmd_pub = self.create_publisher(
            msg_type=TwistStamped,
            topic='/cmd_vel',
            qos_profile=10
        )

        # Subscriber de odometria
        self.odom_sub = self.create_subscription(
            msg_type=Odometry,
            topic='/odom',
            callback=self.odom_callback,
            qos_profile=10
        )
        
        self.error_pub = self.create_publisher(Float64, '/controle/erro_norma', 10)
        self.v_pub = self.create_publisher(Float64, '/controle/v', 10)
        self.w_pub = self.create_publisher(Float64, '/controle/w', 10)
        
        # Estados do robô
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Tempo
        self.start_time = self.get_clock().now()
        self.t = 0.0

        # Parâmetros da curva
        self.a = 1
        self.omega = 0.08

        # Publishers para RViz
        self.curve_pub = self.create_publisher(Path, '/curva_parametrica', 10)
        self.tracking_point_pub = self.create_publisher(PointStamped, '/ponto_rastreio', 10)
        # Timers
        self.timer = self.create_timer(0.01, self.control_loop)
        self.curve_timer = self.create_timer(1.0, self.publish_curve)

        self.time_log = []
        self.error_log = []
        self.v_log = []
        self.w_log = []


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

    def publish_tracking_point(self, pd):
        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = 'odom'

        point.point.x = float(pd[0])
        point.point.y = float(pd[1])
        point.point.z = 0.0

        self.tracking_point_pub.publish(point)     
 
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

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)
        self.get_logger().info('Robô parado.')

    def control_loop(self):
        self.att_time()
        
        # ================================
        # 👉 COLOQUE SEU CONTROLADOR AQUI
        # ================================
        
        
        # ganhos
        d = .03
        
        theta = self.theta
        Ainv = (1.0 / d) * np.array([
            [d * np.cos(theta), d * np.sin(theta)],
            [-np.sin(theta),    np.cos(theta)]
        ])
        
        
        pd, pddot = curva(self.t, self.a, self.omega)
        p = np.array([ self.x, self.y])
        
        error = pd - p
        uv = pddot + 2 * error
                
        u = Ainv @ uv
    
        v=u[0]
        w=u[1]
        # v = np.clip(v, - 0.22,  0.22)
        # w = np.clip(w, -2.84, 2.84)
        # ================================
        # Publicação
        # ================================
        
        now = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        self.time_log.append(now)
        self.error_log.append(np.linalg.norm(error))
        self.v_log.append(v)
        self.w_log.append(w)
        
        self.publish_cmd(v, w)
        self.publish_tracking_point(pd)
        self.get_logger().info(
            f"v={v:.2f}, w={w:.2f}",
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)

    node = ControllerNode()

    try:
        node.get_logger().info('Nó iniciado.')
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('Shutdown recebido.')

    finally:
        t = np.array(node.time_log)
        e = np.array(node.error_log)
        v = np.array(node.v_log)
        w = np.array(node.w_log)

        plt.figure()
        plt.plot(t, e)
        plt.title("Erro")
        plt.xlabel("Tempo [s]")
        plt.ylabel("||e||")
        plt.grid()

        plt.figure()
        plt.plot(t, v, label="v")
        plt.plot(t, w, label="w")
        plt.title("Controle")
        plt.xlabel("Tempo [s]")
        plt.legend()
        plt.grid()

        plt.show()



        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()