import matplotlib.pyplot as plt
import rospy
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_matrix
import matplotlib.animation as animation


class Visualiser:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ln, = plt.plot([], [], 'r--', label="lidar distance")
        self.x_data, self.y_data = [], []

    def plot_init(self):
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(-7, 7)
        return self.ln

    # def getYaw(self, pose):
    #     quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z,
    #                   pose.orientation.w)
    #     euler = tf.transformations.euler_from_quaternion(quaternion)
    #     yaw = euler[2]
    #     return yaw

    def lidar_callback(self, scan_msg):
        """ Callback funtion for laser scan """
        detec_field = {
            "front": min(min(scan_msg.ranges[270:450]), 10),
        }
        # -- Calculated the moving average
        lidar_filter = round((0.05 * detec_field["front"]) + ((1 - 0.05) * detec_field["front"]), 3)
        self.y_data.append(lidar_filter)
        x_index = len(self.x_data)
        self.x_data.append(x_index + 1)

    def odom_callback(self, msg):
        yaw_angle = self.getYaw(msg.pose.pose)
        self.y_data.append(yaw_angle)
        x_index = len(self.x_data)
        self.x_data.append(x_index + 1)

    def update_plot(self, frame):
        plt.legend(loc="upper left")
        self.ln.set_data(self.x_data, self.y_data)
        return self.ln


rospy.init_node('lidar_visual_node')
vis = Visualiser()
sub = rospy.Subscriber('/agv2/scan', LaserScan, vis.lidar_callback)

ani = animation.FuncAnimation(vis.fig, vis.update_plot, init_func=vis.plot_init)
plt.show()
