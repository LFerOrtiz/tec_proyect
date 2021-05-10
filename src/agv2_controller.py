#!/usr/bin/env python
# coding=utf-8
import numpy as np
import rospy
import tf

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, Quaternion
from marker_detector import MarkerDetector
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion

# -- Constants for configuration of Aruco library
KERNEL_SIZE = (3, 3)  # Kernel size for Gaussian Blur
MARKER_LENGTH = 0.2  # Marker size in cm
LEADER_TAG = 0  # ID of preceding marker

# -- AGV parameters
system_params = {
    'm': 150.0,  # Mass of AGV
    'g': 9.81,  # Gravity force (m/s²)
    'Crr': 0.015,  # Coefficient of rolling friction
    'Cd': 0.80,  # Drag coefficient
    'rho': 1.224,  # Density of air (Kg/m³)
    'A': 0.077025,  # Front area (m²)
}

# -- Constants for PID controller and variables
SET_DISTANCE = 0.65
MAX_SPEED = 1.20  # Max speed (m/s)
REFRESH = 15  # Refresh rate
INIT_SPEED = 0.00  # Initial speed (m/s)
ALPHA = 0.2

lidar_filter = 0.0
dist_host_to_lead = 0.0
pitch_angle = 0.0
center_x = 0
width = 0

# CvBridge
bridge = CvBridge()
move = Twist()
process_frame = None
vel_pub = None
camera_sub = None
pose_agv1 = None

# -- Get the calibration parameters of the camera
info_camera = "/home/fer/catkin_ws/src/tec_proyect/info/"
camera_matrix = np.loadtxt(info_camera + 'cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(info_camera + 'cameraDistortion.txt', delimiter=',')

# -- Create a object for the marker detection
marker_detection = MarkerDetector(MARKER_LENGTH, KERNEL_SIZE, camera_matrix, camera_distortion)


class ControlPID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, dt=0.0):
        """ Define the Gains for PID controller

        :param kp: Gain for proportional part
        :param ki: Gain for integral part
        :param kd: Gain for derivative part
        :param dt: Time step
        """
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.past_error = 0
        self.error = 0
        self.sum_error = 0

    def compute(self, actual_value, set_point, debug=False):
        """ Compute the PID controller.

        :param actual_value: Actual value
        :param set_point: The set point for the PID
        :param debug: Print the information about the PID and error
        """

        # Compute error
        self.error = actual_value - set_point
        self.sum_error = (self.error + self.sum_error) * self.dt

        # Proportional error
        p_output = self.kp * self.error
        # Integral error
        i_output = self.ki * self.sum_error * self.dt
        # Derivative error
        d_output = self.kd * ((self.error - self.past_error) / self.dt)
        self.past_error = self.error

        if debug:
            print ("Error Sum: ", self.sum_error)
            print ("Past error: ", self.past_error)
            print ("Error: ", self.error)
            print ("P: ", p_output)
            print ("I: ", i_output)
            print ("D: ", d_output)

        output = p_output + i_output + d_output
        return output


def frame_callback(ros_frame):
    """
    Callback function for read and convert the image format of ROS to OpenCV format.

    :param ros_frame: Get the video frame in ROS format
    """
    global bridge, process_frame, dist_host_to_lead, pitch_angle, center_x, width

    try:
        # -- Convert the image format of ROS to OpenCV form
        cv_frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        # rospy.loginfo("Frame received")
        marker_frame = marker_detection.detection(cv_frame, LEADER_TAG, True, False)
        dist_host_to_lead = marker_detection.distance
        pitch_angle = marker_detection.pitch
        center_x = marker_detection.center_x
        width = marker_detection.width


def lidar_scan_callback(scan_msg):
    """
    Callback function for laser scan.

    :param scan_msg: Message with information of scan topic
    """
    global lidar_filter
    detected_field = {
        'fright': min(min(scan_msg.ranges[100:124]), 10),
        'front':  min(min(scan_msg.ranges[125:145]), 10),
        'fleft':  min(min(scan_msg.ranges[146:170]), 10),
    }

    # -- Calculated the moving average
    lidar_filter = round((0.2 * detected_field["front"]) + ((1 - 0.2) * detected_field["front"]), 3)
    # print("LIDAR: ", lidar_filter)


def lead_odom_callback(msg):
    """ Callback function for odometry of lead vehicle"""
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y,
                        orientation_q.z, orientation_q.w]
    tf.TransformListener()
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    # rospy.loginfo("Yaw: %s", np.degrees(yaw))


def stop():
    """
    Callback function to stop the AGV when evoke shutdown.
    """
    twist = Twist()
    twist.linear.x = 0.0
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0
    vel_pub.publish(twist)


# -- Main function
if __name__ == '__main__':
    # -- Create a node for agv2
    rospy.init_node('agv2_controller', anonymous=True)

    # -- Update
    rate_fresh = rospy.Rate(REFRESH)
    step_time = float(1.0 / REFRESH)

    # -- Topics
    odom_agv1 = 'agv1/odom'                     # Odom of AGV1
    raw_camera_topic = 'agv2/camera/image_raw'  # Topic for the image from the camera
    lidar_scan_topic = 'agv2/scan'              # Topic for the scan of the lidar
    move_controller_topic = 'agv2/cmd_vel'      # Topic of move controller

    # -- Subscribers and Publishers
    vel_pub = rospy.Publisher(move_controller_topic, Twist, queue_size=5)
    camera_sub = rospy.Subscriber(raw_camera_topic, Image, frame_callback)  # Define the subscriber topic for camera
    lidar_sub = rospy.Subscriber(lidar_scan_topic, LaserScan, lidar_scan_callback)
    pose_agv1 = rospy.Subscriber(odom_agv1, Odometry, lead_odom_callback)

    try:
        # -- Initial parameters for move for the vehicle
        pid_control = ControlPID(2.0, 0.006, 0.65, step_time)

        move.linear.x = INIT_SPEED
        move.angular.z = 0
        distance_filter = 0
        angle_filter = 0
        past_vel = 0

        while not rospy.is_shutdown():
            # -- Get the current velocity of AGV
            distance_filter = round((ALPHA * dist_host_to_lead) + ((1 - ALPHA) * distance_filter), 3)
            angle_filter = round(((0.05 * np.degrees(pitch_angle)) + ((1 - 0.05) * angle_filter)), 3)
            policy = SET_DISTANCE

            # -- Longitudinal Control
            dist_cal = (move.linear.x + past_vel) / 2 * step_time
            past_vel = move.linear.x

            control_signal = pid_control.compute(distance_filter, policy, False)
            if control_signal < MAX_SPEED:
                move.linear.x = control_signal
            else:
                if control_signal < 0:
                    move.linear.x = -MAX_SPEED
                else:
                    move.linear.x = MAX_SPEED

            # -- Lateral controller
            err = center_x - width / 2
            move.angular.z = -float(err) / 100
            rospy.loginfo(err)

            # -- Send the update velocity
            vel_pub.publish(move)

            rospy.on_shutdown(stop)
            rate_fresh.sleep()

    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutdown node")
