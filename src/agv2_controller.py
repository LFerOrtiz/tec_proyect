#!/usr/bin/env python
# coding=utf-8
import numpy as np
import rospy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from marker_detector import MarkerDetector
from sensor_msgs.msg import Image

# -- Constants for configuration of Aruco library
KERNEL_SIZE = (3, 3)  # Kernel size for Gaussian Blur
MARKER_LENGTH = 0.2  # Marker size in cm
LEADER_TAG = 0  # ID of leader marker

# -- AGV parameters
system_params = {
    'm': 25.0,  # Mass of AGV
    'g': 9.81,  # Gravity force (m/s²)
    'Crr': 0.015,  # Coefficient of rolling friction
    'Cd': 0.80,  # Drag coefficient
    'rho': 1.224,  # Density of air (Kg/m³)
    'A': 0.077025,  # Front area (m²)
}

# -- Constants for PID controller and variables
LENGTH_PRECEDING = 0.6  # Length of the preceding vehicle
CONST_HEADWAY = 0.8
SET_DISTANCE = 0.45
SET_VELOCITY = 0.6
MAX_SPEED = 1.20  # Max speed (m/s)
REFRESH = 15  # Refresh rate
INIT_SPEED = 0.00  # Initial speed (m/s)
ALPHA = 0.85

dist_host_to_lead = 0.0
pitch_angle = 0.0

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

    def compute(self, actual, set_point, debug=False):
        """ Compute the PID controller.

        :param actual: Actual value
        :param set_point: The set point for the PID
        :param debug: Print the information about the PID and error
        """

        # Compute error
        self.error = actual - set_point
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
    global bridge, process_frame, dist_host_to_lead, pitch_angle

    try:
        # -- Convert the image format of ROS to OpenCV form
        cv_frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        # rospy.loginfo("Frame received")
        marker_frame, tag_lose = marker_detection.detection(cv_frame, LEADER_TAG, True, False)
        dist_host_to_lead = marker_detection.distance
        pitch_angle = marker_detection.pitch


def lidar_scan_callback(scan_msg):
    """
    Callback function for laser scan.

    :param scan_msg: Message with information of scan topic
    """
    detected_field = {
        "front": min(min(scan_msg.ranges[288:431]), 10),
    }
    # -- Calculated the moving average
    lidar_filter = round((0.05 * detected_field["front"]) + ((1 - 0.05) * detected_field["front"]), 3)
    # print("LIDAR: ", lidar_filter)


def lead_odom_callback(msg):
    """ Callback function for odometry of lead vehicle"""
    # print(msg.pose.pose)
    pass


def stop():
    """
    Callback function to stop the AGV when evoke shutdown.
    """
    move = Twist()
    move.linear.x = 0.0
    move.linear.y = 0.0
    move.linear.z = 0.0
    move.angular.x = 0.0
    move.angular.y = 0.0
    move.angular.z = 0.0
    vel_pub.publish(move)


# -- Main function
if __name__ == '__main__':
    max_accel = 0.2
    min_accel = -0.2

    # -- Create a node for agv2
    rospy.init_node('agv2_controller', anonymous=True)

    # -- Update
    rate_fresh = rospy.Rate(REFRESH)
    step_time = float(1.0 / REFRESH)

    # -- Topics
    odom_agv1 = 'agv1/odom'  # Odom of AGV1
    raw_camera_topic = 'agv2/camera/image_raw'  # Topic for the image from the camera
    lidar_scan_topic = 'agv2/scan'  # Topic for the scan of the lidar
    move_controller_topic = 'agv2/cmd_vel'  # Topic of move controller

    # -- Subscribers and Publishers
    vel_pub = rospy.Publisher(move_controller_topic, Twist, queue_size=25)
    camera_sub = rospy.Subscriber(raw_camera_topic, Image, frame_callback)  # Define the subscriber topic for camera
    lidar_sub = rospy.Subscriber(lidar_scan_topic, LaserScan, lidar_scan_callback)
    pose_agv1 = rospy.Subscriber(odom_agv1, Odometry, lead_odom_callback)

    try:
        # -- Initial parameters for move for the vehicle
        pid_control = ControlPID(3.6, 0.00, 0.26, step_time)

        move.linear.x = INIT_SPEED
        move.angular.z = 0
        distance_filter = 0
        angle = 0
        i = 0

        while not rospy.is_shutdown():
            # -- Get the current velocity of AGV
            distance_filter = round((ALPHA * dist_host_to_lead) + ((1 - ALPHA) * distance_filter), 3)
            angle = round(np.degrees((ALPHA * pitch_angle) + ((1 - ALPHA) * angle)), 3)
            policy = LENGTH_PRECEDING

            # -- Control
            control_signal = pid_control.compute(distance_filter, policy, False)

            # Forces of vehicle model
            # f_roll = (system_params['Crr'] * system_params['m'] * system_params['g'])
            # f_aero = (system_params['rho'] * system_params['A'] * system_params['Cd'] * pow(vel, 2)) / 2

            # Sum of forces
            # accel = (control_signal - f_aero - f_roll) / system_params['m']
            # past_accel = accel

            # vel = past_vel + (past_accel + accel) / 2 * step_time

            # if not (check_limits(accel, max_accel, min_accel)):
            #     if accel >= 0:
            #         accel = max_accel
            #     else:
            #         accel = min_accel

            rospy.loginfo("Distance: ")
            rospy.loginfo(distance_filter)
            print ("Headway Policy: ", policy)

            move.linear.x = control_signal
            move.angular.z = 0.0

            # -- Send the update velocity

            vel_pub.publish(move)
            print("AGV2 Vel: ", move.linear.x)
            rospy.on_shutdown(stop)
            rate_fresh.sleep()

    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutdown node")
