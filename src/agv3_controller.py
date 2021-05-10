#!/usr/bin/env python
# coding=utf-8
import numpy as np
import rospy
import tf

from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from marker_detector import MarkerDetector
from sensor_msgs.msg import Image

# -- Constants for configuration of Aruco library
KERNEL_SIZE = (3, 3)  # Kernel size for Gaussian Blur
MARKER_LENGTH = 0.2  # Marker size in cm
LEADER_TAG = 1 # ID of leader marker

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
SET_DISTANCE = 0.5
SET_VELOCITY = 1.2
MAX_SPEED = 1.20  # Max speed (m/s)
REFRESH = 20  # Refresh rate
INIT_SPEED = 0.00  # Initial speed (m/s)
KP = 20.0  # Gain for proportional part
KI = 0.2  # Gain for integral part
KD = 5.0  # Gain for derivative part

error_sum = 0
dist_host_to_lead = 0
past_actual = 0
first_run = True

# CvBridge
bridge = CvBridge()
move = Twist()
process_frame = None

# -- Get the calibration parameters of the camera
info_camera = "/home/fer/catkin_ws/src/tec_proyect/info/"
camera_matrix = np.loadtxt(info_camera + 'cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(info_camera + 'cameraDistortion.txt', delimiter=',')

# -- Create a object for the marker detection
marker_detection = MarkerDetector(MARKER_LENGTH, KERNEL_SIZE, camera_matrix, camera_distortion)


def frame_callback(ros_frame):
    """ Convert the image format of ROS to OpenCV format """
    global bridge, process_frame, dist_host_to_lead

    try:
        # -- Convert the image format of ROS to OpenCV form
        cv_frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        # rospy.loginfo("Frame received")
        marker_frame = marker_detection.detection(cv_frame, LEADER_TAG, True)
        dist_host_to_lead = marker_detection.dist_proyection_avr


def lead_odom_callback(msg):
    """ Callback function for odometry of lead vehicle"""
    # print(msg)
    pass


def compute_pid(actual, setpoint):
    # Get the current safe distance in base of velocity of host AGV
    global error_sum, past_actual, first_run

    # Compute error
    error = setpoint - actual
    error_sum += error
    print ("Error: ", error)

    if first_run:
        first_run = False
        past_actual = actual

    # Proportional error
    p_output = KP * error
    # Integral error
    i_output = KI * error_sum
    # Derivative error
    d_output = KD * (dist_host_to_lead - past_actual)
    past_error = error

    print ("P: ", p_output)
    print ("I: ", i_output)
    print ("D: ", d_output)

    output = p_output + d_output + i_output
    return output


def main():
    # -- Global variables
    global process_frame, REFRESH, SET_DISTANCE, system_params, dist_host_to_lead

    # -- Create a node for agv2
    rospy.init_node('agv3_controller', anonymous=True)

    # -- Update
    rate_fresh = rospy.Rate(REFRESH)
    step_time = float(1.0 / REFRESH)

    # -- Topics
    odom_agv1 = 'agv2/odom'  # Odom of AGV1
    raw_camera_topic = 'agv3/camera/image_raw'  # Topic for the image from the camera
    move_controller_topic = 'agv3/cmd_vel'  # Topic of move controller

    # -- Subscribers and Publishers
    # Define the publisher for the move controller topic
    vel_pub = rospy.Publisher(move_controller_topic, Twist, queue_size=25)
    camera_sub = rospy.Subscriber(raw_camera_topic, Image, frame_callback)  # Define the subscriber topic for camera
    pose_agv1 = rospy.Subscriber(odom_agv1, Odometry, lead_odom_callback)

    # -- Initial parameters for move for the vehicle
    move.linear.x = INIT_SPEED
    move.angular.z = 0

    # -- Variables for control and ACC
    dist_host_to_lead = 0.0
    host_accel = 0.0
    host_vel = 0.0
    max_accel = 2
    min_accel = -2

    while not rospy.is_shutdown():
        # Get the current velocity oof AGV
        host_vel = move.linear.x

        # ACC control
        if dist_host_to_lead >= SET_DISTANCE:
            # Distance between the cars is greater than the safe distance (Speed Control)
            print("Speed Control")
            setpoint = SET_VELOCITY
            control = compute_pid(host_vel, setpoint)
        else:  # Maintain the safe distance (Spacing control mode)
            print ("Spacing control")
            control = compute_pid(dist_host_to_lead, SET_DISTANCE)

        print ("Control: ", control)
        # Final force
        host_accel = 1 / system_params['m'] * (control - 1/2 * (system_params['rho'] * host_vel ** 2 *
                                               system_params['A'] * system_params['Cd']) - (system_params['Crr'] *
                                               system_params['m'] * system_params['g']))

        if not(max_accel > host_accel > min_accel):
            if host_accel >= 0:
                host_accel = max_accel
            else:
                host_accel = min_accel

        host_vel = host_accel * step_time

        print("Host Accel: ", host_accel)
        print("AGV3: ", host_vel)

        move.linear.x = host_vel
        move.angular.z = 0.0

        # -- Send the update velocity
        #vel_pub.publish(move)

        rate_fresh.sleep()


# -- Main function
if __name__ == '__main__':
    try:
        main()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutdown node")
