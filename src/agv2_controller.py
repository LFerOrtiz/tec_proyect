#!/usr/bin/env python
import numpy as np

import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from marker_detector import MarkerDetector
from sensor_msgs.msg import Image

# -- Constants for configuration of Aruco library
KERNEL_SIZE = (3, 3)  # Kernel size for Gaussian Blur
MARKER_LENGTH = 0.2  # Marker size in cm
LEADER_TAG = 0  # ID of leader marker
DIST_REF = 0.45  # Distance Reference

# -- Constants for PD controller and variables
REFRESH = 60  # Refresh rate
INIT_SPEED = 0.00  # Initial speed (m/s)
KP = 1.2  # Gain for proportional part
KD = 10  # Gain for derivative part
error = 0  # Error

# -- Distance
past_distance = 0.0
current_distance = 0.0

# -- Velocity
current_velocity = 0.0

# CvBridge
bridge = CvBridge()
process_frame = None

# -- Get the calibration parameters of the camera
info_camera = "/home/fer/catkin_ws/src/tec_proyect/info/"
camera_matrix = np.loadtxt(info_camera + 'cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(info_camera + 'cameraDistortion.txt', delimiter=',')

# -- Create a object for the marker detection
marker_detection = MarkerDetector(MARKER_LENGTH, KERNEL_SIZE, camera_matrix, camera_distortion)


def frame_callback(ros_frame):
    """ Convert the image format of ROS to OpenCV format """
    global bridge, process_frame, current_distance

    try:
        # -- Convert the image format of ROS to OpenCV form
        cv_frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        # rospy.loginfo("Frame received")
        marker_frame = marker_detection.detection(cv_frame, LEADER_TAG, True)
        current_distance = marker_detection.dist_proyection_avr


def main():
    # -- Global variables
    global error, KP, KD, process_frame, current_distance, past_distance, REFRESH, current_velocity

    # -- Create a node for agv2
    rospy.init_node('agv2_controller', anonymous=True)

    # -- Update
    rate_fresh = rospy.Rate(REFRESH)
    time_step = float(1.0 / REFRESH)

    # -- Topics
    # odom_agv1 = 'agv1/odom'  # Odom of AGV1
    # odom_agv3 = 'agv3/odom'  # Odom of AGV3
    raw_camera_topic = 'agv2/camera/image_raw'  # Topic for the image from the camera
    process_camera_topic = 'agv2/camera/process_image'  # Topic to send the process image
    move_controller_topic = 'agv2/cmd_vel'  # Topic of move controller

    # -- Subscribers and Publishers
    # Define the publisher for the move controller topic
    vel_pub = rospy.Publisher(move_controller_topic, Twist, queue_size=25)
    camera_sub = rospy.Subscriber(raw_camera_topic, Image, frame_callback)  # Define the subscriber topic for camera
    # camera_pub = rospy.Publisher(process_camera_topic, Image, queue_size=30)
    # pose_agv1 = rospy.Subscriber(odom_agv1, Odometry, leader_position_callback)

    # -- Initial linear speed for the vehicle
    speed = Twist()
    speed.linear.x = INIT_SPEED

    while not rospy.is_shutdown():
        # -- Get the current velocity of vehicle
        current_velocity = speed.linear.x

        # -- Compute de error between the reference distance and the distance of the sensor
        error = current_distance - DIST_REF
        current_velocity = KP * error + (KP * (current_distance - past_distance) / time_step)
        past_distance = current_distance
        speed.linear.x = current_velocity
        speed.angular.z = 0.0

        # -- Send the update velocity
        print("AGV2: ", speed.linear.x)
        vel_pub.publish(speed)

        rate_fresh.sleep()


# -- Main function
if __name__ == '__main__':
    try:
        main()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutdown node")
