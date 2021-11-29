#!/usr/bin/env python
# coding=utf-8
import csv
import math
import numpy as np
import timeit
import rospy
from gazebo_msgs.srv import *

# --- Messages create
from tec_proyect.msg import V2VCommunication

# --- Libraries created
from multi_marker_detector import MarkerDetector
from control_pid import ControlPID

# --- ROS packages
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Range
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

# --- Constants for configuration of Aruco library
from tf.transformations import euler_from_quaternion

KERNEL_SIZE = (3, 3)  # Kernel size for Gaussian Blur
MARKER_LENGTH = 0.2  # Marker size in cm

# --- Constants for ID detection and AGV identification
# FRONT_AGV_IDS = np.zeros((3,), dtype=np.int)    # ID of preceding marker
FRONT_AGV_IDS = np.array([0, 1, 2], dtype=np.int)
VEHICLE_NAME = "AGV2"
SELF_IDS = np.array([3, 4, 5], dtype=np.int)    # ID of current agv
has_id = False                                  # If camera detect IDs

# --- Variables for control process
obtained_ids = "False"
init_config = False
WINDOW_SIZE = 5.0

# --- Constants for PID controller and variables
TIME_RESPONSE = 0.1
SET_DISTANCE = 0.7  # Distance between vehicles
MAX_SPEED = 1.20  # Max speed (m/s)
MAX_SPEED_BACKWARDS = 0.5
REFRESH = 30  # Refresh rate (Hz)
INIT_SPEED = 0.00  # Initial speed (m/s)

# --- Sensor variables
sonar_filter_right = 0.0
sonar_filter_left = 0.0
UNKNOWN_ID = 999999999
tag_info = {"id": int(UNKNOWN_ID)}
first_call = False
cnt = 0

# --- Subscribers and publisher global variables
bridge = CvBridge()
move = Twist()
video = Image()

# --- Global variables for publishers
vel_pub = None

# --- Get the calibration parameters of the camera
info_camera = "/home/fer/catkin_ws/src/tec_proyect/info/"
camera_matrix = np.loadtxt(info_camera + 'cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(info_camera + 'cameraDistortion.txt', delimiter=',')

# --- Create a object for the marker detection
marker_detection = MarkerDetector(MARKER_LENGTH, KERNEL_SIZE, camera_matrix, camera_distortion)


def frame_callback(video_frame):
    """
    Callback function for read and convert the image from ROS format to OpenCV format.

    :param video_frame: Video frame in ROS format
    """
    global bridge, has_id, tag_info, first_call, cnt
    # tic = timeit.default_timer()
    try:
        # --- Convert the image format of ROS to OpenCV form
        cv_frame = bridge.imgmsg_to_cv2(video_frame, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        has_id, tag_info = marker_detection.detection(cv_frame, FRONT_AGV_IDS, True, False)

        if cnt < 5:
            cnt += 1
        if first_call is False and cnt > 4:
            first_call = True
            cnt = 0

        # print(round(timeit.default_timer() - tic, 3))


def tag_detection_cb(video_frame):
    """
    Callback for detect tags for the initial configuration

    :param video_frame: Video frame in ROS format
    """
    global init_config
    num_tag_found_ok = 0
    try:
        # --- Convert the image format of ROS to OpenCV form
        cv_frame = bridge.imgmsg_to_cv2(video_frame, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        find_someone, tag_found, _ = marker_detection.single_detector(cv_frame, FRONT_AGV_IDS)

        num_tag_found_ok = [tag_found[i] == FRONT_AGV_IDS[i] for i in range(0, len(tag_found))].count(True)

        if find_someone or num_tag_found_ok >= 1:
            init_config = True


def lidar_scan_callback(scan_msg):
    """
    Callback function for laser scan.

    :param scan_msg: Message with information of scan topic
    :return: None
    """
    detected_field = {
        'fright': min(min(scan_msg.ranges[100:124]), 10),
        'front': min(min(scan_msg.ranges[125:145]), 10),
        'fleft': min(min(scan_msg.ranges[146:170]), 10),
    }


def odom_cb(odom_msg):
    timestamp = odom_msg.header.stamp.nsecs
    x_leader_pose = odom_msg.pose.pose.position.x
    y_leader_pose = odom_msg.pose.pose.position.y
    z_leader_pose = odom_msg.pose.pose.position.z

    orientation_quaternion = odom_msg.pose.pose.orientation
    orientation_list = [orientation_quaternion.x,
                        orientation_quaternion.y,
                        orientation_quaternion.z,
                        orientation_quaternion.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

    # rospy.loginfo("X: %s", x_leader_pose)
    # rospy.loginfo("y: %s", y_leader_pose)
    # rospy.loginfo("Z: %s", z_leader_pose)
    # rospy.loginfo("yaw: %s", yaw)


def sonar_left_cb(sonar_scan):
    """
    Callback for get the information from the left sonar

    :param sonar_scan: Message with information of the range topic
    """
    global sonar_filter_left

    sonar_filter_left = round(((sonar_filter_left * (WINDOW_SIZE - 1)) + sonar_scan.range) / WINDOW_SIZE, 3)


def sonar_right_cb(sonar_scan):
    """
    Callback for get the information from the right sonar

    :param sonar_scan: Message with information of the range topic
    """
    global sonar_filter_right

    sonar_filter_right = round(((sonar_filter_right * (WINDOW_SIZE - 1)) + sonar_scan.range) / WINDOW_SIZE, 3)


def stop():
    """
    Callback function to stop the AGV when evoke shutdown.
    :return: None
    """
    twist = Twist()
    twist.linear.x = 0.0
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0
    vel_pub.publish(twist)


def zero_one(x):
    """
    Change the variable True or False to 1 or 0
    :param x: variable True or False
    :return: 1 or 0
    """
    if x:
        return 1
    else:
        return 0


def ramp_vel(vel_prev, vel_target, t_prev, t_now, ramp_rate):
    """
    :param vel_prev: past velocity
    :param vel_target: current velocity
    :param t_prev: last time
    :param t_now: current time
    :param ramp_rate: # units: meters per second^2
    :return: velocity computed under the acceleration constraint provided as a parameter
    """
    step = ramp_rate * (t_now - t_prev).to_sec()
    sign = 1.0 if (vel_target > vel_prev) else -1.0
    error = math.fabs(vel_target - vel_prev)
    if error < step:
        return vel_target
    else:
        print(vel_prev + sign * step)
        return vel_prev + sign * step


def vehicle_config_cb(msg):
    """
    Callback for get the IDs for detection
    :param msg: message form the topic
    :return: None
    """
    global FRONT_AGV_IDS, obtained_ids

    if msg.vehicle_name == VEHICLE_NAME:
        FRONT_AGV_IDS[0] = msg.id0
        FRONT_AGV_IDS[1] = msg.id1
        FRONT_AGV_IDS[2] = msg.id2

        if not np.all(FRONT_AGV_IDS == 0):
            rospy.loginfo("IDs to detect: %s", FRONT_AGV_IDS)
            obtained_ids = "OK"


# --- Main function
if __name__ == '__main__':
    # --- Create a node for agv_follower1
    rospy.init_node('agv_follower1_controller', anonymous=True)

    # --- Wait until Gazebo run
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    s = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

    # --- Initial configuration for the IDs
    config_sub = rospy.Subscriber('id_assigment', V2VCommunication, vehicle_config_cb, queue_size=1)
    config_status_pub = rospy.Publisher('vehicles_status', V2VCommunication, queue_size=1)
    config_status = V2VCommunication()

    rospy.loginfo("Beginning the configuration...........")
    rospy.loginfo("Waiting...........")

    # --- Movement
    vel_pub = rospy.Publisher('agv2/cmd_vel', Twist, queue_size=1)
    agv_odom = rospy.Subscriber('agv2/odom', Odometry, odom_cb)

    # --- Initial configuration of the AGV
    # while True:
    #     # --- Keep here while
    #     rospy.loginfo("Configuring AGV...........")
    #     rospy.loginfo(obtained_ids)
    #     config_status.vehicle_name = VEHICLE_NAME
    #     config_status.status = obtained_ids
    #
    #     if obtained_ids == "OK":
    #         # --- Break the while loop
    #         config_status_pub.publish(config_status)
    #         camera_sub = rospy.Subscriber('agv2/camera/image_raw', Image, tag_detection_cb)
    #         while not init_config:
    #             if init_config:
    #                 break
    #         break
    #
    #     rospy.sleep(1)

    # --- Start with the AGV controller after the configuration
    # --- Unsubscribe to the topics
    # config_sub.unregister()
    # config_status_pub.unregister()
    # camera_sub.unregister()
    rospy.loginfo("Starting Tasks.........")

    # --- Sensors
    camera_sub = rospy.Subscriber('agv2/camera/image_raw', Image, frame_callback, queue_size=2)
    # lidar_sub = rospy.Subscriber('agv2/scan', LaserScan, lidar_scan_callback)
    sonar_left_sub = rospy.Subscriber('agv2/sonar_left', Range, sonar_left_cb)
    sonar_right_sub = rospy.Subscriber('agv2/sonar_right', Range, sonar_right_cb)

    rospy.loginfo("All OK............")
    # --- Update
    rate_fresh = rospy.Rate(REFRESH)
    t_past = rospy.Time.now()
    step_time = 0.0
    rospy.sleep(1)
    try:
        # --- Initial parameters for PID
        pid_control = ControlPID(2.0, 0.008, 0.5)
        lateral_pid = ControlPID(5.3, 3.2, 0.015)

        # pid_control = ControlPID(2.0, 0.008, 0.65)
        # lateral_pid = ControlPID(5.0, 3.2, 0.01)

        # Initialization of variables
        move.linear.x = INIT_SPEED
        move.angular.z = 0.0
        past_center_x = 0
        distance_filter = np.zeros((1,), dtype=np.float16)
        angle_filter = np.zeros((1,), dtype=np.float16)
        past_id = np.zeros(5, dtype=np.int16)

        past_vel = 0.0
        past_angle = 0.0
        policy = SET_DISTANCE

        while not rospy.is_shutdown():
            # --- Update the step time
            t_now = rospy.Time.now()
            step_time = (t_now - t_past).to_sec()
            has = zero_one(has_id)
            current_angle = 0.0

            if has_id and first_call and tag_info["distance"] <= 2.5:
                # distance_filter[0] = round((0.4 * float(tag_info["distance"])) + ((1 - 0.4) * distance_filter[0]), 3)
                # angle_filter[0] = round(((0.1 * tag_info["pitch"]) + ((1 - 0.1) * angle_filter[0])), 3)
                distance_filter[0] = marker_detection.d_filter
                angle_filter[0] = marker_detection.th_filter

                # --- Longitudinal Control
                control_signal = round(pid_control.compute(distance_filter[0], policy, step_time, 0.05, True), 3)

                if control_signal >= MAX_SPEED:
                    if control_signal < 0.0:
                        move.linear.x = -MAX_SPEED_BACKWARDS
                    else:
                        move.linear.x = MAX_SPEED
                elif distance_filter[0] > policy or control_signal <= MAX_SPEED:
                    if abs(control_signal) < 0.01:
                        control_signal = 0.0
                    move.linear.x = ramp_vel(past_vel, control_signal, t_past, t_now, 0.6)

                past_vel = control_signal

                # --- Lateral controller
                err = tag_info["center_x"] - tag_info["width"] / 2.0
                center = -float(err) / 180.0
                past_center_x = tag_info["center_x"]

                if (0.018 >= angle_filter >= -0.018) and tag_info["id"] == FRONT_AGV_IDS[1] and tag_info["id"] == \
                        past_id[0]:
                    center = -float(err) / 270.0
                    current_angle = 0.0 + center
                elif tag_info["id"] == FRONT_AGV_IDS[1] and tag_info["id"] == past_id[0]:
                    current_angle = angle_filter[0] + center
                elif tag_info["id"] == FRONT_AGV_IDS[0] or tag_info["id"] == FRONT_AGV_IDS[2] and tag_info["id"] == \
                        past_id[0]:
                    current_angle = angle_filter[0] + center
                else:
                    current_angle = center

                angle = round(lateral_pid.compute(current_angle, 0.005, step_time, 0.05, True), 4)
                if math.pi + math.pi / 2 >= angle >= -(math.pi + math.pi / 2):
                    move.angular.z = ramp_vel(past_angle, angle, t_past, t_now, 0.7)
                elif angle < -(math.pi + math.pi / 2):
                    move.angular.z = -((math.pi + math.pi) / 2)
                else:
                    move.angular.z = (math.pi + math.pi) / 2

                past_angle = angle

                past_id[0] = past_id[1]
                past_id[1] = past_id[2]
                past_id[2] = past_id[3]
                past_id[3] = past_id[4]
                past_id[4] = tag_info["id"]

                # --- Send the update velocity
                vel_pub.publish(move)
                rospy.loginfo("ID: %s", tag_info["id"])
                # rospy.loginfo("Vel_X: %s", move.linear.x)
                # rospy.loginfo("Distance: %s", distance_filter[0])
                # rospy.loginfo("Angle: %s", round(move.angular.z, 3))
                # rospy.loginfo("Center: %s", round(err, 3))
                # rospy.loginfo("Left Sonar: %s", sonar_filter_left)
                # rospy.loginfo("Right Sonar: %s", sonar_filter_right)
                # rospy.loginfo("Step time: %s", step_time)
                rospy.loginfo(".............................")

            else:
                stop()
                distance_filter = ([0, ])
                angle_filter = ([0, ])

            # if video_new is not None:
            #     video_pub.publish(video_new)

            # --- Save the angle info in a csv file
            # with open("/home/fer/Pruebas/agv2_test.csv", "a") as cvsfile:
            #     writer = csv.writer(cvsfile)
            #     writer.writerow([str(has), str(tag_info["id"]), str(round(np.degrees(angle_filter[0]), 4))])

            # Run when the node is shutdown to stop the vehicle
            rospy.on_shutdown(stop)
            t_past = t_now
            rate_fresh.sleep()

    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutdown node")
