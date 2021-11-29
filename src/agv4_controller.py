#!/usr/bin/env python
# coding=utf-8
import csv
import math
import numpy as np
import rospy
from gazebo_msgs.srv import *

# --- Messages create
from tec_proyect.msg import V2VCommunication

# --- Libraries created
from multi_marker_detector import MarkerDetector
from control_pid import ControlPID
from move_to_pose import MoveToGoal

# --- ROS packages
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion

# --- Constants for configuration of Aruco library
KERNEL_SIZE = (3, 3)  # Kernel size for Gaussian Blur
MARKER_LENGTH = 0.2  # Marker size in cm

# --- Constants for ID detection and AGV identification
x_front_pose, y_front_pose, front_yaw = 0.0, 0.0, 0.0
x_agv_pose, y_agv_pose, agv_yaw = 0.0, 0.0, 0.0
FRONT_AGV_IDS = np.zeros((3,), dtype=np.int)    # ID of the leading agv
FRONT_AGV_NAME = None                           # Name of the leading agv
VEHICLE_NAME = "agv4"                           # Name of this agv
SELF_IDS = np.array([9, 10, 11], dtype=np.int)  # ID of this agv
has_id = False                                  # If camera detect IDs

# --- Variables for control process
obtained_ids = None
init_config = False
all_config = False
ids_detect = None
first_call = False
disunion = False
disunion_completed = False
cnt = 0
x_goal_disunion, y_goal_disunion, theta_goal = 0.0, 0.0, 0.0
WINDOW_SIZE = 5.0

# --- Constants for PID controller and variables
TIME_RESPONSE = 0.1
SET_DISTANCE = 0.7          # Distance between vehicles
MAX_SPEED = 1.20            # Max speed (m/s)
MAX_SPEED_BACKWARDS = 0.5
REFRESH = 30                # Refresh rate (Hz)
INIT_SPEED = 0.0            # Initial speed (m/s)

# --- Subscribers and publisher global objets
bridge = CvBridge()
move = Twist()
video = Image()
move2goal = None

# --- Global variables for publishers
vel_pub = None

# --- Get the calibration parameters of the camera
info_camera = "/home/fer/catkin_ws/src/tec_proyect/info/"
camera_matrix = np.loadtxt(info_camera + 'cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(info_camera + 'cameraDistortion.txt', delimiter=',')

# --- Create a object for the marker detection
marker_detection = MarkerDetector(MARKER_LENGTH, KERNEL_SIZE, camera_matrix, camera_distortion)


def frame_callback(ros_frame):
    """
    Callback function for read and convert the image from ROS format to OpenCV format.
    :param ros_frame: Video frame in ROS format
    :return: None
    """
    global bridge, has_id, tag_info, first_call, cnt

    try:
        # --- Convert the image format of ROS to OpenCV form
        cv_frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        has_id, tag_info = marker_detection.detection(cv_frame, FRONT_AGV_IDS, True, False)

        if cnt < 5:
            cnt += 1
        if first_call is False and cnt > 4:
            first_call = True
            cnt = 0


def odom_cb(odom_msg):
    """Odometry of the agv"""
    global x_agv_pose, y_agv_pose, agv_yaw
    x_agv_pose = round(odom_msg.pose.pose.position.x, 4)
    y_agv_pose = round(odom_msg.pose.pose.position.y, 4)

    orientation_quaternion = odom_msg.pose.pose.orientation
    orientation_list = [orientation_quaternion.x,
                        orientation_quaternion.y,
                        orientation_quaternion.z,
                        orientation_quaternion.w]
    (agv_roll, agv_pitch, agv_yaw) = euler_from_quaternion(orientation_list)

    # Update the pose of the agv
    move2goal.update_pose(x_agv_pose, y_agv_pose, agv_yaw)


def front_odom_cb(odom_msg):
    """Odometry of the front agv"""
    global x_front_pose, y_front_pose, front_yaw
    x_front_pose = odom_msg.pose.pose.position.x
    y_front_pose = odom_msg.pose.pose.position.y

    orientation_quaternion = odom_msg.pose.pose.orientation
    orientation_list = [orientation_quaternion.x,
                        orientation_quaternion.y,
                        orientation_quaternion.z,
                        orientation_quaternion.w]
    (front_roll, front_pitch, front_yaw) = euler_from_quaternion(orientation_list)


def stop():
    """
    Callback function to stop the AGV when evoke shutdown.
    :return: None
    """
    twist = Twist()
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
        return vel_prev + sign * step


def tag_detection_cb(video_frame):
    """
    Callback for detect tags for the initial configuration

    :param video_frame: Video frame in ROS format
    """
    global ids_detect, FRONT_AGV_IDS
    try:
        # --- Convert the image format of ROS to OpenCV form
        cv_frame = bridge.imgmsg_to_cv2(video_frame, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        past_ids = FRONT_AGV_IDS
        find_someone, tag_found, _ = marker_detection.single_detector(cv_frame, FRONT_AGV_IDS)
        num_tag_found_ok = any(x in tag_found for x in FRONT_AGV_IDS)

        if find_someone and num_tag_found_ok is True:
            ids_detect = "ID_DETECTED"
            FRONT_AGV_IDS = past_ids
        else:
            ids_detect = None


def vehicle_config_cb(msg):
    """
    Callback for get the IDs for detection

    :param msg: message form the topic
    :return: None
    """
    global FRONT_AGV_IDS, obtained_ids, FRONT_AGV_NAME, all_config

    if msg.status == "coupling" and msg.vehicle_name == VEHICLE_NAME:
        all_config = True
        rospy.logerr("AGV4 full config %s", all_config)

    if ids_detect is None and msg.vehicle_name != VEHICLE_NAME:
        FRONT_AGV_NAME = msg.vehicle_name
        FRONT_AGV_IDS[0] = msg.id0
        FRONT_AGV_IDS[1] = msg.id1
        FRONT_AGV_IDS[2] = msg.id2

    if not np.all(FRONT_AGV_IDS == 0):
        rospy.loginfo("IDs to detect: %s", FRONT_AGV_IDS)
        obtained_ids = "OBTAINED_ID"
    else:
        obtained_ids = None


def disconfig_cb(msg):
    global x_goal_disunion, y_goal_disunion, theta_goal, init_config, disunion
    if msg.vehicle_name == VEHICLE_NAME and msg.status == "DISUNION" and not disunion:
        rospy.logerr("Initializing disjointing ......")
        x_goal_disunion = msg.x_point
        y_goal_disunion = msg.y_point
        theta_goal = msg.theta
        init_config = False
        disunion = True


# --- Main function
if __name__ == '__main__':
    # --- Create a node for agv4
    rospy.init_node('agv4_controller', anonymous=True)

    # --- Wait until Gazebo run
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    s = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

    # --- Initial configuration for the IDs
    config_sub = rospy.Subscriber('/id_assigment', V2VCommunication, vehicle_config_cb, queue_size=1)
    config_status_pub = rospy.Publisher('/vehicles_status', V2VCommunication, queue_size=1)
    config_status = V2VCommunication()

    rospy.loginfo("Beginning the configuration...........")
    rospy.loginfo("Waiting...........")

    # --- Movement
    vel_pub = rospy.Publisher('agv4/cmd_vel', Twist, queue_size=1)
    move2goal = MoveToGoal(vel_pub)
    rospy.loginfo("Getting odometry...........")
    agv_odom = rospy.Subscriber('agv4/odom', Odometry, odom_cb)
    front_agv_odom_sub = None

    # --- Initial configuration of the AGV
    while not init_config:
        # --- Keep here while get the ID of the front agv
        count = 0
        rospy.logdebug_once("Configuring AGV4........")

        if obtained_ids == "OBTAINED_ID":
            # --- Activate the camera and detect the ID of the front agv
            camera_sub = rospy.Subscriber('agv4/camera/image_raw', Image, tag_detection_cb)
            while not init_config:
                # --- Keep here while don't get the correct ID of the leading agv
                if not all_config:
                    count += 1
                if ids_detect == "ID_DETECTED":
                    rospy.loginfo("IDs: %s", FRONT_AGV_IDS)
                    # Send a answer to the leader agv
                    config_status.vehicle_name = VEHICLE_NAME
                    config_status.vehicle_detect = FRONT_AGV_NAME
                    config_status.status = ids_detect
                    for i in range(0, 2):
                        config_status_pub.publish(config_status)
                    rospy.sleep(1)
                    if all_config:
                        # All the initial configuration is complete
                        init_config = True
                        # Connect to the odometry of the leading agv
                        front_odom_topic = FRONT_AGV_NAME + '/odom'
                        front_agv_odom_sub = rospy.Subscriber(front_odom_topic, Odometry, front_odom_cb)

                if count >= 15:
                    obtained_ids = None
                    ids_detect = None
                    camera_sub.unregister()
                    break
                rospy.sleep(1)

        if not all_config:
            init_config = False

        rospy.sleep(1)

    # --- Start with the AGV controller after the configuration
    # --- Unsubscribe to the topics
    config_sub.unregister()
    config_status_pub.unregister()
    camera_sub.unregister()
    rospy.loginfo("Starting Tasks.........")

    # --- Sensors
    camera_sub = rospy.Subscriber('agv4/camera/image_raw', Image, frame_callback)
    disconfig_sub = rospy.Subscriber('agv4/disconfig', V2VCommunication, disconfig_cb, queue_size=2)
    disconfig_pub = rospy.Publisher('agv4/disconfig', V2VCommunication, queue_size=2)
    disconfig = V2VCommunication()

    rospy.loginfo("All OK............")
    # --- Update
    rate_fresh = rospy.Rate(REFRESH)
    t_past = rospy.Time.now()
    step_time = 0.0
    rospy.sleep(1)
    try:
        # --- Initial parameters for PID
        pid_control = ControlPID(2.0, 0.008, 0.5)
        pid_velocity = ControlPID(2.0, 0.008, 0.5)
        lateral_pid = ControlPID(5.3, 3.2, 0.015)

        # Initialization of variables
        move.linear.x = INIT_SPEED
        move.angular.z = 0.0
        past_center_x = 0
        distance_filter = np.zeros((1,), dtype=np.float16)  # distance_filter[0]: past
        angle_filter = np.zeros((1,), dtype=np.float16)  # angle_filter[0]: past
        past_id = np.zeros(5, dtype=np.int16)  # angle_filter[0]: past
        obstacle_lateral = False

        past_vel = 0.0
        past_angle = 0.0
        policy = SET_DISTANCE

        while init_config:
            # --- Update the step time
            t_now = rospy.Time.now()
            step_time = (t_now - t_past).to_sec()
            has = zero_one(has_id)
            current_angle = 0.0

            if has_id and first_call and tag_info["distance"] <= 2.5:
                distance_filter[0] = marker_detection.d_filter
                angle_filter[0] = marker_detection.th_filter

                # --- Longitudinal Control
                current_vel = move.linear.x
                vel_control = round(pid_velocity.compute(current_vel, past_vel, step_time, MAX_SPEED, True), 3)
                control_signal = round(pid_control.compute(distance_filter[0], policy, step_time, 0.05, True), 3)

                out_vel = vel_control + control_signal
                if control_signal >= MAX_SPEED:
                    if control_signal < 0.0:
                        move.linear.x = -MAX_SPEED_BACKWARDS
                    else:
                        move.linear.x = MAX_SPEED
                elif distance_filter[0] > policy or control_signal <= MAX_SPEED:
                    if abs(control_signal) < 0.01:
                        control_signal = 0.0
                    move.linear.x = ramp_vel(past_vel, out_vel, t_past, t_now, 0.6)

                past_vel = control_signal

                # --- Lateral controller
                err = tag_info["center_x"] - tag_info["width"] / 2.0
                center = -float(err) / 175.0
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

                angle = round(lateral_pid.compute(current_angle, 0.01, step_time, 0.05, False), 4)
                if math.pi + math.pi / 2 >= angle >= -(math.pi + math.pi / 2):
                    move.angular.z = ramp_vel(past_angle, angle, t_past, t_now, 0.6)
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

            else:
                stop()
                distance_filter = ([0, ])
                angle_filter = ([0, ])

            # Run when the node is shutdown to stop the vehicle
            rospy.on_shutdown(stop)
            t_past = t_now
            rate_fresh.sleep()

        # -- Disunion loop
        while not init_config and disunion and not disunion_completed:
            rospy.logwarn("Disconfiguring the agv")
            move2goal.move_to_pose(x_goal_disunion, y_goal_disunion, theta_goal)
            disconfig_sub.unregister()
            rospy.sleep(1)
            disconfig.vehicle_name = VEHICLE_NAME
            disconfig.status = "DISUNION_COMPLETED"
            disconfig_pub.publish(disconfig)
            rospy.logerr("In the station")
            rospy.sleep(1)
            disunion_completed = True

    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutdown node")
