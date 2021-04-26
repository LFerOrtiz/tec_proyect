#!/usr/bin/env python
import cv2
import math
import numpy as np

from cv2 import aruco

import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

# import tf2

FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font for displaying text
K_SIZE = (3, 3)  # Kernel size for Gaussian Blur
KP = 0.98  # Gain for proportional controller
MARKER_LENGTH = 0.2  # Marker size in cm
LEADER_TAG = 0  # ID of leader marker
DIST_REF = 0.7  # Distance Reference

error = 0  # Error
distance_avr = 0
angle_avr = 0
distance = 0
sample = 0
angle = 0

# CvBridge
bridge = CvBridge()
process_image = None

# --- 180 deg rotation matrix around the X axis
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

# --- Define the Aruco dictionary 
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)  # Use 5x5 dictionary for find the markers
parameters = aruco.DetectorParameters_create()  # Marker detection parameters
# parameters.adaptiveThreshConstant = 5     # Adaptive Threshold Constant
parameters.adaptiveThreshWinSizeMin = 5  # Min window of Adaptive Threshold
parameters.adaptiveThreshWinSizeMax = 6  # Max window of Adaptive Threshold
parameters.adaptiveThreshWinSizeStep = 100
parameters.cornerRefinementMinAccuracy = 0.001  # Minimum error for the stop refinement process
parameters.cornerRefinementMaxIterations = 50  # Maximum number of iterations for stop criteria of the corner refinement process

info_camera = "/home/fer/catkin_ws/src/tec_proyect/info/"
camera_matrix = np.loadtxt(info_camera + 'cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(info_camera + 'cameraDistortion.txt', delimiter=',')


def is_rotation_matrix(r):
    """ Checks if a matrix is a valid rotation matrix """
    rt = np.transpose(r)
    should_be_identity = np.dot(rt, r)
    i = np.identity(3, dtype=r.dtype)
    n = np.linalg.norm(i - should_be_identity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(r):
    """ Calculates rotation matrix to euler angles.
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ). """
    assert (is_rotation_matrix(r))

    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def draw_cube(frame, corners):
    pass


def frame_callback(ros_frame):
    """ Convert the image format of ROS to OpenCV format """
    global MARKER_LENGTH, DIST_REF, bridge, aruco_dict, parameters, distance, distance_avr, error, sample, angle, \
        angle_avr, camera_matrix, camera_distortion

    try:
        cv_frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        # rospy.loginfo("Frame received")
        height, width, channels = cv_frame.shape

        # Camera matrix for calibration (ideal camera)
        # camera_matrix = np.mat([[1.0, 0.0, height / 2.0], [0.0, 1.0, width / 2.0], [0.0, 0.0, 1.0]])
        # camera_distortion = np.mat([0.0, 0.0, 0.0, 0.0, 0.0])

        gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)  # Convert to gray scale
        gray = cv2.GaussianBlur(gray, K_SIZE, 0)  # Applied a Gaussian Filter for High frequency noise

        # List IDs and the corners of each ID
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if np.all(ids is not None) and ids[0] == LEADER_TAG:  # Check if found a marker
            for item in range(0, len(ids)):
                # rotationVect: Pose of the marker respect to camera frame
                # translationVect: Position of the marker in camera frames
                pose_return = aruco.estimatePoseSingleMarkers(corners[item], MARKER_LENGTH, cameraMatrix=camera_matrix,
                                                              distCoeffs=camera_distortion)
                rotation_vet, translation_vet = pose_return[0][0, 0, :], pose_return[1][0, 0, :]

                # for i in range(rotationVect.shape[0]):
                aruco.drawDetectedMarkers(cv_frame, corners)  # Draw a square around the markers
                aruco.drawAxis(cv_frame, camera_matrix,  # Draw the axis in the video frame
                               camera_distortion,
                               rotation_vet, translation_vet, 0.05)
                center_x = (corners[item][0][0][0] + corners[item][0][1][0] + corners[item][0][2][0] +
                            corners[item][0][3][
                                0]) / 4  # X coordinate of marker's center
                center_y = (corners[item][0][0][1] + corners[item][0][1][1] + corners[item][0][2][1] +
                            corners[item][0][3][
                                1]) / 4  # Y coordinate of marker's center
                cv2.putText(cv_frame, "id" + str(ids[item]), (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (50, 225, 250), 2)

                # --- Rotation matrix tag -> camera
                # rospy.loginfo(cv2.Rodrigues(rotation_vet)[0])
                R_ct = np.matrix(cv2.Rodrigues(rotation_vet)[0])
                R_tc = R_ct.T  # Transpose of the matrix

                # --- Get the attitude in terms of euler 321 (Flip first)
                roll_marker, pitch_marker, yaw_marker = rotation_matrix_to_euler_angles(R_flip * R_tc)

                if sample < 5:
                    distance += np.sqrt(translation_vet[0] ** 2 + translation_vet[1] ** 2 + translation_vet[2] ** 2)
                    angle += pitch_marker
                    sample += 1.0
                else:
                    angle = angle / sample
                    distance = distance / sample
                    distance_avr = distance
                    angle_avr = angle
                    distance = 0
                    sample = 0
                    error = DIST_REF - distance_avr

                # --- Show general info of distance and angles
                str_angle = "Angle: %3.4f" % math.degrees(angle_avr)
                cv2.putText(cv_frame, str_angle, (0, 400), FONT, 0.5, (0, 155, 230), 2, cv2.LINE_AA)
                str_distance = "Distance: %3.4f" % distance_avr
                cv2.putText(cv_frame, str_distance, (0, 415), FONT, 0.5, (0, 155, 230), 2, cv2.LINE_AA)

                # --- X: Red Axis (Roll), Y: Green Axis (Pitch), Z: Blue Axis (Yaw)
                str_position = "Marker position X=%3.2f  Y=%3.2f  Z=%3.2f" % (
                    translation_vet[0], translation_vet[1], translation_vet[2])
                cv2.putText(cv_frame, str_position, (0, 430), FONT, 0.5, (0, 155, 255), 2, cv2.LINE_AA)
                str_attitude = "Marker Attitude R=%3.2f  P=%3.2f  Y=%3.2f" % (
                    math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker))
                cv2.putText(cv_frame, str_attitude, (0, 445), FONT, 0.5, (0, 155, 255), 2, cv2.LINE_AA)

                # --- Position and attitude of the camera respect to the marker
                # camera_pose = -R_ct * np.matrix(translation_vet).T
                # roll_camera, pitch_camera, yaw_camera = rotation_matrix_to_euler_angles(R_flip * R_ct)
                # str_position = "Camera position X=%3.2f  Y=%3.2f  Z=%3.2f"%(camera_pose[0], camera_pose[1], camera_pose[2])
                # cv2.putText(cv_frame, str_position, (0, 460), FONT, 0.5, (0, 25, 255), 2, cv2.LINE_AA)
                # str_attitude = "Camera Attitude R=%3.2f  P=%3.2f  Y=%3.2f"%(math.degrees(roll_camera), math.degrees(pitch_camera), math.degrees(yaw_camera))
                # cv2.putText(cv_frame, str_attitude, (0, 475), FONT, 0.5, (0, 25, 255), 2, cv2.LINE_AA)

        # process_image = bridge.cv2_to_imgmsg(cv_frame, "rgb8")
        cv2.imshow("Tag", cv_frame)
        cv2.waitKey(10)


def leader_position(msg):
    pass


def main():
    global error, KP
    # Create a node for agv2
    rospy.init_node('agv2_controller', anonymous=True)
    rate_fresh = rospy.Rate(1)  # 60 Hz

    # Topics
    odom_agv1 = 'agv1/odom'  # Odom of AGV1
    odom_agv3 = 'agv3/odom'  # Odom of AGV3
    raw_camera_topic = 'agv2/camera/image_raw'
    process_camera_topic = 'agv2/camera/process_image'
    move_controller_topic = 'agv2/cmd_vel'

    # Subscribers and Publishers
    vel_pub = rospy.Publisher(move_controller_topic, Twist, queue_size=25)
    camera_sub = rospy.Subscriber(raw_camera_topic, Image, frame_callback)
    camera_pub = rospy.Publisher(process_camera_topic, Image, queue_size=30)
    pose_agv1 = rospy.Subscriber(odom_agv1, Odometry, leader_position)

    speed = Twist()
    speed.linear.x = 0.05

    while not rospy.is_shutdown():
        # camera_pub.publish(process_frame)
        if 0.1 > error > 0.0:
            speed.linear.x = speed.linear.x - 0.01
            speed.angular.z = 0.0
        elif error - 0.05 < -0.2:
            speed.linear.x = speed.linear.x + 0.001 / KP  # High speed (Forward)
            speed.angular.z = 0.0
            rospy.loginfo("High Speed")
        elif error + 0.05 > 0.2:
            speed.linear.x = speed.linear.x * KP  # Low speed (Back)
            speed.angular.z = 0.0
            rospy.loginfo("Low Speed")
        elif speed.linear.x > 0.5 or speed.linear.x < -0.5:
            speed.linear.x = speed.linear.x
            speed.angular.z = 0.0

        # rospy.loginfo(error)
        # rospy.loginfo(speed)
        # vel_pub.publish(speed)

        rate_fresh.sleep()


# Main function
if __name__ == '__main__':
    try:
        main()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutdown node")
