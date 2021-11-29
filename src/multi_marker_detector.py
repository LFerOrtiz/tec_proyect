# coding=utf-8
import csv
import cv2
import math
import numpy as np

from cv2 import aruco

import rospy

cnt = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font for displaying text
FONT_SIZE = 0.7
UNKNOWN_ID = 999999999
WINDOW_SIZE = 5
CYAN = (255, 255, 0)
INIT_INFO = {"id": int(UNKNOWN_ID),
             "distance": 0.0,
             "pitch": 0.0,
             "width": 0.0,
             "center_x": 0.0,
             }


def _get_chunks(length, n):
    """Yield successive n-sized chunks from l."""
    a = []
    for i in range(0, len(length), n):
        a.append(length[i:i + n])

    return a


def _get_contour_center(contours):
    m = cv2.moments(contours)
    cx = -1
    cy = -1
    if (m['m00'] != 0):
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])

    return cx, cy


class MarkerDetector:
    def __init__(self, marker_size, kernel_size, camera_matrix, camera_distortion):
        self.marker_size = marker_size
        self.kernel_size = kernel_size
        self.parameters = None
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion
        self.marker_dict = None
        self.corners = None
        self.ids = None

        # -- Frame info
        self.height = 0
        self.width = 0
        self.channels = 0
        self.center_x = 0
        self.center_y = 0

        # -- Distance and angle average
        self.x_coord = 0
        self.y_coord = 0
        self.z_coord = 0
        self.d_filter = 0.0
        self.th_filter = 0.0
        self.distance = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        # --- 180 deg rotation matrix around the X axis
        self.r_flip = np.zeros((3, 3), dtype=np.float32)
        self.r_flip[0, 0] = 1.0
        self.r_flip[1, 1] = -1.0
        self.r_flip[2, 2] = -1.0
        self._marker_dictionary_config()

    def _marker_dictionary_config(self):
        """ Parameters configuration for aruco detector function """
        # --- Define the Aruco dictionary
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)  # Use 5x5 dictionary for find the markers
        self.parameters = aruco.DetectorParameters_create()  # Marker detection parameters
        self.parameters.adaptiveThreshWinSizeMin = 5  # Min window of Adaptive Threshold
        self.parameters.adaptiveThreshWinSizeMax = 6  # Max window of Adaptive Threshold
        self.parameters.adaptiveThreshWinSizeStep = 100
        self.parameters.cornerRefinementMinAccuracy = 0.001  # Minimum error for the stop refinement process
        self.parameters.cornerRefinementMaxIterations = 50  # Maximum number of iterations for stop criteria of the
        # corner refinement process

    def _is_rotation_matrix(self, r):
        """ Checks if a matrix is a valid rotation matrix """
        rt = np.transpose(r)
        should_be_identity = np.dot(rt, r)
        i = np.identity(3, dtype=r.dtype)
        n = np.linalg.norm(i - should_be_identity)
        return n < 1e-6

    def _rotation_matrix_to_euler_angles(self, r):
        """ Calculates rotation matrix to euler angles.

        The order of the euler angles ( x and z are swapped ). """
        assert (self._is_rotation_matrix(r))

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

    def detection(self, frame, ids_detect, show_frame, debug=False):
        global cnt
        """
        Detection of multiples marker and get the distance and pose of the marker respect to the camera.

        :param frame:       Video frame
        :param ids_detect:  Array of ids to detect
        :param show_frame:  Create a new window to show the frame proceed
        :param debug:       Show the pose information of the tag in the video window
        """
        """ Detect the marker indicate in the video frame
            X: Red Axis (Roll)
            Y: Green Axis (Pitch)
            Z: Blue Axis (Yaw)
        """
        pose_list = []
        parallel_tag = {}
        # -- Get the height, width and color channels of the frame
        has_id = False
        self.height, self.width, self.channels = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the image to gray scale
        gray = cv2.GaussianBlur(gray, self.kernel_size, 0)  # Applied a Gaussian Filter for High frequency noise
        self.corners, self.ids, rejected_img_points = aruco.detectMarkers(gray, self.marker_dict,
                                                                          parameters=self.parameters)

        # Check for at least one id
        if np.all(self.ids is not None):
            # Check if there is at least one matching item
            has_id = any(x in self.ids for x in ids_detect)

            if has_id:
                for item in range(0, len(self.ids)):
                    # rotationVect: Pose of the marker respect to camera frame
                    # translationVect: Position of the marker in camera frames
                    pose_return = aruco.estimatePoseSingleMarkers(self.corners[item], self.marker_size,
                                                                  cameraMatrix=self.camera_matrix,
                                                                  distCoeffs=self.camera_distortion)
                    # print("Pose: ", pose_return)
                    rotation_vet, translation_vet = pose_return[0][0, 0, :], pose_return[1][0, 0, :]
                    # print("Vector de Rotacion: ", rotation_vet)
                    # print("Vector de traslacion: ", translation_vet)

                    # --- Find the center of the marker and draw the ID
                    # X coordinate of marker's center
                    self.center_x = (self.corners[item][0][0][0] + self.corners[item][0][1][0] +
                                     self.corners[item][0][2][0] + self.corners[item][0][3][0]) / 4

                    # Y coordinate of marker's center
                    self.center_y = (self.corners[item][0][0][1] + self.corners[item][0][1][1] +
                                     self.corners[item][0][2][1] + self.corners[item][0][3][1]) / 4

                    # --- Rotation matrix tag -> camera
                    r_ct = np.matrix(cv2.Rodrigues(rotation_vet)[0])
                    r_tc = r_ct.T  # Transpose of the matrix

                    # --- Get the attitude in terms of euler 321 (Flip first)
                    self.roll, self.pitch, self.yaw = self._rotation_matrix_to_euler_angles(self.r_flip * r_tc)

                    # --- Calculated estimated distance
                    self.x_coord = round(translation_vet[0], 4)
                    self.y_coord = round(translation_vet[1], 4)
                    self.z_coord = round(translation_vet[2], 4)
                    self.distance = round(math.sqrt(self.x_coord ** 2 + self.y_coord ** 2 + self.z_coord ** 2),
                                          3) - 0.15

                    # --- Update the list of pose with the current values
                    if self.ids[item] in ids_detect:
                        # if self.ids[item].astype(int)[0] != ids_detect[1]:
                        #     sign = np.sign(self.pitch)
                        #     self.pitch = (math.fabs(self.pitch) + 0.61) * sign
                        pose_list.append({"id": self.ids[item].astype(int)[0],
                                          "distance": self.distance,
                                          "pitch": round(self.pitch, 3),
                                          "width": int(self.width),
                                          "center_x": int(self.center_x),
                                          })

                        # --- Draw the information of tag on the screen
                        cv2.putText(frame, "id" + str(self.ids[item]), (int(self.center_x), int(self.center_y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 225, 250), 2)
                        cv2.circle(frame, (int(self.center_x), int(self.center_y)), 8, (0, 0, 255), -1)
                        aruco.drawDetectedMarkers(frame, self.corners)  # Draw a square around the markers
                        aruco.drawAxis(frame, self.camera_matrix,  # Draw the axis in the video frame
                                       self.camera_distortion,
                                       rotation_vet, translation_vet, 0.2)

                    # --- Show general info of distance and angles of the marker
                    if debug and len(pose_list) == 1:
                        str_angle = "Angle: %3.4f" % math.degrees(self.pitch)
                        cv2.putText(frame, str_angle, (0, 400), FONT, FONT_SIZE, (255, 0, 0), 2, cv2.LINE_AA)
                        str_distance = "Distance: %3.4f" % self.distance
                        cv2.putText(frame, str_distance, (0, 420), FONT, FONT_SIZE, (255, 0, 0), 2, cv2.LINE_AA)

                        str_position = "Marker position X=%3.2f  Y=%3.2f  Z=%3.2f" % (
                            self.x_coord, self.y_coord, self.z_coord)
                        cv2.putText(frame, str_position, (0, 440), FONT, FONT_SIZE, (255, 0, 0), 2, cv2.LINE_AA)
                        str_attitude = "Marker Attitude R=%3.2f  P=%3.2f  Y=%3.2f" % (
                            math.degrees(self.roll), math.degrees(self.pitch), math.degrees(self.yaw))
                        cv2.putText(frame, str_attitude, (0, 460), FONT, FONT_SIZE, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                parallel_tag = INIT_INFO
                has_id = False
        else:
            pose_list = []
            parallel_tag = INIT_INFO
            has_id = False

        # --- Find the tag with the pitch angle closest to zero
        if len(pose_list) >= 2:
            item_min_pitch = 0
            item_max_pitch = 0

            for index in range(0, len(pose_list)):
                if pose_list[index]["id"] <= 99:
                    item_min_pitch = dict(min([abs(d.get('pitch')), d.items()] for d in pose_list
                                              if d.get('pitch') is not None)[1])
                    item_max_pitch = dict(max([abs(d.get('pitch')), d.items()] for d in pose_list
                                              if d.get('pitch') is not None)[1])

            if abs(math.floor(math.degrees(item_min_pitch["pitch"]))) <= 35:
                parallel_tag = item_min_pitch
            elif abs(math.floor(math.degrees(item_min_pitch["pitch"]))) > 50:
                parallel_tag = item_max_pitch
            else:
                mid = int(self.width / 2.0)
                x_offset = (item_max_pitch["center_x"] + item_min_pitch["center_x"]) / 2.0 - mid
                y_offset = item_min_pitch["distance"]

                parallel_tag = item_max_pitch
                parallel_tag["pitch"] = math.atan2(y_offset, x_offset)

        elif len(pose_list) == 1:
            parallel_tag = pose_list[0]

        self.d_filter = round(((self.d_filter * (WINDOW_SIZE - 1)) + float(parallel_tag["distance"])) / WINDOW_SIZE, 3)
        self.th_filter = round(((self.th_filter * (WINDOW_SIZE - 1)) + float(parallel_tag["pitch"])) / WINDOW_SIZE, 4)

        # with open("/home/fer/Pruebas/test_50.csv", "a") as cvsfile:
            # cnt += 1
            # rospy.logwarn(cnt)
            # if cnt < 500:
            #     writer = csv.writer(cvsfile)
            #     writer.writerow([str("Distancia"), str(self.d_filter), str(round(np.degrees(self.th_filter), 4)), str(5.0)])

        # --- If True, create a new window with the video frame processed
        if show_frame:
            cv2.drawMarker(frame, (int(self.width / 2.0), int(self.height / 2.0)), color=CYAN,
                           markerType=cv2.MARKER_CROSS, thickness=3)
            window_resize = cv2.resize(frame, (352, 240))  # 480, 360 resize
            cv2.imshow("Marker ID " + str(ids_detect), window_resize)
            cv2.waitKey(1)

        # --- Return the video frame processed, if the algorithm detect some tag and the information of tag with the
        #     smallest pitch angle
        return has_id, parallel_tag

    def single_detector(self, frame, id_detect):
        """
        Callback function for read and convert the image format of ROS to OpenCV format.

        :param frame:       Video frame
        :param id_detect:   Array of ids to detect
        :return:            has_id, id_found_array, position_tag_x
        """
        # -- Get the height, width and color channels of the frame
        self.height, self.width, self.channels = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the image to gray scale
        gray = cv2.GaussianBlur(gray, self.kernel_size, 0)  # Applied a Gaussian Filter for High frequency noise
        self.corners, self.ids, rejected_img_points = aruco.detectMarkers(gray, self.marker_dict,
                                                                          parameters=self.parameters)

        id_found_array = []
        # Check for at least one id
        if np.all(self.ids is not None):
            # Check if there is at least one matching item
            has_id = any(x in self.ids for x in id_detect)

            if has_id:
                for item in range(0, len(self.ids)):
                    # rotationVect: Pose of the marker respect to camera frame
                    # translationVect: Position of the marker in camera frames
                    pose_return = aruco.estimatePoseSingleMarkers(self.corners[item], self.marker_size,
                                                                  cameraMatrix=self.camera_matrix,
                                                                  distCoeffs=self.camera_distortion)
                    rotation_vet, translation_vet = pose_return[0][0, 0, :], pose_return[1][0, 0, :]

                    # --- Find the center of the marker and draw the ID
                    # X coordinate of marker's center
                    self.center_x = (self.corners[item][0][0][0] + self.corners[item][0][1][0] +
                                     self.corners[item][0][2][0] + self.corners[item][0][3][0]) / 4

                    # Y coordinate of marker's center
                    self.center_y = (self.corners[item][0][0][1] + self.corners[item][0][1][1] +
                                     self.corners[item][0][2][1] + self.corners[item][0][3][1]) / 4

                    self.x_coord = round(translation_vet[0], 4)
                    self.y_coord = round(translation_vet[1], 4)
                    self.z_coord = round(translation_vet[2], 4)
                    self.distance = round(math.sqrt(self.x_coord ** 2 + self.y_coord ** 2 + self.z_coord ** 2),
                                          3) - 0.15

                    if 9.0 > self.distance > 8.0:
                        rospy.logwarn(self.distance)
                        has_id = False

                    # --- Update the list of pose with the current values
                    if self.ids[item] in id_detect:
                        # --- Draw the information of tag on the screen
                        cv2.putText(frame, "id" + str(self.ids[item]), (int(self.center_x), int(self.center_y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 225, 250), 2)
                        cv2.circle(frame, (int(self.center_x), int(self.center_y)), 8, (0, 0, 255), -1)
                        aruco.drawDetectedMarkers(frame, self.corners)  # Draw a square around the markers
                        aruco.drawAxis(frame, self.camera_matrix,  # Draw the axis in the video frame
                                       self.camera_distortion,
                                       rotation_vet, translation_vet, 0.2)

                        id_found_array.append(self.ids[item])    # Found ids

            else:
                has_id = False
        else:
            has_id = False

        id_found_array = np.array(id_found_array)
        position_tag_x = self.width - self.center_x

        return has_id, id_found_array, position_tag_x

    def vehicle_detector(self, frame):
        """
        Detect the blue square of the AGV

        :param frame:   Video frame
        :return:        True or False
        """
        good = False
        # Image Filter
        filtered_frame = cv2.GaussianBlur(frame, self.kernel_size, 0)

        # Binary Mask
        hvs_image = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)

        # --- HSV format
        # Blue color
        low_blue = np.array([120, 80, 20])
        high_blue = np.array([126, 255, 255])
        blue_mask = cv2.inRange(hvs_image, low_blue, high_blue)

        # Get contours
        _, contours, hierarchy = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours
        black_image = np.zeros([blue_mask.shape[0], blue_mask.shape[1], 3], 'uint8')
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if 500 > area > 35:
                cv2.drawContours(filtered_frame, [c], -1, (150, 250, 150), 1)
                cx, cy = _get_contour_center(c)
                cv2.circle(filtered_frame, (cx, cy), (int)(radius), (0, 0, 255), 1)
                cv2.circle(black_image, (cx, cy), (int)(radius), (0, 0, 255), 1)
                cv2.circle(black_image, (cx, cy), 5, (150, 150, 255), -1)
                good = True

                res = cv2.bitwise_and(black_image,black_image, mask= blue_mask)
                cv2.imwrite("result_img.png", hvs_image)
                cv2.imwrite("result_img2.png", res)
                cv2.imwrite("result_img3.png", filtered_frame)

        return good


