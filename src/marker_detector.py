import cv2
import math
import numpy as np
from cv2 import aruco

FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font for displaying text
FONT_SIZE = 0.7


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
        self.filter = 0
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
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)  # Use 5x5 dictionary for find the markers
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
        The result is the same as MATLAB except the order
        of the euler angles ( x and z are swapped ). """
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

    def detection(self, frame, id_detect, show_frame, debug):
        """ Detect the marker indicate in the video frame
            X: Red Axis (Roll),
            Y: Green Axis (Pitch),
            Z: Blue Axis (Yaw)
        """
        # Get the height, width and color channels of the frame
        self.height, self.width, self.channels = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the image to gray scale
        gray = cv2.GaussianBlur(gray, self.kernel_size, 0)  # Applied a Gaussian Filter for High frequency noise
        self.corners, self.ids, rejected_img_points = aruco.detectMarkers(gray, self.marker_dict,
                                                                          parameters=self.parameters)

        if np.all(self.ids is not None) and self.ids[0] == id_detect:
            for item in range(0, len(self.ids)):
                # rotationVect: Pose of the marker respect to camera frame
                # translationVect: Position of the marker in camera frames
                pose_return = aruco.estimatePoseSingleMarkers(self.corners[item], self.marker_size,
                                                              cameraMatrix=self.camera_matrix,
                                                              distCoeffs=self.camera_distortion)
                rotation_vet, translation_vet = pose_return[0][0, 0, :], pose_return[1][0, 0, :]

                aruco.drawDetectedMarkers(frame, self.corners)  # Draw a square around the markers
                aruco.drawAxis(frame, self.camera_matrix,  # Draw the axis in the video frame
                               self.camera_distortion,
                               rotation_vet, translation_vet, 0.2)

                # --- Find the center of the marker and draw the ID
                self.center_x = (self.corners[item][0][0][0] + self.corners[item][0][1][0] + self.corners[item][0][2][0] +
                            self.corners[item][0][3][
                                0]) / 4  # X coordinate of marker's center
                self.center_y = (self.corners[item][0][0][1] + self.corners[item][0][1][1] + self.corners[item][0][2][1] +
                            self.corners[item][0][3][
                                1]) / 4  # Y coordinate of marker's center
                cv2.putText(frame, "id" + str(self.ids[item]), (int(self.center_x), int(self.center_y)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (50, 225, 250), 2)

                # --- Rotation matrix tag -> camera
                r_ct = np.matrix(cv2.Rodrigues(rotation_vet)[0])
                r_tc = r_ct.T  # Transpose of the matrix

                # --- Get the attitude in terms of euler 321 (Flip first)
                self.roll, self.pitch, self.yaw = self._rotation_matrix_to_euler_angles(self.r_flip * r_tc)

                # -- Calculated estimated distance
                self.x_coord = translation_vet[0]
                self.y_coord = translation_vet[1]
                self.z_coord = translation_vet[2]
                self.distance = math.sqrt(self.x_coord ** 2 + self.y_coord ** 2 + self.z_coord ** 2)

                # --- Show general info of distance and angles of the marker
                if debug:
                    str_angle = "Angle: %3.4f" % math.degrees(self.pitch)
                    cv2.putText(frame, str_angle, (0, 400), FONT, FONT_SIZE, (255, 0, 0), 2, cv2.LINE_AA)
                    str_distance = "Distance: %3.4f" % self.distance
                    cv2.putText(frame, str_distance, (0, 420), FONT, FONT_SIZE, (255, 0, 0), 2, cv2.LINE_AA)

                    # --- X: Red Axis (Roll), Y: Green Axis (Pitch), Z: Blue Axis (Yaw)
                    str_position = "Marker position X=%3.2f  Y=%3.2f  Z=%3.2f" % (self.x_coord, self.y_coord, self.z_coord)
                    cv2.putText(frame, str_position, (0, 440), FONT, FONT_SIZE, (255, 0, 0), 2, cv2.LINE_AA)
                    str_attitude = "Marker Attitude R=%3.2f  P=%3.2f  Y=%3.2f" % (
                        math.degrees(self.roll), math.degrees(self.pitch), math.degrees(self.yaw))
                    cv2.putText(frame, str_attitude, (0, 460), FONT, FONT_SIZE, (255, 0, 0), 2, cv2.LINE_AA)

                # --- Position and attitude of the camera respect to the marker
                # camera_pose = -R_ct * np.matrix(translation_vet).T
                # roll_camera, pitch_camera, yaw_camera = rotation_matrix_to_euler_angles(R_flip * R_ct)
                # str_position = "Camera position X=%3.2f  Y=%3.2f  Z=%3.2f"%(camera_pose[0], camera_pose[1], camera_pose[2])
                # cv2.putText(cv_frame, str_position, (0, 460), FONT, 0.5, (0, 25, 255), 2, cv2.LINE_AA)
                # str_attitude = "Camera Attitude R=%3.2f  P=%3.2f  Y=%3.2f"%(math.degrees(roll_camera), math.degrees(pitch_camera), math.degrees(yaw_camera))
                # cv2.putText(cv_frame, str_attitude, (0, 475), FONT, 0.5, (0, 25, 255), 2, cv2.LINE_AA)

        if show_frame:
            window_resize = cv2.resize(frame, (480, 360))
            cv2.imshow("Marker ID " + str(id_detect), window_resize)
            cv2.waitKey(10)

        return frame
