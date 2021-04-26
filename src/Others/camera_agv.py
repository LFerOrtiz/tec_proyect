#!/usr/bin/env python 
import rospy
import numpy as np
import cv2
import tag_detection

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# initialize the known distance from the camera to the object, which
# in this case is 1.0 m
KNOWN_DISTANCE = 1.0
# initialize the known object width, which in this case is 10.8 cm
KNOWN_WIDTH = 0.108
focal = 0.0
num_frame = 0
estimate_dist = 0.0

#CvBridge
bridge = CvBridge()

def frame_callback(ros_frame):
    """ Convert the image format of ROS to OpenCV format """
    global bridge, num_frame, focal, KNOWN_WIDTH, KNOWN_DISTANCE
    try:
        cv_frame = bridge.imgmsg_to_cv2(ros_frame, desired_encoding = "bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        rospy.loginfo("Recibo un frame")
        # Calculated distance
        if num_frame == 0:
            num_frame += 1
            focal = tag_detection.focal_length(cv_frame, KNOWN_DISTANCE, KNOWN_WIDTH)

        tag_detection.distance_to_camera(cv_frame, KNOWN_DISTANCE, focal)
        # Resize the image 
        resize_frame = cv2.resize(cv_frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
        # Find the Tag in the world
        tag_detection.process_tag(resize_frame)
        cv2.waitKey(10)


def main():
    # Crea el nodo con nombre "tennis_ball_image"
    rospy.init_node('camera_one_agv_image', anonymous = True)

    # Topico en el cual se recibira cada frame del video
    topic_sub_one = '/agv2/camera/image_raw'
    subscriber_one = rospy.Subscriber(topic_sub_one, Image, frame_callback)
    rospy.spin()

    # Cierra todas la ventanas que abrio openCV
    cv2.destroyAllWindows()

# Cuando se pone como interprete, este modulo funciona como funcion principal
if __name__ == '__main__':
    main()