#!/usr/bin/env python 
""" Generate Aruco Markers """
import numpy as np
import cv2
import cv2.aruco as aruco

# Select type of aruco marker (size)
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

ID = input("ID of the Marker (0-999): ")
img_size = input("Size of the Marker (px): ")
name_img = raw_input("Name for the img file: ")

# Create an image from the marker
# second param is ID number
# last param is total image size
img = aruco.drawMarker(aruco_dict, ID, img_size)
cv2.imwrite((name_img+".png"), img)

# Display the image to us
cv2.imshow('frame', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()