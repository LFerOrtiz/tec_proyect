""" Find and track the ARTag """
import numpy as np
import cv2

dimension = 200
p1 = np.array([
    [0, 0],
    [dimension - 1, 0],
    [dimension - 1, dimension - 1],
    [0, dimension - 1]], dtype="float32")

def _find_contours(image):
	# convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 140, 250, 7)
    cv2.imshow("canny", edged)

	# find the contours in the edged image and keep the largest one;
    _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for contours how don't have a parent contour (-1)
    index = []
    contour_list = []
    temp_contours = []
    final_contour_list = []
    area_min = None
    for item in hierarchy[0]:
        if item[3] != -1:
            index.append(item[3])

    for item in index:
        # Get the perimeter of each closed contour
        perimeter = cv2.arcLength(contours[item], True)
        approx = cv2.approxPolyDP(contours[item], 0.02 * perimeter, True)

        # Filter for each contour have more of 4 edges
        if len(approx) > 4:
            new_perimeter = cv2.arcLength(contours[item - 1], True)
            corners = cv2.approxPolyDP(contours[item - 1], 0.02 * new_perimeter, True)
            contour_list.append(corners)

    for contour in contour_list:
        if len(contour) == 4:
            temp_contours.append(contour)

    # Filter for the contours how have a specific area
    for element in temp_contours:
        if 10000 > cv2.contourArea(element) > 1:
            final_contour_list.append(element)
            area_min = cv2.minAreaRect(element)

    # Show four points in every edge of the square
    # for item in final_contour_list:
    #     for puntos in item:
    #         for punto in puntos:
    #             x, y = punto
    #             cv2.circle(gray,(x,y), 3, (0,0,255), -1)
    # cv2.imshow('Puntos', gray)
    # c = max(contours, key = cv2.contourArea)
    print(area_min)
    return final_contour_list, area_min

def _order(pts):
    """ Function to return the order of points in camera frame """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # print("Rect points: {} {} {} {} ").format(rect[0], rect[2], rect[1], rect[3])

    return rect

def _homography(p, p1):
    """ Function to compute homography between world and camera frame """
    A = []
    p2 = _order(p)

    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1, :] / Vh[-1, -1]
    h = np.reshape(l, (3, 3))
    return h

def process_tag(frame):
    """ Main process for tag """
    # Store the parameters of the video frame
    frame_rows, frame_cols, frame_channels = frame.shape
    # load the frame, find the marker in the image
    markers_list, _ = _find_contours(frame)
    for contour in range(len(markers_list)):
        cv2.drawContours(frame, [markers_list[contour]], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", frame)

        # print("Contours list: {}").format(markers_list)
        # [ first_row:last_row , column_0 ]
        contour_rez = markers_list[contour][:, 0]
        homography_matrix = _homography(p1, _order(contour_rez))

        # Execute a perspective transformation using the homography matrix
        tag = cv2.warpPerspective(frame, homography_matrix, (200, 200))
        cv2.imshow("Tag after Homo", tag)

def focal_length(image, knowDist, knownWidth):
    """ Get the focal length using a know distance and width """
    _, marker = _find_contours(image)
    focalLength = (marker[1][0] * knowDist) / knownWidth
    return focalLength

def distance_to_camera(image, knownWidth, focalLength):
    """ Compute and return the distance from the marker to the camera """
    _, marker = _find_contours(image)
    dist = (knownWidth * focalLength/ marker[1][0])

	# draw a bounding box around the image and display it
    cv2.putText(image, "%.2fft" % (dist / 1.0),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
    #cv2.imshow("image", image)
    