"""
Python script to generate ArUco model for Gazebo,
to run this file use the console and please enter the follow command
> python ArucoMesh.py --num X -- dict Y

for help:
> python ArucoMesh.py -h

X: number of tags you want
Y: select the Aruco dictionary you want
"""

import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
import os

tag_dictionary = {
    "0": "DICT_4X4_50",
    "1": "DICT_4X4_100",
    "2": "DICT_4X4_250",
    "3": "DICT_4X4_1000",
    "4": "DICT_5X5_50",
    "5": "DICT_5X5_100",
    "6": "DICT_5X5_250",
    "7": "DICT_5X5_1000",
    "8": "DICT_6X6_50",
    "9": "DICT_6X6_100",
    "10": "DICT_6X6_250",
    "11": "DICT_6X6_1000""",
    "12": "DICT_7X7_50""",
    "13": "DICT_7X7_100",
    "14": "DICT_7X7_250",
    "15": "DICT_7X7_1000",
    "16": "DICT_ARUCO_ORIGINAL",
    "17": "DICT_APRILTAG_16H5",
    "18": "DICT_APRILTAG_25h9",
    "19": "DICT_APRILTAG_36h10",
    "20": "DICT_APRILTAG_36h11",
}


def make_dir(whatever):
    try:
        os.makedirs(whatever)
    except OSError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Aruco tags')
    parser.add_argument('--num', "--NUM_TAGS", type=int, required=False, default=2,
                        help='Integer for the amount of tags to generate. Default [2].')
    parser.add_argument('--dict', "--DICT_TAGS", type=int, required=False, default=7,
                        help='Dictionary of ArUco Tag of OpenCV. Default ["7": '
                             '"DICT_5X5_1000"]. Options: ' + str(tag_dictionary))

    args = parser.parse_args()
    tag_num = args.num
    tag_dict = args.dict

    # Make folder for models
    make_dir("models")

    # Get parameters for ArUco library
    aruco_dict = aruco.Dictionary_get(tag_dict)

    for i in range(0, tag_num):
        # Make folders for tags models
        make_dir("models/tag_" + str(i) + "/materials/textures")
        make_dir("models/tag_" + str(i) + "/materials/scripts")

        # *********** Generate tag image ***********
        # Last parameter is total image size
        img = aruco.drawMarker(aruco_dict, i, 900)
        image = np.zeros((900, 900, 1), np.uint8)
        image[:] = 0
        image = cv2.addWeighted(image, 0.9, img, 0.9, 0.0)

        # Border parameter
        top = 50
        bottom = 50
        left = 50
        right = 50
        borderType = cv2.BORDER_CONSTANT
        value = [255, 255, 255]
        dst = cv2.copyMakeBorder(image, top, bottom, left, right, borderType, None, value)

        cv2.imwrite("models/tag_" + str(i) +
                    "/materials/textures/aruco_marker_" + str(i) + ".png", dst)

        # *********** Generate tag material script ***********
        file = open("models/tag_" + str(i) + "/materials/scripts/tag.material", 'w')

        file.write("\n material aruco_tag_" + str(i) + "\n \
            {\n \
                technique\n \
                {\n \
                pass\n \
                {\n \
                    texture_unit\n \
                    {\n \
                    // Relative to the location of the material script\n \
                    texture ../textures/aruco_marker_" + str(i) + ".png\n \
                    // Repeat the texture over the surface (4 per face)\n \
                    scale 1 1\n \
                    }\n \
                }\n \
                }\n \
            }\n")
        file.close()

        # *********** Generate tag config ***********
        with open("models/tag_" + str(i) + "/model.config", 'w') as file:
            file.write("\n \
                <?xml version=\"1.0\"?>\n \
                \n \
                <model>\n \
                    <name>Aruco tag" + str(i) + "</name>\n \
                    <version>1.0</version>\n \
                    <sdf version=\"1.6\">model.sdf</sdf>\n \
                \n \
                    <author>\n \
                        <name>Luis Fernando</name>\n \
                        <email>A01769971@itesm.mx</email>\n \
                    </author>\n \
                \n \
                    <description>\n \
                        Aruco tag " + str(i) + "\n \
                    </description>\n \
                \n \
                </model>\n")

        # *********** Generate tag SDF model ***********
        with open("models/tag_" + str(i) + "/model.sdf", 'w') as file:
            file.write("\n \
                <?xml version=\"1.0\"?>\n \
                <sdf version=\"1.6\">\n \
                    <model name=\"Aruco tag" + str(i) + "\">\n \
                        <static>true</static>\n \
                        <link name=\"robot_link\">\n \
                            <visual name=\"body_visual\">\n \
                                <geometry>\n \
                                    <box>\n \
                                        <size>0.3 0.3 0.02</size> <!-- Dimensions 30x30x2cm -->\n \
                                    </box>\n \
                                </geometry>\n \
                                <material> <!-- Body material -->\n \
                                    <script>\n \
                                        <uri>model://tag_" + str(i) + "/materials"
                                                                      "/scripts/tag.material</uri>\n \
                                        <name>aruco_tag_" + str(i) + "</name>\n \
                                    </script>\n \
                                </material> <!-- End Body Material -->\n \
                            </visual>\n \
                        </link>\n \
                    </model>\n \
                </sdf>\n \
                \n")