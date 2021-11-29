#!/usr/bin/env python
# coding=utf-8
import math
import numpy as np
import threading
import actionlib
import rospy
import smach_ros
from pydub.playback import play
from pydub import AudioSegment

from gazebo_msgs.srv import DeleteModel, SpawnModel

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

from smach import State, StateMachine

# -- Libraries created
from multi_marker_detector import MarkerDetector

# -- Messages create
from tec_proyect.msg import V2VCommunication

# --- Variables to send to the station selected
station_id = 0
map_to_use = None
is_there_a_goal = False
action_on_station = ""

# -- Constants for configuration of Aruco library
KERNEL_SIZE = (3, 3)  # Kernel size for Gaussian Blur
MARKER_LENGTH = 0.3  # Marker size in cm
SELF_IDS = np.array([0, 1, 2], dtype=np.int)  # ID of current agv
V2V_DIST = 0.8
roll = 0.0
pitch = 0.0
yaw = 0.0

# --- Constants for platoon
VEHICLE_NAME = "agv1"
list_agv_platoon = [VEHICLE_NAME]  # list of agv in the platoon
list_agv_index = 0  # index of the last agv in the platoon
audio_union = '/home/fer/Música/doorbell.wav'
audio_newgoal = '/home/fer/Música/DoorBell_Ding.wav'

agv_info = {
    "agv1": {'ID': np.array([0, 1, 2], dtype=np.int)},
    "agv2": {'ID': np.array([3, 4, 5], dtype=np.int)},
    "agv3": {'ID': np.array([6, 7, 8], dtype=np.int)},
    "agv4": {'ID': np.array([9, 10, 11], dtype=np.int)},
}

disunion_points = {"almacen": [-8.79, -7.3, 3.18],
                   "laboratorio": [1.013, -11.95, -1.52],
                   "hospital": [-7.785, 4.481, 1.557]}

union_points = {"almacen": [-8.79, -7.3, 3.18],
                "laboratorio": [1.013, -11.95, -1.52],
                "hospital": [-8.6257, 4.481, 1.557]}

# --- Counter to know the number of agv in the platoon
num_of_vehicles = 1
initial_number_of_agv = 0
current_station = 0

maps = {
    "probe_circuit": {
        1: [(10.4798, 1.2655, 0.0), (0.0, 0.0, -0.5071, 0.8618)],
        2: [(12.3344, -4.9908, 0.0), (0.0, 0.0, 0.8992, -0.4374)],
        3: [(8.1682, -7.1354, 0.0), (0.0, 0.0, 0.9873, -0.1588)],
        4: [(4.5945, -7.8362, 0.0), (0.0, 0.0, 0.9686, 0.2483)],
        5: [(2.8779, -7.8059, 0.0), (0.0, 0.0, 0.9894, -0.1445)],
        6: [(-7.9114, -12.2948, 0.0), (0.0, 0.0, 0.9781, 0.2077)],
        7: [(-8.6148, -8.0312, 0.0), (0.0, 0.0, 0.3266, 0.9451)],
        8: [(2.8032, -4.8726, 0.0), (0.0, 0.0, 0.3993, 0.9167)],
        9: [(3.0327, 0.5831, 0.0), (0.0, 0.0, 0.8966, 0.4428)],
        10: [(-6.5344, 0.9898, 0.0), (0.0, 0.0, 0.99920, -0.0399)],
        11: [(-12.1028, -1.3840, 0.0), (0.0, 0.0, -0.7824, 0.6226)],
        12: [(-9.2468, -6.2203, 0.0), (0.0, 0.0, -0.0406, 0.9991)],
        13: [(2.4854, -3.2960, 0.0), (0.0, 0.0, 0.1338, 0.9910)],
        14: [(8.713, -2.1600, 0.0), (0.0, 0.0, 0.1250, 0.9921)],
        15: [(11.535, -0.3220, 0.0), (0.0, 0.0, 0.7200, 0.6938)],
        16: [(9.0998, 1.9646, 0.0), (0.0, 0.0, 0.9997, 0.02082)],
        17: [(-6.2884, 5.1194, 0.0), (0.0, 0.0, 0.9916, -0.1293)],
        18: [(-11.8708, 3.4869, 0.0), (0.0, 0.0, -0.7809, 0.6245)],
        19: [(-10.2596, 1.7278, 0.0), (0.0, 0.0, -0.0119, 0.9999)],
        20: [(-0.0700, 3.5394, 0.0), (0.0, 0.0, 0.1210, 0.9926)],
        21: [(10.4798, 1.2655, 0.0), (0.0, 0.0, -0.5071, 0.8618)],
    },

    "almacen": {
        100: [(3.3362, -0.2601, 0.0), (0.0, 0.0, 0.9999, 0.0022)],
        101: [(-3.036, -0.320, 0.0), (0.0, 0.0, 0.9999, 0.0092)],
        102: [(7.0193, -15.4057, 0.0), (0.0, 0.0, 0.0090, 0.9999)],
        103: [(-8.0891, -11.0379, 0.0), (0.0, 0.0, -0.7177, 0.6962)],
        200: [(-7.8659, -3.9052, 0.0), (0.0, 0.0, -0.6924, 0.72150)],
    },

    "laboratorio": {
        100: [(-4.4708, -4.7465, 0.0), (0.0, 0.0, -0.7042, 0.7099)],
        101: [(-5.3536, 5.0826, 0.0), (0.0, 0.0, 0.9999, 0.0003)],
        102: [(2.307, 2.018, 0.0), (0.0, 0.0, 0.9999, 0.094)],
        103: [(6.1594, 5.8784, 0.0), (0.0, 0.0, 0.7040, 0.7101)],
        1031: [(29.213, -4.512, 0.0), (0.000, 0.000, 0.229, 0.973)],
        1032: [(29.437, -2.777, 0.0), (0.000, 0.000, 0.973, 0.232)],
        200: [(-2.54037, -11.1666, 0.0), (0.0, 0.0, 0.00807, 0.99996)],
    },

    "hospital": {
        100: [(-9.183, -4.815, 0.0), (0.0, 0.0, 0.0453, 0.999)],
        101: [(-0.8623, -4.4386, 0.0), (0.0, 0.0, 0.00348, 0.999)],
        102: [(7.155, -4.398, 0.0), (0.0, 0.0, 0.058, 0.999)],
        103: [(16.3394, -4.3872, 0.0), (0.0, 0.0, 0.0544, 0.9985)],
        104: [(25.2922, -4.3232, 0.0), (0.0, 0.0, 0.0256, 0.9996)],
        105: [(29.0745, -2.5237, 0.0), (0.0, 0.0, 0.9984, -0.0560)],
        1051: [(29.213, -4.512, 0.0), (0.000, 0.000, 0.229, 0.973)],
        1052: [(29.437, -2.777, 0.0), (0.000, 0.000, 0.973, 0.232)],
        106: [(19.247, -2.6171, 0.0), (0.0, 0.0, 0.9990, -0.04396)],
        107: [(10.3157, -2.6915, 0.0), (0.0, 0.0, 0.9985, -0.0535)],
        108: [(3.0753, -2.7118, 0.0), (0.0, 0.0, 0.9985, -0.0535)],
        200: [(-7.3786, 3.7734, 0.0), (0.0, 0.0, 0.9973, -0.0726)],
    }
}

camera_info = "/home/fer/catkin_ws/src/tec_proyect/info/"
camera_matrix = np.loadtxt(camera_info + 'cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(camera_info + 'cameraDistortion.txt', delimiter=',')
marker_detection = MarkerDetector(MARKER_LENGTH, KERNEL_SIZE, camera_matrix, camera_distortion)


class InitParamsForPlatoon(State):
    def __init__(self):
        State.__init__(self, outcomes=['getting_new_params', 'waiting'])
        rospy.loginfo("Initialization of AGV Leader and Platoon...........")
        self._init_config = rospy.Subscriber("/params_for_agv_leader", V2VCommunication, self._status_cb)
        self._current_status = ""
        self._trigger_event = threading.Event()
        rospy.sleep(2)

    def execute(self, ud):
        global initial_number_of_agv, num_of_vehicles, list_agv_platoon
        rospy.loginfo("Please indicate agv initial number..................")
        init_params_pub = rospy.Publisher("/params_for_platoon", V2VCommunication, queue_size=1)
        get_init_params = V2VCommunication()

        # --- Waiting for the initial number of AGV
        get_init_params.status = "WAITING_INIT_NUM_AGV"
        init_params_pub.publish(get_init_params)

        # Clear the flag of the trigger event
        self._trigger_event.clear()

        # Wait until the flag of the trigger event is Set
        self._trigger_event.wait()

        if self._current_status == "GIVING_INIT_NUM_AGV" and initial_number_of_agv > 0:
            rospy.loginfo("Initial Number of AGV: %s", initial_number_of_agv)
            get_init_params.status = "ALL_OK"
            init_params_pub.publish(get_init_params)
            rospy.sleep(1)
            return 'getting_new_params'
        else:
            return 'waiting'

    def _status_cb(self, msg):
        global initial_number_of_agv, num_of_vehicles, list_agv_platoon
        self._current_status = msg.status
        initial_number_of_agv = msg.num_of_initial_agv
        self._trigger_event.set()

    def request_preempt(self):
        State.request_preempt(self)
        self._trigger_event.set()


class InitConfigAGV(State):
    def __init__(self, timeout=15):
        """
        :param timeout: maximum waiting time for get information from the topic
        """
        State.__init__(self, outcomes=['finishing_setup', 'waiting'])
        self._timeout = timeout
        self._config_status = None
        self._name_vehicle = None
        self._timeout_timer = None
        self._trigger_event = threading.Event()
        self._sound = AudioSegment.from_file(audio_union)
        rospy.loginfo("Connecting to the others AGV...........")

    def execute(self, userdata):
        global initial_number_of_agv, num_of_vehicles, list_agv_platoon, list_agv_index

        rospy.loginfo("SetUp a new AGV .............")
        rospy.logwarn("List AGV: %s", list_agv_platoon)
        rospy.logwarn("Init AGV: %s", initial_number_of_agv)

        # Create a publisher for send the ids
        vehicle_status_sub = rospy.Subscriber('/vehicles_status', V2VCommunication, self._vehicle_status_cb,
                                              queue_size=1)
        config_id_pub = rospy.Publisher('/id_assigment', V2VCommunication, queue_size=2)
        config_id = V2VCommunication()
        rospy.sleep(0.5)

        if initial_number_of_agv > 1:
            # Get the IDs to configure the new AGV
            for key, value in agv_info.iteritems():
                if key == list_agv_platoon[-1]:
                    vehicle_name = key
                    config_id.vehicle_name = vehicle_name
                    config_id.id0 = agv_info[vehicle_name]["ID"][0]
                    config_id.id1 = agv_info[vehicle_name]["ID"][1]
                    config_id.id2 = agv_info[vehicle_name]["ID"][2]

                    for i in range(0, 2):
                        config_id_pub.publish(config_id)
                        rospy.sleep(1)

            # Clear send a empty message
            config_id = V2VCommunication()
            config_id_pub.publish(config_id)

            # Clear the flag of the trigger event
            self._trigger_event.clear()

            # Set the flag of the trigger event after the time expired
            if self._timeout:
                self._timeout_timer = rospy.Timer(rospy.Duration(self._timeout),
                                                  lambda _: self._trigger_event.set(),
                                                  oneshot=True)

            # Wait until the flag of the trigger event is Set
            self._trigger_event.wait()

            # shutdown timer
            if self._timeout:
                self._timeout_timer.shutdown()

        if self._config_status == "NEW_AGV" and initial_number_of_agv > 1:
            # Add a new vehicle to the platoon
            list_agv_platoon.append(self._name_vehicle)
            num_of_vehicles = len(list_agv_platoon)
            list_agv_index += 1
            initial_number_of_agv = len(list_agv_platoon) + 1
            config_id.vehicle_name = self._name_vehicle
            config_id.status = "coupling"
            rospy.logwarn("Action %s %s", self._name_vehicle, config_id.status)
            config_id_pub.publish(config_id)
            rospy.sleep(2)

            # Clean data
            for i in range(0, 2):
                config_id_pub.publish()
                rospy.sleep(1)

            # config_id_pub.unregister()
            # vehicle_status_sub.unregister()
            self.clean()
            # play a audio file to alert
            play(self._sound)
            rospy.logerr("Finish SetUp ...........")
            return 'finishing_setup'
        elif initial_number_of_agv == 1:
            self.clean()
            rospy.logerr("Finish SetUp ...........")
            return 'finishing_setup'
        else:
            for i in range(0, 2):
                config_id_pub.publish()
            rospy.sleep(1)
            return 'waiting'

    def _vehicle_status_cb(self, msg_data):
        # If the one of the agv detect the IDs of the last agv of the platoon
        if list_agv_platoon[-1] == msg_data.vehicle_detect and msg_data.status == "ID_DETECTED":
            self._config_status = "NEW_AGV"
            self._name_vehicle = msg_data.vehicle_name
            rospy.logwarn("Index: %s", list_agv_index)
            rospy.logwarn("Checking Status... %s", self._config_status)
            rospy.logwarn("Vehicle name... %s", self._name_vehicle)
            rospy.logwarn("Front Vehicle name... %s", msg_data.vehicle_detect)
            rospy.sleep(1)
            self._trigger_event.set()

    # def request_preempt(self):
    #     State.request_preempt(self)
    #     self._trigger_event.set()

    def clean(self):
        self._config_status = None
        self._name_vehicle = None
        self._timeout_timer = None


class CountNumAGV(State):
    def __init__(self):
        State.__init__(self, outcomes={'succeed', 'waiting'})

    def execute(self, ud):
        global initial_number_of_agv, num_of_vehicles, list_agv_platoon
        rospy.logerr("Number of vehicles in the platoon: %s", num_of_vehicles)
        rospy.logerr("List of vehicles in the platoon: %s", list_agv_platoon)
        rospy.logerr("Init num of vehicles in the platoon: %s", initial_number_of_agv)
        if num_of_vehicles >= initial_number_of_agv:
            return 'succeed'
        else:
            return 'waiting'


class StationGoal(State):
    def __init__(self):
        """
        Get a new goal to travel from the user
        """
        State.__init__(self, outcomes=["new goal", "failed", "waiting"],
                       output_keys=['new_goal_out', 'id_station_out', 'veh_discharge_out'])

        # Parameters for the goal
        self.id_marker = None
        self.map_to_use = None
        self.is_there_a_goal = False
        self._action_on_station = ""
        self.new_goal = None
        self._status = ""

        # Subscribe to get the new goal
        self.goal_sub = rospy.Subscriber("/params_for_agv_leader", V2VCommunication, self._goal_cb, queue_size=2)
        self._status_goal_pub = rospy.Publisher("/params_for_platoon", V2VCommunication, queue_size=1)
        self._status_goal = V2VCommunication()
        rospy.loginfo("Getting a new goal........")

    def _goal_cb(self, goal_data):
        self.id_marker = goal_data.id_marker
        self.map_to_use = goal_data.map_to_use
        self._status = goal_data.status
        type_station = math.floor(self.id_marker / 100.0)
        # rospy.loginfo("Status: %s", self._status)
        if 1 == type_station or 2 == type_station:
            self.is_there_a_goal = True
            if type_station == 2:
                self._action_on_station = goal_data.coupling_or_decoupling
                rospy.loginfo("Action: %s", self._action_on_station)

    def _clean_data(self):
        self.id_marker = None
        self.veh_discharge = None
        self.is_there_a_goal = False
        self.action_on_station = ""

    def execute(self, userdata):
        global action_on_station, map_to_use

        self._status_goal.status = "WAITING_NEW_GOAL"
        self._status_goal_pub.publish(self._status_goal)

        rospy.sleep(1)
        rospy.loginfo_once("Executing.....")
        if self.is_there_a_goal and self._status == "SENDING_NEW_GOAL":
            if self.map_to_use in maps:
                map_used_out = maps.get(self.map_to_use)
                map_to_use = self.map_to_use
                rospy.loginfo("what map going to use?: %s", self.map_to_use)
                if self.id_marker in map_used_out:
                    if self.id_marker != current_station:
                        userdata.new_goal_out = map_used_out.get(self.id_marker)
                        userdata.id_station_out = self.id_marker
                        action_on_station = self._action_on_station
                        for i in range(0, 5):
                            self._status_goal.status = "ALL_OK"
                            self._status_goal_pub.publish(self._status_goal)
                            rospy.sleep(1)
                        rospy.loginfo("ID of the station: %s", self.id_marker)
                        rospy.loginfo("To a new goal")
                        rospy.sleep(1)
                        self._clean_data()
                        return "new goal"
                    else:
                        rospy.loginfo("You are in the current station")
                        self._status_goal.status = "WAITING_NEW_GOAL"
                        if map_to_use is None:
                            self._status_goal.map_to_use = "THERE_IS_A_MAP"
                        self._status_goal_pub.publish(self._status_goal)
                        self._clean_data()
                        rospy.sleep(1)
                        return "waiting"
                else:
                    rospy.loginfo("Dont exist the id")
                    return "failed"

            else:
                rospy.loginfo("Don't exist the map")
                return "failed"

        else:
            self._status_goal.status = "WAITING_NEW_GOAL"
            self._status_goal_pub.publish(self._status_goal)
            rospy.loginfo_once("There is not a goal")
            return "waiting"


class NavToGoal(State):
    def __init__(self):
        """
        Send a new goal to travel
        """
        State.__init__(self, outcomes=["goal reached", "aborted", 'timeout'], input_keys=['nav_goal_in'])

        # Create an action client with action definition file "MoveBaseAction"
        self.move_base = actionlib.SimpleActionClient('/agv1/move_base', MoveBaseAction)

        # Waits until the action server has started up and started listening for goals.
        self.move_base.wait_for_server()

        rospy.loginfo("Connecting to the move_base server")

        # Creates a new goal with the MoveBaseGoal constructor
        self.goal = MoveBaseGoal()
        # Use the "map" topic as global coordinate frame
        self.goal.target_pose.header.frame_id = 'map'
        self.cnt = 0
        self._sound = AudioSegment.from_file(audio_newgoal)

    def execute(self, userdata):
        rospy.sleep(1)
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'

        elif userdata.nav_goal_in is None:
            return "aborted"

        else:
            # Move along the "map" coordinate frame
            self.goal.target_pose.pose.position.x = userdata.nav_goal_in[0][0]
            self.goal.target_pose.pose.position.y = userdata.nav_goal_in[0][1]
            self.goal.target_pose.pose.position.z = userdata.nav_goal_in[0][2]
            self.goal.target_pose.pose.orientation.x = userdata.nav_goal_in[1][0]
            self.goal.target_pose.pose.orientation.y = userdata.nav_goal_in[1][1]
            self.goal.target_pose.pose.orientation.z = userdata.nav_goal_in[1][2]
            self.goal.target_pose.pose.orientation.w = userdata.nav_goal_in[1][3]

            # Sends the goal to the action server.
            self.move_base.send_goal(self.goal)

            # Waits 90s to finish performing the action.
            wait = self.move_base.wait_for_result(rospy.Duration(90))

            if not wait:
                self.move_base.cancel_goal()
                rospy.loginfo("Timed out achieving goal")
                self.cnt += 1
                return 'timeout'

            elif self.cnt > 3:
                rospy.loginfo("Num of Tries: %s", self.cnt)
                return 'aborted'

            else:
                state = self.move_base.get_state()
                rospy.loginfo("Goal Succeeded!")
                rospy.sleep(1)
                self.cnt = 0
                play(self._sound)
                return 'goal reached'


class StationDetection(State):
    def __init__(self, topic, msg_type, timeout=5, max_checks=1):
        """
        State for detect the marker in the station.

        :param topic:       topic of the video camera
        :param msg_type:    message type for the topic
        :param timeout:     maximum time for the detection
        :param max_checks:  number of pass to declare the detection as good
        """
        State.__init__(self, outcomes=['load_unload', 'union_disunion', 'failed', 'preempted'], input_keys=['id_in'],
                       output_keys=['tag_position_out'])

        self._topic = topic
        self._msg_type = msg_type
        self._received_msg_cnt = max_checks
        self._timeout = timeout

        self._id_detected = None  # ID detected
        self._id = None  # ID to detect
        self._has_id = False  # One ID detect
        self._sub = None  # Subscriber
        self._timeout_timer = None
        self._position_tag_x = 0.0

        self._trigger_event = threading.Event()

        self.bridge = CvBridge()
        rospy.sleep(1)
        rospy.loginfo("Waiting for images...")

    def _clean_data(self):
        self._id_detected = None  # ID detected
        self._id = None  # ID to detect
        self._has_id = False  # One ID detect
        self._sub = None  # Subscriber
        self._timeout_timer = None
        self._position_tag_x = 0.0

    def execute(self, userdata):
        global current_station
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'

        self._received_msg_cnt = 0
        # Clear the flag of the trigger event
        self._trigger_event.clear()

        self._id = userdata.id_in  # ID for detect
        self._sub = rospy.Subscriber(self._topic, self._msg_type, self.execute_cb)

        # Set the flag of the trigger event after the time expired
        if self._timeout:
            self._timeout_timer = rospy.Timer(rospy.Duration(self._timeout),
                                              lambda _: self._trigger_event.set(),
                                              oneshot=True)

        # Wait until the flag of the trigger event is Set
        self._trigger_event.wait()
        # Unsubscribe to the topic
        self._sub.unregister()

        if self._timeout:
            self._timeout_timer.shutdown()

        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        elif self._has_id:
            userdata.tag_position_out = self._position_tag_x
            rospy.loginfo(self._id_detected[0])
            type_station = math.floor(self._id_detected[0] / 100.0)
            current_station = type_station
            if type_station == 1:
                rospy.loginfo("Load or discharge Station...")
                self._clean_data()
                return 'load_unload'
            elif type_station == 2:
                self._clean_data()
                rospy.logerr("union or disunion Station...")
                return 'union_disunion'
        elif not self._has_id or self._received_msg_cnt != self._received_msg_cnt or self._id_detected is None:
            return 'failed'

    def execute_cb(self, img):
        """
        Callback function for the subscriber

        :param img: video frame
        :return: None
        """
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        try:
            frame = self.bridge.imgmsg_to_cv2(img, "bgr8")

        except CvBridgeError, e:
            print e
        else:
            self._has_id, self._id_detected, self._position_tag_x = marker_detection.single_detector(frame, np.array(
                [self._id]))
            rospy.sleep(2)

            self._received_msg_cnt += 1
            if self._received_msg_cnt == self._received_msg_cnt:
                self._trigger_event.set()

    def request_preempt(self):
        State.request_preempt(self)
        self._trigger_event.set()


class LoadUnloadStation(State):
    def __init__(self, timeout=5):
        """
        Wait a amount of time, simulated the load or unload of items

        :param timeout: waiting time
        """
        State.__init__(self, outcomes=['succeed'])
        self._timeout = timeout
        self._timeout_timer = None
        self._trigger_event = threading.Event()

    def execute(self, ud):
        # Clear the flag of the trigger event
        self._trigger_event.clear()

        # Set the flag of the trigger event after the time expired
        if self._timeout:
            self._timeout_timer = rospy.Timer(rospy.Duration(self._timeout),
                                              lambda _: self._trigger_event.set(),
                                              oneshot=True)
        self._trigger_event.wait()

        if self._timeout:
            self._timeout_timer.shutdown()

        return 'succeed'


class VehicleRelocation(State):
    def __init__(self, topic, msg_type, timeout=10):
        """
        Relocated the AGV in front to the station

        :param topic: topic of the odometry topic
        :param msg_type: type of topical message
        :param timeout: maximum waiting time for get information from the topic
        """
        State.__init__(self, outcomes=['succeed', 'failed'], input_keys=['tag_position_in'])
        self._topic = topic
        self._msg_type = msg_type
        self._timeout = timeout
        self._timeout_timer = None
        self._sub = None  # Subscriber

        # Create an action client with action definition file "MoveBaseAction"
        self.move_base = actionlib.SimpleActionClient('/agv1/move_base', MoveBaseAction)
        # Waits until the action server has started up and started listening for goals.
        self.move_base.wait_for_server()
        rospy.loginfo("Connecting to the move_base server")
        # Creates a new goal with the MoveBaseGoal constructor
        self.goal = MoveBaseGoal()
        # Use the "map" topic as global coordinate frame
        self.goal.target_pose.header.frame_id = 'map'

        # --- Current position and orientation
        self.x_0 = 0.0
        self.y_0 = 0.0
        self.yaw_0 = 0.0
        self.rot_q_0 = 0.0

        # --- Number of vehicles to move
        self.veh_to_move = 0
        self.veh_to_discharge = 0

        self._trigger_event = threading.Event()

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'

        # Clear the flag of the trigger event
        self._trigger_event.clear()

        self._sub = rospy.Subscriber(self._topic, self._msg_type, self.execute_cb)

        # Set the flag of the trigger event after the time expired
        if self._timeout:
            self._timeout_timer = rospy.Timer(rospy.Duration(self._timeout),
                                              lambda _: self._trigger_event.set(),
                                              oneshot=True)

        # Wait until the flag of the trigger event is Set
        self._trigger_event.wait()
        # Unsubscribe to the topic
        self._sub.unregister()
        if self._timeout:
            self._timeout_timer.shutdown()
        rospy.sleep(1)

        if userdata.tag_position_in > 0:
            sign = +1
        else:
            sign = -1

        if 0.261 >= self.yaw_0 >= -0.261:
            self.goal.target_pose.pose.position.x = self.x_0 + (sign * (1.5 * V2V_DIST))
            self.goal.target_pose.pose.position.y = self.y_0 + (sign * 0.5)
            rospy.loginfo("0")
            # WORKING

        elif 1.91 >= self.yaw_0 >= 1.31:
            self.goal.target_pose.pose.position.x = self.x_0 - (sign * 0.5)
            self.goal.target_pose.pose.position.y = self.y_0 + (sign * (1.5 * V2V_DIST))
            rospy.loginfo("90")

        elif 3.15 >= self.yaw_0 >= 2.841:
            self.goal.target_pose.pose.position.x = self.x_0 - (sign * (1.5 * V2V_DIST))
            self.goal.target_pose.pose.position.y = self.y_0 - (sign * 0.5)
            rospy.loginfo("180")
            # WORKING

        elif -1.255 >= self.yaw_0 >= -1.926 or -2.9 >= self.yaw_0 >= (-3.2):
            self.goal.target_pose.pose.position.x = self.x_0 + (sign * 0.55)
            self.goal.target_pose.pose.position.y = self.y_0 - (sign * (1.5 * V2V_DIST))
            rospy.loginfo("270")

        self.goal.target_pose.pose.orientation.x = self.rot_q_0.x
        self.goal.target_pose.pose.orientation.y = self.rot_q_0.y
        self.goal.target_pose.pose.orientation.z = self.rot_q_0.z
        self.goal.target_pose.pose.orientation.w = self.rot_q_0.w

        # rospy.loginfo(self.goal)

        # Sends the goal to the action server.
        self.move_base.send_goal(self.goal)

        # Waits 90s to finish performing the action.
        wait = self.move_base.wait_for_result(rospy.Duration(90))
        # Clean the variables
        self.x_0 = 0.0
        self.y_0 = 0.0
        self.yaw_0 = 0.0
        self.rot_q_0 = 0.0

        if not wait:
            self.move_base.cancel_goal()
            rospy.loginfo("Timed out achieving goal")
            return 'failed'
        else:
            state = self.move_base.get_state()
            rospy.loginfo("Goal Succeeded!")
            return "succeed"

    def execute_cb(self, odom_msg):
        """
        Callback function for get the current position and orientation of robot

        :param odom_msg: odometry message
        """
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'

        self.x_0 = odom_msg.pose.pose.position.x
        self.y_0 = odom_msg.pose.pose.position.y

        self.rot_q_0 = odom_msg.pose.pose.orientation
        # Transform from quaternions to euler angles (roll, pitch and yaw)
        (_, _, self.yaw_0) = euler_from_quaternion([self.rot_q_0.x, self.rot_q_0.y, self.rot_q_0.z, self.rot_q_0.w])

        rospy.loginfo("X_0: %s", self.x_0)
        rospy.loginfo("Y_0: %s", self.y_0)
        rospy.loginfo("Yaw: %s", self.yaw_0)
        self._trigger_event.set()


class VehicleDetection(State):
    def __init__(self, topic, msg_type, timeout=5, max_checks=2):
        """
        State for detect the vehicle in the station.

        :param topic:       topic of the video camera
        :param msg_type:    message type for the topic
        :param timeout:     maximum time for the detection
        :param max_checks:  number of pass to declare the detection as good
        """
        State.__init__(self, outcomes=['detected', 'nothing', 'preempted', 'station_empty'])
        self._topic = topic
        self._msg_type = msg_type
        self._max_checks = max_checks
        self._cnt_checks = 0
        self._timeout = timeout
        self._good_detect = False
        self._sub = None
        self._timeout_timer = None

        self._trigger_event = threading.Event()

        self.bridge = CvBridge()
        rospy.sleep(1)
        rospy.loginfo("Waiting for images...")

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'

        # Clear the flag of the trigger event
        self._trigger_event.clear()

        self._sub = rospy.Subscriber(self._topic, self._msg_type, self.execute_cb)

        # Set the flag of the trigger event after the time expired
        if self._timeout:
            self._timeout_timer = rospy.Timer(rospy.Duration(self._timeout),
                                              lambda _: self._trigger_event.set(),
                                              oneshot=True)

        # Wait until the flag of the trigger event is Set
        self._trigger_event.wait()
        # Unsubscribe to the topic
        self._sub.unregister()

        if self._timeout:
            self._timeout_timer.shutdown()

        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        elif self._good_detect:
            self._cnt_checks = 0
            return 'detected'
        elif self._cnt_checks > self._max_checks:
            self._cnt_checks = 0
            return 'station_empty'
        elif not self._good_detect:
            return 'nothing'

    def execute_cb(self, img):
        """
        Callback function for the subscriber

        :param img: video frame
        :return: None
        """
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        try:
            frame = self.bridge.imgmsg_to_cv2(img, "bgr8")

        except CvBridgeError, e:
            print e
        else:
            self._good_detect = marker_detection.vehicle_detector(frame)
            rospy.loginfo("Detection: %s", self._good_detect)
            rospy.sleep(2)

            self._cnt_checks += 1
            rospy.loginfo("Tries: %s", self._cnt_checks)
            self._trigger_event.set()

    def request_preempt(self):
        State.request_preempt(self)
        self._trigger_event.set()


class StraightLineMove(State):
    def __init__(self, vel_x=0.0, timeout=3):
        """
        Straight line movement for an indicated amount of time

        :param vel_x: velocity for X coordinate
        :param timeout: amount of time to move
        """
        State.__init__(self, outcomes=['succeed', 'failed'])
        self._timeout = timeout
        self._timeout_timer = None
        self._vel_x = vel_x

    def execute(self, ud):
        if self._vel_x < 0.0:
            rospy.loginfo("Move Backward...")
        else:
            rospy.loginfo("Move Forward...")

        cmd_pub = rospy.Publisher('agv1/cmd_vel', Twist, queue_size=1)
        rospy.sleep(1)
        vel = Twist()
        vel.linear.x = self._vel_x

        result = cmd_pub.publish(vel)
        rospy.sleep(self._timeout)

        vel.linear.x = 0.0
        cmd_pub.publish(vel)

        if result is None:
            return 'succeed'
        else:
            return 'failed'


class ActionPlatoon(State):
    def __init__(self):
        """
        Select the action for executing: union or disunion
        """
        State.__init__(self, outcomes=['union', 'disunion', 'preempted'])

    def execute(self, ud):
        global action_on_station, initial_number_of_agv, num_of_vehicles, list_agv_platoon

        rospy.logwarn("Running %s", action_on_station)
        if action_on_station == 'union':
            action_on_station = ''
            initial_number_of_agv = len(list_agv_platoon) + 1
            rospy.logerr("Init number of vehicles in the platoon: %s", initial_number_of_agv)
            return 'union'
        elif action_on_station == 'disunion' and len(list_agv_platoon) > 1:
            action_on_station = ''
            initial_number_of_agv = len(list_agv_platoon) - 1
            rospy.logerr("Number of vehicles in the platoon: %s", num_of_vehicles)
            return 'disunion'
        else:
            action_on_station = ''
            return 'preempted'


class RelocationUnion(State):
    def __init__(self, topic, msg_type, timeout=10):
        """
        Move forward the AGV to let the new AGV join to the platoon

        :param topic: topic of the odometry topic
        :param msg_type: type of topical message
        :param timeout: maximum waiting time for get information from the topic
        """
        State.__init__(self, outcomes=['succeed'])
        self._topic = topic
        self._msg_type = msg_type
        self._timeout = timeout
        self._timeout_timer = None
        self._sub = None  # Subscriber

        # Create an action client with action definition file "MoveBaseAction"
        self.move_base = actionlib.SimpleActionClient('/agv1/move_base', MoveBaseAction)
        # Waits until the action server has started up and started listening for goals.
        self.move_base.wait_for_server()
        rospy.loginfo("Connecting to the move_base server")
        # Creates a new goal with the MoveBaseGoal constructor
        self.goal = MoveBaseGoal()
        # Use the "map" topic as global coordinate frame
        self.goal.target_pose.header.frame_id = 'map'

        # --- Current position and orientation
        self.x_0 = 0.0
        self.y_0 = 0.0
        self.yaw_0 = 0.0
        self.rot_q_0 = 0.0

        # --- Number of vehicles to move
        self.veh_to_move = 0
        self.veh_to_discharge = 0

        self._trigger_event = threading.Event()

    def execute(self, userdata):
        global num_of_vehicles, list_agv_platoon
        # Clear the flag of the trigger event
        self._trigger_event.clear()

        self._sub = rospy.Subscriber(self._topic, self._msg_type, self.execute_cb)

        # Set the flag of the trigger event after the time expired
        if self._timeout:
            self._timeout_timer = rospy.Timer(rospy.Duration(self._timeout),
                                              lambda _: self._trigger_event.set(),
                                              oneshot=True)

        # Wait until the flag of the trigger event is Set
        self._trigger_event.wait()
        # Unsubscribe to the topic
        self._sub.unregister()
        if self._timeout:
            self._timeout_timer.shutdown()
        rospy.sleep(1)

        if 0.261 >= self.yaw_0 >= -0.261:
            self.goal.target_pose.pose.position.x = self.x_0 + ((num_of_vehicles * 0.7) +
                                                                num_of_vehicles * V2V_DIST)
            self.goal.target_pose.pose.position.y = self.y_0 + 0.1
            rospy.loginfo("0")

        elif 1.91 >= self.yaw_0 >= 1.31:
            self.goal.target_pose.pose.position.x = self.x_0 - 0.1
            self.goal.target_pose.pose.position.y = self.y_0 + ((num_of_vehicles * 0.7) +
                                                                num_of_vehicles * V2V_DIST)
            rospy.loginfo("90")

        elif 3.15 >= self.yaw_0 >= 2.851:
            self.goal.target_pose.pose.position.x = self.x_0 - ((num_of_vehicles * 0.7) +
                                                                num_of_vehicles * V2V_DIST)
            self.goal.target_pose.pose.position.y = self.y_0 - 0.1
            rospy.loginfo("180")

        elif -1.255 >= self.yaw_0 >= -1.926:
            self.goal.target_pose.pose.position.x = self.x_0 + 0.1
            self.goal.target_pose.pose.position.y = self.y_0 - ((num_of_vehicles * 0.7) +
                                                                num_of_vehicles * V2V_DIST)
            rospy.loginfo("270")

        self.goal.target_pose.pose.orientation.x = self.rot_q_0.x
        self.goal.target_pose.pose.orientation.y = self.rot_q_0.y
        self.goal.target_pose.pose.orientation.z = self.rot_q_0.z
        self.goal.target_pose.pose.orientation.w = self.rot_q_0.w

        # Sends the goal to the action server.
        self.move_base.send_goal(self.goal)

        # Waits 90s to finish performing the action.
        wait = self.move_base.wait_for_result(rospy.Duration(90))
        # Clean the variables
        self.x_0 = 0.0
        self.y_0 = 0.0
        self.yaw_0 = 0.0
        self.rot_q_0 = 0.0

        if wait:
            state = self.move_base.get_state()
            rospy.loginfo("Goal Succeeded!")
            return "succeed"

    def execute_cb(self, odom_msg):
        """
        Callback function for get the current position and orientation of robot

        :param odom_msg: odometry message
        """
        self.x_0 = odom_msg.pose.pose.position.x
        self.y_0 = odom_msg.pose.pose.position.y

        self.rot_q_0 = odom_msg.pose.pose.orientation
        # Transform from quaternions to euler angles (roll, pitch and yaw)
        (_, _, self.yaw_0) = euler_from_quaternion([self.rot_q_0.x, self.rot_q_0.y, self.rot_q_0.z, self.rot_q_0.w])

        rospy.loginfo("X_0: %s", self.x_0)
        rospy.loginfo("Y_0: %s", self.y_0)
        rospy.loginfo("Yaw: %s", math.degrees(self.yaw_0))
        self._trigger_event.set()


class MovingForward(State):
    def __init__(self, vel_x=0.25, timeout=0.1, tolerance=0.6):
        State.__init__(self, outcomes=['succeed', 'moving'])
        self._timeout = timeout
        self._timeout_timer = None
        self._vel_x = vel_x
        self._x_agv_pose = 0.0
        self._y_agv_pose = 0.0
        self.diff_x = 0.0
        self.diff_y = 0.0
        self.x_goal = 0.0
        self.y_goal = 0.0
        self._moving = True

        self.tolerance = tolerance

    def execute(self, ud):
        global list_agv_platoon

        rospy.loginfo("Moving to disunion point ...")
        last_agv_odom_sub = rospy.Subscriber(list_agv_platoon[-1] + '/odom', Odometry, self._last_agv_odom_cb)
        cmd_pub = rospy.Publisher('agv1/cmd_vel', Twist, queue_size=1)
        vel = Twist()

        # Goal coordinates
        self.x_goal = disunion_points[map_to_use][0]
        self.y_goal = disunion_points[map_to_use][1]
        rospy.sleep(1)
        while self._moving:
            if math.fabs(self.diff_x) > 0.01 or math.fabs(self.diff_y) > 0.01:
                vel.linear.x = self._vel_x
            else:
                vel.linear.x = -self._vel_x

            cmd_pub.publish(vel)
            rospy.sleep(self._timeout)

        # Stopping agv
        last_agv_odom_sub.unregister()
        vel.linear.x = 0.0
        cmd_pub.publish(vel)
        rospy.loginfo("Stopping ......")
        cmd_pub.unregister()
        rospy.sleep(1)

        if not self._moving:
            rospy.logerr("In position ........")
            self.reset_var()
            rospy.sleep(1)
            return 'succeed'
        else:
            return 'moving'

    def _last_agv_odom_cb(self, odom_msg):
        self._x_agv_pose = round(odom_msg.pose.pose.position.x, 3)
        self._y_agv_pose = round(odom_msg.pose.pose.position.y, 3)

        # Calculated the difference between the current position of the last agv an d the point to disunion

        self.diff_x = round(self.x_goal - self._x_agv_pose, 3)
        self.diff_y = round(self.y_goal - self._y_agv_pose, 3)
        rospy.logwarn("Difference X: %s, Y: %s", math.fabs(self.diff_x), math.fabs(self.diff_y))
        if math.fabs(self.diff_x) <= self.tolerance or math.fabs(self.diff_y) <= self.tolerance:
            self._moving = False

    def reset_var(self):
        self._x_agv_pose = 0.0
        self._y_agv_pose = 0.0
        self.diff_x = 0.0
        self.diff_y = 0.0
        self.x_goal = 0.0
        self.y_goal = 0.0
        self._moving = True


class Disunion(State):
    def __init__(self, timeout=15):
        State.__init__(self, outcomes=['disunion_succeed', 'waiting'])
        self._timeout = timeout
        self._timeout_timer = None
        self._trigger_event = threading.Event()
        self._vehicle_status = None

    def execute(self, ud):
        global initial_number_of_agv, num_of_vehicles, list_agv_platoon, list_agv_index
        # Create a publisher for send the ids
        vehicle_status_sub = rospy.Subscriber(list_agv_platoon[-1] + '/disconfig', V2VCommunication,
                                              self._vehicle_status_cb,
                                              queue_size=1)
        disconfig_pub = rospy.Publisher(list_agv_platoon[-1] + '/disconfig', V2VCommunication, queue_size=2)
        disconfig = V2VCommunication()
        rospy.sleep(1)

        # Send the data of the point for disunion
        disconfig.vehicle_name = list_agv_platoon[-1]
        disconfig.x_point = disunion_points[map_to_use][0]
        disconfig.y_point = disunion_points[map_to_use][1]
        disconfig.theta = disunion_points[map_to_use][2]
        disconfig.status = "DISUNION"

        disconfig_pub.publish(disconfig)

        # Clear the flag of the trigger event
        self._trigger_event.clear()

        # Set the flag of the trigger event after the time expired
        if self._timeout:
            self._timeout_timer = rospy.Timer(rospy.Duration(self._timeout),
                                              lambda _: self._trigger_event.set(),
                                              oneshot=True)

        # Wait until get a answer of the agv and the flag of the trigger event is Set
        self._trigger_event.wait()

        # shutdown timer
        if self._timeout:
            self._timeout_timer.shutdown()

        if self._vehicle_status == "DISUNION_COMPLETED" and len(list_agv_platoon) > 1:
            agv_disunion = list_agv_platoon.pop()
            num_of_vehicles = len(list_agv_platoon)
            initial_number_of_agv = len(list_agv_platoon)
            list_agv_index -= 1
            rospy.logerr("Number of vehicles in the platoon: %s", num_of_vehicles)
            rospy.logerr("List of vehicles in the platoon: %s", list_agv_platoon)
            rospy.logerr("Init num of vehicles in the platoon: %s", initial_number_of_agv)
            rospy.logerr("AGV disunion: %s", agv_disunion)
            vehicle_status_sub.unregister()
            disconfig_pub.unregister()
            rospy.sleep(1)
            self._vehicle_status = None
            return 'disunion_succeed'
        else:
            return 'waiting'

    def _vehicle_status_cb(self, msg):
        global initial_number_of_agv, num_of_vehicles, list_agv_platoon, list_agv_index
        if msg.status == "DISUNION_COMPLETED" and list_agv_platoon[-1] == msg.vehicle_name:
            self._vehicle_status = "DISUNION_COMPLETED"
        self._trigger_event.set()


class MainProcess:
    def __init__(self):
        # Initializes a rospy node to let the SimpleActionClient publish and subscribe
        rospy.init_node('agv_leader_controller', anonymous=True)

        # Shutdown function
        rospy.on_shutdown(self.shutdown)

        self.move = rospy.Publisher("/agv1/cmd_vel", Twist, queue_size=1)

        # Waits until the action server has started up and started listening for goals.
        # Wait until Gazebo run
        rospy.wait_for_service("gazebo/delete_model")
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        s = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

        rospy.loginfo("Configuring the platoon.......")
        rospy.sleep(1)

        # --- First State Machine for configure the initial number of AGV for the platoon
        sm_init_params = StateMachine(outcomes=["finish"])

        with sm_init_params:
            StateMachine.add("INIT_PARAMETERS", InitParamsForPlatoon(),
                             transitions={'getting_new_params': 'SET_UP_AGV',
                                          'waiting': 'INIT_PARAMETERS'})

            # --- State Machine for initial configuration of the platoon
            sm_platoon = StateMachine(outcomes=['finish'])

            with sm_platoon:
                # --- Initial configuration for AGV
                StateMachine.add("INIT_CONFIG", InitConfigAGV(),
                                 transitions={'finishing_setup': 'COUNT_AGV',
                                              'waiting': 'INIT_CONFIG'})

                # --- Count the number of AGV and compare to the initial number of AGV indicated
                StateMachine.add("COUNT_AGV", CountNumAGV(),
                                 transitions={'succeed': 'CONTROL_MOVE',
                                              'waiting': 'INIT_CONFIG'})

                # --- Creating a main State Machine for control the travel to the stations
                # --- State Machine for travel to the new goal
                sm_control = StateMachine(outcomes=['succeed'])
                sm_control.userdata.nav_goal = None  # Navigation coordinates
                sm_control.userdata.id_station = None  # ID of the station
                sm_control.userdata.tag_position = 0.0  # Position along X axis of the Tag detected

                with sm_control:
                    # General ..........................................................................................
                    # --- Get a new goal for navigation
                    StateMachine.add("NEW_GOAL", StationGoal(),
                                     transitions={'new goal': "NAV_GOAL",
                                                  'failed': "NEW_GOAL",
                                                  'waiting': "NEW_GOAL"},
                                     remapping={'new_goal_out': 'nav_goal',
                                                'id_station_out': 'id_station'})

                    # Navigation to the new goal ~
                    StateMachine.add("NAV_GOAL", NavToGoal(),
                                     transitions={'goal reached': 'STATION_DETECTION',
                                                  'aborted': "NEW_GOAL",
                                                  'timeout': "NAV_GOAL"},
                                     remapping={'nav_goal_in': 'nav_goal'})

                    # --- Identify the type station
                    StateMachine.add("STATION_DETECTION", StationDetection('agv1/camera/image_raw', Image),
                                     transitions={'load_unload': 'LOAD_UNLOAD_STATION',
                                                  'union_disunion': 'SELECTING_ACTION',
                                                  'failed': 'MOVE_BACKWARD',
                                                  'preempted': 'NAV_GOAL'},
                                     remapping={'id_in': 'id_station',
                                                'tag_position_out': 'tag_position'})
                    # ..................................................................................................

                    # Load or unload ***********************************************************************************
                    # --- Drive backward
                    StateMachine.add("MOVE_BACKWARD", StraightLineMove(vel_x=-0.1),
                                     transitions={'succeed': 'STATION_DETECTION',
                                                  'failed': 'MOVE_BACKWARD'})

                    # --- Relocated the AGV
                    StateMachine.add("LOAD_UNLOAD_STATION", VehicleRelocation('agv1/odom', Odometry),
                                     transitions={'succeed': 'LOAD_UNLOAD_WAITING',
                                                  'failed': 'NEW_GOAL'},
                                     remapping={'tag_position_in': 'tag_position'})

                    # --- Wait for load or unload items
                    StateMachine.add("LOAD_UNLOAD_WAITING", LoadUnloadStation(timeout=5),
                                     transitions={'succeed': 'NEW_GOAL'})

                    # **************************************************************************************************

                    # Union or disunion ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # --- Choose for union or disunion from the platoon
                    StateMachine.add("SELECTING_ACTION", ActionPlatoon(),
                                     transitions={'union': 'AGV_DETECT',
                                                  'disunion': 'RELOCATED_AGV_DISUNION',
                                                  'preempted': 'NEW_GOAL'})

                    # --- Union !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # --- Drive forward
                    StateMachine.add("MOVE_FORWARD", StraightLineMove(vel_x=0.1),
                                     transitions={'succeed': 'AGV_DETECT',
                                                  'failed': 'RELOCATED_AGV_DISUNION'})

                    # --- Detection of vehicle in the station
                    StateMachine.add("AGV_DETECT", VehicleDetection('agv1/camera/image_raw', Image),
                                     transitions={'detected': 'RELOCATED_AGV_UNION',
                                                  'nothing': 'MOVE_FORWARD',
                                                  'preempted': 'NAV_GOAL',
                                                  'station_empty': 'NEW_GOAL'})

                    # --- Relocated the AGV front of station tag
                    StateMachine.add("RELOCATED_AGV_UNION", VehicleRelocation('agv1/odom', Odometry),
                                     transitions={'succeed': 'RELOCATED_FORWARD_AGV_UNION',
                                                  'failed': 'NEW_GOAL'},
                                     remapping={'tag_position_in': 'tag_position'})

                    # --- Relocate for union of a agv
                    StateMachine.add("RELOCATED_FORWARD_AGV_UNION", RelocationUnion('agv1/odom', Odometry),
                                     transitions={'succeed': 'MOVE_FORWARD_UNION'})

                    # --- Drive forward
                    StateMachine.add("MOVE_FORWARD_UNION", StraightLineMove(vel_x=0.16),
                                     transitions={'succeed': 'CONFIG_NEW_AGV',
                                                  'failed': 'MOVE_FORWARD_UNION'})

                    # --- Initial configuration for the new AGV
                    StateMachine.add("CONFIG_NEW_AGV", InitConfigAGV(),
                                     transitions={'finishing_setup': 'COUNT_AGV',
                                                  'waiting': 'CONFIG_NEW_AGV'})

                    # --- Count the number of AGV and compare to the initial number of AGV indicated
                    StateMachine.add("COUNT_AGV", CountNumAGV(),
                                     transitions={'succeed': 'NEW_GOAL',
                                                  'waiting': 'CONFIG_NEW_AGV'})

                    # --- Disunion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # --- Relocated the AGV front of station tag
                    StateMachine.add("RELOCATED_AGV_DISUNION", VehicleRelocation('agv1/odom', Odometry),
                                     transitions={'succeed': 'MOVING_DISUNION',
                                                  'failed': 'NEW_GOAL'},
                                     remapping={'tag_position_in': 'tag_position'})

                    # --- Relocate for disunion of a agv
                    # StateMachine.add("RELOCATED_FORWARD_AGV_DISUNION", RelocationUnion('agv1/odom', Odometry),
                    #                  transitions={'succeed': 'MOVING_DISUNION'})

                    StateMachine.add("MOVING_DISUNION", MovingForward(tolerance=0.6),
                                     transitions={'succeed': 'DISUNION',
                                                  'moving': 'MOVING_DISUNION'})

                    StateMachine.add("DISUNION", Disunion(),
                                     transitions={'disunion_succeed': 'NEW_GOAL',
                                                  'waiting': 'DISUNION'})
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                # --- Transition to the AGV control
                StateMachine.add("CONTROL_MOVE", sm_control,
                                 transitions={'succeed': 'finish'})

            StateMachine.add("SET_UP_AGV", sm_platoon,
                             transitions={'finish': 'finish'})

        # Create and start the introspection server
        intro_server = smach_ros.IntrospectionServer('control', sm_init_params, '/SM_ROOT')
        intro_server.start()

        # Execute the state machine
        outcome = sm_init_params.execute()

        # Wait for ctrl-c to stop the application
        rospy.spin()
        intro_server.stop()

    def shutdown(self):
        rospy.loginfo("Stopping the vehicle ....")
        self.move.publish(Twist())
        rospy.sleep(1)


if __name__ == '__main__':
    try:
        MainProcess()

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
