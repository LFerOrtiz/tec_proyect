#!/usr/bin/env python
# coding=utf-8

"""
Send goal thought the "/goal_station/" topic for the AGV lead
"""
import math
import rospy
from tec_proyect.msg import V2VCommunication

current_status = ""
map_status = ""
count = 0

maps = {
    0: "prueba",
    1: "almacen",
    2: "hospital",
    3: "laboratorio",
}


def status_platoon_cb(msg):
    global current_status, count, map_status
    current_status = msg.status
    map_status = msg.map_to_use
    if current_status == "WAITING_NEW_GOAL":
        count = 1


if __name__ == "__main__":
    rospy.init_node("station_goal_pub", anonymous=True)

    # --- Publisher for send information to AGV leader
    agv_leader_params_pub = rospy.Publisher("/params_for_agv_leader", V2VCommunication, queue_size=1)
    agv_leader_params = V2VCommunication()

    # --- Subscriber to know the Status of the AGV leader of the platoon
    give_params = rospy.Subscriber("/params_for_platoon", V2VCommunication, status_platoon_cb)
    rate_fresh = rospy.Rate(1)
    map_selected = 999999

    try:
        while not rospy.is_shutdown():
            if current_status == "WAITING_INIT_NUM_AGV" and count == 0:
                agv_leader_params.num_of_initial_agv = int(input("Initial number of AGV: "))
                agv_leader_params.status = "GIVING_INIT_NUM_AGV"
                count = 1

            elif current_status == "WAITING_NEW_GOAL" and count == 1:
                print("*************************************************************")
                if map_status != "THERE_IS_A_MAP" and map_selected == 999999:
                    agv_leader_params.map_to_use = maps[int(raw_input("Map to use, "
                                                                      "\n[0]: prueba, "
                                                                      "\n[1]: almacen, "
                                                                      "\n[2]: hospital, "
                                                                      "\n[3]: laboratorio"
                                                                      "\nOption: "))]
                    map_selected = agv_leader_params.map_to_use
                else:
                    agv_leader_params.map_to_use = map_selected
                agv_leader_params.id_marker = int(raw_input("ID of station: "))
                type_station = math.floor(agv_leader_params.id_marker / 100.0)
                if type_station == 2:
                    option_action = int(raw_input("Union [0] or Disunion [1]: "))
                    if option_action == 0:
                        agv_leader_params.coupling_or_decoupling = "union"
                    elif option_action == 1:
                        agv_leader_params.coupling_or_decoupling = "disunion"
                agv_leader_params.status = "SENDING_NEW_GOAL"
                count = 2
                print("*************************************************************")

            elif current_status == "ALL_OK":
                agv_leader_params.id_marker = 0
                agv_leader_params.status = ""
                rospy.loginfo("Waiting")

            agv_leader_params_pub.publish(agv_leader_params)
            rate_fresh.sleep()

    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutdown node")

