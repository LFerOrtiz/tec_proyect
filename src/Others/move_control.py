#!/usr/bin/env python

import rospy
import cv_bridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time

detec_field = 0
state = 0

# *************************************** Funtions ***************************************
def LaserScanCallBack(scan_data):
    """ Callback funtion for laser scan """
    global detec_field
    # Define the detection field
    detec_field = {
        "front_right": min(min(scan_data.ranges[160:287]), 10),
        "front": min(min(scan_data.ranges[288:431]), 10),
        "front_left": min(min(scan_data.ranges[432:500]), 10),
    }
    print(detec_field)
    print(state)
    move(detec_field)


def move(detec_field):
    """ Move the avg depends of the obstacules detected """
    global state
    min_distance = 2.0

    if detec_field["front"] > min_distance and (detec_field["front_right"]  and detec_field["front_left"]) > min_distance:
        state = 1
    elif detec_field["front"] < min_distance and detec_field["front_right"] < min_distance and detec_field["front_left"] > min_distance:
        state = 2
    elif detec_field["front"] < min_distance and detec_field["front_right"] > min_distance and detec_field["front_left"] < min_distance:
        state = 3
    elif detec_field["front"] < min_distance and (detec_field["front_right"] and detec_field["front_left"]) < min_distance:
        state = 0
    

def stop():
    """ Make stop the agv """
    vel_msg = Twist()
    vel_msg.linear.x = 0.0
    return vel_msg


def linear_move():
    """ Make the agv move in straight line """
    vel_msg = Twist()
    vel_msg.linear.x = abs(0.8)         # Velocity of 0.5 m/s
    return vel_msg

def back_move():
    """ Make the agv move in straight line """
    vel_msg = Twist()
    vel_msg.linear.x = -abs(0.8)         # Velocity of 0.5 m/s
    return vel_msg

def turn_left():
    """ Make the avg turn to the left """
    vel_msg = Twist()
    vel_msg.linear.x = -abs(0.8)         # Velocity of 0.5 m/s
    vel_msg.angular.z = 8.5
    return vel_msg


def turn_right():
    """ Make the avg turn to the left """
    vel_msg = Twist()
    vel_msg.linear.x = -abs(0.5)         # Velocity of 0.5 m/s
    vel_msg.angular.z = -8.5
    return vel_msg


def main():
    """ Main program """
    global state

    # Init a new node
    rospy.init_node('move_control', anonymous="True")

    # Subcribers and publisher
    vel_pub = rospy.Publisher('agv1/cmd_vel', Twist, queue_size=25)
    laser_sub = rospy.Subscriber("agv1/scan", LaserScan, LaserScanCallBack)

    # vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=25)
    # laser_sub = rospy.Subscriber("/scan", LaserScan, LaserScanCallBack)


    rate_fresh = rospy.Rate(120)         #120 Hz

    while not rospy.is_shutdown():
        if state == 1:
            msg = linear_move()
        elif state == 2:
            msg = turn_left()
        elif state == 3:
            msg = turn_right()

        elif state == 0:
            msg = stop()
            msg = back_move()
            state = 2
            


        vel_pub.publish(msg)
        rate_fresh.sleep()


# *************************************** Main loop ***************************************
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInternalException:
        rospy.loginfo("Error en inicializar.")


    