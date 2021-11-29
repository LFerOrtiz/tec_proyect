#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

xAnt = 0.0
yAnt = 0.0
cont = 0


def callback(data):
    """
    Callback for get odometry and publish the path

    :param data: data from odometry topic
    """

    global xAnt
    global yAnt
    global cont

    pose = PoseStamped()

    pose.header.frame_id = "main"
    pose.pose.position.x = float(data.pose.pose.position.x)
    pose.pose.position.y = float(data.pose.pose.position.y)
    pose.pose.position.z = float(data.pose.pose.position.z)
    pose.pose.orientation.x = float(data.pose.pose.orientation.x)
    pose.pose.orientation.y = float(data.pose.pose.orientation.y)
    pose.pose.orientation.z = float(data.pose.pose.orientation.z)
    pose.pose.orientation.w = float(data.pose.pose.orientation.w)

    if xAnt != pose.pose.position.x and yAnt != pose.pose.position.y:
        pose.header.seq = path.header.seq + 1
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        pose.header.stamp = path.header.stamp
        path.poses.append(pose)
        # Published the msg

    cont += 1
    rospy.loginfo("Hit: %i" % cont)
    if cont > max_append:
        path.poses.pop(0)

    pub.publish(path)
    xAnt = pose.pose.orientation.x
    yAnt = pose.pose.position.y
    # return path


if __name__ == '__main__':
    # Initializing node
    rospy.init_node('path_plotter')
    max_append = 1000
    if not (max_append > 0):
        rospy.logwarn('The parameter max_list_append is not correct')
        sys.exit()
    pub = rospy.Publisher('agv1/path', Path, queue_size=1)

    path = Path()
    msg = Odometry()

    odom_sub = rospy.Subscriber('agv1/odom', Odometry, callback)
    rate = rospy.Rate(30)  # 30hz

    try:
        while not rospy.is_shutdown():
            rate.sleep()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutdown node")
