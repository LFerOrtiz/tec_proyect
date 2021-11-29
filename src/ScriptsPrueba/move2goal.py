#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Pose2D
from math import pow, atan2, sqrt, fabs
from numpy import arctan2
from control_pid import ControlPID


class Move2Goal:
    def __init__(self, vel_publisher):
        self.velocity_publisher = vel_publisher
        self.rate = rospy.Rate(10)
        self.pose = Pose2D()

    def update_pose(self, x_point, y_point, theta):
        """
        Update the Pose of the agv
        """
        self.pose.x = round(x_point, 4)
        self.pose.y = round(y_point, 4)
        self.pose.theta = round(theta, 4)
        # rospy.logwarn("X: %s ,Y: %s , Yaw: %s", self.pose.x, self.pose.y, self.pose.theta)

    def euclidean_distance(self, goal_pose):
        """Euclidean distance between current pose and the goal."""
        return sqrt(pow((goal_pose.x - self.pose.x), 2) + pow((goal_pose.y - self.pose.y), 2))

    def linear_vel(self, goal_pose, kp=0.3):
        """
        Proportional controller for linear velocity
        """
        vel = kp * self.euclidean_distance(goal_pose)
        if fabs(vel) >= 1.2:
            vel = 1.2
        return vel

    def steering_angle(self, goal_pose):
        delta_x = goal_pose.x - self.pose.x
        delta_y = goal_pose.y - self.pose.y
        point_heading = arctan2(delta_y, delta_x)
        return point_heading

    def angular_vel(self, goal_pose, kp=1.0):
        return kp * (self.steering_angle(goal_pose) - self.pose.theta)

    def move2goal(self, goal_point_x, goal_point_y, goal_theta, dist_tolerance=0.45):
        # Get the goal pose
        goal_point = Pose2D()
        goal_point.x = round(float(goal_point_x), 3)
        goal_point.y = round(float(goal_point_y), 3)
        goal_point.theta = round(float(goal_theta), 3)

        rospy.logerr("Goal x: %s, Goal y: %s", goal_point.x, goal_point.y)

        vel_msg = Twist()
        control = 0

        while self.euclidean_distance(goal_point) >= dist_tolerance:
            rospy.logerr("Theta diff: %s", fabs(self.steering_angle(goal_point) - self.pose.theta))
            if fabs(self.steering_angle(goal_point) - self.pose.theta) > 0.15 and control == 0:
                vel_msg.linear.x = 0.01
                vel_msg.angular.z = self.angular_vel(goal_point)
            else:
                control = 1
                vel_msg.linear.x = self.linear_vel(goal_point)
                vel_msg.angular.z = 0.4

            # Publishing our vel_msg
            self.velocity_publisher.publish(vel_msg)

            # Publish at the desired rate.
            self.rate.sleep()

        # Stopping agv after the movement is over.
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)
        rospy.sleep(2)
