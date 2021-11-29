import math

import rospy
from geometry_msgs.msg import Twist, Pose2D
import numpy as np

# simulation parameters

dt = 0.1


class MoveToGoal:
    def __init__(self, vel_publisher):
        self.velocity_publisher = vel_publisher
        self.rate = rospy.Rate(10)
        self.pose = Pose2D()
        self.move = Twist()
        self.goal = Pose2D()
        self.turn = True
        self.Kp_rho = 0.02
        self.Kp_alpha = 0.18
        self.Kp_beta = 5

    def update_pose(self, x_point, y_point, theta):
        """
        Update the Pose of the agv
        """
        self.pose.x = round(x_point, 4)
        self.pose.y = round(y_point, 4)
        self.pose.theta = round(theta, 4)

    def move_to_pose(self, x_goal, y_goal, theta_goal):
        """
        rho is the distance between the robot and the goal position
        alpha is the angle to the goal relative to the heading of the robot
        beta is the angle between the robot's position and the goal position plus the goal angle

        Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards the goal
        Kp_beta*beta rotates the line so that it is parallel to the goal angle
        """

        if self.goal.y == 0.0 and self.goal.y == 0.0:
            self.goal.x = round(float(x_goal), 3)
            self.goal.y = round(float(y_goal), 3)
            self.goal.theta = round(float(theta_goal), 3)
            rospy.logerr("Goal x: %s, Goal y: %s", self.goal.x, self.goal.y)

        theta = self.pose.theta

        x_diff = self.goal.x - self.pose.x
        y_diff = self.goal.y - self.pose.y

        rho = np.hypot(x_diff, y_diff)

        while rho > 0.4:
            x_diff = self.goal.x - self.pose.x
            y_diff = self.goal.y - self.pose.y

            # Restrict alpha and beta (angle differences) to the range
            # [-pi, pi] to prevent unstable behavior e.g. difference going
            # from 0 rad to 2*pi rad with slight turn

            rho = round(np.hypot(x_diff, y_diff), 4)
            alpha = round((np.arctan2(y_diff, x_diff) - theta + np.pi) % (2 * np.pi) - np.pi, 3)
            beta = round((self.goal.theta - theta - alpha + np.pi) % (2 * np.pi) - np.pi, 3)
            rospy.logerr("AcTan2: %s", (np.arctan2(y_diff, x_diff) - self.pose.theta))
            rospy.logwarn("control: %s", self.turn)
            if np.fabs(round((np.arctan2(y_diff, x_diff) - self.pose.theta), 1)) <= 0.25 and self.turn:
                w = 0.0
                self.turn = False
                self.Kp_rho = 0.4
            elif self.turn:
                w = self.Kp_alpha * alpha + self.Kp_beta * beta

            v = self.Kp_rho * rho
            self.move.linear.x = v
            self.move.angular.z = w

            rospy.logerr("Vel x: %s, Vel W: %s", self.move.linear.x, self.move.angular.z)
            rospy.logerr("Rho : %s, Alpha : %s, Beta : %s", rho, alpha, beta)

            # Publishing our vel_msg
            self.velocity_publisher.publish(self.move)

            # Publish at the desired rate.
            self.rate.sleep()

        # Stopping agv after the movement is over.
        self.move.linear.x = 0
        self.move.angular.z = 0
        self.velocity_publisher.publish(self.move)
        rospy.sleep(2)
