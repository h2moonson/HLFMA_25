#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import rospkg

import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped

from datetime import datetime

from tf.transformations import euler_from_quaternion, quaternion_from_euler

class PathMaker:
    def __init__(self):
        rospy.init_node('morai_maker', anonymous=True)

        self.curr_pose = Pose()
        rospy.Subscriber("/current_pose", Pose, self.status_callback)
        self.rviz_global_path_pub = rospy.Publisher('/rviz_global_path', Path, queue_size=1)
        self.rviz_global_path = Path()
        self.rviz_global_path.header.stamp = rospy.Time.now()
        self.rviz_global_path.header.frame_id = 'map'

        self.is_status = False
        self.prev_x = 0
        self.prev_y = 0
        self.idx = 0

        self.start_x = 0
        self.start_y = 0

        rospack = rospkg.RosPack()
        ROS_HOME = rospack.get_path('decision')
        
        now = datetime.now()
        self.f = open('{}/path/{}-{}-{}_{}-{}.txt'.format(ROS_HOME, now.year, now.month, now.day, now.hour, now.minute), 'w')


        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.is_status:
                self.path_make()
            rate.sleep()

        self.f.close()
        
    def path_make(self):
        x = self.curr_pose.position.x
        y = self.curr_pose.position.y

        if self.start_x == 0 and self.start_y == 0:
            self.start_x = x
            self.start_y = y

        quat = self.curr_pose.orientation 
        orientation_list = [quat.x, quat.y, quat.z, quat.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        mode = 9
        
        distance = np.hypot(x - self.prev_x, y - self.prev_y)
        
        self.rviz_global_path.header.stamp = rospy.Time.now()

        if distance > 0.2:
            self.prev_x = x
            self.prev_y = y

            rviz_pose = PoseStamped()
            rviz_pose.header.frame_id = 'map'
            rviz_pose.header.stamp = rospy.Time.now()
            rviz_pose.header.seq = self.idx
            rviz_pose.pose.position.x = x - self.start_x
            rviz_pose.pose.position.y = y - self.start_y
            rviz_pose.pose.position.z = 0.

            rviz_pose.pose.orientation.w = quat.w
            rviz_pose.pose.orientation.x = quat.x
            rviz_pose.pose.orientation.y = quat.y
            rviz_pose.pose.orientation.z = quat.z

            self.rviz_global_path.poses.append(rviz_pose)            

            data = '{0} {1} {2} {3} \n'.format(x, y, yaw, mode)
            self.f.write(data)
            
            self.idx += 1
            # rospy.loginfo(f'{self.idx}, {x}, {y}, {yaw}')

        self.rviz_global_path_pub.publish(self.rviz_global_path)

    def status_callback(self, msg):
        self.curr_pose = msg
        self.is_status = True

if __name__ == '__main__':
    try:
        test_track = PathMaker()
        
    except rospy.ROSInterruptException:
        pass