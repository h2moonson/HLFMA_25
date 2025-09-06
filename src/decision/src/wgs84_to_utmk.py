#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ros 관련 라이브러리
import sys
import rospy

# numpy
import numpy as np

# tranfrom coordinate
from pyproj import Transformer, CRS

# 쿼터니언 <-> 오일러 변환 함수
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import Point, Pose, TwistWithCovarianceStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

import message_filters as mf

from morai_msgs.msg import EgoVehicleStatus
from sensor_msgs.msg import NavSatFix

from decision.srv import InitializePose, InitializePoseResponse

class TransformCoordinate:
    def __init__(self):
        self.is_erp = sys.argv[1] == 'e'

        self.curr_pose_pub = rospy.Publisher('/current_pose', Pose, queue_size=1)
        self.speed_pub = rospy.Publisher("/gps_velocity", Float64, queue_size=1)
        self.utmk_coordinate_pub = rospy.Publisher('/utmk_coordinate', Point, queue_size=1)

        # WGS84
        proj_WGS84 = CRS('EPSG:4326')

        # UTM52N
        proj_UTM52N = CRS('EPSG:32652')

        # UTM-K
        proj_UTMK = CRS('EPSG:5179')

        self.prev_pose = Pose()
        self.prev_pose.position.x = 0
        self.prev_pose.position.y = 0
        
        self.quat = None

        self.curr_pose = Pose()

        if self.is_erp:
            self.transformer = Transformer.from_crs(proj_WGS84, proj_UTMK, always_xy=True)
            rospy.Subscriber('/gps_front/fix', NavSatFix, self.erp_callback)
            rospy.Subscriber('/gps_front/fix_velocity', TwistWithCovarianceStamped, self.velocity_callback)

        else:
            print('hi?')
            self.transformer = Transformer.from_crs(proj_UTM52N, proj_UTMK, always_xy=True)
            rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.morai_callback)

            # ego_sub = mSubscriber('/Ego_topic', EgoVehicleStatus)
            # imu_sub = mf.Subscriber('/imu', Imu)
            
            # ats = mf.ApproximateTimeSynchronizer([ego_sub, imu_sub], queue_size=10, slop=0.01)
            # ats.registerCallback(self.x_to_utmk)
            
        rospy.Service('InitializePose', InitializePose, self.initialize_curr_pose)
        
    def erp_callback(self, msg):
        long, lat = msg.longitude, msg.latitude
        self.x_to_utmk(long, lat)

    def velocity_callback(self, msg):
        self.xVelocity = msg.twist.twist.linear.x
        self.yVelocity = msg.twist.twist.linear.y
        self.zVelocity = msg.twist.twist.linear.z
        self.speed = np.hypot(self.xVelocity, self.yVelocity) * 3.6

        data = Float64()
        data.data = self.speed
        self.speed_pub.publish(data)

    def morai_callback(self, msg):
        x, y = msg.position.x, msg.position.y
        
        x += 302459.942
        y += 4122635.537
        
        self.x_to_utmk(x, y)

    def x_to_utmk(self, long, lat):
        x, y = long, lat

        utmx, utmy = self.transformer.transform(x, y)
        pub_data = Point(utmx, utmy, 0.0)
        self.utmk_coordinate_pub.publish(pub_data)

        # 이전 좌표와 0.2 이상 차이가 나면 current_pose 갱신 
        prev_x, prev_y = self.prev_pose.position.x, self.prev_pose.position.y
        curr_x, curr_y = pub_data.x, pub_data.y

        self.curr_pose.position.x = curr_x
        self.curr_pose.position.y = curr_y
        self.curr_pose.position.z = 0

        dist = np.hypot(curr_x - prev_x, curr_y - prev_y)
        if dist > 0.2:
            yaw = np.arctan2(curr_y - prev_y, curr_x - prev_x)
            self.quat = quaternion_from_euler(0., 0., yaw)

            self.prev_pose.position.x = curr_x
            self.prev_pose.position.y = curr_y
            self.prev_pose.position.z = 0.0

        if self.quat is not None:
            self.curr_pose.orientation.x = self.quat[0]
            self.curr_pose.orientation.y = self.quat[1]
            self.curr_pose.orientation.z = self.quat[2]
            self.curr_pose.orientation.w = self.quat[3]

        self.curr_pose_pub.publish(self.curr_pose)
        
    def initialize_curr_pose(self):
        self.prev_pose.position.x = 0.0
        self.prev_pose.position.y = 0.0
        self.prev_pose.position.z = 0.0
        
        self.prev_pose.orientation.x = 0.0
        self.prev_pose.orientation.y = 0.0
        self.prev_pose.orientation.z = 0.0
        self.prev_pose.orientation.w = 0.0
        
        return InitializePoseResponse(True)

if __name__ == "__main__":
    rospy.init_node("coordinate")
    pub = TransformCoordinate()
    rospy.spin()
