#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Int32
from collections import deque
from morai_msgs.msg import EgoVehicleStatus

class ModeDeciderMorai:
    def __init__(self):
        rospy.init_node('mode_decider_morai', anonymous=True)

        self.window_size = rospy.get_param('~history_size', 50)
        self.rate_hz = rospy.get_param('~decide_rate', 10.0)
        self.gps_valid_ratio = 0.9
        self.lane_valid_ratio = 0.7

        self.mode = 'wait'
        self.lane_history = deque([0] * self.window_size, maxlen=self.window_size)
        self.gps_history = deque([0] * self.window_size, maxlen=self.window_size)

        rospy.Subscriber('/lane_valid', Int32, self.lane_cb)
        rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.gps_ok_callback)
        self.mode_pub = rospy.Publisher('/driving_mode', String, queue_size=1, latch=True)

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.decide_mode)
        rospy.loginfo("✅ Mode Decider for Morai is running.")
        # self.mode_pub.publish(String(data='gps'))

    def lane_cb(self, msg):

        valid = 1 if msg.data != 0 else 0

        self.lane_history.append(valid)

    def gps_ok_callback(self, msg):
        self.gps_history.append(1)
        # GPS가 불안정할 경우를 시뮬레이션 하려면 아래 주석을 해제
        # if len(self.gps_history) > 0: self.gps_history.popleft()

    def _get_ratio(self, history):
        if not history: return 0.0
        return sum(history) / float(len(history))

    def decide_mode(self, _):
        lane_ok = self._get_ratio(self.lane_history) >= self.lane_valid_ratio
        gps_ok = self._get_ratio(self.gps_history) >= self.gps_valid_ratio

        # 맨 처음에는 모드를 즉시 발행
        if self.mode == 'wait' and self._get_ratio(self.gps_history) > 0:
            self.mode = 'gps' # 기본 시작 모드
            rospy.loginfo("[ModeDecider] Initial Mode Set: {}".format(self.mode))
            self.mode_pub.publish(String(data=self.mode))
            return

        new_mode = self.mode

        rospy.loginfo("lane ok: {}".format(lane_ok))

    
        if lane_ok:
            new_mode = 'cam'
        else:
            if gps_ok:
                new_mode = 'gps'
            else:
                new_mode = 'lidar_only'
        
        if new_mode != self.mode:
            rospy.loginfo("[ModeDecider] Mode Changed: {} -> {}".format(self.mode, new_mode))
            self.mode = new_mode
            self.mode_pub.publish(String(data=self.mode))

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        ModeDeciderMorai().spin()
    except rospy.ROSInterruptException:
        pass