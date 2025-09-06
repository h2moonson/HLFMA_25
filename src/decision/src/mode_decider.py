#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import math
from std_msgs.msg import Bool, String, Int32
from sensor_msgs.msg import NavSatFix
from collections import deque

class ModeDecider:
    def __init__(self):
        rospy.init_node('mode_decider_node')

        # ───── 파라미터 ─────
        self.gps_rms_thresh     = rospy.get_param('~gps_rms_threshold', 0.03)  # meters (3 cm)
        self.window_size        = rospy.get_param('~history_size', 100)
        self.rate_hz            = rospy.get_param('~decide_rate', 10.0)
        self.gps_valid_ratio    = rospy.get_param('~gps_valid_ratio', 1.0)
        self.lane_valid_ratio   = rospy.get_param('~lane_valid_ratio', 1.0)

        # ───── 내부 상태 ─────
        self.mode = 'wait'
        self.lane_history = deque([0] * self.window_size, maxlen=self.window_size)
        self.gps_history = deque([0] * self.window_size, maxlen=self.window_size)

        # ───── ROS 통신 ─────
        rospy.Subscriber('/lane_valid', Int32, self.lane_cb)
        rospy.Subscriber('/ntrip_rops/ublox_gps/fix', NavSatFix, self.gps_cb)
        self.mode_pub = rospy.Publisher('/driving_mode', String, queue_size=1, latch=True)

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.decide_mode)
        rospy.loginfo("✅ Mode Decider for Morai is running.")

    def lane_cb(self, msg):
        digit = msg.data % 10
        valid = 1 if digit == 1 else 0  # 1: valid, 0: invalid
        self.lane_history.append(valid)

    def gps_cb(self, msg):
        # 공분산 배열에서 X, Y 방향의 분산을 추출
        cov_x = msg.position_covariance[0]  # pos_cov[0,0]
        cov_y = msg.position_covariance[4]  # pos_cov[1,1]
        rms = math.sqrt(cov_x + cov_y)
        valid = rms <= self.gps_rms_thresh
        self.gps_history.append(valid)

    def _ratio(self, history):
        if not history: #초기에 비어있을 때 0으로 리턴
            return 0.0
        return sum(history) / float(len(history))

    def decide_mode(self, _):
        lane_ok = self._ratio(self.lane_history) >= self.lane_valid_ratio
        gps_ok = self._ratio(self.gps_history) >= self.gps_valid_ratio

        # 맨 처음에는 모드를 즉시 발행
        if self.mode == 'wait' and self._ratio(self.gps_history) > 0:
            self.mode = 'gps' # 기본 시작 모드
            rospy.loginfo("[ModeDecider] Initial Mode Set: {}".format(self.mode))
            self.mode_pub.publish(String(data=self.mode))
            return

        new_mode = self.mode
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
        ModeDecider().spin()
    except rospy.ROSInterruptException:
        pass
