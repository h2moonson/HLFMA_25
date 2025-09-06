#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import rospy
import rospkg
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32

class SlidingWindow:
    def __init__(self):
        self.cv_brdige = CvBridge()
        
        # --- 슬라이딩 윈도우 파라미터 ---
        self.nwindows = 12
        self.win_width = 100 // 2
        self.win_height = 0 # ROI 높이에 따라 동적으로 계산됨
        self.threshold = 0  # 윈도우 내 최소 픽셀 수 (동적 계산)

        # --- ROS 통신 설정 ---
        self.ipm_pub = rospy.Publisher('/ipm', Image, queue_size=1)
        self.lane_valid_pub = rospy.Publisher('/lane_valid', Int32, queue_size=1)
        self.sliding_window_pub = rospy.Publisher('/sliding_window', Image, queue_size=1)
        
        rospy.Subscriber('/lane_segment', Image, self.cam_callback)

        self.lane_valid = Int32()
        self.ipm = None
        self.crop = None

    def cam_callback(self, msg):
        img = self.cv_brdige.imgmsg_to_cv2(msg)
        img_h, img_w = img.shape[:2]

        src = np.array([
            [img_w * 0.25, img_h * 0.7],
            [img_w * 0.75, img_h * 0.7],
            [img_w * 1.0, img_h * 0.95],
            [img_w * 0.0, img_h * 0.95]
        ], dtype=np.float32)

        ipm_w, ipm_h = 600, 500
        dst = np.array([[0, 0], [ipm_w, 0], [ipm_w, ipm_h], [0, ipm_h]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        self.ipm = cv2.warpPerspective(img, M, (ipm_w, ipm_h), flags=cv2.INTER_LINEAR)

        debug_img = cv2.polylines(img.copy(), [src.astype(int)], True, (0, 255, 0), 2)
        cv2.imshow('Source Area', debug_img)

        crop = self.ipm
        self.crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        self.win_height = self.crop.shape[0] // self.nwindows
        self.threshold = int(self.win_width * 2 * self.win_height * 0.05)

        # [수정] 반환값이 centers 리스트로 변경됨
        left_valid_cnt, right_valid_cnt, left_centers, right_centers = self.sliding_window(crop)

        offset = 210 # 튜닝 필요
        valid_threshold = 8

        ## !! 수정된 부분: 사용할 윈도우 인덱스 지정 !! ##
        # 아래에서 4번째 윈도우를 사용하려면 인덱스 3을 선택 (0부터 시작)
        # 3번째 윈도우를 사용하려면 2로 변경
        target_window_idx = 0

        # 양쪽 차선 모두 인식
        if left_valid_cnt > valid_threshold and right_valid_cnt > valid_threshold and len(left_centers) > target_window_idx and len(right_centers) > target_window_idx:
            # [수정] 전체 평균 대신 특정 윈도우의 중심값 사용
            left_x = left_centers[target_window_idx]
            right_x = right_centers[target_window_idx]
            mid = (left_x + right_x) // 2
            self.lane_valid.data = mid * 2 + 1
            cv2.circle(self.crop, (mid, ipm_h - 10), 10, (0, 255, 0), -1)

        # 왼쪽 차선만 인식
        elif left_valid_cnt > valid_threshold and len(left_centers) > target_window_idx:
            # [수정] 전체 평균 대신 특정 윈도우의 중심값 사용
            left_x = left_centers[target_window_idx]
            mid = left_x + offset
            self.lane_valid.data = mid * 2 + 1 # 값은 0으로 유지
            cv2.circle(self.crop, (mid, ipm_h - 10), 10, (255, 0, 0), -1)

        # 오른쪽 차선만 인식
        elif right_valid_cnt > valid_threshold and len(right_centers) > target_window_idx:
            # [수정] 전체 평균 대신 특정 윈도우의 중심값 사용
            right_x = right_centers[target_window_idx]
            mid = right_x - offset
            self.lane_valid.data = mid * 2 + 1 # 값은 0으로 유지
            cv2.circle(self.crop, (mid, ipm_h - 10), 10, (0, 0, 255), -1)

        else:
            # 차선 미인식
            self.lane_valid.data = 0

        self.lane_valid_pub.publish(self.lane_valid)
        
        cv2.imshow('Sliding Window', self.crop)
        cv2.waitKey(1)
        rospy.loginfo('Lane Valid: %d | Left Cnt: %d | Right Cnt: %d', self.lane_valid.data, left_valid_cnt, right_valid_cnt)

    def sliding_window(self, img):
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        peak_threshold = 50 
        if histogram[left_base] < peak_threshold:
            left_base = -1 
        if histogram[right_base] < peak_threshold:
            right_base = -1

        left_current = left_base
        right_current = right_base

        left_valid_cnt, right_valid_cnt = 0, 0
        
        # [수정] 각 윈도우의 중심점과 모든 픽셀 인덱스를 저장할 리스트 생성
        left_centers, right_centers = [], []
        all_left_inds, all_right_inds = [], []

        for window in range(self.nwindows):
            if left_current != -1:
                win_y_low = img.shape[0] - (window + 1) * self.win_height
                win_y_high = img.shape[0] - window * self.win_height
                win_left_low  = left_current - self.win_width
                win_left_high = left_current + self.win_width
                
                cv2.rectangle(self.crop, (win_left_low, win_y_low), (win_left_high, win_y_high), (0, 255, 255), 2)

                good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                  (nonzero_x >= win_left_low) & (nonzero_x < win_left_high)).nonzero()[0]
                all_left_inds.append(good_left_inds)

                if len(good_left_inds) > self.threshold:
                    left_current = int(np.mean(nonzero_x[good_left_inds]))
                    left_centers.append(left_current) # [수정] 유효한 윈도우의 중심점 저장
                    left_valid_cnt += 1

            if right_current != -1:
                win_y_low = img.shape[0] - (window + 1) * self.win_height
                win_y_high = img.shape[0] - window * self.win_height
                win_right_low = right_current - self.win_width
                win_right_high = right_current + self.win_width

                cv2.rectangle(self.crop, (win_right_low, win_y_low), (win_right_high, win_y_high), (255, 255, 0), 2)
                
                good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                   (nonzero_x >= win_right_low) & (nonzero_x < win_right_high)).nonzero()[0]
                all_right_inds.append(good_right_inds)

                if len(good_right_inds) > self.threshold:
                    right_current = int(np.mean(nonzero_x[good_right_inds]))
                    right_centers.append(right_current) # [수정] 유효한 윈도우의 중심점 저장
                    right_valid_cnt += 1

        # 시각화를 위해 전체 픽셀 좌표는 계속 계산
        if len(all_left_inds) > 0:
            left_lane_inds = np.concatenate(all_left_inds)
            left_x, left_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
            if len(left_x) > 0:
                self.crop[left_y, left_x] = [255, 0, 0]

        if len(all_right_inds) > 0:
            right_lane_inds = np.concatenate(all_right_inds)
            right_x, right_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]
            if len(right_x) > 0:
                self.crop[right_y, right_x] = [0, 0, 255]

        # [수정] 반환값 변경
        return left_valid_cnt, right_valid_cnt, left_centers, right_centers

if __name__ == '__main__':
    rospy.init_node('sliding_window')
    node = SlidingWindow()
    rospy.spin()