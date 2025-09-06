#!/usr/bin/env python3
import rospy
import math
import cv2
import numpy as np
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import splprep, splev  # 3차 보간에 사용

class GridSlidingWindow:
    def __init__(self):
        self.occ_grid = None
        self.occ_grid_img = None
        self.occ_grid_height = None  # 원본 occupancy grid의 height (100)
        self.occ_grid_width = None   # 원본 occupancy grid의 width (70)
        self.occ_grid_resolution = None
        self.origin_x = 0.0  # OccupancyGrid의 원점 x
        self.origin_y = 0.0  # OccupancyGrid의 원점 y

        self.window_width = 22 # 이전에는 16이었음
        self.window_height = 6
        self.min_pixel_threshold = 1  # 이전에는 1이었음
        self.max_empty_windows = 4

        self.local_path_pub = rospy.Publisher("local_path", Path, queue_size=1)
        rospy.Subscriber("occupancy_grid", OccupancyGrid, self.occupancy_grid_callback)

    def convert_processed_to_original(self, point):
        x_proc, y_proc = point
        x_orig = self.occ_grid_width - 1 - y_proc
        y_orig = self.occ_grid_height - 1 - x_proc
        return (x_orig, y_orig)

    def occupancy_grid_callback(self, msg):
            self.occ_grid = msg 
            self.occ_grid_resolution = msg.info.resolution
            self.occ_grid_height = msg.info.height
            self.occ_grid_width = msg.info.width
            self.origin_x = msg.info.origin.position.x
            self.origin_y = msg.info.origin.position.y

            lane_side = ""
            lane_points = None
            right_orig = None
            left_orig = None
            forced_point = None
            interpolated_path = Path()
            invasion_threshold = self.occ_grid_height // 2 
            
            # 회전된 이미지 좌표계에서 lane point 검출
            self.occ_grid_img, self.left_lane_points, self.right_lane_points, last_left_point, last_right_point = self.process_occupancy_grid(self.occ_grid)

            # smoothing 제거: 단순히 현재 프레임의 lane point 사용
            smoothed_left = self.left_lane_points
            smoothed_right = self.right_lane_points

            # 두 lane 모두 없으면 그냥 리턴
            if len(smoothed_left) == 0 and len(smoothed_right) == 0:
                return

            # XOR 조건 : 한쪽 영역만 valid 한 경우
            if (len(smoothed_left) == 0) != (len(smoothed_right) == 0):
                # 왼쪽 lane이 없을 때
                if len(smoothed_left) == 0:
                    # 만약 left_count가 0이면 직선(오른쪽만 존재)으로 가정
                    if last_right_point == None or last_right_point[0] > invasion_threshold:
                        lane_points = np.array([self.convert_processed_to_original(pt) for pt in smoothed_right])
                        print("shift (왼쪽 없음)") 
                        lane_side = 'right'
                    else:
                        # left_count가 0이 아니면, 급격한 좌회전 상황으로 판단
                        print("급격한 좌회전")
                        forced_point = (self.occ_grid_height / 5, self.occ_grid_width - 1)
                        forced_point = self.convert_processed_to_original(forced_point)

                        right_orig = np.array([self.convert_processed_to_original(pt) for pt in smoothed_right])
                        right_orig = right_orig[right_orig[:, 0].argsort()]

                        left_orig = np.array([forced_point])
                        lane_side = 'abnormal'

                # 오른쪽 lane이 없을 때
                elif len(smoothed_right) == 0:
                    if last_left_point == None or last_left_point[0] < invasion_threshold:
                        lane_points = np.array([self.convert_processed_to_original(pt) for pt in smoothed_left])
                        print("shift (오른쪽 없음)")
                        lane_side = 'left'
                    else:
                        print("급격한 우회전")
                        forced_point = (self.occ_grid_height * (4/5), self.occ_grid_width - 1)
                        forced_point = self.convert_processed_to_original(forced_point)

                        left_orig = np.array([self.convert_processed_to_original(pt) for pt in smoothed_left])
                        left_orig = left_orig[left_orig[:, 0].argsort()]

                        right_orig = np.array([forced_point])
                        lane_side = 'abnormal'

                # straight-line 경우: lane_points가 할당된 상태
                if lane_side != 'abnormal':
                    # lane_points가 numpy array임을 보장
                    if not isinstance(lane_points, np.ndarray):
                        lane_points = np.array(lane_points)
                    lane_points = lane_points[lane_points[:, 0].argsort()]
                    interpolated_path = self.make_shifted_path(lane_points, lane_side)
                else:
                    interpolated_path = self.make_center_path(left_orig, right_orig)
            else:
                # 양쪽 lane point가 모두 존재하는 경우 기존 로직 사용
                left_orig = np.array([self.convert_processed_to_original(pt) for pt in smoothed_left])
                right_orig = np.array([self.convert_processed_to_original(pt) for pt in smoothed_right])
                left_orig = left_orig[left_orig[:, 0].argsort()]
                right_orig = right_orig[right_orig[:, 0].argsort()]
                interpolated_path = self.make_center_path(left_orig, right_orig)

            # [수정 완료] 경로 시작점을 차량 전방 0.3m로 강제하는 로직
            if interpolated_path.poses: # 생성된 경로가 비어있지 않다면
                # 경로의 첫 번째 포인트의 좌표를 차량 기준(velodyne) 0.3m 앞으로 설정
                interpolated_path.poses[0].pose.position.x = 0.3
                interpolated_path.poses[0].pose.position.y = 0.0
                interpolated_path.poses[0].pose.position.z = 0.0 # z축은 0으로 설정

            self.local_path_pub.publish(interpolated_path)

    def make_shifted_path(self, lane_points, lane_side):
        # 보간 및 접선 계산
        if len(lane_points) >= 3:
            tck, u = splprep([lane_points[:, 0], lane_points[:, 1]], s=1, k=2)
            u_new = np.linspace(0, 1, num=100)
            x_new, y_new = splev(u_new, tck)
            if len(x_new) >= 2:
                dx_new = np.gradient(x_new)
                dy_new = np.gradient(y_new)
            else:
                dx_new = [0] * len(x_new)
                dy_new = [0] * len(y_new)
        else:
            x_new, y_new = lane_points[:, 0], lane_points[:, 1]
            if len(x_new) >= 2:
                dx_new = np.gradient(x_new)
                dy_new = np.gradient(y_new)
            else:
                dx_new = [0] * len(x_new)
                dy_new = [0] * len(y_new)

        # 법선 오프셋 적용: 오른쪽 lane만 있으면 왼쪽으로, 왼쪽 lane만 있으면 오른쪽으로 이동
        offset_distance = 15.0  # 필요에 따라 조정
        x_new_shifted, y_new_shifted = [], []
        for x_val, y_val, dx, dy in zip(x_new, y_new, dx_new, dy_new):
            norm = np.sqrt(dx**2 + dy**2)
            if norm == 0:
                nx, ny = 0, 0
            else:
                if lane_side == 'right':
                    nx, ny = -dy/norm, dx/norm
                else:
                    nx, ny = dy/norm, -dx/norm
            x_new_shifted.append(x_val + offset_distance * nx)
            y_new_shifted.append(y_val + offset_distance * ny)

        x_new, y_new = x_new_shifted, y_new_shifted

        # 최종 경로 메시지 생성 (grid index → 실제 좌표 변환)
        interpolated_path = Path()
        interpolated_path.header.frame_id = 'map'
        interpolated_path.header.stamp = rospy.Time.now()
        for x, y in zip(x_new, y_new):
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.pose.position.x = self.origin_x + x * self.occ_grid_resolution
            pose_stamped.pose.position.y = self.origin_y + y * self.occ_grid_resolution
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation = Quaternion(0, 0, 0, 1)
            interpolated_path.poses.append(pose_stamped)
        return interpolated_path

    def make_center_path(self, left_orig, right_orig): 
        raw_waypoints = []
        left_idx, right_idx = 0, 0
        while left_idx < len(left_orig) and right_idx < len(right_orig):
            avg_x = (left_orig[left_idx][0] + right_orig[right_idx][0]) / 2
            avg_y = (left_orig[left_idx][1] + right_orig[right_idx][1]) / 2
            x = self.origin_x + avg_x * self.occ_grid_resolution 
            y = self.origin_y + avg_y * self.occ_grid_resolution
            raw_waypoints.append((x, y))
            left_idx += 1
            right_idx += 1

        while left_idx < len(left_orig):
            avg_x = (left_orig[left_idx][0] + right_orig[-1][0]) / 2
            avg_y = (left_orig[left_idx][1] + right_orig[-1][1]) / 2
            x = self.origin_x + avg_x * self.occ_grid_resolution 
            y = self.origin_y + avg_y * self.occ_grid_resolution
            raw_waypoints.append((x, y))
            left_idx += 1

        while right_idx < len(right_orig):
            avg_x = (left_orig[-1][0] + right_orig[right_idx][0]) / 2
            avg_y = (left_orig[-1][1] + right_orig[right_idx][1]) / 2
            x = self.origin_x + avg_x * self.occ_grid_resolution 
            y = self.origin_y + avg_y * self.occ_grid_resolution
            raw_waypoints.append((x, y))
            right_idx += 1

        raw_waypoints = np.array(raw_waypoints)
        if len(raw_waypoints) >= 3:
            tck, u = splprep([raw_waypoints[:, 0], raw_waypoints[:, 1]], s=1, k=2)
            u_new = np.linspace(0, 1, num=100)
            x_new, y_new = splev(u_new, tck)
        else:
            x_new, y_new = raw_waypoints[:, 0], raw_waypoints[:, 1]

        interpolated_path = Path()
        interpolated_path.header.frame_id = 'map'
        interpolated_path.header.stamp = rospy.Time.now()
        for x, y in zip(x_new, y_new):
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation = Quaternion(0, 0, 0, 1)
            interpolated_path.poses.append(pose_stamped)

        return interpolated_path   
    

    def sliding_window_lane_detection_region(self, img, region, window_width, window_height, min_pixel_threshold):
        """
        img: 전체 단일 채널 이미지 (CV_8U)
        region: 'left' 또는 'right'
        """
        height, width = img.shape
        # 좌우 영역의 x 범위 결정
        if region == 'left':
            x_start, x_end = 0, width // 2
        else:  # 'right'
            x_start, x_end = width // 2, width

        # 해당 영역 내에서 시작점(current_x) 계산
        region_img = img[:, x_start:x_end]
        ret, binary = cv2.threshold(region_img, 127, 255, cv2.THRESH_BINARY_INV)

        occupied_count = cv2.countNonZero(binary)
        nonzero_region = cv2.findNonZero(binary)
        if nonzero_region is None:
            return [], cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None  # None으로 처리
        nonzero_region = nonzero_region.reshape(-1, 2)
        
        start_y = height - 1 - window_height
        # 영역 내에서 bottom_indices 계산 (여기서는 해당 영역만 사용)
        bottom_indices = np.where(nonzero_region[:, 1] >= start_y - 25)[0]
        if len(bottom_indices) > 0:
            region_current_x = int(np.mean(nonzero_region[bottom_indices, 0]))
            current_x = region_current_x + x_start  # 전체 이미지 좌표로 변환
        else:
            current_x = (x_start + x_end) // 2

        # 전체 이미지에서 min_y_occupied를 계산 (좌우 동일한 기준)
        ret_all, binary_all = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        nonzero_all = cv2.findNonZero(binary_all)
        if nonzero_all is None:
            min_y_occupied = 0
        else:
            nonzero_all = nonzero_all.reshape(-1, 2)
            min_y_occupied = int(np.min(nonzero_all[:, 1]))

        out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lane_points = []
        current_y = start_y
        consecutive_empty = 0

        # 마지막 window 좌표 저장 변수 초기화
        last_window_coord = None

        while True:
            if current_y - window_height < min_y_occupied or current_y < 0:
                break

            win_y_low = current_y - window_height
            win_y_high = current_y
            win_x_low = max(0, current_x - window_width // 2)
            win_x_high = min(width, current_x + window_width // 2)

            window_img = img[win_y_low:win_y_high, win_x_low:win_x_high]
            ret, window_binary = cv2.threshold(window_img, 127, 255, cv2.THRESH_BINARY_INV)
            nonzero_window = cv2.findNonZero(window_binary)

            if nonzero_window is not None and len(nonzero_window) >= min_pixel_threshold:
                nonzero_window = nonzero_window.reshape(-1, 2)
                current_x = win_x_low + int(np.mean(nonzero_window[:, 0]))
                consecutive_empty = 0
                color = (0, 255, 0)
                center_y = (win_y_low + win_y_high) // 2
                lane_points.append(((current_x, center_y), "n"))
                last_window_coord = (current_x, center_y)
            else:
                consecutive_empty += 1
                color = (0, 0, 255)
                lane_points.append(((current_x, current_y), "e"))
                if consecutive_empty >= self.max_empty_windows:
                    # max empty가 연속되면 탈출 (빈 window들도 저장)
                    last_window_coord = (current_x, current_y)
                    break

            cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), color, 2)
            current_y -= window_height

        # lane_points를 y좌표 기준 오름차순 정렬
        lane_points_sorted = sorted(lane_points, key=lambda p: p[0][1])

        # non-empty("n")인 구간의 양 끝 추출
        first_n_idx = None
        last_n_idx = None
        for i, (pt, flag) in enumerate(lane_points_sorted):
            if flag == "n":
                first_n_idx = i
                break

        for i in range(len(lane_points_sorted) - 1, -1, -1):
            if lane_points_sorted[i][1] == "n":
                last_n_idx = i
                break

        if first_n_idx is not None and last_n_idx is not None:
            lane_points_final = lane_points_sorted[first_n_idx:last_n_idx + 1]
        else:
            lane_points_final = []

        # 좌표만 추출
        lane_points_final = [pt for pt, flag in lane_points_final]

        # occupied_count 대신 마지막 sliding window 좌표 반환
        return lane_points_final, out_img, last_window_coord

    def process_occupancy_grid(self, msg: OccupancyGrid):
        grid_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        img = 255 - (grid_data * 255 // 100).astype(np.uint8)
        grid_data_flipped = img[::-1]
        processed_img = np.transpose(grid_data_flipped)[::-1]

        # 왼쪽과 오른쪽 영역에서 lane point 검출
        left_lane_points, left_vis, last_left_point = self.sliding_window_lane_detection_region(processed_img, 'left',
                                                                                self.window_width,
                                                                                self.window_height,
                                                                                self.min_pixel_threshold)
        right_lane_points, right_vis, last_right_point = self.sliding_window_lane_detection_region(processed_img, 'right',
                                                                                  self.window_width,
                                                                                  self.window_height,
                                                                                  self.min_pixel_threshold)
        combined_vis = cv2.addWeighted(left_vis, 0.5, right_vis, 0.5, 0)
        expanded_img = cv2.resize(combined_vis, (350, 500), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("Occupancy Map with ROI and Sliding Windows", expanded_img)
        # cv2.waitKey(1)
    
        return processed_img, left_lane_points, right_lane_points, last_left_point, last_right_point

if __name__ == '__main__':
    rospy.init_node("grid_sliding_window", anonymous=True)
    grid_sliding_window = GridSlidingWindow()
    rospy.spin()
