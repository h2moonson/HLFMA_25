#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[실제 차량용 최종 버전]
실제 차량의 GPS(WGS84)와 인지 정보를 바탕으로 자율주행을 수행합니다.
장애물 정보는 라이다 기준 로컬 좌표로 가정하고 올바르게 변환합니다.
제어 명령은 아두이노가 수신할 수 있는 Twist 메시지 형태로 발행합니다.
"""

import rospy
import numpy as np
import math

from nav_msgs.msg import Path
from std_msgs.msg import String, Int32, Int64
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped, Vector3, Twist
from visualization_msgs.msg import Marker, MarkerArray
from pyproj import Proj
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# utils.py는 실제 차량용으로 별도 관리하거나, 경로가 같다면 공용으로 사용합니다.
from utils import PurePursuit, PidController, PathReader
from utils import compute_s_and_yaw, cartesian_to_frenet, frenet_to_cartesian, generate_quintic_path

# 라이다 인지 노드가 발행하는 메시지 타입 (실제 환경에 맞게 수정)
from lidar_object_detection.msg import ObjectInfo


class EgoStatus:
    """차량의 현재 상태(위치, 헤딩, 속도)를 저장하는 데이터 클래스"""
    def __init__(self):
        self.position = Vector3()
        self.heading = 0.0
        self.velocity = 0.0 # km/h

class AutonomousDriver:
    """자율주행의 모든 판단과 제어를 총괄하는 메인 클래스"""
    def __init__(self):
        rospy.init_node('real_vehicle_driver_node', anonymous=True)

        self.on_dynamic_obs = 0

        # Publisher
        self.cmd_vel_publisher = rospy.Publisher('/teleop_cmd_vel', Twist, queue_size=1)
        self.ego_marker_publisher = rospy.Publisher('/ego_marker', Marker, queue_size=1)
        self.target_point_publisher = rospy.Publisher('/pure_pursuit_target_point', Marker, queue_size=1)
        self.global_path_publisher = rospy.Publisher('/global_path', Path, queue_size=1) 

        self.local_path_publisher = rospy.Publisher('/local_path_viz', Path, queue_size=1)
        self.local_path_publisher = rospy.Publisher('/chk_path_viz', Path, queue_size=1)
        self.obst_marker_publisher = rospy.Publisher('/obst_marker_viz', MarkerArray, queue_size=1)
        
        # Subscriber
        rospy.Subscriber("/ntrip_rops/ublox_gps/fix", NavSatFix, self.gps_callback)
        rospy.Subscriber("/lane_valid", Int32, self.lane_error_callback)
        rospy.Subscriber("/driving_mode", String, self.driving_mode_callback)
        rospy.Subscriber("/obstacle_info", ObjectInfo, self.obstacle_callback)
        rospy.Subscriber("/local_path", Path, self.local_path_callback)

        # [핵심] WGS84 -> UTM-52N 좌표계 변환기
        self.wgs84_to_utm52n = Proj(proj='utm', zone=52, ellps='WGS84', preserve_units=False)
        
        # [중요] 차량 중심(base_link)에서 라이다 센서까지의 오프셋 (x:앞쪽+, y:왼쪽+)
        self.lidar_offset_x = 0.8  # 값은 실제 차량에 맞게 측정 후 수정해야 함

        # 경로 파일의 기준이 되는 원점 (UTM-52N)
        self.map_origin = [249947.5672, 3688367.483] # 예: jeju_cw_origin

        # 클래스 멤버 변수 초기화
        self.status = EgoStatus()
        self.current_mode = 'wait'
        self.is_gps_received = False
        self.is_avoiding = False
        self.avoidance_path = Path()
        self.current_waypoint = 0

        # 속도 추정을 위한 변수
        self.prev_gps_time = None
        self.prev_utm_pos = None

        # 제어 파라미터
        self.max_velocity_kph = 10.0
        self.min_velocity_kph = 2.0
        self.lane_steering_deg = 0.0
        self.gps_steering_deg = 0.0
        self.lane_steering_gain_1 = rospy.get_param('~lane_k1', 2.0e-3)
        self.lane_steering_gain_2 = rospy.get_param('~lane_k2', 1.0e-5)
        
        # 제어기 및 경로 객체 생성
        self.pure_pursuit_controller = PurePursuit()
        self.pid_controller = PidController()
        path_reader = PathReader('decision', self.map_origin)
        self.global_path = path_reader.read_txt("clockwise.txt")
        # self.global_path = path_reader.read_txt("counter_clockwise.txt")
        
        # 메인 루프
        rate = rospy.Rate(40)
        while not rospy.is_shutdown():
            if not self.is_gps_received or self.current_mode == 'wait':
                rate.sleep()
                continue
            
            # 1. 모드에 따라 주행할 local_path 결정
            local_path = Path()
            if self.current_mode == 'lidar_only':
                local_path = self.lidar_path

            elif self.current_mode == 'gps':
                path_to_follow = self.global_path
                if self.is_avoiding:
                    if self.is_avoidance_complete():
                        self.is_avoiding = False

                    else:
                        path_to_follow = self.avoidance_path
                
                self.current_waypoint, local_path = self.pure_pursuit_controller.findLocalPath(path_to_follow, self.status)
                self.local_path_publisher.publish(local_path)

            # 2. 경로 유효성 검사 (cam 모드는 경로 기반이 아니므로 제외)
            if len(local_path.poses) < 2 and (self.current_mode != 'cam' or self.is_avoiding):
                self.is_avoiding = False
                self.publish_arduino_command(0.0, 0.0, 1.0)
                rate.sleep()
                continue
            
            # 3. [핵심 수정] 모든 경로 기반 제어의 공통 전처리
            #    - getPath()를 먼저 호출해서 제어기에 경로를 설정해야 함
            if self.current_mode in ['gps', 'lidar_only']:
                self.pure_pursuit_controller.getPath(local_path)
                self.pure_pursuit_controller.getEgoStatus(self.status)

            # 4. 모드에 따라 세부 제어값 계산
            target_velocity_kph = self.min_velocity_kph
            path_based_steering_deg = 0.0
            
            if self.current_mode == 'gps':
                # getPath() 이후에 호출되어야 안전함
                curvature_info = self.pure_pursuit_controller.estimateCurvature()
                if curvature_info:
                    corner_theta_degree, _, _ = curvature_info
                    target_velocity_kph = self.corner_speed_controller(corner_theta_degree)
                
                path_based_steering_deg, target_x, target_y, _ = self.pure_pursuit_controller.steeringAngle(0)
                self.visualize_target_point(target_x, target_y)

            elif self.current_mode == 'lidar_only':
                target_velocity_kph = 5.0
                path_based_steering_deg, target_x, target_y, _ = self.pure_pursuit_controller.steeringAngle(0)
                self.visualize_target_point(target_x, target_y)
            
            # 5. 최종 제어값 선택
            final_steering_degree = 0.0
            if self.current_mode == 'cam':
                final_steering_degree = self.lane_steering_deg
                target_velocity_kph = 4.0 - self.on_dynamic_obs * 4.0
            elif self.current_mode in ['gps', 'lidar_only']:
                final_steering_degree = path_based_steering_deg
            else: # 'wait' 또는 알 수 없는 모드
                self.publish_arduino_command(0.0, 0.0, 1.0)
                rate.sleep()
                continue
            
            # 6. PID 및 최종 명령 발행
            pid_output = self.pid_controller.pid(target_velocity_kph, self.status.velocity)
            brake_cmd = -pid_output if pid_output < 0 else 0.0
            self.publish_arduino_command(target_velocity_kph, final_steering_degree, brake_cmd)

            # 7. 시각화
            self.visualize_ego_marker()
            self.global_path_publisher.publish(self.global_path)
            rate.sleep()

    def gps_callback(self, msg):
        if msg.status.status < 0: # GPS 수신 불량
            self.is_gps_received = False
            return

        utm_x, utm_y = self.wgs84_to_utm52n(msg.longitude, msg.latitude)
        current_time = msg.header.stamp.to_sec()

        if self.prev_utm_pos is not None and self.prev_gps_time is not None:
            dt = current_time - self.prev_gps_time
            if dt > 0.01:
                dx, dy = utm_x - self.prev_utm_pos[0], utm_y - self.prev_utm_pos[1]
                distance = np.hypot(dx, dy)
                
                velocity_ms = distance / dt
                self.status.velocity = velocity_ms * 3.6

                if distance > 0.2:
                    self.status.heading = np.degrees(np.arctan2(dy, dx))
        
        self.status.position.x = utm_x - self.map_origin[0]
        self.status.position.y = utm_y - self.map_origin[1]
        
        self.prev_utm_pos, self.prev_gps_time = (utm_x, utm_y), current_time
        self.is_gps_received = True

    def obstacle_callback(self, msg):
        if self.is_avoiding: return

        # [수정] 'cam' 모드의 장애물 처리 로직
        if self.current_mode == 'cam':
            if msg.objectCounts == 0:
                self.on_dynamic_obs = 0
                return
            
            obs_local_x, obs_local_y = np.array([[msg.centerX[i], msg.centerY[i]] for i in range(msg.objectCounts)]).T

            # TODO : lower_x, upper_x, lower_y, upper_y 값 꼭 바꾸기
            # 지금은 center 점을 가지고 겹치는 지 테스트를 하는데, 이 bbox 8개의 꼭짓점이 이 범위 안에 들어오는지 체크하는 것도 괜찮을듯
            lower_x, upper_x, lower_y, upper_y = 0., 3., -1., 1.
            ind = ((lower_x < obs_local_x) & (obs_local_x < upper_x)) & ((lower_y < obs_local_y) & (obs_local_y < upper_y))
            
            self.on_dynamic_obs = int(np.any(ind))
            if self.on_dynamic_obs == 1:
                rospy.logwarn("[Obstacle] CAM Mode: Obstacle in ROI. Stopping.")

        elif self.current_mode == 'gps':
            is_path_blocked, blocked_idx = self.check_path_collision(msg, self.global_path)
            if is_path_blocked:
                rospy.logwarn("[Obstacle] GPS Mode: Path blocked at index {}. Generating avoidance path.".format(blocked_idx))
                self.generate_avoidance_path(blocked_idx)

    def check_path_collision(self, obstacle_msg, ref_path):
        VEHICLE_RADIUS, ROI_DISTANCE = 0.7, 15.0
        ego_x, ego_y, ego_yaw_rad = self.status.position.x, self.status.position.y, np.deg2rad(self.status.heading)
        
        # 현재 차량의 위치(waypoint)를 기준으로 검사 범위를 설정합니다.
        start_idx = self.current_waypoint
        end_idx = min(start_idx + len(ref_path.poses) - 1, start_idx + int(ROI_DISTANCE / 0.2))

        viz_path = Path()
        viz_path.header.frame_id = 'map'
        viz_path.header.stamp = rospy.Time.now()
        for i in range(start_idx, end_idx):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = viz_path.header.stamp
            pose.pose.position.x = ref_path.poses[i].pose.position.x
            pose.pose.position.y = ref_path.poses[i].pose.position.y
            viz_path.poses.append(pose)
        self.local_path_publisher.publish(viz_path)

        markers = MarkerArray()

        # [수정] 'gps' 모드의 장애물 처리 로직
        # .markers 대신 .objectCounts 만큼 반복합니다.
        for obs_idx in range(obstacle_msg.objectCounts):
            # .centerX, .centerY로 장애물 위치를 가져옵니다.
            obs_local_x = obstacle_msg.centerX[obs_idx] + self.lidar_offset_x
            obs_local_y = obstacle_msg.centerY[obs_idx]

            # 라이다 좌표계 기준의 장애물 위치를 맵 좌표계로 변환합니다.
            delta_x = obs_local_x * np.cos(ego_yaw_rad) - obs_local_y * np.sin(ego_yaw_rad)
            delta_y = obs_local_x * np.sin(ego_yaw_rad) + obs_local_y * np.cos(ego_yaw_rad)
            obs_global_x = ego_x + delta_x
            obs_global_y = ego_y + delta_y

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = rospy.Time.now()
            marker.pose.position.x = obs_global_x
            marker.pose.position.y = obs_global_y
            marker.pose.position.z = 1.

            marker.type = Marker.CUBE

            quat = quaternion_from_euler(0, 0, ego_yaw_rad)
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            
            marker.scale.x = obstacle_msg.lengthX[obs_idx]
            marker.scale.y = obstacle_msg.lengthY[obs_idx]
            marker.scale.z = obstacle_msg.lengthZ[obs_idx]
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = .5

            markers.markers.append(marker)
            self.obst_marker_publisher.publish(markers)

            # .scale 대신 .lengthX, .lengthY로 장애물 크기를 가져와 반지름을 계산합니다.
            obs_radius = max(obstacle_msg.lengthX[obs_idx], obstacle_msg.lengthY[obs_idx]) / 2.0

            for i in range(start_idx, end_idx):
                path_point = ref_path.poses[i].pose.position
                
                # 경로점과 장애물 사이의 거리가 충돌 반경보다 작으면 충돌로 판단합니다.
                if np.hypot(path_point.x - obs_global_x, path_point.y - obs_global_y) < (VEHICLE_RADIUS + obs_radius):
                    return True, i # 충돌 발생, 해당 경로 인덱스 반환
                    
        return False, -1 # 충돌 없음

    def generate_avoidance_path(self, blocked_idx):
        try:
            ref_path = self.global_path
            s_list, yaw_list = compute_s_and_yaw(ref_path)
            s0, l0 = cartesian_to_frenet(self.status.position.x, self.status.position.y, ref_path, s_list)
            AVOIDANCE_LENGTH, LATERAL_OFFSET = 0.0, 2.0
            target_s, target_l = s_list[blocked_idx] + AVOIDANCE_LENGTH, LATERAL_OFFSET
            l_path_func = generate_quintic_path(s0, l0, target_s, target_l)
            new_path = Path(); new_path.header.frame_id = 'map'
            s_points = np.linspace(s0, target_s, int((target_s - s0) / 0.2))
            for s in s_points:
                l = l_path_func(s)
                x, y, yaw = frenet_to_cartesian(s, l, ref_path, s_list, yaw_list)
                pose = PoseStamped(); pose.header.frame_id = 'map'
                pose.pose.position.x, pose.pose.position.y = x, y
                q = quaternion_from_euler(0, 0, yaw)
                pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
                new_path.poses.append(pose)
            self.avoidance_path, self.is_avoiding = new_path, True
        except Exception as e:
            rospy.logerr(f"Failed to generate avoidance path: {e}")
            self.is_avoiding = False
    
    def driving_mode_callback(self, msg):
        self.current_mode = msg.data
        if self.current_mode != 'gps': self.is_avoiding = False

    def lane_error_callback(self, msg):
        PIXEL_ERROR = msg.data // 2 - 320
        self.lane_steering_deg = np.rad2deg(self.lane_steering_gain_1 * PIXEL_ERROR + self.lane_steering_gain_2 * PIXEL_ERROR * abs(PIXEL_ERROR))

    def local_path_callback(self, msg):
        """'수동 TF 변환'을 통해 /local_path를 'map' 기준 경로로 만듭니다."""
        if not self.is_status_received:
            return

        transformed_path = Path()
        transformed_path.header.frame_id = 'map'
        
        ego_yaw_rad = math.radians(self.status.heading)
        cos_yaw = math.cos(ego_yaw_rad)
        sin_yaw = math.sin(ego_yaw_rad)

        for pose in msg.poses:
            p_v = pose.pose.position
            p_b_x = p_v.x + self.lidar_offset_x
            p_b_y = p_v.y
            
            p_m_x = self.status.position.x + p_b_x * cos_yaw - p_b_y * sin_yaw
            p_m_y = self.status.position.y + p_b_x * sin_yaw + p_b_y * cos_yaw
            
            new_pose = PoseStamped()
            new_pose.header.frame_id = 'map'
            new_pose.pose.position.x = p_m_x
            new_pose.pose.position.y = p_m_y
            new_pose.pose.position.z = self.status.position.z
            new_pose.pose.orientation = pose.pose.orientation
            
            transformed_path.poses.append(new_pose)

        self.lidar_path = transformed_path 
    
    def publish_arduino_command(self, target_vel_kph, steering_deg, brake_val):
        twist_msg = Twist()
        if brake_val > 0.1:
            twist_msg.linear.x = 0.0
        else:
            scaled_velocity = (target_vel_kph / self.max_velocity_kph) * 255.0
            twist_msg.linear.x = np.clip(scaled_velocity, 0, 255)
        
        # 아두이노가 받는 값 = angular.z * 0.2 라고 가정하면, angular.z = steer_angle / 0.2
        # steer_angle은 라디안 단위여야 함.
        steering_rad = np.deg2rad(steering_deg)
        twist_msg.angular.z = steering_rad / 0.2 # 이 값은 실제 아두이노 코드에 맞춰 튜닝 필요
        self.cmd_vel_publisher.publish(twist_msg)

    def corner_speed_controller(self, corner_theta_degree):
        corner_theta_degree = min(corner_theta_degree, 30)
        target_vel = -0.5 * corner_theta_degree + 15
        return np.clip(target_vel, self.min_velocity_kph, self.max_velocity_kph)
        
    def is_avoidance_complete(self):
        if not self.avoidance_path.poses:
            return False
        dist_to_end = np.hypot(
            self.status.position.x - self.avoidance_path.poses[-1].pose.position.x,
            self.status.position.y - self.avoidance_path.poses[-1].pose.position.y
        )
        if dist_to_end < 1.5:
            rospy.loginfo("[Main Loop] Avoidance complete. Returning to global path.")
            return True
        return False

    def visualize_target_point(self, x, y):
        """Pure Pursuit 목표점을 vkfkstor 구 마커로 발행합니다."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        self.target_point_publisher.publish(marker)

    def visualize_ego_marker(self):
        """'map' 좌표계 기준의 현재 차량 위치를 파란색 구 마커로 발행합니다."""
        if not self.is_status_received:
            return
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.8
        marker.scale.y = 0.8
        marker.scale.z = 0.8
        marker.color.a = 1.0
        marker.color.b = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.status.position.x
        marker.pose.position.y = self.status.position.y
        self.ego_marker_publisher.publish(marker)

if __name__ == '__main__':
    try:
        AutonomousDriver()
    except rospy.ROSInterruptException:
        pass