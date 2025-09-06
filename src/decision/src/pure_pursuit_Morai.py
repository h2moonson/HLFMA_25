#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math

from nav_msgs.msg import Path
from std_msgs.msg import String, Int32, Int64
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus
from geometry_msgs.msg import PoseStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from pyproj import Transformer, CRS
from lidar_object_detection.msg import ObjectInfo
from utils_morai import PurePursuit, PidController, PathReader
from utils_morai import compute_s_and_yaw, cartesian_to_frenet, frenet_to_cartesian, generate_quintic_path
from tf.transformations import quaternion_matrix, quaternion_from_euler

class EgoStatus(object):
    """차량의 현재 상태(위치, 헤딩, 속도)를 저장하는 데이터 클래스"""
    def __init__(self):
        self.position = Vector3()
        self.heading = 0.0
        self.velocity = 0.0

class AutonomousDriver(object):
    def __init__(self):
        rospy.init_node('morai_driver_node', anonymous=True)

        self.on_dynamic_obs = 0

        # ----------------------------------------------------------------
        # 1. Publisher 선언
        # ----------------------------------------------------------------
        self.control_publisher = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.ego_marker_publisher = rospy.Publisher('/ego_marker', Marker, queue_size=1)
        self.target_point_publisher = rospy.Publisher('/pure_pursuit_target_point', Marker, queue_size=1)
        self.global_path_publisher = rospy.Publisher('/global_path', Path, queue_size=1)
        self.local_path_publisher = rospy.Publisher('/local_path_viz', Path, queue_size=1)

        self.chk_path_publisher = rospy.Publisher('/chk_path_viz', Path, queue_size=1)
        self.obst_marker_publisher = rospy.Publisher('/obst_marker_viz', MarkerArray, queue_size=1)

        # ----------------------------------------------------------------
        # 2. [수정] 모든 변수 및 객체 우선 초기화
        # Subscriber를 선언하기 전에 콜백 함수에서 사용할 모든 것을 정의합니다.
        # ----------------------------------------------------------------
        
        # 좌표 변환 및 파라미터 설정
        self.morai_offset = [302459.942, 4122635.537]
        proj_UTM52N = CRS('EPSG:32652')
        proj_UTMK = CRS('EPSG:5179')
        self.utm52n_to_utmk_transformer = Transformer.from_crs(proj_UTM52N, proj_UTMK, always_xy=True)
        self.map_origin = [935718.8406, 1916164.8082]
        # self.heading_offset_deg = 0.0
        self.lidar_offset_x = 0.8 # 실제 마운트 위치는 0.8
        
        # 주행 경로 읽기 (global_path 생성)
        path_reader = PathReader('decision', self.map_origin)
        self.global_path = path_reader.read_txt("test2.txt")

        # 클래스 멤버 변수(상태 변수) 초기화
        self.status = EgoStatus()
        self.current_mode = 'wait'
        self.is_status_received = False
        self.is_avoiding = False
        self.avoidance_path = Path()
        self.current_waypoint = 0
        self.lidar_path = Path()
        self.lane_steering_deg = 0.0
        self.lane_steering_gain_1 = rospy.get_param('~lane_k1', 0.0005)
        self.lane_steering_gain_2 = rospy.get_param('~lane_k2', 0.0001)
        
        # 주행 속도 파라미터
        self.min_velocity_kph = 2.0
        self.max_velocity_kph = 10.0
        
        # 제어기 및 메시지 객체 생성
        self.ctrl_cmd = CtrlCmd()
        self.ctrl_cmd.longlCmdType = 2
        self.pure_pursuit_controller = PurePursuit()
        self.pid_controller = PidController()

        # ----------------------------------------------------------------
        # 3. Subscriber 선언
        # 모든 변수가 준비된 후, Subscriber를 선언하여 콜백이 안전하게 호출되도록 합니다.
        # ----------------------------------------------------------------
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.ego_status_callback)
        rospy.Subscriber("/lane_valid", Int32, self.lane_error_callback)
        rospy.Subscriber("/driving_mode", String, self.driving_mode_callback)
        rospy.Subscriber("/obstacle_info", ObjectInfo, self.obstacle_callback)
        rospy.Subscriber("/local_path", Path, self.local_path_callback)
        
        # ----------------------------------------------------------------
        # 4. 메인 실행 루프
        # ----------------------------------------------------------------
        rate = rospy.Rate(40)
        while not rospy.is_shutdown():
            if not self.is_status_received or self.current_mode == 'wait':
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
                self.publish_morai_command( 0.0, 0.0, 1.0)
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
                target_velocity_kph = 6.0 - self.on_dynamic_obs * 6.0
            elif self.current_mode in ['gps', 'lidar_only']:
                final_steering_degree = path_based_steering_deg
            else: # 'wait' 또는 알 수 없는 모드
                self.publish_morai_command( 0.0, 0.0, 1.0)
                rate.sleep()
                continue
            
            # 6. PID 및 최종 명령 발행
            pid_output = self.pid_controller.pid(target_velocity_kph, self.status.velocity)
            brake_cmd = -pid_output if pid_output < 0 else 0.0
            self.publish_morai_command(target_velocity_kph, final_steering_degree, brake_cmd)
            
            # 7. 시각화
            self.visualize_ego_marker()
            self.global_path_publisher.publish(self.global_path)
            rate.sleep()


    def ego_status_callback(self, msg):
        """'map' 기준의 차량 상태와 헤딩 보정"""
        sim_x, sim_y = msg.position.x, msg.position.y
        vehicle_x_utm52n = sim_x + self.morai_offset[0]
        vehicle_y_utm52n = sim_y + self.morai_offset[1]
        vehicle_x_utmk, vehicle_y_utmk = self.utm52n_to_utmk_transformer.transform(vehicle_x_utm52n, vehicle_y_utm52n)
        
        self.status.position.x = vehicle_x_utmk - self.map_origin[0]
        self.status.position.y = vehicle_y_utmk - self.map_origin[1]
        self.status.velocity = max(0, msg.velocity.x * 3.6)
        
        # [수정] 헤딩 값에 오프셋을 적용하여 보정하는 로직 복원
        raw_heading_deg = msg.heading
        corrected_heading_deg = raw_heading_deg # - self.heading_offset_deg
        self.status.heading = corrected_heading_deg
        
        self.is_status_received = True

    # --- (이하 나머지 함수들은 수정할 필요가 없습니다) ---
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
        VEHICLE_RADIUS, ROI_DISTANCE = 0.6, 5.0
        ego_x, ego_y, ego_yaw_rad = self.status.position.x, self.status.position.y, np.deg2rad(self.status.heading)
        
        # 현재 차량의 위치(waypoint)를 기준으로 검사 범위를 설정합니다.
        start_idx = self.current_waypoint

        # ----------------------------------------------------------------
        # [수정] end_idx 계산 로직 수정
        # '경로의 전체 길이'와 '현재 위치 + 검사 거리' 중 작은 값을 선택하여
        # 인덱스가 경로 길이를 벗어나지 않도록 수정합니다.
        # ----------------------------------------------------------------
        look_ahead_points = int(ROI_DISTANCE / 0.2) # 약 7미터 앞의 포인트 수
        end_idx = min(len(ref_path.poses), start_idx + look_ahead_points)

        # --- (이하 코드는 동일합니다) ---
        
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
        self.chk_path_publisher.publish(viz_path) # Publisher 이름 chk_path_publisher로 수정 제안

        markers = MarkerArray()

        for obs_idx in range(obstacle_msg.objectCounts):
            obs_local_x = obstacle_msg.centerX[obs_idx] + self.lidar_offset_x
            obs_local_y = obstacle_msg.centerY[obs_idx]

            delta_x = obs_local_x * np.cos(ego_yaw_rad) - obs_local_y * np.sin(ego_yaw_rad)
            delta_y = obs_local_x * np.sin(ego_yaw_rad) + obs_local_y * np.cos(ego_yaw_rad)
            obs_global_x = ego_x + delta_x
            obs_global_y = ego_y + delta_y

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = rospy.Time.now()
            marker.id = obs_idx # 각 마커에 고유 ID 부여
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
            
            obs_radius = max(obstacle_msg.lengthX[obs_idx], obstacle_msg.lengthY[obs_idx]) / 2.0

            for i in range(start_idx, end_idx):
                path_point = ref_path.poses[i].pose.position
                if np.hypot(path_point.x - obs_global_x, path_point.y - obs_global_y) < (VEHICLE_RADIUS + obs_radius):
                    self.obst_marker_publisher.publish(markers) # 충돌 시점에만 발행
                    return True, i 
                    
        self.obst_marker_publisher.publish(markers) # 충돌이 없을 때도 마커 발행
        return False, -1
                

    def generate_avoidance_path(self, blocked_idx):
        try:
            ref_path = self.global_path
            s_list, yaw_list = compute_s_and_yaw(ref_path)
            s0, l0 = cartesian_to_frenet(self.status.position.x, self.status.position.y, ref_path, s_list)
            AVOIDANCE_LENGTH, LATERAL_OFFSET = 0.3, 1.65
            target_s = s_list[blocked_idx] + AVOIDANCE_LENGTH
            target_l = LATERAL_OFFSET
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
            rospy.logerr("Failed to generate avoidance path: {}".format(e))
            self.is_avoiding = False
    
    def driving_mode_callback(self, msg):
        self.current_mode = msg.data
        if self.current_mode != 'gps': self.is_avoiding = False

    def lane_error_callback(self, msg):
        # ----------------------------------------------------------------
        # [추가] 차선이 인식되지 않았을 때(msg.data == 0), 조향각을 갱신하지 않고 이전 값을 유지합니다.
        # ----------------------------------------------------------------
        if msg.data == 0:
            rospy.logwarn("Lane data is 0. Maintaining previous steering angle.")
            return

        # --- (이하 기존 계산 로직은 그대로 실행됩니다) ---
        
        # lane_valid 토픽 데이터 형식에 맞춰 중앙 x좌표를 복원합니다.
        lane_center_x = (msg.data - 1) // 2

        rospy.loginfo("얼마?: {}".format(msg.data))
        
        # 실제 이미지 중심(300)을 기준으로 오차를 계산합니다.
        image_center_x = 300 
        PIXEL_ERROR = image_center_x - lane_center_x
        
        steer_rad = self.lane_steering_gain_1 * PIXEL_ERROR + self.lane_steering_gain_2 * PIXEL_ERROR * abs(PIXEL_ERROR)
        self.lane_steering_deg = np.rad2deg(steer_rad)

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

    def publish_morai_command(self, velocity_kph, steering_deg, brake_val):
        # rospy.loginfo('caller : {}'.format(caller))
        
        self.ctrl_cmd.velocity = velocity_kph
        self.ctrl_cmd.steering = np.deg2rad(steering_deg)
        self.ctrl_cmd.brake = brake_val
        self.control_publisher.publish(self.ctrl_cmd)

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
        if dist_to_end < 0.5:
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