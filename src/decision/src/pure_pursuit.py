#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import tf

from nav_msgs.msg import Path
from std_msgs.msg import Int64MultiArray, Float64, Int64, Bool, Int32, String
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from pyproj import Proj
from morai_msgs.msg import GPSMessage, CtrlCmd, EventInfo, EgoVehicleStatus
from morai_msgs.srv import MoraiEventCmdSrv
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# --- [수정] 필요한 클래스 및 함수 추가 임포트 ---
from utils import PurePursuit, PidController, PathReader
from utils import compute_s_and_yaw, cartesian_to_frenet, frenet_to_cartesian, generate_quintic_path

# --- [참고] 사용자 환경에 맞는 장애물 메시지 타입으로 수정해야 합니다 ---
# 예: from your_lidar_pkg.msg import ObstacleInfo
# 지금은 시각화 메시지로 임시 대체합니다. 실제 객체 정보(위치, 크기)를 포함해야 합니다.
from visualization_msgs.msg import MarkerArray as ObjectInfo

class EgoStatus:
    def __init__(self):
        self.heading = 0.0
        self.position = Vector3()
        self.velocity = Vector3()


class PurePursuitNode:
    def __init__(self):
        rospy.init_node('pure_pursuit_with_avoidance', anonymous=True)

        # Publisher
        self.ego_marker_pub                 = rospy.Publisher('/ego_marker', Marker, queue_size=1)
        self.ctrl_cmd_pub                   = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.current_waypoint_pub           = rospy.Publisher('/current_waypoint', Int64, queue_size=1)
        self.pure_pursuit_target_point_pub  = rospy.Publisher('/pure_pursuit_target_point', Marker, queue_size=1)
        self.global_path_pub                = rospy.Publisher('/global_path', Path, queue_size=1)
        
        self.map_origin = [302473.4671541687, 4123735.5805772855] # K-City Origin
        self.jeju_cw_origin = [249947.5672, 3688367.483] # UTM변환 시 East, North
        self.jeju_ccw_origin = [249961.5167, 3688380.892] # UTM변환 시 East, North

        self.status_msg = EgoStatus()
        self.ctrl_cmd_msg = CtrlCmd()
        self.current_mode = 'wait'

        self.is_gps = False
        self.is_gps_status = False

        self.lfd = 0.0
        self.target_velocity = 5.0
        self.steering = 0.0

        self.accel_msg = 0.0
        self.brake_msg = 0.0
        self.max_velocity = 5.0
        self.min_velocity = 2.0

        self.proj_UTM = Proj(proj='utm', zone=52, ellps='WGS84', preserve_units=False)
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Subscriber
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.egoStatusCB)
        rospy.Subscriber("/ntrip_rops/ublox_gps/fix", NavSatFix, self.gpsCB)
        rospy.Subscriber("/lane_valid", Int32, self.calcLaneDetectSteeringCB)
        rospy.Subscriber("/driving_mode", String, self.modeCB)
        
        # --- [신규] 장애물 정보 Subscriber 추가 ---
        rospy.Subscriber("/obstacle_info", ObjectInfo, self.obstacle_callback)

        self.pid_control_input = 0.0
        self.steering_offset = 0.03
        self.current_velocity = 0
        
        self.lane_k1 = rospy.get_param('~lane_k1', 2.0e-3)
        self.lane_k2 = rospy.get_param('~lane_k2', 1.0e-5)
        self.lane_steering = 0.0
        
        self.pp_global = PurePursuit()
        pid = PidController()

        path_reader = PathReader('decision', self.map_origin)
        self.cw_path = path_reader.read_txt("clockwise.txt")
        self.ccw_path = path_reader.read_txt("counter_clockwise.txt")

        # --- [신규] 장애물 회피 관련 상태 변수 ---
        self.is_avoiding = False
        self.avoidance_path = Path()
        
        rate = rospy.Rate(40) 
        
        # --- [수정] 메인 루프 구조 변경 ---
        while not rospy.is_shutdown():
            self.getEgoCoord()
            
            if self.is_gps_status and self.current_mode != 'wait':
                # === [핵심 수정] 1. 추종 경로 선택 로직 ===
                path_to_follow = self.cw_path # 기본은 전역 경로
                if self.is_avoiding:
                    path_to_follow = self.avoidance_path
                    # 회피 완료 조건 확인
                    if len(path_to_follow.poses) > 0:
                        dist_to_end = np.hypot(
                            self.status_msg.position.x - path_to_follow.poses[-1].pose.position.x,
                            self.status_msg.position.y - path_to_follow.poses[-1].pose.position.y
                        )
                        if dist_to_end < 1.5: # 회피 경로 끝 1.5m 이내 도착 시
                            rospy.loginfo("[Main Loop] Avoidance complete. Returning to global path.")
                            self.is_avoiding = False
                            path_to_follow = self.cw_path # 즉시 전역 경로로 복귀

                # === 2. 선택된 경로 기반으로 제어 계산 ===
                self.current_waypoint, local_path = self.pp_global.findLocalPath(path_to_follow, self.status_msg)
                
                if len(local_path.poses) < 2:
                    rospy.logwarn("Local path is too short. Skipping control.")
                    rate.sleep()
                    continue

                self.pp_global.getPath(local_path)
                self.pp_global.getEgoStatus(self.status_msg)
                
                steer, self.target_x, self.target_y, self.lfd = self.pp_global.steeringAngle(0) # 동적 LFD 사용
                self.steer_gps = (steer + 2.7) * self.steering_offset
                
                # 곡률 기반 속도 제어
                curvature_info = self.pp_global.estimateCurvature()
                if curvature_info:
                    corner_theta_degree, _, _ = curvature_info
                    self.target_velocity = self.cornerController(corner_theta_degree)
                
                # 현재 모드에 따른 최종 제어값 선택
                if self.current_mode == 'cam':
                    steering_msg = self.lane_steering # calcLaneDetectSteeringCB 에서 계산됨
                    target_vel = 4.0
                elif self.current_mode == 'gps':
                    steering_msg = self.steer_gps
                    target_vel = self.target_velocity
                else: # 'wait', 'lidar_only' 등
                    self.brake()
                    rate.sleep()
                    continue

                # PID 속도 제어
                self.pid_control_input = pid.pid(target_vel, self.current_velocity)
                if self.pid_control_input > 0:
                    self.accel_msg = self.pid_control_input
                    self.brake_msg = 0.0
                else:
                    self.accel_msg = 0.0
                    self.brake_msg = -self.pid_control_input

                # 최종 명령 발행
                self.publishCtrlCmd(self.accel_msg, steering_msg, self.brake_msg)
                self.visualizeTargetPoint()
            
            self.visualizeEgoPoint()
            self.global_path_pub.publish(self.cw_path)
            rate.sleep()
    
    # --- [신규] 장애물 처리 콜백 함수 ---
    def obstacle_callback(self, msg):
        if self.is_avoiding:
            return

        if self.current_mode == 'cam':
            # TODO: 전방의 특정 거리 내 장애물 확인 로직 필요
            # 예: if len(msg.markers) > 0 and self.is_obstacle_in_front(msg.markers[0]):
            if len(msg.markers) > 0: # 임시로 장애물이 하나라도 감지되면 정지
                rospy.logwarn("[Obstacle] CAM Mode: Obstacle detected. Stopping vehicle.")
                self.brake()

                return

        elif self.current_mode == 'gps':
            is_path_blocked, blocked_idx = self.check_path_collision(msg, self.cw_path)
            if is_path_blocked:
                rospy.logwarn(f"[Obstacle] GPS Mode: Path blocked at index {blocked_idx}. Generating avoidance path.")
                self.generate_avoidance_path(blocked_idx)

    # --- [신규] 경로-장애물 충돌 확인 함수 ---
    def check_path_collision(self, obstacle_msg, ref_path):
        vehicle_radius = 0.7 # 차량 폭의 절반 정도
        roi_distance = 15.0 # 전방 15m 내의 경로만 확인

        start_idx = self.current_waypoint
        end_idx = min(start_idx + int(roi_distance / 0.2), len(ref_path.poses)) # 20cm 간격 가정

        for i in range(start_idx, end_idx):
            path_point = ref_path.poses[i].pose.position
            for obstacle in obstacle_msg.markers:
                # obstacle_msg가 MarkerArray라고 가정
                obs_point = obstacle.pose.position
                obs_radius = max(obstacle.scale.x, obstacle.scale.y) / 2.0
                
                dist = np.hypot(path_point.x - obs_point.x, path_point.y - obs_point.y)
                
                if dist < (vehicle_radius + obs_radius):
                    return True, i # 충돌 발생, 충돌한 경로 인덱스 반환

        return False, -1 # 충돌 없음

    # --- [신규] 회피 경로 생성 함수 ---
    def generate_avoidance_path(self, blocked_idx):
        try:
            ref_path = self.cw_path
            s_list, yaw_list = compute_s_and_yaw(ref_path)

            s0, l0 = cartesian_to_frenet(self.status_msg.position.x, self.status_msg.position.y, ref_path, s_list)

            avoidance_length = 15.0  # 15m에 걸쳐 회피 기동
            lateral_offset = 2.0     # 왼쪽으로 2m 회피

            target_s = s_list[blocked_idx] + avoidance_length
            target_l = lateral_offset

            l_path_func = generate_quintic_path(s0, l0, target_s, target_l)

            new_path = Path()
            new_path.header.frame_id = 'map'
            resolution = int((target_s - s0) / 0.2)
            s_points = np.linspace(s0, target_s, resolution)

            for s in s_points:
                l = l_path_func(s)
                x, y, yaw = frenet_to_cartesian(s, l, ref_path, s_list, yaw_list)
                
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.pose.position.x, pose.pose.position.y = x, y
                q = quaternion_from_euler(0, 0, yaw)
                pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
                new_path.poses.append(pose)

            self.avoidance_path = new_path
            self.is_avoiding = True

        except Exception as e:
            rospy.logerr(f"Failed to generate avoidance path: {e}")
            self.is_avoiding = False

    def getEgoCoord(self):
        if self.is_gps:
            self.status_msg.position.x = self.xy_zone[0] - self.map_origin[0]
            self.status_msg.position.y = self.xy_zone[1] - self.map_origin[1]
            self.status_msg.velocity.x = self.current_velocity
            self.is_gps_status = True
        else:
            self.is_gps_status = False

    def gpsCB(self, msg: NavSatFix):
        if msg.status.status == 0:
            self.is_gps = False
            return
        
        self.is_gps = True
        self.xy_zone = self.proj_UTM(msg.longitude, msg.latitude)

        if hasattr(self, 'prev_xy'):
            dx = self.xy_zone[0] - self.prev_xy[0]
            dy = self.xy_zone[1] - self.prev_xy[1]
            if np.hypot(dx, dy) > 0.2:
                self.status_msg.heading = np.degrees(np.arctan2(dy, dx))
                self.prev_xy = self.xy_zone
        else:
            self.prev_xy = self.xy_zone

    def modeCB(self, msg:String): 
        self.current_mode = msg.data
        if self.current_mode != 'gps':
            # GPS 모드가 아니면 회피 상태 강제 종료
            self.is_avoiding = False

    def egoStatusCB(self, msg: EgoVehicleStatus): 
        self.current_velocity = max(0, msg.velocity.x * 3.6) # kph

    def calcLaneDetectSteeringCB(self, msg:Int32):
        err_px = msg.data - 320 # 이미지 너비의 절반(640/2)을 빼서 에러 계산
        steer_rad = self.lane_k1 * err_px + self.lane_k2 * err_px * abs(err_px)
        self.lane_steering = (steer_rad + 2.7) * self.steering_offset
    
    def publishCtrlCmd(self, accel, steer, brake):
        self.ctrl_cmd_msg.longlCmdType = 2
        self.ctrl_cmd_msg.accel = accel
        self.ctrl_cmd_msg.steering = steer
        self.ctrl_cmd_msg.brake = brake
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

    def cornerController(self, corner_theta_degree):
        corner_theta_degree = min(corner_theta_degree, 30)
        target_velocity = -0.5 * corner_theta_degree + 10
        return np.clip(target_velocity, self.min_velocity, self.max_velocity)
    
    def brake(self):
        self.publishCtrlCmd(0.0, 0.0, 1.0)

    def visualizeTargetPoint(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 0.5, 0.5, 0.5
        marker.color.a, marker.color.r = 1.0, 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x, marker.pose.position.y = self.target_x, self.target_y
        self.pure_pursuit_target_point_pub.publish(marker)

    def visualizeEgoPoint(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 0.5, 0.5, 0.5
        marker.color.a, marker.color.b = 1.0, 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x, marker.pose.position.y = self.status_msg.position.x, self.status_msg.position.y
        self.ego_marker_pub.publish(marker)

if __name__ == '__main__':
    try:
        PurePursuitNode()
    except rospy.ROSInterruptException:
        pass