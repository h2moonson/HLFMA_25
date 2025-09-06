# -*- coding: utf-8 -*-

import rospy
import rospkg
import math
from nav_msgs.msg import Path,Odometry
from geometry_msgs.msg import PoseStamped,Point
from std_msgs.msg import Float64,Int16,Float32MultiArray
import numpy as np
from math import cos,sin,sqrt,pow,atan2,pi
import tf
import copy
from scipy.interpolate import interp1d

DEG2RAD = 1 / 180 * pi
RAD2DEG = 1 / DEG2RAD

class PathReader :  ## 텍스트 파일에서 경로를 출력 ##
    def __init__(self,pkg_name, path_offset):
        rospack=rospkg.RosPack()
        self.file_path=rospack.get_path(pkg_name)
        self.path_offset = path_offset
        rospy.loginfo(f"path_offset: {self.path_offset}")

    def read_txt(self,file_name):
        full_file_name=self.file_path+"/path/"+file_name
        openFile = open(full_file_name, 'r')
        global_path = Path()
        global_path.header.frame_id='map'
        line=openFile.readlines()
        for i in line :
            tmp=i.split(",")
            read_pose=PoseStamped()
            read_pose.pose.position.x=float(tmp[3]) - self.path_offset[0]
            read_pose.pose.position.y=float(tmp[4]) - self.path_offset[1]
            read_pose.pose.position.z=0
            read_pose.pose.orientation.x=0
            read_pose.pose.orientation.y=0
            read_pose.pose.orientation.z=0
            read_pose.pose.orientation.w=1
            global_path.poses.append(read_pose)

        openFile.close()
        return global_path 
    

class PurePursuit: ## purePursuit 알고리즘 적용 ##
    def __init__(self):
        self.forward_point = Point()
        self.current_position = Point()
        self.is_look_forward_point = False
        self.vehicle_length = 0.72
        self.lfd = 2
        self.min_lfd = 1.0
        self.max_lfd = 6.0
        self.steering = 0

    def getPath(self, msg):
        self.path = msg  #nav_msgs/Path
    
    def getEgoStatus(self, msg):
        self.current_vel = msg.velocity.x  #kph
        self.vehicle_yaw = msg.heading * DEG2RAD  # rad
        self.current_position.x = msg.position.x
        self.current_position.y = msg.position.y
        self.current_position.z = 0.0

    def steeringAngle(self, _static_lfd=1.0):
        vehicle_position = self.current_position
        rotated_point = Point()
        self.is_look_forward_point = False

        ego_yaw_negative = -self.vehicle_yaw
        cos_yaw = math.cos(ego_yaw_negative)
        sin_yaw = math.sin(ego_yaw_negative)
        
        for i in self.path.poses:
            path_point = i.pose.position
            dx = path_point.x - vehicle_position.x
            dy = path_point.y - vehicle_position.y
        
            rotated_point.x = dx * cos_yaw - dy * sin_yaw
            rotated_point.y = dx * sin_yaw + dy * cos_yaw

            if rotated_point.x > 0:
                dis = math.sqrt(pow(rotated_point.x, 2) + pow(rotated_point.y, 2))
                
                if _static_lfd > 0:
                    self.lfd = _static_lfd
                else:
                    self.lfd = self.current_vel * 0.9
                    if self.lfd < self.min_lfd:
                        self.lfd = self.min_lfd
                    elif self.lfd > self.max_lfd:
                        self.lfd = self.max_lfd

                if dis >= self.lfd:
                    self.forward_point = path_point
                    self.is_look_forward_point = True
                    break
        
        theta = math.atan2(rotated_point.y, rotated_point.x)
        # [수정] 표준 좌표계(X:전방, Y:좌측)에서는 보통 -1을 곱하지 않음
        # 만약 수정 후 조향이 반대가 되면 이 부분의 -1을 다시 추가하거나 제거하여 튜닝
        self.steering = math.atan2((2 * self.vehicle_length * math.sin(theta)), self.lfd) * RAD2DEG
        
        return self.steering, self.forward_point.x, self.forward_point.y, self.lfd

    def findLocalPath(self, ref_path,status_msg):
        out_path=Path()
        current_x=status_msg.position.x
        current_y=status_msg.position.y
        current_waypoint=0
        min_dis=float('inf')

        for i in range(len(ref_path.poses)) :
            dx=current_x - ref_path.poses[i].pose.position.x
            dy=current_y - ref_path.poses[i].pose.position.y
            dis=sqrt(dx*dx + dy*dy)
            if dis < min_dis :
                min_dis=dis
                current_waypoint=i

        if current_waypoint + 80 > len(ref_path.poses) :
            last_local_waypoint= len(ref_path.poses)
        else :
            last_local_waypoint=current_waypoint+80
        
        out_path.header.frame_id='map'
        for i in range(current_waypoint,last_local_waypoint) :
            tmp_pose=PoseStamped()
            tmp_pose.pose.position.x=ref_path.poses[i].pose.position.x
            tmp_pose.pose.position.y=ref_path.poses[i].pose.position.y
            tmp_pose.pose.position.z=ref_path.poses[i].pose.position.z
            tmp_pose.pose.orientation.x=0
            tmp_pose.pose.orientation.y=0
            tmp_pose.pose.orientation.z=0
            tmp_pose.pose.orientation.w=1
            out_path.poses.append(tmp_pose)

        return current_waypoint, out_path

    def estimateCurvature(self):
        if len(self.path.poses) < 3:
            return None

        # [수정] 곡률 계산을 위한 전방 주시 거리 (미터 단위)
        LOOKAHEAD_DISTANCE = 5.0 
        
        vehicle_position = self.current_position
        target_point = None

        # [수정] 경로의 끝이 아닌, 차량 앞에서부터 일정 거리의 점을 찾습니다.
        for point in self.path.poses:
            dx = point.pose.position.x - vehicle_position.x
            dy = point.pose.position.y - vehicle_position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance >= LOOKAHEAD_DISTANCE:
                target_point = point.pose.position
                break
        
        # 만약 적절한 점을 찾지 못했다면(경로의 끝), 경로의 마지막 점을 사용합니다.
        if target_point is None:
            target_point = self.path.poses[-1].pose.position

        dx = target_point.x - vehicle_position.x
        dy = target_point.y - vehicle_position.y

        rotated_point = Point()
        rotated_point.x = math.cos(self.vehicle_yaw) * dx + math.sin(self.vehicle_yaw) * dy
        rotated_point.y = math.sin(self.vehicle_yaw) * dx - math.cos(self.vehicle_yaw) * dy
    
        # atan2의 입력 순서는 (y, x) 입니다.
        corner_theta = abs(math.atan2(rotated_point.y, rotated_point.x))
        return corner_theta * RAD2DEG, target_point.x, target_point.y

class PidController:
    def __init__(self):
        self.p_gain = 0.7
        self.i_gain = 0.0        
        self.d_gain = 0.05
        self.controlTime = 0.025 
        self.prev_error = 0
        self.i_control = 0

    def pid(self, target_velocity, current_velocity):
        error = target_velocity - current_velocity
        p_control = self.p_gain * error
        self.i_control += self.i_gain * error * self.controlTime
        d_control = self.d_gain * (error-self.prev_error) / self.controlTime
        output = p_control + self.i_control + d_control
        self.prev_error = error
        return output

# =====================================================================
# Frenet Frame 관련 신규 추가 함수들
# =====================================================================

def compute_s_and_yaw(ref_path):
    """
    전역 경로의 각 점에 대한 누적 거리(s)와 yaw 값을 계산합니다.
    """
    s_list = [0.0]
    yaw_list = []
    path_points = [p.pose.position for p in ref_path.poses]

    for i in range(1, len(path_points)):
        dx = path_points[i].x - path_points[i-1].x
        dy = path_points[i].y - path_points[i-1].y
        s_list.append(s_list[-1] + np.hypot(dx, dy))
        yaw_list.append(np.arctan2(dy, dx))
    
    if yaw_list:
        yaw_list.append(yaw_list[-1])  # 마지막 yaw 값 복사
    
    return np.array(s_list), np.array(yaw_list)

def cartesian_to_frenet(x, y, ref_path, s_list):
    """
    직교 좌표(x, y)를 Frenet 좌표(s, l)로 변환합니다.
    """
    path_poses = ref_path.poses
    min_dist = float('inf')
    min_idx = 0
    for i, pose in enumerate(path_poses):
        dx = x - pose.pose.position.x
        dy = y - pose.pose.position.y
        dist = dx**2 + dy**2
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    ref_pose = path_poses[min_idx].pose
    ref_x = ref_pose.position.x
    ref_y = ref_pose.position.y

    # Calculate reference yaw
    if min_idx < len(path_poses) - 1:
        next_pose = path_poses[min_idx + 1].pose
        ref_yaw = np.arctan2(next_pose.position.y - ref_y, next_pose.position.x - ref_x)
    else: # Reached the end of the path
        prev_pose = path_poses[min_idx - 1].pose
        ref_yaw = np.arctan2(ref_y - prev_pose.position.y, ref_x - prev_pose.position.x)

    dx = x - ref_x
    dy = y - ref_y

    s = s_list[min_idx]
    l = dx * -np.sin(ref_yaw) + dy * np.cos(ref_yaw)
    return s, l

def frenet_to_cartesian(s, l, ref_path, s_list, yaw_list):
    """
    Frenet 좌표(s, l)를 직교 좌표(x, y, yaw)로 변환합니다.
    """
    path_points = [p.pose.position for p in ref_path.poses]
    # interp1d를 사용하여 s 값에 해당하는 ref_path 상의 점을 보간
    fx = interp1d(s_list, [p.x for p in path_points], fill_value="extrapolate")
    fy = interp1d(s_list, [p.y for p in path_points], fill_value="extrapolate")
    fyaw = interp1d(s_list, yaw_list, fill_value="extrapolate")

    x_ref = fx(s)
    y_ref = fy(s)
    yaw_ref = fyaw(s)

    x = x_ref - l * np.sin(yaw_ref)
    y = y_ref + l * np.cos(yaw_ref)
    
    # 경로의 yaw와 l 방향을 고려하여 최종 yaw 계산 (단순화된 접근)
    final_yaw = yaw_ref 
    return x, y, final_yaw

def generate_quintic_path(s0, l0, s1, l1):
    """
    Quintic Polynomial (5차 다항식) 경로를 생성하는 함수를 반환합니다.
    시작과 끝 지점의 위치, 속도, 가속도가 0이라고 가정합니다.
    """
    A = np.array([
        [1, s0, s0**2,   s0**3,     s0**4,      s0**5],
        [0, 1,  2*s0,    3*s0**2,   4*s0**3,    5*s0**4],
        [0, 0,  2,       6*s0,     12*s0**2,   20*s0**3],
        [1, s1, s1**2,   s1**3,     s1**4,      s1**5],
        [0, 1,  2*s1,    3*s1**2,   4*s1**3,    5*s1**4],
        [0, 0,  2,       6*s1,     12*s1**2,   20*s1**3],
    ])
    b = np.array([l0, 0, 0, l1, 0, 0])
    coeffs = np.linalg.solve(A, b)
    return lambda s: np.polyval(coeffs[::-1], s)