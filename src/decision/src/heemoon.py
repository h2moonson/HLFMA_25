#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import tf

from nav_msgs.msg import Path
from std_msgs.msg import Int64MultiArray, Float64, Int64, Bool, Int32, String
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import  Vector3
from visualization_msgs.msg import Marker, MarkerArray
from pyproj import Proj, transform
from morai_msgs.msg import GPSMessage, CtrlCmd, EventInfo, EgoVehicleStatus
from morai_msgs.srv import MoraiEventCmdSrv
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from utils import PurePursuit, PidController, PathReader

class EgoStatus:
    def __init__(self):
        self.heading = 0.0
        self.position = Vector3()
        self.velocity = Vector3()


class PurePursuitNode:
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        # Publisher
        self.ego_marker_pub                 = rospy.Publisher('/ego_marker', Marker, queue_size=1)
        self.ctrl_cmd_pub                   = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.current_waypoint_pub           = rospy.Publisher('/current_waypoint', Int64, queue_size=1)
        self.pure_pursuit_target_point_pub  = rospy.Publisher('/pure_pursuit_target_point', Marker, queue_size=1)
        self.curvature_target_point_pub     = rospy.Publisher('/curvature_target_point', Marker, queue_size=1)
        self.global_path_pub                = rospy.Publisher('/global_path', Path, queue_size=1) 
        ########################  모라이 K-city - 제주도 보정용 파라미터  ########################
        self.map_origin = [0.0, 0.0] # 실행 시 최초 GPS 시작점 좌표 받음 - 실제로는 position covariance 값 안정화 될 때 까지 기다릴 것
        self.k_city_origin = [302473.4671541687, 4123735.5805772855] # UTM변환 시 East, North
        self.jeju_cw_origin = [249947.5672, 3688367.483] # UTM변환 시 East, North
        self.jeju_ccw_origin = [249961.5167, 3688380.892] # UTM변환 시 East, North


        self.status_msg = EgoStatus()
        self.ctrl_cmd_msg = CtrlCmd()
        self.current_mode = String('wait')

        ### Param - Sensor Connection ###
        self.is_gps_status = False
        self.is_gps = False

        ### Param - Lateral Controller ###
        self.lfd = 0.0
        self.current_waypoint = 0
        self.target_x = 0.0
        self.target_y = 0.0
        self.curvature_target_x = 0.0
        self.curvature_target_y = 0.0
        self.corner_theta_degree = 0.0

        self.target_velocity = 5.0
        self.steering = 0.0

        ### Param - Longitudinal Controller ###
        self.accel_msg = 0.0
        self.brake_msg = 0.0
        self.max_velocity = 5.0
        self.min_velocity = 2.0
    
        self.euler_data = [0,0,0,0]
        self.quaternion_data = [0,0,0,0]

        ### Param - tf ### 
        self.proj_UTM = Proj(proj='utm', zone = 52, ellps='WGS84', preserve_units=False)
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Subscriber
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.egoStatusCB) ## PID테스트 위해서 모라이에서 속도 받아오기  
        # rospy.Subscriber("/gps", GPSMessage, self.gpsCB) #gps모라이
        rospy.Subscriber("/ntrip_rops/ublox_gps/fix", NavSatFix, self.gpsCB) #실제센서
        rospy.Subscriber("/grid_sliding_window_path", Path, self.grid_sliding_window_pathCB)
        rospy.Subscriber("/lane_valid", Int32, self.calcLaneDetectSteeringCB) # 받은 값에서 2로 나누면 차선의 현재 center x 좌표가 나옴
        rospy.Subscriber("/driving_mode", String, self.modeCB)

        self.pid_control_input = 0.0
        self.steering_offset = 0.03
        self.current_velocity = 0
        
        ### For Lane Detection Control ### 
        self.lane_k1 = rospy.get_param('~lane_k1', 2.0e-3) #1차 계수
        self.lane_k2 = rospy.get_param('~lane_k2', 1.0e-5) #2차 계수
        self.lane_steering = 0.0
        
        
        ### Class ###
        self.grid_sliding_window_path = Path()
        path_reader = PathReader('decision', self.jeju_cw_origin) ## 경로 파일의 위치, offset 적용
        
        
        # ── PurePursuit 인스턴스 2개 ─────────────────────────
        self.pp_global = PurePursuit()
        self.pp_lidar  = PurePursuit()

        # ── steering / target-velocity 저장용 ───────────────
        self.steer_cam,   self.vel_cam   = 0.0, 4.0
        self.steer_gps,   self.vel_gps   = 0.0, 5.0
        self.steer_lidar, self.vel_lidar = 0.0, 3.0

        pid = PidController()
        ### Read path ###
        self.cw_path = path_reader.read_txt("clockwise.txt") ## 출력할 경로의 이름
        self.ccw_path = path_reader.read_txt("counter_clockwise.txt") ## 출력할 경로의 이름
        
        rate = rospy.Rate(40) 
        
        while not rospy.is_shutdown():
            self.getEgoCoord()
            
            if self.is_gps_status == True and self.current_mode != 'wait': 
                # self.ctrl_cmd_msg.longlCmdType = 1
                self.current_waypoint, local_path = self.pp_global.findLocalPath(self.cw_path, self.status_msg)
                self.pp_global.getPath(local_path) 
                self.pp_global.getEgoStatus(self.status_msg) # utils에서 계산하도록 속도, 위치 넘겨줌
                # @TODO getEgoStatus에 임시로 local path테스트만 하기에 0, 0 으로 함수 내부에서 설정해둠 > 추후 수정
                steer, self.target_x, self.target_y, self.lfd = self.pp_global.steeringAngle(1.5)
                self.steer_gps = (self.steering + 2.7) * self.steering_offset
                estimate_curvature = self.pp_global.estimateCurvature() #local path에서 곡률 계산해서 속도 동적으로 조절하기 위함

                if not estimate_curvature:
                    continue

                self.corner_theta_degree, self.curvature_target_x, self.curvature_target_y = estimate_curvature
                self.target_velocity = self.cornerController(self.corner_theta_degree)

                #항상 세 종류의 Steering 게산
                self.updateLidarOnlyPP() #1. self.steer_lidar갱신
                #2. self.steer_cam 은 laneErrorCB가 수시 갱신
                #3. GPS PurePursuit 계산 -> self.steer_gps에서 갱신 

                if  self.current_mode == 'cam':
                    self.steering_msg = self.steer_cam
                    target_vel        = self.vel_cam
                elif self.current_mode == 'gps':
                    self.steering_msg = self.steer_gps
                    target_vel        = self.vel_gps
                elif self.current_mode == 'lidar_only':
                    self.steering_msg = self.steer_lidar
                    target_vel        = self.vel_lidar
                else:                               # 'wait' 등
                    self.brake(); rate.sleep(); continue

                            
                self.pid_control_input = pid.pid(target_vel, self.current_velocity) ## 속도 제어를 위한 PID 적용 (target Velocity, Status Velocity)
                
                if self.pid_control_input > 0 :
                    self.accel_msg= self.pid_control_input
                    self.brake_msg = 0

                else :
                    self.accel_msg = 0
                    self.brake_msg = -self.pid_control_input



                self.current_waypoint_pub.publish(self.current_waypoint)
                self.global_path_pub.publish(self.cw_path)
                self.visualizeEgoPoint()
                self.visualizeTargetPoint()
                self.publishCtrlCmd(self.accel_msg, self.steering_msg, self.brake_msg)
            rate.sleep()

    def getEgoCoord(self):  # Vehicle Status Subscriber
        if self.is_gps == True:
            # 위치 업데이트: gpsCB에서 받은 좌표에 기초
            self.status_msg.position.x = self.xy_zone[0] - self.k_city_origin[0]
            self.status_msg.position.y = self.xy_zone[1] - self.k_city_origin[1]
            self.status_msg.position.z = 0.0
            # 여기서 self.status_msg.heading은 이미 gpsCB에서 갱신된 값 사용
            self.status_msg.velocity.x = self.current_velocity

            self.tf_broadcaster.sendTransform(
                (self.status_msg.position.x, self.status_msg.position.y, self.status_msg.position.z),
                tf.transformations.quaternion_from_euler(0, 0, (self.status_msg.heading) / 180 * np.pi),
                rospy.Time.now(),
                "base_link",
                "map"
            )
            self.is_gps_status = True

        elif self.is_gps is False:
            self.is_gps_status = False

        else:
            rospy.loginfo("Waiting for GPS")
            self.is_gps_status = False

    def updateLidarOnlyPP(self): 
        if not self.grid_sliding_window_path.poses:
            return 
        self.pp_lidar.getPath(self.grid_sliding_window_path)
        # LiDAR 경로는 차량 좌표계라 ego status 를 (0,0,heading=0) 로 고정
        dummy_status = EgoStatus();  dummy_status.velocity.x = self.current_velocity
        self.pp_lidar.getEgoStatus(dummy_status)
        steer, tx, ty, _ = self.pp_lidar.steeringAngle(1.2)
        self.steer_lidar = (steer + 2.7)*self.steering_offset

    def grid_sliding_window_pathCB(self, msg): 
        self.grid_sliding_window_path = msg

    def modeCB(self, msg:String): 
        # mode : cam / gps/ lidar_only
        self.current_mode = msg.data
        
    def gpsCB(self, msg: NavSatFix):
        # mode와 상관없이 계속 받고 있어야함
        if msg.status == 0: 
            # GPS 상태 불량
            self.is_gps = False

        else:
            new_xy = self.proj_UTM(msg.longitude, msg.latitude)

            # 이전 좌표와의 차이를 계산
            if hasattr(self, 'prev_xy'):
                dx = new_xy[0] - self.prev_xy[0]
                dy = new_xy[1] - self.prev_xy[1]
                distance = np.sqrt(dx**2 + dy**2)
                # 0.2 m 이상 차이가 날 경우에만 heading 업데이트
                if distance >= 0.2:
                    # arctan2 함수를 이용해 heading 계산 (라디안 -> 도 변환)
                    self.status_msg.heading = np.degrees(np.arctan2(dy, dx))
                    self.prev_xy = new_xy
            else:
                # 첫번째 GPS 메시지인 경우 이전 좌표 저장 + map 원점 반영
                rospy.loginfo(f"First GPS message received{new_xy}")
                self.prev_xy = new_xy

            self.xy_zone = new_xy

            self.tf_broadcaster.sendTransform((0, 0, 1.2),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "gps",
                            "base_link")
            
            self.tf_broadcaster.sendTransform((1.20, 0, 0.20),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "velodyne",
                            "base_link")
            
            self.is_gps = True

    def calcLaneDetectSteeringCB(self, msg:Int32): 
        '''
        ADD HERE
        #msg 를 2로 나눈 값을 차선의 중앙픽셀 x좌표 라고 생각
        #카메라 이미지의 전체 W의 절반과 위에서 나온 중앙 x좌표 차이를 error로 계산해서> steering_angle을 2차식 형태로 결정(error가 클수록 많이 조향하도록)
        '''
        err_px = msg.data #@TODO 이건 단순 중앙 좌표, 카메라의 width /2값을 뺴줘야 error
        steer_rad = self.lane_k1*err_px + self.lane_k2*err_px*abs(err_px)
        self.steer_cam = (steer_rad + 2.7) * self.steering_offset
    
    def publishCtrlCmd(self, accel_msg, steering_msg, brake_msg):
        self.ctrl_cmd_msg.accel = accel_msg
        self.ctrl_cmd_msg.steering = steering_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

    def cornerController(self, corner_theta_degree):
        corner_theta_degree = min(corner_theta_degree, 30)

        target_velocity = -0.5 * corner_theta_degree + 10

        if target_velocity < self.min_velocity:
            target_velocity = self.min_velocity
        
        if target_velocity > self.max_velocity : 
            target_velocity = self.max_velocity

        return target_velocity
    
    def brake(self) :
        self.ctrl_cmd_msg.longlCmdType = 1
        self.accel_msg = 0.0
        self.steering_msg = 0.0
        self.brake_msg = 1.0

        self.publishCtrlCmd(self.accel_msg, self.steering_msg, self.brake_msg)

    def egoStatusCB(self, msg): 
        self.current_velocity = max(0, msg.velocity.x * 3.6)
        self.current_wheel_angle = msg.wheel_angle

    def visualizeTargetPoint(self):
        pure_pursuit_target_point = Marker()
        
        pure_pursuit_target_point.header.frame_id = "map"
        pure_pursuit_target_point.action = pure_pursuit_target_point.ADD
        pure_pursuit_target_point.type = pure_pursuit_target_point.SPHERE
        
        pure_pursuit_target_point.scale.x = 0.5
        pure_pursuit_target_point.scale.y = 0.5
        pure_pursuit_target_point.scale.z = 0.5
        
        pure_pursuit_target_point.pose.orientation.w = 1.0
        
        pure_pursuit_target_point.color.r = 1.0
        pure_pursuit_target_point.color.g = 0.0
        pure_pursuit_target_point.color.b = 0.0
        pure_pursuit_target_point.color.a = 1.0 
        
        pure_pursuit_target_point.pose.position.x = self.target_x
        pure_pursuit_target_point.pose.position.y = self.target_y
        pure_pursuit_target_point.pose.position.z = 0.0
        
        self.pure_pursuit_target_point_pub.publish(pure_pursuit_target_point)

    def visualizeEgoPoint(self):
        ego_point = Marker()
        
        ego_point.header.frame_id = "map"
        ego_point.action = ego_point.ADD
        ego_point.type = ego_point.SPHERE
        
        ego_point.scale.x = 0.5
        ego_point.scale.y = 0.5
        ego_point.scale.z = 0.5
        
        ego_point.pose.orientation.w = 1.0
        
        ego_point.color.r = 0.0
        ego_point.color.g = 0.0
        ego_point.color.b = 1.0
        ego_point.color.a = 1.0 
        
        ego_point.pose.position.x = self.status_msg.position.x 
        ego_point.pose.position.y = self.status_msg.position.y
        ego_point.pose.position.z = 0.0
        
        self.ego_marker_pub.publish(ego_point)

if __name__ == '__main__':
    try:
        pure_pursuit_= PurePursuitNode()

    except rospy.ROSInterruptException:
        pass