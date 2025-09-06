#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion

class OccupancyGridMapper:
    def __init__(self):
        # 기존 파라미터들
        self.resolution = rospy.get_param("~resolution", 0.1)
        self.width = rospy.get_param("~width", 70)
        self.height = rospy.get_param("~height", 100)
        self.origin_x = rospy.get_param("~origin_x", -1.0)
        self.origin_y = rospy.get_param("~origin_y", -5.0)
        self.ground_threshold = rospy.get_param("~ground_threshold", -10.0)
        self.padding_distance = rospy.get_param("~padding_distance", 0.2)
        
        # Temporal smoothing 관련 파라미터
        self.history_length = rospy.get_param("~history_length", 5)  # 몇 프레임을 사용할지 결정
        self.grid_history = []  # occupancy grid 히스토리 저장 리스트
        
        self.pub = rospy.Publisher("occupancy_grid", OccupancyGrid, queue_size=1)
        rospy.Subscriber("roi_raw", PointCloud2, self.pc_callback)

    def apply_padding(self, grid, padding_cells):
        padded_grid = grid.copy()
        height, width = grid.shape
        threshold = int((3/4) * width)
        
        for y in range(height):
            for x in range(width):
                if grid[y, x] == 100:
                    if x < threshold:
                        y_min = max(0, y - 2 * padding_cells)
                        y_max = min(height, y + 2 * padding_cells + 1)
                        x_min = max(0, x)
                        x_max = min(width, x + padding_cells * 3 + 1)
                    else:
                        y_min = max(0, y - 4 * padding_cells)
                        y_max = min(height, y + 4 * padding_cells + 1)
                        x_min = max(0, x - padding_cells * 2)
                        x_max = min(width, x + padding_cells + 1)
                    
                    padded_grid[y_min:y_max, x_min:x_max] = 100
        return padded_grid

    def pc_callback(self, msg):
        # PointCloud2 메시지를 numpy array로 변환 (x, y, z 추출)
        points = np.array(list(point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))))
        if points.size == 0:
            return

        # 지면 제거
        filtered_points = points[points[:, 2] > self.ground_threshold]

        # 빈 2D 그리드 생성 (0: free, 100: occupied)
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        # (x, y) 좌표를 격자 셀 인덱스로 변환
        xs = filtered_points[:, 0] - self.origin_x
        ys = filtered_points[:, 1] - self.origin_y
        cell_x = np.floor(xs / self.resolution).astype(np.int32)
        cell_y = np.floor(ys / self.resolution).astype(np.int32)

        valid = (cell_x >= 0) & (cell_x < self.width) & (cell_y >= 0) & (cell_y < self.height)
        cell_x = cell_x[valid]
        cell_y = cell_y[valid]

        grid[cell_y, cell_x] = 100

        # 패딩 적용
        padding_cells = int(self.padding_distance / self.resolution)
        if padding_cells > 0:
            grid = self.apply_padding(grid, padding_cells)

        # Temporal smoothing: 히스토리에 현재 grid 추가
        self.grid_history.append(grid)
        if len(self.grid_history) > self.history_length:
            self.grid_history.pop(0)

        # 여러 프레임의 grid를 평균내어 smoothing 수행
        # 각 셀별 평균을 구한 후 임계값 (예: 50 이상이면 occupied) 적용
        avg_grid = np.mean(np.stack(self.grid_history, axis=0), axis=0)
        smoothed_grid = np.where(avg_grid >= 40, 100, 0).astype(np.int8)

        # OccupancyGrid 메시지 구성
        occ_grid = OccupancyGrid()
        occ_grid.header = msg.header
        occ_grid.header.frame_id = "velodyne"
        occ_grid.info.resolution = self.resolution
        occ_grid.info.width = self.width
        occ_grid.info.height = self.height
        occ_grid.info.origin = Pose(Point(self.origin_x, self.origin_y, 0.0),
                                    Quaternion(0, 0, 0, 1))

        occ_grid.data = smoothed_grid.flatten().tolist()

        self.pub.publish(occ_grid)

if __name__ == "__main__":
    rospy.init_node("occupancy_grid_mapper")
    mapper = OccupancyGridMapper()
    rospy.spin()
