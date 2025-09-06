#include "./dbscan.h"
#include "./header.h"
#include "./processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "./processPointClouds.cpp"
#include <cstdlib>
#include <ctime>

struct Color{
    int r;
    int g;
    int b;
};

Color getRandomcolor() {
    Color color;
    color.r = rand() % 256;
    color.g = rand() % 256;
    color.b = rand() % 256;
    return color;
}

// pcl point type
typedef pcl::PointXYZI PointT;
typedef pcl::PointXYZI clusterPointT;

// ROI parameter
double zMinROI, zMaxROI, xMinROI, xMaxROI, yMinROI, yMaxROI;
double xMinBoundingBox, xMaxBoundingBox, yMinBoundingBox, yMaxBoundingBox, zMinBoundingBox, zMaxBoundingBox;
// DBScan parameter
int minPoints;
double epsilon, minClusterSize, maxClusterSize;
// VoxelGrid parameter
float leafSize;
// segmentPlane parameter
int maxIterations;
float distanceThreshold;


// publisher
ros::Publisher pubROI;
ros::Publisher pubCluster;
ros::Publisher pubObjectInfo;
ros::Publisher pubPointInfo;
ros::Publisher pubObjectMarkerArray;
ros::Publisher pubPlaneInfo;

//MSG
lidar_object_detection::ObjectInfo objectInfoMsg;
lidar_object_detection::PointInfo pointInfoMsg; // 이미지에 옮길 bbox 각 꼭짓점 정보


void cfgCallback(lidar_object_detection::labacon3DetectionConfig &config_tunnel_static, int32_t level) {
    xMinROI = config_tunnel_static.xMinROI;
    xMaxROI = config_tunnel_static.xMaxROI;
    yMinROI = config_tunnel_static.yMinROI;
    yMaxROI = config_tunnel_static.yMaxROI;
    zMinROI = config_tunnel_static.zMinROI;
    zMaxROI = config_tunnel_static.zMaxROI;

    minPoints = config_tunnel_static.minPoints;
    epsilon = config_tunnel_static.epsilon;
    minClusterSize = config_tunnel_static.minClusterSize;
    maxClusterSize = config_tunnel_static.maxClusterSize;

    xMinBoundingBox = config_tunnel_static.xMinBoundingBox;
    xMaxBoundingBox = config_tunnel_static.xMaxBoundingBox;
    yMinBoundingBox = config_tunnel_static.yMinBoundingBox;
    yMaxBoundingBox = config_tunnel_static.yMaxBoundingBox;
    zMinBoundingBox = config_tunnel_static.zMinBoundingBox;
    zMaxBoundingBox = config_tunnel_static.zMaxBoundingBox;

    leafSize  = config_tunnel_static.leafSize;

    maxIterations = config_tunnel_static.maxIterations;
    distanceThreshold = config_tunnel_static.distanceThreshold;
}

pcl::PointCloud<PointT>::Ptr ROI (const sensor_msgs::PointCloud2ConstPtr& input) {
    // ... do data processing
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    pcl::fromROSMsg(*input, *cloud); // sensor_msgs -> PointCloud 형변환

    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr center(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr outskirt(new pcl::PointCloud<PointT>);

    // pcl::PointCloud<PointT>::Ptr *retPtr = &cloud_filtered;
    // std::cout << "Loaded : " << cloud->width * cloud->height << '\n';




    // X축 ROI
    pcl::PassThrough<PointT> filter;
    filter.setInputCloud(cloud);                //입력 
    filter.setFilterFieldName("x");             //적용할 좌표 축 (eg. X축)
    filter.setFilterLimits(xMinROI, 0.0);          //적용할 값 (최소, 최대 값)
    filter.filter(*center);             //필터 적용 

    filter.setFilterLimitsNegative (true);     //적용할 값 외 
    filter.filter(*outskirt);

    // Y축 ROI
    // pcl::PassThrough<PointT> filter;
    filter.setInputCloud(center);                //입력 
    filter.setFilterFieldName("y");             //적용할 좌표 축 (eg. Y축)
    filter.setFilterLimits(-0.7, 0.7);          //적용할 값 (최소, 최대 값)
    filter.setFilterLimitsNegative (true);     //적용할 값 외 
    filter.filter(*cloud_filtered);             //필터 적용 

    *cloud_filtered += *outskirt;

    // 오브젝트 생성 
    // Z축 ROI
    // pcl::PassThrough<PointT> filter;
    filter.setInputCloud(cloud_filtered);                //입력 
    filter.setFilterFieldName("z");             //적용할 좌표 축 (eg. Z축)
    filter.setFilterLimits(zMinROI, zMaxROI);          //적용할 값 (최소, 최대 값)
    filter.setFilterLimitsNegative (false);     //적용할 값 외 
    filter.filter(*cloud_filtered);             //필터 적용 

    // X축 ROI
    // pcl::PassThrough<PointT> filter;
    filter.setInputCloud(cloud_filtered);                //입력 
    filter.setFilterFieldName("x");             //적용할 좌표 축 (eg. X축)
    filter.setFilterLimits(xMinROI, xMaxROI);          //적용할 값 (최소, 최대 값)
    filter.setFilterLimitsNegative (false);     //적용할 값 외 
    filter.filter(*cloud_filtered);             //필터 적용 

    // Y축 ROI
    // pcl::PassThrough<PointT> filter;
    filter.setInputCloud(cloud_filtered);                //입력 
    filter.setFilterFieldName("y");             //적용할 좌표 축 (eg. Y축)
    filter.setFilterLimits(yMinROI, yMaxROI);          //적용할 값 (최소, 최대 값)
    filter.setFilterLimitsNegative (false);     //적용할 값 외 
    filter.filter(*cloud_filtered);             //필터 적용 

    // 포인트수 출력
    // std::cout << "ROI Filtered :" << cloud_filtered->width * cloud_filtered->height  << '\n'; 

    sensor_msgs::PointCloud2 roi_raw;
    pcl::toROSMsg(*cloud_filtered, roi_raw);

    pubROI.publish(roi_raw);

    return cloud_filtered;
}

pcl::PointCloud<PointT>::Ptr segmentPlane(pcl::PointCloud<PointT>::Ptr input) {
    ProcessPointClouds<PointT> pointProcessor;
    std::pair<pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<PointT>::Ptr> segmentCloud = pointProcessor.SegmentPlane(input, maxIterations, distanceThreshold);

    sensor_msgs::PointCloud2 pointCloudSegmentPlane;
    pcl::toROSMsg(*segmentCloud.first, pointCloudSegmentPlane);
    pubPlaneInfo.publish(pointCloudSegmentPlane);

    return segmentCloud.first;
}

pcl::PointCloud<PointT>::Ptr voxelGrid(pcl::PointCloud<PointT>::Ptr input) {
    //Voxel Grid를 이용한 DownSampling
    pcl::VoxelGrid<PointT> vg;    // VoxelGrid 선언
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>); //Filtering 된 Data를 담을 PointCloud 선언
    vg.setInputCloud(input);             // Raw Data 입력
    vg.setLeafSize(leafSize, leafSize, leafSize); // 사이즈를 너무 작게 하면 샘플링 에러 발생
    vg.filter(*cloud_filtered);          // Filtering 된 Data를 cloud PointCloud에 삽입

    // std::cout << "After Voxel Filtered :" << cloud_filtered->width * cloud_filtered->height  << '\n'; 

    return cloud_filtered;
}

void cluster(pcl::PointCloud<PointT>::Ptr input) {
    if (input->empty()) {
        objectInfoMsg.objectCounts = 0;
        pointInfoMsg.bboxCounts = 0;
        pubObjectInfo.publish(objectInfoMsg);
        pubPointInfo.publish(pointInfoMsg);
        
        sensor_msgs::PointCloud2 cluster_point;
        pcl::PointCloud<clusterPointT> totalcloud_clustered;
        pcl::toROSMsg(totalcloud_clustered, cluster_point);
        cluster_point.header.frame_id = "velodyne";
        pubCluster.publish(cluster_point);
        return;
    }

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(input);
    std::vector<pcl::PointIndices> cluster_indices;
    DBSCANKdtreeCluster<PointT> dc;
    dc.setCorePointMinPts(minPoints);
    dc.setClusterTolerance(epsilon);
    dc.setMinClusterSize(minClusterSize);
    dc.setMaxClusterSize(maxClusterSize);
    dc.setSearchMethod(tree);
    dc.setInputCloud(input);
    dc.extract(cluster_indices);

    pcl::PointCloud<clusterPointT> totalcloud_clustered;
    int valid_cluster_count = 0;

    // 각 클러스터에 접근
    for (const auto& it : cluster_indices) {
        if (valid_cluster_count >= 100) {
            ROS_WARN("Maximum number of objects (100) detected. Some clusters are ignored.");
            break; 
        }

        pcl::PointCloud<clusterPointT> eachcloud_clustered;
        for (int pit : it.indices) {
            clusterPointT tmp;
            // [수정] input.points -> input->points
            tmp.x = input->points[pit].x;
            tmp.y = input->points[pit].y;
            tmp.z = input->points[pit].z;
            tmp.intensity = valid_cluster_count;
            eachcloud_clustered.push_back(tmp);
        }


        clusterPointT minPoint, maxPoint;
        pcl::getMinMax3D(eachcloud_clustered, minPoint, maxPoint);
        
        float lengthX = maxPoint.x - minPoint.x;
        float lengthY = maxPoint.y - minPoint.y;
        float lengthZ = maxPoint.z - minPoint.z;

        if (xMinBoundingBox <= lengthX && lengthX <= xMaxBoundingBox &&
            yMinBoundingBox <= lengthY && lengthY <= yMaxBoundingBox &&
            zMinBoundingBox <= lengthZ && lengthZ <= zMaxBoundingBox) {
            
            totalcloud_clustered += eachcloud_clustered;
            
            objectInfoMsg.lengthX[valid_cluster_count] = lengthX;
            objectInfoMsg.lengthY[valid_cluster_count] = lengthY;
            objectInfoMsg.lengthZ[valid_cluster_count] = lengthZ;
            objectInfoMsg.centerX[valid_cluster_count] = (minPoint.x + maxPoint.x) / 2;
            objectInfoMsg.centerY[valid_cluster_count] = (minPoint.y + maxPoint.y) / 2;
            objectInfoMsg.centerZ[valid_cluster_count] = (minPoint.z + maxPoint.z) / 2;
            
            valid_cluster_count++;
        }
    }

    objectInfoMsg.objectCounts = valid_cluster_count;
    pubObjectInfo.publish(objectInfoMsg);

    // pointInfoMsg 발행 로직은 필요에 따라 추가
    // pointInfoMsg.bboxCounts = valid_cluster_count;
    // pubPointInfo.publish(pointInfoMsg);

    sensor_msgs::PointCloud2 cluster_point;
    pcl::toROSMsg(totalcloud_clustered, cluster_point);
    cluster_point.header.frame_id = "velodyne";
    pubCluster.publish(cluster_point);
}

void visualizeObject() {
    visualization_msgs::MarkerArray objectMarkerArray;
    visualization_msgs::Marker objectMarker;

    objectMarker.header.frame_id = "velodyne"; 
    objectMarker.ns = "object_shape";
    objectMarker.type = visualization_msgs::Marker::CUBE;
    objectMarker.action = visualization_msgs::Marker::ADD;

    for (int i = 0; i < objectInfoMsg.objectCounts; i++) {

        if (xMinBoundingBox <= objectInfoMsg.lengthX[i] && objectInfoMsg.lengthX[i] <= xMaxBoundingBox &&
            yMinBoundingBox <= objectInfoMsg.lengthY[i] && objectInfoMsg.lengthY[i] <= yMaxBoundingBox &&
            zMinBoundingBox <= objectInfoMsg.lengthZ[i] && objectInfoMsg.lengthZ[i] <= zMaxBoundingBox) {

            // Set the namespace and id for this marker.  This serves to create a unique ID
            // Any marker sent with the same namespace and id will overwrite the old one
            objectMarker.header.stamp = ros::Time::now();
            objectMarker.id = 100+i; // 

            // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
            objectMarker.pose.position.x = objectInfoMsg.centerX[i];
            objectMarker.pose.position.y = objectInfoMsg.centerY[i];
            objectMarker.pose.position.z = objectInfoMsg.centerZ[i];
            objectMarker.pose.orientation.x = 0.0;
            objectMarker.pose.orientation.y = 0.0;
            objectMarker.pose.orientation.z = 0.0;
            objectMarker.pose.orientation.w = 1.0;

            // Set the scale of the marker -- 1x1x1 here means 1m on a side
            objectMarker.scale.x = objectInfoMsg.lengthX[i];
            objectMarker.scale.y = objectInfoMsg.lengthY[i];
            objectMarker.scale.z = objectInfoMsg.lengthZ[i];

            // Set the color -- be sure to set alpha to something non-zero!
            objectMarker.color.r = 0.0;
            objectMarker.color.g = 1.0;
            objectMarker.color.b = 0.0;
            objectMarker.color.a = 0.5;

            objectMarker.lifetime = ros::Duration(0.1);
            objectMarkerArray.markers.emplace_back(objectMarker);
        }
    }

    // Publish the marker
    pubObjectMarkerArray.publish(objectMarkerArray);
}

void mainCallback(const sensor_msgs::PointCloud2ConstPtr& input) {
    pcl::PointCloud<PointT>::Ptr cloudPtr;

    // main process method
    cloudPtr = ROI(input);
    // cloudPtr = voxelGrid(cloudPtr);
    // cloudPtr = segmentPlane(cloudPtr);
    cluster(cloudPtr);

    // visualize method
    visualizeObject();
}

int main (int argc, char** argv) {
    // Initialize ROS
    ros::init (argc, argv, "labacon3_detection");
    ROS_INFO("<<<<< labacon3_detection node is starting up! >>>>>"); 

    ros::NodeHandle nh;

    dynamic_reconfigure::Server<lidar_object_detection::labacon3DetectionConfig> server;
    dynamic_reconfigure::Server<lidar_object_detection::labacon3DetectionConfig>::CallbackType f;

    f = boost::bind(&cfgCallback, _1, _2);
    server.setCallback(f);
    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe ("velodyne_points", 1, mainCallback);

    // Create a ROS publisher for the output point cloud
    pubROI = nh.advertise<sensor_msgs::PointCloud2> ("roi_raw_static", 1);
    pubCluster = nh.advertise<sensor_msgs::PointCloud2>("cluster_static", 1);
    pubObjectInfo = nh.advertise<lidar_object_detection::ObjectInfo>("obstacle_info", 1);
    pubPointInfo = nh.advertise<lidar_object_detection::PointInfo>("bbox_point_info_static", 1);
    pubObjectMarkerArray = nh.advertise<visualization_msgs::MarkerArray>("bounding_box_static", 1);
    pubPlaneInfo = nh.advertise<sensor_msgs::PointCloud2> ("plane_static", 1);

    // Spin
    ros::spin();
}