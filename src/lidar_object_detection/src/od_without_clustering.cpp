#include "./dbscan.h"
#include "./header.h"
#include "./processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "./processPointClouds.cpp"

// pcl point type
typedef pcl::PointXYZ PointT;
// cluster point type
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

float orientation_x = 0.0;
float orientation_y = 0.0;
float orientation_z = 0.0;
float orientation_w = 0.0;


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


void cfgCallback(lidar_object_detection::objectDetectorConfig &config_without_clustering, int32_t level) {
    xMinROI = config_without_clustering.xMinROI;
    xMaxROI = config_without_clustering.xMaxROI;
    yMinROI = config_without_clustering.yMinROI;
    yMaxROI = config_without_clustering.yMaxROI;
    zMinROI = config_without_clustering.zMinROI;
    zMaxROI = config_without_clustering.zMaxROI;

    minPoints = config_without_clustering.minPoints;
    epsilon = config_without_clustering.epsilon;
    minClusterSize = config_without_clustering.minClusterSize;
    maxClusterSize = config_without_clustering.maxClusterSize;

    xMinBoundingBox = config_without_clustering.xMinBoundingBox;
    xMaxBoundingBox = config_without_clustering.xMaxBoundingBox;
    yMinBoundingBox = config_without_clustering.yMinBoundingBox;
    yMaxBoundingBox = config_without_clustering.yMaxBoundingBox;
    zMinBoundingBox = config_without_clustering.zMinBoundingBox;
    zMaxBoundingBox = config_without_clustering.zMaxBoundingBox;

    leafSize  = config_without_clustering.leafSize;

    maxIterations = config_without_clustering.maxIterations;
    distanceThreshold = config_without_clustering.distanceThreshold;
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
    filter.setFilterLimits(-1.5, 0.0);          //적용할 값 (최소, 최대 값)
    filter.filter(*center);             //필터 적용 

    filter.setFilterLimitsNegative (true);     //적용할 값 외 
    filter.filter(*outskirt);

    // Y축 ROI
    // pcl::PassThrough<PointT> filter;
    filter.setInputCloud(center);                //입력 
    filter.setFilterFieldName("y");             //적용할 좌표 축 (eg. Y축)
    filter.setFilterLimits(-1.0, 1.0);          //적용할 값 (최소, 최대 값)
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

void mainCallback(const sensor_msgs::PointCloud2ConstPtr& input) {
    pcl::PointCloud<PointT>::Ptr cloudPtr;
    // main process method
    cloudPtr = ROI(input);
    cloudPtr = voxelGrid(cloudPtr);
}

void imuCallback(const sensor_msgs::Imu& msg) {
    orientation_x = msg.orientation.x;
    orientation_y = msg.orientation.y;
    orientation_z = msg.orientation.z;
    orientation_w = msg.orientation.w;
}



int main (int argc, char** argv) {
    // Initialize ROS
    ros::init (argc, argv, "od_without_clustering");
    ros::NodeHandle nh;
    
    dynamic_reconfigure::Server<lidar_object_detection::objectDetectorConfig> server;
    dynamic_reconfigure::Server<lidar_object_detection::objectDetectorConfig>::CallbackType f;

    f = boost::bind(&cfgCallback, _1, _2);
    server.setCallback(f);

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber main_sub = nh.subscribe("velodyne_points", 1, mainCallback);
    
    // Create a ROS publisher for the output point cloud
    pubROI = nh.advertise<sensor_msgs::PointCloud2> ("roi_raw", 1);

    // Spin
    ros::spin();
}