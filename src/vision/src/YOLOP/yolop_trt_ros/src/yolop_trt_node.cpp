#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include "tensorrt/yolop_trt.hpp"

class YolopNode {
public:
  YolopNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : it_(nh)
  {
    // params
    pnh.param<std::string>("engine_path", enginePath_, std::string("../yolop.plan"));
    pnh.param<std::string>("image_topic", imageTopic_, std::string("/camera/image_raw"));
    pnh.param<bool>("publish_overlay", pubOverlay_, true);
    pnh.param<bool>("publish_masks",   pubMasks_,   true);

    pnh.param("seg_th",   params_.segTh,   0.45f);
    pnh.param("det_conf", params_.detConf, 0.30f);
    pnh.param("nms_iou",  params_.nmsIou,  0.45f);
    pnh.param("da_min_area", params_.daMinA, 600);
    pnh.param("ll_min_area", params_.llMinA, 150);

    ROS_INFO_STREAM("Loading TensorRT engine: " << enginePath_);
    infer_.reset(new yolop::YolopTRT(enginePath_));

    if (pubOverlay_)
      pubOverlayImg_ = it_.advertise("/yolop/overlay", 1);
    if (pubMasks_) {
      pubDaMask_ = it_.advertise("/yolop/drive_area_mask", 1);
      pubLlMask_ = it_.advertise("/yolop/lane_line_mask", 1);
    }

    sub_ = it_.subscribe(imageTopic_, 1, &YolopNode::imageCb, this);
    ROS_INFO_STREAM("Subscribed: " << imageTopic_);
  }

private:
  void imageCb(const sensor_msgs::ImageConstPtr& msg){
    cv::Mat bgr;
    try {
      bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const std::exception& e){
      ROS_WARN_STREAM("cv_bridge error: " << e.what());
      return;
    }

    cv::Mat overlay, daMask, llMask;
    auto dets = infer_->infer(bgr, overlay, daMask, llMask, params_);

    ros::Time stamp = msg->header.stamp;
    std_msgs::Header header = msg->header;

    if (pubOverlay_) {
      sensor_msgs::ImagePtr out = cv_bridge::CvImage(header, "bgr8", overlay).toImageMsg();
      pubOverlayImg_.publish(out);
    }
    if (pubMasks_) {
      sensor_msgs::ImagePtr da = cv_bridge::CvImage(header, "mono8", daMask).toImageMsg();
      sensor_msgs::ImagePtr ll = cv_bridge::CvImage(header, "mono8", llMask).toImageMsg();
      pubDaMask_.publish(da);
      pubLlMask_.publish(ll);
    }
    // 필요하면 dets를 별도 메시지로 내보내도록 확장 가능(vision_msgs 등)
  }

  std::string enginePath_, imageTopic_;
  bool pubOverlay_{true}, pubMasks_{true};

  image_transport::ImageTransport it_;
  image_transport::Subscriber sub_;
  image_transport::Publisher pubOverlayImg_, pubDaMask_, pubLlMask_;

  yolop::Params params_;
  std::unique_ptr<yolop::YolopTRT> infer_;
};

int main(int argc, char** argv){
  ros::init(argc, argv, "yolop_trt_node");
  ros::NodeHandle nh, pnh("~");
  try{
    YolopNode node(nh, pnh);
    ros::spin();
  } catch(const std::exception& e){
    ROS_FATAL_STREAM("Fatal: " << e.what());
    return 1;
  }
  return 0;
}
