#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2DArray.h>
#include <opencv2/opencv.hpp>

#include "yolop_trt.hpp"   // ↓ 2) 래퍼 헤더

class YolopTRTNode {
public:
  YolopTRTNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : it_(nh)
  {
    // 파라미터
    pnh.param<std::string>("engine_path", engine_path_, "yolop.engine");
    pnh.param<int>("input_w", in_w_, 640);
    pnh.param<int>("input_h", in_h_, 640);
    pnh.param<float>("conf_thresh", conf_thresh_, 0.25f);
    pnh.param<float>("nms_thresh",  nms_thresh_,  0.45f);
    pnh.param<bool>("fp16", fp16_, true);

    // TensorRT 핸들 준비 (엔진 없으면 내부에서 빌드되게 구현 가능)
    yolop_.reset(new YolopTRT(engine_path_, in_w_, in_h_, conf_thresh_, nms_thresh_, fp16_));

    // 토픽
    sub_ = it_.subscribe("image", 1, &YolopTRTNode::imageCb, this);
    pub_overlay_ = it_.advertise("overlay", 1);
    pub_da_      = it_.advertise("da_mask", 1);
    pub_ll_      = it_.advertise("ll_mask", 1);
    pub_det_     = nh.advertise<vision_msgs::Detection2DArray>("detections", 1);
  }

private:
  void imageCb(const sensor_msgs::ImageConstPtr& msg) {
    cv::Mat bgr;
    try {
      bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (...) {
      ROS_WARN("cv_bridge failed");
      return;
    }

    YolopOutput out;
    yolop_->infer(bgr, out);  // TensorRT 추론 (아래 2) 참조)

    // 퍼블리시: overlay
    cv::Mat overlay = yolop_->overlay(bgr, out);
    sensor_msgs::ImagePtr ov_msg = cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg();
    pub_overlay_.publish(ov_msg);

    // DA/LL mask
    sensor_msgs::ImagePtr da_msg = cv_bridge::CvImage(msg->header, "mono8", out.da_mask).toImageMsg();
    sensor_msgs::ImagePtr ll_msg = cv_bridge::CvImage(msg->header, "mono8", out.ll_mask).toImageMsg();
    pub_da_.publish(da_msg);
    pub_ll_.publish(ll_msg);

    // Detection2DArray (간단 변환)
    vision_msgs::Detection2DArray detarr;
    detarr.header = msg->header;
    for (auto& d : out.dets) {
      vision_msgs::Detection2D det;
      det.bbox.center.x = (d.bbox.x + d.bbox.width/2.0);
      det.bbox.center.y = (d.bbox.y + d.bbox.height/2.0);
      det.bbox.size_x = d.bbox.width;
      det.bbox.size_y = d.bbox.height;
      vision_msgs::ObjectHypothesisWithPose hyp;
      hyp.id = d.cls;
      hyp.score = d.conf;
      det.results.push_back(hyp);
      detarr.detections.push_back(det);
    }
    pub_det_.publish(detarr);
  }

  image_transport::ImageTransport it_;
  image_transport::Subscriber sub_;
  image_transport::Publisher pub_overlay_, pub_da_, pub_ll_;
  ros::Publisher pub_det_;

  std::unique_ptr<YolopTRT> yolop_;
  std::string engine_path_;
  int in_w_{640}, in_h_{640};
  float conf_thresh_{0.25f}, nms_thresh_{0.45f};
  bool fp16_{true};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "yolop_trt_node");
  ros::NodeHandle nh, pnh("~");
  YolopTRTNode node(nh, pnh);
  ros::spin();
  return 0;
}
