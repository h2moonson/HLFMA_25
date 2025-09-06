#include "yolop_trt.hpp"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

class YolopNode {
public:
  YolopNode(ros::NodeHandle& nh): nh_(nh){
    nh_.param<std::string>("engine_path", engine_, "/path/yolop.plan");
    nh_.param<std::string>("image_topic", imgTopic_, "/camera/image_raw");
    pubVis_ = nh_.advertise<sensor_msgs::Image>("/yolop/vis",1);
    sub_    = nh_.subscribe(imgTopic_,1,&YolopNode::cb,this);

    YolopConfig cfg; trt_.reset(new YolopTRT(engine_, cfg));
  }

  void cb(const sensor_msgs::ImageConstPtr& msg){
    cv::Mat bgr = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    std::vector<Box> boxes; cv::Mat da,ll,vis;
    trt_->infer(bgr, boxes, da, ll, &vis);

    // 간단 시각화
    for(auto& b: boxes)
      cv::rectangle(bgr, cv::Rect(cv::Point(b.x1,b.y1), cv::Point(b.x2,b.y2)), {0,255,0}, 2);

    auto m = cv_bridge::CvImage(msg->header, "bgr8", bgr).toImageMsg();
    pubVis_.publish(m);
  }
private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher  pubVis_;
  std::unique_ptr<YolopTRT> trt_;
  std::string engine_, imgTopic_;
};

int main(int argc, char** argv){
  ros::init(argc, argv, "yolop_trt");
  ros::NodeHandle nh("~");
  YolopNode node(nh);
  ros::spin();
  return 0;
}
