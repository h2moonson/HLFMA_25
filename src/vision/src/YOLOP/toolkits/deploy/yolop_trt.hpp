#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Det {
  cv::Rect bbox;
  int cls;
  float conf;
};

struct YolopOutput {
  std::vector<Det> dets;   // NMS 결과
  cv::Mat da_mask;         // CV_8UC1
  cv::Mat ll_mask;         // CV_8UC1
};

class YolopTRT {
public:
  YolopTRT(const std::string& engine_path, int in_w, int in_h,
           float conf_thresh, float nms_thresh, bool fp16);

  ~YolopTRT();

  // 입력 BGR1장 → 추론 결과
  void infer(const cv::Mat& bgr, YolopOutput& out);

  // 디버그 오버레이
  cv::Mat overlay(const cv::Mat& bgr, const YolopOutput& out);

private:
  // 내부 구현 포인터(PImpl) 또는 직접 멤버들:
  struct Impl; Impl* impl_;
};
