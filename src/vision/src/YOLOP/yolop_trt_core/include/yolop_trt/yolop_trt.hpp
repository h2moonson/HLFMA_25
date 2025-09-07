#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace yolop_trt {

struct Detection {
  cv::Rect2f box;
  float score;
  int   cls;
};

struct Params {
  // 하이퍼파라미터
  float segTh     = 0.45f;  // 세그 확률 임계값
  float detConf   = 0.30f;  // obj*cls 임계값
  float nmsIou    = 0.45f;  // NMS IoU
  int   daMinArea = 600;    // drive-area 마스크 잡티 최소 면적(px)
  int   llMinArea = 150;    // lane-line  마스크 잡티 최소 면적(px)

  // 입력 크기
  int inputW = 640;
  int inputH = 640;
};

class YolopTRT {
public:
  explicit YolopTRT(const std::string& enginePlanPath, const Params& p = Params{});
  ~YolopTRT();

  bool valid() const;

  // 단일 프레임 추론
  void infer(const cv::Mat& bgr,
             cv::Mat* overlay,                          // 시각화 결과 (원본 크기)
             cv::Mat* daMask = nullptr,                 // 0/255 마스크 (원본 크기)
             cv::Mat* llMask = nullptr,                 // 0/255 마스크 (원본 크기)
             std::vector<Detection>* dets = nullptr);   // 박스 결과
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace yolop_trt
