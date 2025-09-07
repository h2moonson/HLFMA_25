#pragma once
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace yolop {

struct Params {
  float segTh   = 0.45f;  // 세그 확률 임계
  float detConf = 0.30f;  // 디텍션 confidence(obj*cls) 임계
  float nmsIou  = 0.45f;  // NMS IoU
  int   daMinA  = 600;    // 주행영역 최소 면적(픽셀)
  int   llMinA  = 150;    // 차선 최소 면적(픽셀)
};

struct Detection {
  cv::Rect2f box;
  float score;
  int cls; // YOLOP는 nc=1 기준 0
};

class YolopTRT {
public:
  YolopTRT(const std::string& planPath);
  ~YolopTRT();

  // 한 프레임 추론
  // inBGR: 입력 BGR 이미지
  // outOverlay: 오버레이(BGR)
  // outDaMask/outLlMask: 0/255 마스크(MONO8)
  // returns: 디텍션 리스트
  std::vector<Detection> infer(
    const cv::Mat& inBGR,
    cv::Mat& outOverlay,
    cv::Mat& outDaMask,
    cv::Mat& outLlMask,
    const Params& p = Params());

  int inputW() const { return W_; }
  int inputH() const { return H_; }

private:
  // utils
  static cv::Mat letterbox(const cv::Mat& img, int newW, int newH, int& padX, int& padY, float& scale);
  static void toCHWFP16(const cv::Mat& bgr, int H, int W, __half* dst);
  static void softmax2(const cv::Mat& c0, const cv::Mat& c1, cv::Mat& p1);
  static cv::Mat cleanupMask(const cv::Mat& bin01, int minArea, int morphK=3, int morphIters=1);
  static std::vector<Detection> nms(const std::vector<Detection>& src, float iouTh);
  static bool isNormalizedXYWH(float cx, float cy, float w, float h);

private:
  struct TRTLogger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
      if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << "\n";
    }
  };

  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  cudaStream_t stream_{};

  // tensor names (ONNX export와 동일)
  const char* IN_      = "images";
  const char* OUT_DET_ = "det_out";         // [1,25200,6]
  const char* OUT_DA_  = "drive_area_seg";  // [1,2,640,640] or [1,1,Ho,Wo]
  const char* OUT_LL_  = "lane_line_seg";   // [1,2,640,640] or [1,1,Ho,Wo]

  // dims
  int B_=1, C_=3, H_=640, W_=640;
  int daC_=2, llC_=2, Ho_=640, Wo_=640, H2_=640, W2_=640;

  // device buffers
  void *dIn_=nullptr, *dDa_=nullptr, *dLl_=nullptr, *dDet_=nullptr;
  size_t inBytes_=0, daBytes_=0, llBytes_=0, detBytes_=0;
  nvinfer1::DataType dtype_da_, dtype_ll_, dtype_det_;

  // helpers
  void allocBuffers_();
  void freeBuffers_();
};

} // namespace yolop
