#pragma once
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>

struct Box { float x1,y1,x2,y2,score; int cls; };

struct YolopConfig {
    int inputW = 640;
    int inputH = 640;
    float confTh = 0.25f;
    float nmsTh  = 0.45f;
    int   numClasses = 1; // 차량만이면 1, 필요시 수정
    // YOLOP가 쓰는 stride/anchors는 모델 구현에 맞게 조정 (예: {8,16,32})
    std::vector<int> strides = {8,16,32};
    // anchors도 모델 기준으로 채워넣기 (여기선 placeholder)
    std::vector<float> anchors = {
        10,13, 16,30, 33,23,  // P3
        30,61, 62,45, 59,119, // P4
        116,90, 156,198, 373,326 // P5
    };
};

class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
};

class YolopTRT {
public:
    YolopTRT(const std::string& enginePath, const YolopConfig& cfg);
    ~YolopTRT();
    // 단일 이미지 추론
    void infer(const cv::Mat& bgr,
               std::vector<Box>& outBoxes,
               cv::Mat& daMask, cv::Mat& llMask,
               cv::Mat* vis=nullptr);

private:
    YolopConfig cfg_;
    Logger logger_;
    nvinfer1::IRuntime*     runtime_ = nullptr;
    nvinfer1::ICudaEngine*  engine_  = nullptr;
    nvinfer1::IExecutionContext* ctx_ = nullptr;

    int inIndex_=-1, detIndex_=-1, daIndex_=-1, llIndex_=-1;
    void* bindings_[4]{};
    size_t inSizeBytes_=0, detSizeBytes_=0, daSizeBytes_=0, llSizeBytes_=0;

    // letterbox + CHW float
    cv::Mat letterbox(const cv::Mat& img, float& scale, int& dx, int& dy);
    void postprocess(const cv::Size& origSz, float scale, int dx, int dy,
                     const float* det, int detCnt,
                     const float* da, const float* ll,
                     std::vector<Box>& outBoxes,
                     cv::Mat& daMask, cv::Mat& llMask,
                     cv::Mat* vis);
};
