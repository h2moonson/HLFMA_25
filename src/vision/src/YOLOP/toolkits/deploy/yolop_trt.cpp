#include "yolop_trt.hpp"

// 여기에 필요 헤더들 포함(당신 레포 기준)
// - TensorRT / CUDA 헤더 (NvInfer.h, cuda_runtime.h 등)
// - 기존 헬퍼: yolov5.hpp (APIToModel, doInference, nms, 상수들)
// - common.hpp/common.cpp에서 제공하는 유틸(전처리/버퍼)

#include "yolov5.hpp"     // ← 반드시 제공 필요
#include "common.hpp"     // ← 반드시 제공 필요 (없다면 보내 주세요)

#include <NvInfer.h>
#include <cuda_runtime.h>

struct YolopTRT::Impl {
  // TensorRT core
  nvinfer1::IRuntime* runtime{nullptr};
  nvinfer1::ICudaEngine* engine{nullptr};
  nvinfer1::IExecutionContext* context{nullptr};
  cudaStream_t stream{};

  // 입력 크기/임계값
  int in_w{640}, in_h{640};
  float conf_thresh{0.25f}, nms_thresh{0.45f};

  // GPU buffers
  void* buffers[4]{};
  // host outputs
  std::vector<float> det_out;
  std::vector<int>   seg_out, lane_out;

  Impl(int w, int h, float conf, float nms)
    : in_w(w), in_h(h), conf_thresh(conf), nms_thresh(nms) {}
};

YolopTRT::YolopTRT(const std::string& engine_path, int in_w, int in_h,
                   float conf_thresh, float nms_thresh, bool fp16) {
  impl_ = new Impl(in_w, in_h, conf_thresh, nms_thresh);

  // 1) .engine 로드 (없으면 .wts로 APIToModel 빌드 → 저장) - 당신 레포 방식 재사용
  //    여기서는 간단히 .engine 존재한다고 가정. 없다면 main.cpp 로직을 옮겨오세요.
  std::ifstream f(engine_path, std::ios::binary);
  if (!f.good()) {
    throw std::runtime_error("engine file not found: " + engine_path);
  }
  f.seekg(0, f.end); size_t size = f.tellg(); f.seekg(0, f.beg);
  std::vector<char> trt(size); f.read(trt.data(), size); f.close();

  impl_->runtime = createInferRuntime(gLogger);
  impl_->engine  = impl_->runtime->deserializeCudaEngine(trt.data(), size);
  impl_->context = impl_->engine->createExecutionContext();
  cudaStreamCreate(&impl_->stream);

  // 2) 바인딩/버퍼 준비 (main.cpp와 동일)
  int inputIndex        = impl_->engine->getBindingIndex(INPUT_BLOB_NAME);
  int output_det_index  = impl_->engine->getBindingIndex(OUTPUT_DET_NAME);
  int output_seg_index  = impl_->engine->getBindingIndex(OUTPUT_SEG_NAME);
  int output_lane_index = impl_->engine->getBindingIndex(OUTPUT_LANE_NAME);

  // host side sizes (main.cpp의 OUTPUT_SIZE/IMG_W/IMG_H 상수 사용)
  impl_->det_out.resize(BATCH_SIZE * OUTPUT_SIZE);
  impl_->seg_out.resize(BATCH_SIZE * IMG_H * IMG_W);
  impl_->lane_out.resize(BATCH_SIZE * IMG_H * IMG_W);

  // GPU malloc
  CUDA_CHECK(cudaMalloc(&impl_->buffers[inputIndex],        BATCH_SIZE * 3 * impl_->in_h * impl_->in_w * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&impl_->buffers[output_det_index],  impl_->det_out.size()  * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&impl_->buffers[output_seg_index],  impl_->seg_out.size()  * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&impl_->buffers[output_lane_index], impl_->lane_out.size() * sizeof(int)));
}

YolopTRT::~YolopTRT() {
  if (!impl_) return;
  cudaStreamDestroy(impl_->stream);
  for (void*& b : impl_->buffers) if (b) cudaFree(b);
  if (impl_->context) impl_->context->destroy();
  if (impl_->engine)  impl_->engine->destroy();
  if (impl_->runtime) impl_->runtime->destroy();
  delete impl_;
}

void YolopTRT::infer(const cv::Mat& bgr, YolopOutput& out) {
  // 1) 전처리 (GPU letterbox) → input buffer 채우기
  cv::Mat bgr_in = bgr; // 필요 시 복사/리사이즈
  preprocess_img_gpu(bgr_in, (float*)impl_->buffers[0], impl_->in_w, impl_->in_h); // 당신 레포의 함수 사용

  // 2) 엔진 실행
  doInference(*impl_->context, impl_->stream, impl_->buffers,
              impl_->det_out.data(), impl_->seg_out.data(), impl_->lane_out.data(),
              BATCH_SIZE);

  // 3) 후처리 (NMS, seg/lane resize)
  std::vector<Yolo::Detection> dets;
  nms(dets, impl_->det_out.data(), impl_->conf_thresh, impl_->nms_thresh);
  out.dets.clear();
  out.dets.reserve(dets.size());
  for (auto& d : dets) {
    Det dd;
    dd.bbox = cv::Rect((int)d.bbox[0], (int)d.bbox[1], (int)d.bbox[2], (int)d.bbox[3]);
    dd.cls  = d.class_id;
    dd.conf = d.conf;
    out.dets.push_back(dd);
  }

  cv::Mat seg_in(IMG_H, IMG_W, CV_32S, impl_->seg_out.data());
  cv::Mat lane_in(IMG_H, IMG_W, CV_32S, impl_->lane_out.data());
  cv::resize(seg_in,  out.da_mask,  bgr.size(), 0,0, cv::INTER_NEAREST);
  cv::resize(lane_in, out.ll_mask,  bgr.size(), 0,0, cv::INTER_NEAREST);
}

cv::Mat YolopTRT::overlay(const cv::Mat& bgr, const YolopOutput& out) {
  cv::Mat vis; bgr.copyTo(vis);
  // seg 색칠
  cv::Mat color = vis.clone();
  color.setTo(cv::Scalar(0,255,0), out.da_mask > 0);
  color.setTo(cv::Scalar(0,0,255), out.ll_mask > 0);
  cv::addWeighted(color, 0.3, vis, 0.7, 0, vis);
  // bbox
  for (auto& d : out.dets) cv::rectangle(vis, d.bbox, {255,200,0}, 2);
  return vis;
}
