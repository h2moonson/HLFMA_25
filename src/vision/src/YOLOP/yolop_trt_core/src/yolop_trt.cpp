#include <yolop_trt/yolop_trt.hpp>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace yolop_trt {

// ── Logger ───────────────────────────────────────────────────────────────────
struct TRTLogger : public ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << "\n";
  }
};
static TRTLogger gLogger;

// ── Utils ────────────────────────────────────────────────────────────────────
static std::vector<char> readFile(const std::string& path){
  std::ifstream ifs(path, std::ios::binary);
  if(!ifs) throw std::runtime_error("cannot open: " + path);
  return std::vector<char>(std::istreambuf_iterator<char>(ifs), {});
}

static cv::Mat letterbox(const cv::Mat& img, int newW, int newH,
                         int& padX, int& padY, float& scale){
  const int w = img.cols, h = img.rows;
  scale = std::min(1.0f * newW / w, 1.0f * newH / h);
  const int rw = int(w * scale);
  const int rh = int(h * scale);
  cv::Mat resized; cv::resize(img, resized, cv::Size(rw, rh));
  cv::Mat out(newH, newW, img.type(), cv::Scalar(114,114,114)); // pad=114
  padX = (newW - rw) / 2;
  padY = (newH - rh) / 2;
  resized.copyTo(out(cv::Rect(padX, padY, rw, rh)));
  return out;
}

static void toCHWFP16(const cv::Mat& bgr, int H, int W, __half* dst){
  cv::Mat resized; cv::resize(bgr, resized, cv::Size(W, H));
  cv::Mat rgb; cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0/255.0);
  for(int c=0;c<3;++c){
    for(int y=0;y<H;++y){
      const float* row = rgb.ptr<float>(y);
      for(int x=0;x<W;++x){
        float v = row[x*3 + c];
        dst[c*H*W + y*W + x] = __float2half(v);
      }
    }
  }
}

static inline bool isNormalizedXYWH(float cx, float cy, float w, float h){
  return (cx <= 1.5f && cy <= 1.5f && w <= 1.5f && h <= 1.5f);
}

static float iouRect(const cv::Rect2f& a, const cv::Rect2f& b){
  float inter = (a & b).area();
  float uni   = a.area() + b.area() - inter;
  return uni > 0.f ? inter/uni : 0.f;
}

static std::vector<Detection> nms(const std::vector<Detection>& src, float iouTh){
  std::vector<Detection> dets = src, keep;
  std::sort(dets.begin(), dets.end(), [](auto& A, auto& B){return A.score > B.score;});
  std::vector<int> alive(dets.size(), 1);
  for (size_t i=0;i<dets.size();++i){
    if(!alive[i]) continue;
    keep.push_back(dets[i]);
    for (size_t j=i+1;j<dets.size();++j){
      if(!alive[j]) continue;
      if (iouRect(dets[i].box, dets[j].box) > iouTh) alive[j] = 0;
    }
  }
  return keep;
}

static cv::Mat cleanupMask(const cv::Mat& bin01, int minArea, int morphK=3, int morphIters=1){
  CV_Assert(bin01.type()==CV_8U);
  cv::Mat m = bin01.clone();
  if (morphK > 1){
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morphK, morphK));
    cv::morphologyEx(m, m, cv::MORPH_OPEN, k, {-1,-1}, morphIters);
  }
  if (minArea <= 1) return m;

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(m, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::Mat out = cv::Mat::zeros(m.size(), CV_8U);
  for (auto& c : contours){
    if (std::fabs(cv::contourArea(c)) >= minArea)
      cv::drawContours(out, std::vector<std::vector<cv::Point>>{c}, -1, 255, cv::FILLED);
  }
  return out;
}

static void softmax2(const cv::Mat& c0, const cv::Mat& c1, cv::Mat& p1){
  CV_Assert(c0.size()==c1.size() && c0.type()==CV_32F && c1.type()==CV_32F);
  p1.create(c0.size(), CV_32F);
  for(int y=0;y<c0.rows;++y){
    const float* a = c0.ptr<float>(y);
    const float* b = c1.ptr<float>(y);
    float* p = p1.ptr<float>(y);
    for(int x=0;x<c0.cols;++x){
      float m = std::max(a[x], b[x]);
      float ea = std::exp(a[x]-m);
      float eb = std::exp(b[x]-m);
      p[x] = eb / (ea + eb);
    }
  }
}

// ── Impl ─────────────────────────────────────────────────────────────────────
class YolopTRT::Impl {
public:
  Params P;

  std::unique_ptr<IRuntime> runtime;
  std::unique_ptr<ICudaEngine> engine;
  std::unique_ptr<IExecutionContext> ctx;

  void* dIn = nullptr;
  void* dDa = nullptr;
  void* dLl = nullptr;
  void* dDet= nullptr;
  size_t inBytes=0, daBytes=0, llBytes=0, detBytes=0;

  DataType dtype_da{}, dtype_ll{}, dtype_det{};
  Dims inD{}, daD{}, llD{}, detD{};

  cudaStream_t stream = nullptr;
  int W=640, H=640;

  Impl(const std::string& plan, const Params& p): P(p), W(p.inputW), H(p.inputH){
    auto blob = readFile(plan);
    runtime.reset(createInferRuntime(gLogger));
    if(!runtime) throw std::runtime_error("createInferRuntime failed");
    engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if(!engine) throw std::runtime_error("deserializeCudaEngine failed");
    ctx.reset(engine->createExecutionContext());
    if(!ctx) throw std::runtime_error("createExecutionContext failed");

    const char* IN      = "images";
    const char* OUT_DET = "det_out";
    const char* OUT_DA  = "drive_area_seg";
    const char* OUT_LL  = "lane_line_seg";

    // 입력 shape 설정(동적일 수 있음)
    inD = engine->getTensorShape(IN);
    if (inD.nbDims == 0 || inD.d[2] <= 0 || inD.d[3] <= 0){
      ctx->setInputShape(IN, Dims4{1,3,H,W});
      inD = ctx->getTensorShape(IN);
    }
    if (inD.d[0] != 1 || inD.d[1] != 3) throw std::runtime_error("Expected input [1,3,H,W]");

    daD = ctx->getTensorShape(OUT_DA);
    llD = ctx->getTensorShape(OUT_LL);
    detD= ctx->getTensorShape(OUT_DET);

    dtype_da = engine->getTensorDataType(OUT_DA);
    dtype_ll = engine->getTensorDataType(OUT_LL);
    dtype_det= engine->getTensorDataType(OUT_DET);

    auto safeMul = [](const Dims& d)->size_t{
      size_t prod=1; for (int i=0;i<d.nbDims;++i){ long long v=d.d[i]; if(v<1) v=1; prod*= (size_t)v; } return prod;
    };

    inBytes  = (size_t)inD.d[0]*inD.d[1]*inD.d[2]*inD.d[3]*sizeof(__half);
    daBytes  = safeMul(daD) * (dtype_da==DataType::kHALF ? sizeof(__half) : sizeof(float));
    llBytes  = safeMul(llD) * (dtype_ll==DataType::kHALF ? sizeof(__half) : sizeof(float));
    detBytes = safeMul(detD)* (dtype_det==DataType::kHALF? sizeof(__half): sizeof(float));

    cudaStreamCreate(&stream);
    cudaMalloc(&dIn,  inBytes);
    cudaMalloc(&dDa,  daBytes);
    cudaMalloc(&dLl,  llBytes);
    cudaMalloc(&dDet, detBytes);

    ctx->setTensorAddress(IN,      dIn);
    ctx->setTensorAddress(OUT_DA,  dDa);
    ctx->setTensorAddress(OUT_LL,  dLl);
    ctx->setTensorAddress(OUT_DET, dDet);
  }

  ~Impl(){
    if (stream) cudaStreamDestroy(stream);
    if (dIn)  cudaFree(dIn);
    if (dDa)  cudaFree(dDa);
    if (dLl)  cudaFree(dLl);
    if (dDet) cudaFree(dDet);
  }
};

YolopTRT::YolopTRT(const std::string& planPath, const Params& p)
: impl_(std::make_unique<Impl>(planPath, p)) {}

YolopTRT::~YolopTRT() = default;

bool YolopTRT::valid() const { return (bool)impl_; }

void YolopTRT::infer(const cv::Mat& imgBGR,
                     cv::Mat* overlay,
                     cv::Mat* daMaskOut,
                     cv::Mat* llMaskOut,
                     std::vector<Detection>* detsOut){
  auto& I = *impl_;
  if (imgBGR.empty()) throw std::runtime_error("empty image");

  int padX=0, padY=0; float scale=1.0f;
  cv::Mat lb = letterbox(imgBGR, I.W, I.H, padX, padY, scale);

  std::vector<__half> inHost((size_t)I.inD.d[0]*I.inD.d[1]*I.inD.d[2]*I.inD.d[3]);
  toCHWFP16(lb, I.inD.d[2], I.inD.d[3], inHost.data());
  cudaMemcpyAsync(I.dIn, inHost.data(), I.inBytes, cudaMemcpyHostToDevice, I.stream);

  if(!I.ctx->enqueueV3(I.stream)) throw std::runtime_error("enqueueV3 failed");
  cudaStreamSynchronize(I.stream);

  auto copyOut = [&](void* dptr, size_t bytes, nvinfer1::DataType dt, std::vector<float>& host){
    if (dt == DataType::kHALF){
      std::vector<__half> tmp(bytes/sizeof(__half));
      cudaMemcpyAsync(tmp.data(), dptr, bytes, cudaMemcpyDeviceToHost, I.stream);
      cudaStreamSynchronize(I.stream);
      host.resize(tmp.size());
      for(size_t i=0;i<tmp.size();++i) host[i] = __half2float(tmp[i]);
    } else {
      host.resize(bytes/sizeof(float));
      cudaMemcpyAsync(host.data(), dptr, bytes, cudaMemcpyDeviceToHost, I.stream);
      cudaStreamSynchronize(I.stream);
    }
  };

  std::vector<float> daHost, llHost, detHost;
  copyOut(I.dDa,  I.daBytes,  I.dtype_da,  daHost);
  copyOut(I.dLl,  I.llBytes,  I.dtype_ll,  llHost);
  copyOut(I.dDet, I.detBytes, I.dtype_det, detHost);

  int daC = (I.daD.nbDims>=2? I.daD.d[1]:-1);
  int llC = (I.llD.nbDims>=2? I.llD.d[1]:-1);
  int Ho  = (I.daD.nbDims>=4? I.daD.d[2]:-1), Wo=(I.daD.nbDims>=4? I.daD.d[3]:-1);
  int H2  = (I.llD.nbDims>=4? I.llD.d[2]:-1), W2=(I.llD.nbDims>=4? I.llD.d[3]:-1);

  // ── 세그 확률 → 마스크 (letterbox 공간)
  cv::Mat daMaskLb, llMaskLb;
  if (daC == 2){
    cv::Mat c0(Ho, Wo, CV_32F, daHost.data() + 0*Ho*Wo);
    cv::Mat c1(Ho, Wo, CV_32F, daHost.data() + 1*Ho*Wo);
    cv::Mat prob1; softmax2(c0, c1, prob1);
    daMaskLb = (prob1 > I.P.segTh);
  } else if (daC == 1){
    cv::Mat prob(Ho, Wo, CV_32F, daHost.data());
    daMaskLb = (prob > I.P.segTh);
  } else throw std::runtime_error("Unexpected DA channels");

  if (llC == 2){
    cv::Mat c0(H2, W2, CV_32F, llHost.data() + 0*H2*W2);
    cv::Mat c1(H2, W2, CV_32F, llHost.data() + 1*H2*W2);
    cv::Mat prob1; softmax2(c0, c1, prob1);
    llMaskLb = (prob1 > I.P.segTh);
  } else if (llC == 1){
    cv::Mat prob(H2, W2, CV_32F, llHost.data());
    llMaskLb = (prob > I.P.segTh);
  } else throw std::runtime_error("Unexpected LL channels");

  // ── unletterbox → 원본 크기
  int rw = int(imgBGR.cols * scale);
  int rh = int(imgBGR.rows * scale);
  int x0 = std::max(0, padX), y0 = std::max(0, padY);
  int w0 = std::min(I.W - x0, rw);
  int h0 = std::min(I.H - y0, rh);
  cv::Mat daCrop = daMaskLb(cv::Rect(x0, y0, w0, h0));
  cv::Mat llCrop = llMaskLb(cv::Rect(x0, y0, w0, h0));

  cv::Mat daMask, llMask;
  cv::resize(daCrop, daMask, imgBGR.size(), 0, 0, cv::INTER_NEAREST);
  cv::resize(llCrop, llMask, imgBGR.size(), 0, 0, cv::INTER_NEAREST);

  daMask = cleanupMask(daMask, I.P.daMinArea, 3, 1);
  llMask = cleanupMask(llMask, I.P.llMinArea, 3, 1);

  // ── 디텍션 decode → unletterbox → NMS
  std::vector<Detection> dets;
  int stride = I.detD.d[2]; // 6
  int num    = I.detD.d[1]; // 25200
  bool norm  = true;

  for (int i=0;i<num; ++i){
    float cx = detHost[i*stride + 0];
    float cy = detHost[i*stride + 1];
    float w  = detHost[i*stride + 2];
    float h  = detHost[i*stride + 3];
    float obj= detHost[i*stride + 4];
    float cls= detHost[i*stride + 5];

    if (i==0) norm = isNormalizedXYWH(cx,cy,w,h);
    float score = obj * cls;
    if (score < I.P.detConf) continue;

    if (norm){ cx *= I.W; cy *= I.H; w *= I.W; h *= I.H; }

    float x1 = cx - w*0.5f, y1 = cy - h*0.5f;
    float x2 = cx + w*0.5f, y2 = cy + h*0.5f;

    x1 -= padX; x2 -= padX; y1 -= padY; y2 -= padY;
    if (scale > 0){ x1/=scale; x2/=scale; y1/=scale; y2/=scale; }

    x1 = std::clamp(x1, 0.f, (float)imgBGR.cols-1);
    y1 = std::clamp(y1, 0.f, (float)imgBGR.rows-1);
    x2 = std::clamp(x2, 0.f, (float)imgBGR.cols-1);
    y2 = std::clamp(y2, 0.f, (float)imgBGR.rows-1);

    if (x2-x1 < 1.f || y2-y1 < 1.f) continue;
    dets.push_back( Detection{ cv::Rect2f(x1,y1, x2-x1, y2-y1), score, 0 } );
  }
  dets = nms(dets, I.P.nmsIou);

  // ── overlay 출력
  if (overlay){
    cv::Mat vis; imgBGR.copyTo(vis);
    cv::Mat tintDA(vis.size(), CV_8UC3, cv::Scalar(0,0,255));
    cv::Mat tintLL(vis.size(), CV_8UC3, cv::Scalar(0,255,0));

    cv::Mat visDA = vis.clone(); tintDA.copyTo(visDA, daMask);
    cv::addWeighted(vis, 0.55, visDA, 0.45, 0.0, vis);

    cv::Mat visLL = vis.clone(); tintLL.copyTo(visLL, llMask);
    cv::addWeighted(vis, 0.55, visLL, 0.45, 0.0, vis);

    for (auto& d : dets){
      cv::rectangle(vis, d.box, cv::Scalar(255,0,0), 2);
    }
    *overlay = std::move(vis);
  }
  if (daMaskOut) *daMaskOut = std::move(daMask);
  if (llMaskOut) *llMaskOut = std::move(llMask);
  if (detsOut)   *detsOut   = std::move(dets);
}

} // namespace yolop_trt
