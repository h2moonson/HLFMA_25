// yolop_trt_min.cpp — YOLOP TensorRT (ONNX-plan) 데모
// Build: see CMakeLists.txt
// Run:
//   ./yolop_trt_min ../yolop.plan ../../../../inference/images ./out \
//       --seg-th 0.45 --det-conf 0.30 --nms-iou 0.45 --da-min-area 600 --ll-min-area 150
//
// 기능 요약:
//  - 입력 letterbox(114) → 출력 언레터박스(원본 좌표 복원)
//  - BGR->RGB, /255.0 정규화
//  - 세그: softmax/sigmoid 자동 분기 + 마스크 클린업
//  - 디텍션: [1,25200,6] 디코드 + NMS + 드로잉
//  - FP16/FP32 출력 자동 처리
//  - 중간 확률맵/마스크 PNG 저장

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <cuda_fp16.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace nvinfer1;

// ── CLI 파라미터 ─────────────────────────────────────────────────────────────
struct Args {
  std::string planPath = "../yolop.plan";
  std::string inDir    = "../../../inference/images";
  std::string outDir   = "./out";
  float segTh   = 0.40f;
  float detConf = 0.30f;
  float nmsIou  = 0.45f;
  int   daMinA  = 600;
  int   llMinA  = 100;
};
static Args parse_args(int argc, char** argv){
  Args a;
  if (argc>=2) a.planPath = argv[1];
  if (argc>=3) a.inDir    = argv[2];
  if (argc>=4) a.outDir   = argv[3];
  for (int i=4;i<argc;i++){
    if (!strcmp(argv[i],"--seg-th") && i+1<argc)    a.segTh = std::stof(argv[++i]);
    else if (!strcmp(argv[i],"--det-conf") && i+1<argc) a.detConf = std::stof(argv[++i]);
    else if (!strcmp(argv[i],"--nms-iou") && i+1<argc)  a.nmsIou = std::stof(argv[++i]);
    else if (!strcmp(argv[i],"--da-min-area") && i+1<argc) a.daMinA = std::stoi(argv[++i]);
    else if (!strcmp(argv[i],"--ll-min-area") && i+1<argc) a.llMinA = std::stoi(argv[++i]);
  }
  return a;
}

// ── TRT Logger ───────────────────────────────────────────────────────────────
struct Log : public ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << "\n";
  }
} gLogger;

// ── 유틸 ─────────────────────────────────────────────────────────────────────
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
  cv::Mat out(newH, newW, img.type(), cv::Scalar(114,114,114));  // pad=114
  padX = (newW - rw) / 2;
  padY = (newH - rh) / 2;
  resized.copyTo(out(cv::Rect(padX, padY, rw, rh)));
  return out;
}

// BGR->RGB, /255.0, CHW FP16
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

static void printTensorInfo(ICudaEngine* engine, const char* name){
  auto mode = engine->getTensorIOMode(name);
  auto dtype = engine->getTensorDataType(name);
  auto dims = engine->getTensorShape(name);
  std::cout << "  - " << name
            << " (" << (mode==TensorIOMode::kINPUT?"INPUT":"OUTPUT")
            << ") dtype=" << (int)dtype << " dims=[";
  for(int k=0;k<dims.nbDims;++k){ std::cout << dims.d[k] << (k+1<dims.nbDims?",":""); }
  std::cout << "]\n";
}

template<typename T>
static void print_minmax(const std::vector<T>& v, const char* tag){
  if(v.empty()){ std::cout << "[DBG] " << tag << " empty\n"; return; }
  double mn=1e30, mx=-1e30;
  for(auto &x: v){ double f = (double)x; mn = std::min(mn,f); mx = std::max(mx,f); }
  std::cout << "[DBG] " << tag << " min="<< mn << " max="<< mx << "\n";
}

// ── Detection helpers ────────────────────────────────────────────────────────
struct Det {
  cv::Rect2f box;
  float score;
  int cls;
};

static float iouRect(const cv::Rect2f& a, const cv::Rect2f& b){
  float inter = (a & b).area();
  float uni   = a.area() + b.area() - inter;
  return uni > 0.f ? inter/uni : 0.f;
}

static std::vector<Det> nms(const std::vector<Det>& src, float iouTh=0.45f){
  std::vector<Det> dets = src, keep;
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

static inline bool isNormalizedXYWH(float cx, float cy, float w, float h){
  // 아주 러프하게: 값이 ~1.5 이하이면 정규화된 것으로 간주(640 스케일링 전)
  return (cx <= 1.5f && cy <= 1.5f && w <= 1.5f && h <= 1.5f);
}

static void drawDetections(cv::Mat& vis,
                           const std::vector<Det>& dets,
                           const cv::Scalar& color = cv::Scalar(255,0,0)) {
  for (auto& d : dets){
    cv::rectangle(vis, d.box, color, 2);
    char buf[64]; std::snprintf(buf, sizeof(buf), "cls=%d %.2f", d.cls, d.score);
    int base=0; auto size = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    cv::Rect2f r = d.box;
    cv::Rect bg((int)r.x, std::max(0, (int)r.y- size.height-4), size.width+6, size.height+4);
    bg &= cv::Rect(0,0, vis.cols, vis.rows);
    cv::rectangle(vis, bg, color, cv::FILLED);
    cv::putText(vis, buf, {bg.x+3, bg.y+bg.height-3}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);
  }
}

// ── Mask cleanup: morphology + min-area ──────────────────────────────────────
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

// ── 2-채널 softmax에서 channel1 확률 ───────────────────────────────────────
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

int main(int argc, char** argv){
  try{
    Args args = parse_args(argc, argv);
    std::filesystem::create_directories(args.outDir);

    // 1) Load engine
    std::vector<char> blob = readFile(args.planPath);
    std::unique_ptr<IRuntime> runtime(createInferRuntime(gLogger));
    if(!runtime) throw std::runtime_error("createInferRuntime failed");
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if(!engine) throw std::runtime_error("deserializeCudaEngine failed");

    std::unique_ptr<IExecutionContext> ctx(engine->createExecutionContext());
    if(!ctx) throw std::runtime_error("createExecutionContext failed");

    // 바인딩 이름 (export_onnx_trt.py와 동일)
    const char* IN      = "images";
    const char* OUT_DET = "det_out";         // [1,25200,6]
    const char* OUT_DA  = "drive_area_seg";  // [1,2,640,640] or [1,1,Ho,Wo]
    const char* OUT_LL  = "lane_line_seg";   // [1,2,640,640] or [1,1,Ho,Wo]

    // 바인딩 전체 출력
    std::cout << "[TRT] I/O tensors:\n";
    int nIO = engine->getNbIOTensors();
    for(int i=0;i<nIO;++i){
      const char* nm = engine->getIOTensorName(i);
      auto mode = engine->getTensorIOMode(nm);
      if (mode==TensorIOMode::kINPUT || mode==TensorIOMode::kOUTPUT)
        printTensorInfo(engine.get(), nm);
    }

    // 2) 입력/출력 shape
    Dims inD = engine->getTensorShape(IN); // [-1,3,640,640] or [1,3,640,640]
    int B = (inD.d[0] > 0) ? inD.d[0] : 1;
    int C = inD.d[1], H = inD.d[2], W = inD.d[3];
    if (H < 0 || W < 0) {
      ctx->setInputShape(IN, Dims4{1,3,640,640});
      Dims inD2 = ctx->getTensorShape(IN);
      B = inD2.d[0]; C = inD2.d[1]; H = inD2.d[2]; W = inD2.d[3];
    }
    if (B != 1 || C != 3) throw std::runtime_error("Expected input [1,3,H,W]");
    std::cout << "[DBG] Input shape: " << B << "x" << C << "x" << H << "x" << W << "\n";

    auto daD = ctx->getTensorShape(OUT_DA);
    auto llD = ctx->getTensorShape(OUT_LL);
    int daC = (daD.nbDims>=2? daD.d[1]:-1);
    int llC = (llD.nbDims>=2? llD.d[1]:-1);
    int Ho  = (daD.nbDims>=4? daD.d[2]:-1), Wo = (daD.nbDims>=4? daD.d[3]:-1);
    int H2  = (llD.nbDims>=4? llD.d[2]:-1), W2 = (llD.nbDims>=4? llD.d[3]:-1);

    // 3) GPU buffer alloc
    size_t inBytes = (size_t)B*C*H*W*sizeof(__half);

    auto safeMulElems = [](const Dims& d)->size_t{
      size_t prod = 1;
      for (int i=0;i<d.nbDims;++i) {
        int64_t vi = static_cast<int64_t>(d.d[i]);
        if (vi < 1) vi = 1;
        prod *= static_cast<size_t>(vi);
      }
      return prod;
    };

    size_t daElems = safeMulElems(daD);
    size_t llElems = safeMulElems(llD);

    Dims detD = ctx->getTensorShape(OUT_DET);               // [1,25200,6]
    auto dtype_det = engine->getTensorDataType(OUT_DET);
    size_t detElems = safeMulElems(detD);
    size_t detBytes = detElems * (dtype_det==DataType::kHALF ? sizeof(__half) : sizeof(float));

    auto dtype_da = engine->getTensorDataType(OUT_DA);
    auto dtype_ll = engine->getTensorDataType(OUT_LL);
    size_t daBytes = daElems * (dtype_da==DataType::kHALF ? sizeof(__half) : sizeof(float));
    size_t llBytes = llElems * (dtype_ll==DataType::kHALF ? sizeof(__half) : sizeof(float));

    std::cout << "[DBG] DET dtype="<<(int)dtype_det<<" elems="<<detElems<<" bytes="<<detBytes<<"\n";
    std::cout << "[DBG] DA  dtype="<<(int)dtype_da <<" elems="<<daElems <<" bytes="<<daBytes <<"\n";
    std::cout << "[DBG] LL  dtype="<<(int)dtype_ll <<" elems="<<llElems <<" bytes="<<llBytes <<"\n";

    void *dIn=nullptr, *dDa=nullptr, *dLl=nullptr, *dDet=nullptr;
    cudaMalloc(&dIn, inBytes);
    cudaMalloc(&dDa, daBytes);
    cudaMalloc(&dLl, llBytes);
    cudaMalloc(&dDet, detBytes);

    cudaStream_t stream; cudaStreamCreate(&stream);

    ctx->setTensorAddress(IN, dIn);
    ctx->setTensorAddress(OUT_DET, dDet);
    ctx->setTensorAddress(OUT_DA, dDa);
    ctx->setTensorAddress(OUT_LL, dLl);

    // 4) 추론 루프
    for (auto& file : std::filesystem::directory_iterator(args.inDir)){
      if(!file.is_regular_file()) continue;
      std::string ext = file.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

      cv::Mat img = cv::imread(file.path().string());
      if(img.empty()){
        std::cerr << "skip " << file.path() << " (cannot read)\n";
        continue;
      }

      int padX=0, padY=0; float scale=1.0f;
      cv::Mat lb = letterbox(img, W, H, padX, padY, scale);

      std::vector<__half> inHost((size_t)B*C*H*W);
      toCHWFP16(lb, H, W, inHost.data());

      cudaMemcpyAsync(dIn, inHost.data(), inBytes, cudaMemcpyHostToDevice, stream);
      if(!ctx->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
      cudaStreamSynchronize(stream);

      // ── D2H (세그)
      std::vector<float> daHostF, llHostF;
      if (dtype_da == DataType::kHALF){
        std::vector<__half> h(daElems);
        cudaMemcpyAsync(h.data(), dDa, daBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        daHostF.resize(daElems);
        for(size_t i=0;i<daElems;++i) daHostF[i] = __half2float(h[i]);
      } else {
        daHostF.resize(daElems);
        cudaMemcpyAsync(daHostF.data(), dDa, daBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
      }
      if (dtype_ll == DataType::kHALF){
        std::vector<__half> h(llElems);
        cudaMemcpyAsync(h.data(), dLl, llBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        llHostF.resize(llElems);
        for(size_t i=0;i<llElems;++i) llHostF[i] = __half2float(h[i]);
      } else {
        llHostF.resize(llElems);
        cudaMemcpyAsync(llHostF.data(), dLl, llBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
      }

      // ── D2H (디텍션)
      std::vector<float> detHostF;
      if (dtype_det == DataType::kHALF){
        std::vector<__half> tmp(detElems);
        cudaMemcpyAsync(tmp.data(), dDet, detBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        detHostF.resize(detElems);
        for(size_t i=0;i<detElems;++i) detHostF[i] = __half2float(tmp[i]);
      } else {
        detHostF.resize(detElems);
        cudaMemcpyAsync(detHostF.data(), dDet, detBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
      }

      if (Ho <= 0 || Wo <= 0 || H2 <= 0 || W2 <= 0) {
        daD = ctx->getTensorShape(OUT_DA);
        llD = ctx->getTensorShape(OUT_LL);
        daC = daD.d[1]; llC = llD.d[1];
        Ho = daD.d[2]; Wo = daD.d[3];
        H2 = llD.d[2]; W2 = llD.d[3];
      }

      std::cout << "[DBG] OUT_DA shape: ["<< daD.d[0]<<","<<daC<<","<<Ho<<","<<Wo<<"]\n";
      std::cout << "[DBG] OUT_LL shape: ["<< llD.d[0]<<","<<llC<<","<<H2<<","<<W2<<"]\n";
      print_minmax(daHostF, "DA raw");
      print_minmax(llHostF, "LL raw");
      print_minmax(detHostF, "DET raw");

      // ── 세그: 확률/마스크 생성
      std::string base = file.path().stem().string();

      auto save_prob_png = [&](const cv::Mat& prob, const std::string& name){
        cv::Mat u8; prob.convertTo(u8, CV_8U, 255.0);
        cv::imwrite((std::filesystem::path(args.outDir)/name).string(), u8);
      };

      cv::Mat daMaskLb, llMaskLb;
      if (daC == 2){
        cv::Mat c0(Ho, Wo, CV_32F, daHostF.data() + 0*Ho*Wo);
        cv::Mat c1(Ho, Wo, CV_32F, daHostF.data() + 1*Ho*Wo);
        cv::Mat prob1; softmax2(c0, c1, prob1);
        daMaskLb = (prob1 > args.segTh);
        save_prob_png(prob1, base+"__da_prob_lb.png");
      } else if (daC == 1){
        cv::Mat prob(Ho, Wo, CV_32F, daHostF.data());
        daMaskLb = (prob > args.segTh);
        save_prob_png(prob, base+"__da_prob_lb.png");
      } else {
        throw std::runtime_error("Unexpected DA channels (expected 1 or 2)");
      }

      if (llC == 2){
        cv::Mat c0(H2, W2, CV_32F, llHostF.data() + 0*H2*W2);
        cv::Mat c1(H2, W2, CV_32F, llHostF.data() + 1*H2*W2);
        cv::Mat prob1; softmax2(c0, c1, prob1);
        llMaskLb = (prob1 > args.segTh);
        save_prob_png(prob1, base+"__ll_prob_lb.png");
      } else if (llC == 1){
        cv::Mat prob(H2, W2, CV_32F, llHostF.data());
        llMaskLb = (prob > args.segTh);
        save_prob_png(prob, base+"__ll_prob_lb.png");
      } else {
        throw std::runtime_error("Unexpected LL channels (expected 1 or 2)");
      }

      // 레터박스 마스크 저장
      cv::imwrite((std::filesystem::path(args.outDir)/(base+"__da_mask_lb.png")).string(), daMaskLb*255);
      cv::imwrite((std::filesystem::path(args.outDir)/(base+"__ll_mask_lb.png")).string(), llMaskLb*255);

      // ── 언레터박스: pad 제거 → 원본 크기
      int rw = int(img.cols * scale);
      int rh = int(img.rows * scale);
      int x0 = std::max(0, padX), y0 = std::max(0, padY);
      int w0 = std::min(W - x0, rw);
      int h0 = std::min(H - y0, rh);
      if (w0<=0 || h0<=0) {
        std::cerr << "[WARN] invalid crop size: w0="<<w0<<" h0="<<h0
                  << " (padX="<<padX<<" padY="<<padY<<" rw="<<rw<<" rh="<<rh<<")\n";
        continue;
      }

      cv::Mat daCrop = daMaskLb(cv::Rect(x0, y0, w0, h0));
      cv::Mat llCrop = llMaskLb(cv::Rect(x0, y0, w0, h0));
      cv::Mat daMask, llMask;
      cv::resize(daCrop, daMask, img.size(), 0, 0, cv::INTER_NEAREST);
      cv::resize(llCrop, llMask, img.size(), 0, 0, cv::INTER_NEAREST);

      // ── 마스크 클린업 (잡티 제거)
      daMask = cleanupMask(daMask, args.daMinA, /*morphK=*/3, /*iters=*/1);
      llMask = cleanupMask(llMask, args.llMinA, /*morphK=*/3, /*iters=*/1);

      // 원본 공간 마스크 저장
      cv::imwrite((std::filesystem::path(args.outDir)/(base+"__da_mask.png")).string(), daMask*255);
      cv::imwrite((std::filesystem::path(args.outDir)/(base+"__ll_mask.png")).string(), llMask*255);

      // ── 디텍션: 디코드 → 언레터박스 → NMS
      const int stride = detD.d[2]; // = 6
      const int num    = detD.d[1]; // = 25200

      std::vector<Det> dets; dets.reserve(256);
      bool isNorm = true;
      for (int i=0;i<num; ++i){
        const float cx = detHostF[i*stride + 0];
        const float cy = detHostF[i*stride + 1];
        const float w  = detHostF[i*stride + 2];
        const float h  = detHostF[i*stride + 3];
        const float obj= detHostF[i*stride + 4];
        const float cls= detHostF[i*stride + 5]; // nc=1

        if (i==0) isNorm = isNormalizedXYWH(cx,cy,w,h);
        float score = obj * cls;
        if (score < args.detConf) continue;

        float ccx=cx, ccy=cy, ww=w, hh=h;
        if (isNorm){ ccx *= W; ccy *= H; ww *= W; hh *= H; } // letterbox 공간

        float x1 = ccx - ww*0.5f;
        float y1 = ccy - hh*0.5f;
        float x2 = ccx + ww*0.5f;
        float y2 = ccy + hh*0.5f;

        x1 -= padX; x2 -= padX;
        y1 -= padY; y2 -= padY;
        if (scale > 0.f){
          x1 /= scale; x2 /= scale;
          y1 /= scale; y2 /= scale;
        }
        x1 = std::clamp(x1, 0.f, (float)img.cols-1);
        y1 = std::clamp(y1, 0.f, (float)img.rows-1);
        x2 = std::clamp(x2, 0.f, (float)img.cols-1);
        y2 = std::clamp(y2, 0.f, (float)img.rows-1);

        if (x2-x1 < 1.f || y2-y1 < 1.f) continue;
        dets.push_back( Det{ cv::Rect2f(x1,y1, x2-x1, y2-y1), score, 0 } );
      }

      std::cout << "[DBG] det candidates="<<dets.size()<<"\n";
      dets = nms(dets, args.nmsIou);
      std::cout << "[DBG] det kept="<<dets.size()<<"\n";

      // ── 시각화 (overlay + bbox)
      cv::Mat vis; img.copyTo(vis);
      cv::Mat tintDA(vis.size(), CV_8UC3, cv::Scalar(0,0,255));  // 빨강
      cv::Mat tintLL(vis.size(), CV_8UC3, cv::Scalar(0,255,0));  // 초록

      cv::Mat visDA = vis.clone();
      tintDA.copyTo(visDA, daMask);
      cv::addWeighted(vis, 0.55, visDA, 0.45, 0.0, vis);

      cv::Mat visLL = vis.clone();
      tintLL.copyTo(visLL, llMask);
      cv::addWeighted(vis, 0.55, visLL, 0.45, 0.0, vis);

      drawDetections(vis, dets, cv::Scalar(255,0,0)); // 파랑

      const std::string outPath = (std::filesystem::path(args.outDir) / file.path().filename()).string();
      cv::imwrite(outPath, vis);
      std::cout << "Saved: " << outPath << "\n";
    }

    cudaStreamDestroy(stream);
    cudaFree(dIn); cudaFree(dDa); cudaFree(dLl); cudaFree(dDet);
    return 0;

  } catch(const std::exception& e){
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
