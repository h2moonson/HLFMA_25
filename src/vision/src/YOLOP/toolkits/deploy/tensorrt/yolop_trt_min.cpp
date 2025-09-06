// TensorRT 10.x / C++17
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;
using namespace nvinfer1;

struct Logger : ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
  }
} gLogger;

static void checkCuda(cudaError_t e){ if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

static size_t vol(const Dims& d){ size_t v=1; for(int i=0;i<d.nbDims;i++) v*=static_cast<size_t>(d.d[i]); return v; }

struct Box { float x1,y1,x2,y2,score; int cls; };

static float iou(const Box& a, const Box& b){
  float xx1=std::max(a.x1,b.x1), yy1=std::max(a.y1,b.y1);
  float xx2=std::min(a.x2,b.x2), yy2=std::min(a.y2,b.y2);
  float w=std::max(0.f, xx2-xx1), h=std::max(0.f, yy2-yy1);
  float inter=w*h;
  float areaA=std::max(0.f,a.x2-a.x1)*std::max(0.f,a.y2-a.y1);
  float areaB=std::max(0.f,b.x2-b.x1)*std::max(0.f,b.y2-b.y1);
  float u=areaA+areaB-inter; return u>0? inter/u:0.f;
}
static std::vector<Box> nms(std::vector<Box> v, float iouThr){
  std::sort(v.begin(),v.end(),[](auto& a,auto& b){return a.score>b.score;});
  std::vector<Box> keep; std::vector<char> rem(v.size(),0);
  for(size_t i=0;i<v.size();++i){ if(rem[i]) continue; keep.push_back(v[i]);
    for(size_t j=i+1;j<v.size();++j) if(!rem[j] && iou(v[i],v[j])>iouThr) rem[j]=1;
  } return keep;
}

static bool guessDetIsNormalized(const float* det, int n, int step=6, float thr=2.0f){
  int samples=0; double acc=0.0;
  for(int i=0;i<n && samples<50;i++){
    const float* p=&det[i*step];
    if(p[4] < 0.2f) continue; // conf filter
    acc += std::max(p[2],p[3]); // w/h
    samples++;
  }
  if(samples==0) return true;
  return (acc/samples) < thr; // 평균 w/h가 2 미만이면 0~1 정규화로 추정
}

static std::vector<char> loadFile(const std::string& path){
  std::ifstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("Cannot open file: "+path);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

int main(int argc, char** argv){
  if(argc < 3){
    std::cerr << "Usage: " << argv[0] << " <engine.plan> <input_dir> [output_dir]\n";
    return 1;
  }
  const std::string enginePath = argv[1];
  const fs::path inDir = argv[2];
  const fs::path outDir = (argc>=4)? fs::path(argv[3]) : (inDir / "outputs");

  if(!fs::exists(inDir) || !fs::is_directory(inDir)){
    std::cerr << "Input dir not found: " << inDir << "\n";
    return 1;
  }
  fs::create_directories(outDir);

  // ── 엔진 로드 (TRT10)
  auto blob = loadFile(enginePath);

  // 모든 TensorRT 객체는 TRT10에서 destroy()가 아니라 delete 사용!
  auto runtime = std::unique_ptr<IRuntime, void(*)(IRuntime*)>(
      createInferRuntime(gLogger),
      [](IRuntime* p){ delete p; }
  );

  auto engine = std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)>(
      runtime ? runtime->deserializeCudaEngine(blob.data(), blob.size()) : nullptr,
      [](ICudaEngine* p){ delete p; }
  );

  auto ctx = std::unique_ptr<IExecutionContext, void(*)(IExecutionContext*)>(
      engine ? engine->createExecutionContext() : nullptr,
      [](IExecutionContext* p){ delete p; }
  );

  if(!engine || !ctx){
    std::cerr<<"Engine/Context creation failed\n";
    return 2;
  }

  // ── 텐서 이름 (dump 결과 반영)
  const char* IN  = "images";
  const char* DET = "det_out";         // [1,25200,6]
  const char* DA  = "drive_area_seg";  // [1,2,640,640]
  const char* LL  = "lane_line_seg";   // [1,2,640,640]

  // ── 텐서 shape
  Dims inDims  = engine->getTensorShape(IN);
  Dims detDims = engine->getTensorShape(DET);
  Dims daDims  = engine->getTensorShape(DA);
  Dims llDims  = engine->getTensorShape(LL);

  if(inDims.nbDims!=4 || inDims.d[0]!=1 || inDims.d[1]!=3){ std::cerr<<"Expect IN [1,3,H,W]\n"; return 3; }
  if(detDims.nbDims!=3 || detDims.d[2]!=6){ std::cerr<<"Expect DET [1,N,6]\n"; return 3; }
  if(daDims.nbDims!=4 || daDims.d[1]!=2 || llDims.nbDims!=4 || llDims.d[1]!=2){ std::cerr<<"Expect seg [1,2,H,W]\n"; return 3; }

  const int C=inDims.d[1], H=inDims.d[2], W=inDims.d[3];
  const int Ndet=detDims.d[1];

  const size_t inBytes  = vol(inDims)*sizeof(float);
  const size_t detBytes = vol(detDims)*sizeof(float);
  const size_t daBytes  = vol(daDims)*sizeof(float);
  const size_t llBytes  = vol(llDims)*sizeof(float);

  // ── 디바이스 버퍼
  void *dIn=nullptr, *dDet=nullptr, *dDa=nullptr, *dLl=nullptr;
  checkCuda(cudaMalloc(&dIn,  inBytes));
  checkCuda(cudaMalloc(&dDet, detBytes));
  checkCuda(cudaMalloc(&dDa,  daBytes));
  checkCuda(cudaMalloc(&dLl,  llBytes));

  // 동적 입력 대비 (여기서는 고정이라 그대로)
  ctx->setInputShape(IN, inDims);

  // 엔진에 텐서 주소 연결 (TRT10: 이름 기반)
  ctx->setTensorAddress(IN,  dIn);
  ctx->setTensorAddress(DET, dDet);
  ctx->setTensorAddress(DA,  dDa);
  ctx->setTensorAddress(LL,  dLl);

  // ── 입력 폴더의 모든 jpg/jpeg 처리
  int processed=0;
  for(auto& entry : fs::directory_iterator(inDir)){
    if(!entry.is_regular_file()) continue;
    auto ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if(ext != ".jpg" && ext != ".jpeg") continue;

    const fs::path inPath = entry.path();
    const fs::path outPath = outDir / inPath.filename();

    cv::Mat img = cv::imread(inPath.string());
    if(img.empty()){ std::cerr<<"Cannot read image: "<<inPath<<"\n"; continue; }

    // 전처리 BGR->RGB, HWC->CHW, 0~1
    cv::Mat resized; cv::resize(img, resized, cv::Size(W,H));
    std::vector<float> inHost(C*H*W);
    for(int y=0;y<H;y++){
      const uchar* p = resized.ptr<uchar>(y);
      for(int x=0;x<W;x++){
        // B,G,R -> RGB
        float r = p[x*3+2]/255.f, g=p[x*3+1]/255.f, b=p[x*3+0]/255.f;
        inHost[0*H*W + y*W + x] = r;
        inHost[1*H*W + y*W + x] = g;
        inHost[2*H*W + y*W + x] = b;
      }
    }
    checkCuda(cudaMemcpy(dIn, inHost.data(), inBytes, cudaMemcpyHostToDevice));

    // 추론 (TRT10: enqueueV3(stream)만 받음)
    cudaStream_t stream = 0;
    if(!ctx->enqueueV3(stream)){ std::cerr<<"enqueueV3 failed on "<<inPath<<"\n"; continue; }
    checkCuda(cudaStreamSynchronize(stream));

    // Host로 결과 복사
    std::vector<float> detHost(detBytes/sizeof(float));
    std::vector<float> daHost (daBytes /sizeof(float));
    std::vector<float> llHost (llBytes /sizeof(float));
    checkCuda(cudaMemcpy(detHost.data(), dDet, detBytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy( daHost.data(),  dDa,  daBytes,  cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy( llHost.data(),  dLl,  llBytes,  cudaMemcpyDeviceToHost));

    // 세그: 2채널 argmax → 바이너리 마스크
    const int sH=daDims.d[2], sW=daDims.d[3];
    cv::Mat daMask(sH,sW,CV_8U), llMask(sH,sW,CV_8U);
    for(int y=0;y<sH;y++){
      for(int x=0;x<sW;x++){
        float da0 = daHost[0*sH*sW + y*sW + x];
        float da1 = daHost[1*sH*sW + y*sW + x];
        float ll0 = llHost[0*sH*sW + y*sW + x];
        float ll1 = llHost[1*sH*sW + y*sW + x];
        daMask.at<uchar>(y,x) = (da1>da0)? 255:0;
        llMask.at<uchar>(y,x) = (ll1>ll0)? 255:0;
      }
    }

    // 디텍션 decode + NMS
    bool isNormalized = guessDetIsNormalized(detHost.data(), Ndet, 6, 2.0f);
    float confThr=0.35f, iouThr=0.45f;
    std::vector<Box> boxes; boxes.reserve(256);
    for(int i=0;i<Ndet;i++){
      const float* p = &detHost[i*6];
      float conf=p[4]; if(conf<confThr) continue;
      int cls = int(std::round(p[5]));

      float cx=p[0], cy=p[1], w=p[2], h=p[3];
      if(isNormalized){ cx*=W; cy*=H; w*=W; h*=H; }

      float x1 = std::clamp(cx-w*0.5f, 0.f, float(W-1));
      float y1 = std::clamp(cy-h*0.5f, 0.f, float(H-1));
      float x2 = std::clamp(cx+w*0.5f, 0.f, float(W-1));
      float y2 = std::clamp(cy+h*0.5f, 0.f, float(H-1));
      boxes.push_back({x1,y1,x2,y2,conf,cls});
    }
    boxes = nms(std::move(boxes), iouThr);

    // 시각화: 세그 오버레이 + 박스
    cv::Mat daR, llR; 
    cv::resize(daMask, daR, img.size(), 0,0, cv::INTER_NEAREST);
    cv::resize(llMask, llR, img.size(), 0,0, cv::INTER_NEAREST);

    cv::Mat vis = img.clone();
    vis.setTo(cv::Scalar(0,0,255), daR);     // 도로: 빨강
    vis.setTo(cv::Scalar(0,255,0), llR);     // 차선: 초록

    float sx=float(img.cols)/float(W), sy=float(img.rows)/float(H);
    for(const auto& b: boxes){
      cv::Point p1(int(b.x1*sx), int(b.y1*sy));
      cv::Point p2(int(b.x2*sx), int(b.y2*sy));
      cv::rectangle(vis, p1, p2, cv::Scalar(255,200,0), 2);
      char buf[64]; std::snprintf(buf,sizeof(buf),"c%d %.2f", b.cls, b.score);
      cv::putText(vis, buf, p1+cv::Point(0,-4), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,200,0), 1);
    }

    if(!cv::imwrite(outPath.string(), vis)){
      std::cerr<<"Failed to save: "<<outPath<<"\n"; continue;
    }
    std::cout<<"Processed: "<<inPath<<" -> "<<outPath<<"\n";
    processed++;
  }

  std::cout<<"Done. images processed = "<<processed<<"\n";
  cudaFree(dIn); cudaFree(dDet); cudaFree(dDa); cudaFree(dLl);
  return 0;
}
