#include <yolop_trt/yolop_trt.hpp>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nvinfer1;
namespace fs = std::filesystem;

namespace yolop {

static std::vector<char> readFile_(const std::string& path){
  std::ifstream ifs(path, std::ios::binary);
  if(!ifs) throw std::runtime_error("cannot open: " + path);
  return std::vector<char>(std::istreambuf_iterator<char>(ifs), {});
}

YolopTRT::YolopTRT(const std::string& planPath){
  auto blob = readFile_(planPath);
  runtime_.reset(createInferRuntime(logger_));
  if(!runtime_) throw std::runtime_error("createInferRuntime failed");
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if(!engine_) throw std::runtime_error("deserializeCudaEngine failed");
  ctx_.reset(engine_->createExecutionContext());
  if(!ctx_) throw std::runtime_error("createExecutionContext failed");

  // input dims
  Dims inD = engine_->getTensorShape(IN_);
  if (inD.d[2] <= 0 || inD.d[3] <= 0) {
    ctx_->setInputShape(IN_, Dims4{1,3,640,640});
    inD = ctx_->getTensorShape(IN_);
  }
  B_ = (inD.d[0] > 0) ? inD.d[0] : 1;
  C_ = inD.d[1]; H_ = inD.d[2]; W_ = inD.d[3];
  if(B_!=1 || C_!=3) throw std::runtime_error("Expected input [1,3,H,W]");

  // outputs
  auto daD = ctx_->getTensorShape(OUT_DA_);
  auto llD = ctx_->getTensorShape(OUT_LL_);
  daC_ = (daD.nbDims>=2? daD.d[1]:-1);
  llC_ = (llD.nbDims>=2? llD.d[1]:-1);
  Ho_  = (daD.nbDims>=4? daD.d[2]:-1);
  Wo_  = (daD.nbDims>=4? daD.d[3]:-1);
  H2_  = (llD.nbDims>=4? llD.d[2]:-1);
  W2_  = (llD.nbDims>=4? llD.d[3]:-1);

  dtype_da_ = engine_->getTensorDataType(OUT_DA_);
  dtype_ll_ = engine_->getTensorDataType(OUT_LL_);
  dtype_det_ = engine_->getTensorDataType(OUT_DET_);

  allocBuffers_();
}

YolopTRT::~YolopTRT(){
  freeBuffers_();
}

void YolopTRT::allocBuffers_(){
  auto safeMul = [](const Dims& d)->size_t{
    size_t prod=1; for(int i=0;i<d.nbDims;++i){ long long v=d.d[i]; if(v<1) v=1; prod*= (size_t)v; } return prod;
  };
  inBytes_ = (size_t)B_*C_*H_*W_*sizeof(__half);

  auto daD = ctx_->getTensorShape(OUT_DA_);
  auto llD = ctx_->getTensorShape(OUT_LL_);
  auto detD= ctx_->getTensorShape(OUT_DET_);

  size_t daElems = safeMul(daD);
  size_t llElems = safeMul(llD);
  size_t detElems= safeMul(detD);

  daBytes_ = daElems * (dtype_da_==DataType::kHALF ? sizeof(__half): sizeof(float));
  llBytes_ = llElems * (dtype_ll_==DataType::kHALF ? sizeof(__half): sizeof(float));
  detBytes_= detElems* (dtype_det_==DataType::kHALF ? sizeof(__half): sizeof(float));

  cudaStreamCreate(&stream_);
  cudaMalloc(&dIn_,  inBytes_);
  cudaMalloc(&dDa_,  daBytes_);
  cudaMalloc(&dLl_,  llBytes_);
  cudaMalloc(&dDet_, detBytes_);

  ctx_->setTensorAddress(IN_, dIn_);
  ctx_->setTensorAddress(OUT_DA_, dDa_);
  ctx_->setTensorAddress(OUT_LL_, dLl_);
  ctx_->setTensorAddress(OUT_DET_, dDet_);
}

void YolopTRT::freeBuffers_(){
  if (stream_) cudaStreamDestroy(stream_);
  if (dIn_)  cudaFree(dIn_), dIn_=nullptr;
  if (dDa_)  cudaFree(dDa_), dDa_=nullptr;
  if (dLl_)  cudaFree(dLl_), dLl_=nullptr;
  if (dDet_) cudaFree(dDet_), dDet_=nullptr;
}

cv::Mat YolopTRT::letterbox(const cv::Mat& img, int newW, int newH, int& padX, int& padY, float& scale){
  int w=img.cols, h=img.rows;
  scale = std::min(1.f*newW/w, 1.f*newH/h);
  int rw=int(w*scale), rh=int(h*scale);
  cv::Mat resized; cv::resize(img, resized, cv::Size(rw,rh));
  cv::Mat out(newH,newW, img.type(), cv::Scalar(114,114,114));
  padX = (newW-rw)/2; padY=(newH-rh)/2;
  resized.copyTo(out(cv::Rect(padX,padY,rw,rh)));
  return out;
}

void YolopTRT::toCHWFP16(const cv::Mat& bgr, int H, int W, __half* dst){
  cv::Mat resized; cv::resize(bgr, resized, cv::Size(W,H));
  cv::Mat rgb; cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0/255.0);
  for(int c=0;c<3;++c){
    for(int y=0;y<H;++y){
      const float* row = rgb.ptr<float>(y);
      for(int x=0;x<W;++x){
        dst[c*H*W + y*W + x] = __float2half(row[x*3 + c]);
      }
    }
  }
}

void YolopTRT::softmax2(const cv::Mat& c0, const cv::Mat& c1, cv::Mat& p1){
  p1.create(c0.size(), CV_32F);
  for(int y=0;y<c0.rows;++y){
    const float* a = c0.ptr<float>(y);
    const float* b = c1.ptr<float>(y);
    float* p = p1.ptr<float>(y);
    for(int x=0;x<c0.cols;++x){
      float m = std::max(a[x], b[x]);
      float ea = std::exp(a[x]-m);
      float eb = std::exp(b[x]-m);
      p[x] = eb / (ea+eb);
    }
  }
}

cv::Mat YolopTRT::cleanupMask(const cv::Mat& bin01, int minArea, int morphK, int morphIters){
  CV_Assert(bin01.type()==CV_8U);
  cv::Mat m = bin01.clone();
  if (morphK>1){
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morphK,morphK));
    cv::morphologyEx(m,m,cv::MORPH_OPEN,k, {-1,-1}, morphIters);
  }
  if (minArea<=1) return m;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(m, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::Mat out = cv::Mat::zeros(m.size(), CV_8U);
  for (auto& c: contours){
    if (std::fabs(cv::contourArea(c)) >= minArea)
      cv::drawContours(out, std::vector<std::vector<cv::Point>>{c}, -1, 255, cv::FILLED);
  }
  return out;
}

static float iouRect(const cv::Rect2f& a, const cv::Rect2f& b){
  float inter=(a & b).area();
  float uni=a.area()+b.area()-inter;
  return uni>0? inter/uni : 0.f;
}

std::vector<Detection> YolopTRT::nms(const std::vector<Detection>& src, float iouTh){
  std::vector<Detection> dets=src, keep;
  std::sort(dets.begin(), dets.end(), [](auto& A, auto& B){return A.score>B.score;});
  std::vector<int> alive(dets.size(),1);
  for(size_t i=0;i<dets.size();++i){
    if(!alive[i]) continue;
    keep.push_back(dets[i]);
    for(size_t j=i+1;j<dets.size();++j){
      if(!alive[j]) continue;
      if (iouRect(dets[i].box, dets[j].box) > iouTh) alive[j]=0;
    }
  }
  return keep;
}

bool YolopTRT::isNormalizedXYWH(float cx, float cy, float w, float h){
  return (cx<=1.5f && cy<=1.5f && w<=1.5f && h<=1.5f);
}

std::vector<Detection> YolopTRT::infer(
  const cv::Mat& inBGR, cv::Mat& outOverlay, cv::Mat& outDaMask, cv::Mat& outLlMask, const Params& p)
{
  // letterbox
  int padX=0, padY=0; float scale=1.f;
  cv::Mat lb = letterbox(inBGR, W_, H_, padX, padY, scale);

  // H2D
  std::vector<__half> inHost((size_t)B_*C_*H_*W_);
  toCHWFP16(lb, H_, W_, inHost.data());
  cudaMemcpyAsync(dIn_, inHost.data(), inBytes_, cudaMemcpyHostToDevice, stream_);
  if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
  cudaStreamSynchronize(stream_);

  // D2H
  auto daD = ctx_->getTensorShape(OUT_DA_);
  auto llD = ctx_->getTensorShape(OUT_LL_);
  int Ho = daD.d[2], Wo=daD.d[3], H2=llD.d[2], W2=llD.d[3];
  size_t daElems = (size_t)daD.d[0]*daD.d[1]*daD.d[2]*daD.d[3];
  size_t llElems = (size_t)llD.d[0]*llD.d[1]*llD.d[2]*llD.d[3];

  std::vector<float> daHostF(daElems), llHostF(llElems), detHostF;
  // seg DA
  if (dtype_da_==DataType::kHALF){
    std::vector<__half> tmp(daElems);
    cudaMemcpyAsync(tmp.data(), dDa_, daBytes_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    for(size_t i=0;i<daElems;++i) daHostF[i]=__half2float(tmp[i]);
  } else {
    cudaMemcpyAsync(daHostF.data(), dDa_, daBytes_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }
  // seg LL
  if (dtype_ll_==DataType::kHALF){
    std::vector<__half> tmp(llElems);
    cudaMemcpyAsync(tmp.data(), dLl_, llBytes_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    for(size_t i=0;i<llElems;++i) llHostF[i]=__half2float(tmp[i]);
  } else {
    cudaMemcpyAsync(llHostF.data(), dLl_, llBytes_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }
  // det
  auto detD = ctx_->getTensorShape(OUT_DET_);
  size_t detElems = (size_t)detD.d[0]*detD.d[1]*detD.d[2];
  detHostF.resize(detElems);
  if (dtype_det_==DataType::kHALF){
    std::vector<__half> tmp(detElems);
    cudaMemcpyAsync(tmp.data(), dDet_, detBytes_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    for(size_t i=0;i<detElems;++i) detHostF[i]=__half2float(tmp[i]);
  } else {
    cudaMemcpyAsync(detHostF.data(), dDet_, detBytes_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }

  // seg prob → mask(threshold)
  cv::Mat daMaskLb, llMaskLb;
  if (daD.d[1]==2){
    cv::Mat c0(Ho, Wo, CV_32F, daHostF.data());
    cv::Mat c1(Ho, Wo, CV_32F, daHostF.data()+Ho*Wo);
    cv::Mat prob1; softmax2(c0,c1,prob1);
    daMaskLb = (prob1 > p.segTh);
  } else {
    cv::Mat prob(Ho, Wo, CV_32F, daHostF.data());
    daMaskLb = (prob > p.segTh);
  }
  if (llD.d[1]==2){
    cv::Mat c0(H2, W2, CV_32F, llHostF.data());
    cv::Mat c1(H2, W2, CV_32F, llHostF.data()+H2*W2);
    cv::Mat prob1; softmax2(c0,c1,prob1);
    llMaskLb = (prob1 > p.segTh);
  } else {
    cv::Mat prob(H2, W2, CV_32F, llHostF.data());
    llMaskLb = (prob > p.segTh);
  }

  // unletterbox to original size + cleanup
  int rw=int(inBGR.cols*scale), rh=int(inBGR.rows*scale);
  int x0=std::max(0,padX), y0=std::max(0,padY);
  int w0=std::min(W_ - x0, rw);
  int h0=std::min(H_ - y0, rh);
  cv::Mat daCrop = daMaskLb(cv::Rect(x0,y0,w0,h0));
  cv::Mat llCrop = llMaskLb(cv::Rect(x0,y0,w0,h0));

  cv::Mat daMask, llMask;
  cv::resize(daCrop, daMask, inBGR.size(), 0,0, cv::INTER_NEAREST);
  cv::resize(llCrop, llMask, inBGR.size(), 0,0, cv::INTER_NEAREST);

  daMask = cleanupMask(daMask, p.daMinA, 3, 1);
  llMask = cleanupMask(llMask, p.llMinA, 3, 1);

  // det decode → unletterbox → NMS
  const int stride = detD.d[2]; //6
  const int num    = detD.d[1];
  std::vector<Detection> dets; dets.reserve(256);
  bool isNorm=true;
  for(int i=0;i<num;++i){
    float cx = detHostF[i*stride+0];
    float cy = detHostF[i*stride+1];
    float w  = detHostF[i*stride+2];
    float h  = detHostF[i*stride+3];
    float obj= detHostF[i*stride+4];
    float cls= detHostF[i*stride+5];
    if (i==0) isNorm = isNormalizedXYWH(cx,cy,w,h);
    float score = obj*cls;
    if (score < p.detConf) continue;

    float ccx=cx, ccy=cy, ww=w, hh=h;
    if (isNorm){ ccx*=W_; ccy*=H_; ww*=W_; hh*=H_; }
    float x1=ccx-ww*0.5f, y1=ccy-hh*0.5f, x2=ccx+ww*0.5f, y2=ccy+hh*0.5f;
    x1-=padX; x2-=padX; y1-=padY; y2-=padY;
    x1/=scale; x2/=scale; y1/=scale; y2/=scale;
    x1 = std::clamp(x1,0.f,(float)inBGR.cols-1);
    y1 = std::clamp(y1,0.f,(float)inBGR.rows-1);
    x2 = std::clamp(x2,0.f,(float)inBGR.cols-1);
    y2 = std::clamp(y2,0.f,(float)inBGR.rows-1);
    if (x2-x1<1 || y2-y1<1) continue;
    dets.push_back({cv::Rect2f(x1,y1,x2-x1,y2-y1), score, 0});
  }
  dets = nms(dets, p.nmsIou);

  // overlay
  cv::Mat vis; inBGR.copyTo(vis);
  cv::Mat tintDA(vis.size(), CV_8UC3, cv::Scalar(0,0,255));
  cv::Mat tintLL(vis.size(), CV_8UC3, cv::Scalar(0,255,0));

  cv::Mat visDA = vis.clone();  tintDA.copyTo(visDA, daMask);
  cv::addWeighted(vis, 0.55, visDA, 0.45, 0.0, vis);
  cv::Mat visLL = vis.clone();  tintLL.copyTo(visLL, llMask);
  cv::addWeighted(vis, 0.55, visLL, 0.45, 0.0, vis);

  for (auto& d: dets){
    cv::rectangle(vis, d.box, {255,0,0}, 2);
    char buf[64]; std::snprintf(buf,sizeof(buf),"cls=%d %.2f", d.cls, d.score);
    int base=0; auto sz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    cv::Rect bg((int)d.box.x, std::max(0,(int)d.box.y - sz.height-4), sz.width+6, sz.height+4);
    bg &= cv::Rect(0,0,vis.cols, vis.rows);
    cv::rectangle(vis, bg, {255,0,0}, cv::FILLED);
    cv::putText(vis, buf, {bg.x+3, bg.y+bg.height-3}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);
  }

  outOverlay = vis;
  outDaMask = daMask;   // 0/255
  outLlMask = llMask;   // 0/255
  return dets;
}

} // namespace yolop
