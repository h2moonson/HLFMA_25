#include "yolop_trt.hpp"
#include <cuda_runtime.h>
#include <algorithm>

static void check(cudaError_t e){ if(e!=cudaSuccess){throw std::runtime_error(cudaGetErrorString(e));}}

YolopTRT::YolopTRT(const std::string& enginePath, const YolopConfig& cfg):cfg_(cfg){
    std::ifstream f(enginePath, std::ios::binary);
    if(!f.good()) throw std::runtime_error("Engine not found");
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    runtime_ = nvinfer1::createInferRuntime(logger_);
    engine_  = runtime_->deserializeCudaEngine(data.data(), data.size());
    ctx_     = engine_->createExecutionContext();

    // 바인딩 인덱스
    inIndex_  = engine_->getBindingIndex("images");
    detIndex_ = engine_->getBindingIndex("det_out");
    daIndex_  = engine_->getBindingIndex("da_seg_out");
    llIndex_  = engine_->getBindingIndex("ll_seg_out");

    auto inDims  = engine_->getBindingDimensions(inIndex_);   // [N,3,640,640]
    auto detDims = engine_->getBindingDimensions(detIndex_);  // [N, M, 5+K]
    auto daDims  = engine_->getBindingDimensions(daIndex_);   // [N, 1, H, W]
    auto llDims  = engine_->getBindingDimensions(llIndex_);   // [N, 1, H, W]

    size_t n = 1;
    inSizeBytes_  = 1; for(int i=0;i<inDims.nbDims;i++)  inSizeBytes_  *= inDims.d[i];
    detSizeBytes_ = 1; for(int i=0;i<detDims.nbDims;i++) detSizeBytes_ *= detDims.d[i];
    daSizeBytes_  = 1; for(int i=0;i<daDims.nbDims;i++)  daSizeBytes_  *= daDims.d[i];
    llSizeBytes_  = 1; for(int i=0;i<llDims.nbDims;i++)  llSizeBytes_  *= llDims.d[i];
    inSizeBytes_  *= sizeof(float);
    detSizeBytes_ *= sizeof(float);
    daSizeBytes_  *= sizeof(float);
    llSizeBytes_  *= sizeof(float);

    check(cudaMalloc(&bindings_[inIndex_],  inSizeBytes_));
    check(cudaMalloc(&bindings_[detIndex_], detSizeBytes_));
    check(cudaMalloc(&bindings_[daIndex_],  daSizeBytes_));
    check(cudaMalloc(&bindings_[llIndex_],  llSizeBytes_));
}

YolopTRT::~YolopTRT(){
    for(void*& b: bindings_) if(b) cudaFree(b);
    if(ctx_) ctx_->destroy();
    if(engine_) engine_->destroy();
    if(runtime_) runtime_->destroy();
}

cv::Mat YolopTRT::letterbox(const cv::Mat& img, float& scale, int& dx, int& dy){
    int w=img.cols, h=img.rows;
    float r = std::min((float)cfg_.inputW/w, (float)cfg_.inputH/h);
    int nw = (int)(w*r), nh = (int)(h*r);
    scale=r; dx=(cfg_.inputW-nw)/2; dy=(cfg_.inputH-nh)/2;

    cv::Mat resized; cv::resize(img, resized, cv::Size(nw,nh));
    cv::Mat out(cfg_.inputH, cfg_.inputW, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(out(cv::Rect(dx,dy,nw,nh)));
    return out;
}

static void hwc_to_chw(const cv::Mat& img, float* dst){
    // BGR -> CHW float32 [0,1]
    for(int c=0;c<3;c++)
      for(int y=0;y<img.rows;y++){
        const uchar* p = img.ptr<uchar>(y);
        for(int x=0;x<img.cols;x++) dst[c*img.rows*img.cols + y*img.cols + x] = p[x*3 + (2-c)]/255.f;
      }
}

static void nms(std::vector<Box>& boxes, float iouTh){
    std::sort(boxes.begin(), boxes.end(), [](auto&a, auto&b){return a.score>b.score;});
    std::vector<Box> keep;
    auto iou=[&](const Box& A, const Box& B){
        float xx1=std::max(A.x1,B.x1), yy1=std::max(A.y1,B.y1);
        float xx2=std::min(A.x2,B.x2), yy2=std::min(A.y2,B.y2);
        float w=std::max(0.f, xx2-xx1), h=std::max(0.f, yy2-yy1);
        float inter=w*h, areaA=(A.x2-A.x1)*(A.y2-A.y1), areaB=(B.x2-B.x1)*(B.y2-B.y1);
        return inter/(areaA+areaB-inter+1e-6f);
    };
    std::vector<int> removed(boxes.size(),0);
    for(size_t i=0;i<boxes.size();++i){
        if(removed[i]) continue;
        keep.push_back(boxes[i]);
        for(size_t j=i+1;j<boxes.size();++j)
            if(!removed[j] && iou(boxes[i],boxes[j])>iouTh) removed[j]=1;
    }
    boxes.swap(keep);
}

void YolopTRT::postprocess(const cv::Size& orig, float s, int dx, int dy,
                           const float* det, int detCnt,
                           const float* da, const float* ll,
                           std::vector<Box>& outBoxes,
                           cv::Mat& daMask, cv::Mat& llMask,
                           cv::Mat* vis){
    // det: [M, 5+K] = [cx,cy,w,h,obj, cls...]
    for(int i=0;i<detCnt;i++){
        const float* p = det + i*(5+cfg_.numClasses);
        float obj = p[4];
        if(obj<cfg_.confTh) continue;
        int   best=0; float bestc=0.f;
        for(int c=0;c<cfg_.numClasses;c++){
            float sc = obj * p[5+c];
            if(sc>bestc){ bestc=sc; best=c; }
        }
        if(bestc<cfg_.confTh) continue;

        float cx=p[0], cy=p[1], w=p[2], h=p[3];
        // letterbox 역보정
        float x1=(cx - w/2 - dx)/s, y1=(cy - h/2 - dy)/s;
        float x2=(cx + w/2 - dx)/s, y2=(cy + h/2 - dy)/s;
        x1 = std::clamp(x1,0.f,(float)orig.width-1);
        x2 = std::clamp(x2,0.f,(float)orig.width-1);
        y1 = std::clamp(y1,0.f,(float)orig.height-1);
        y2 = std::clamp(y2,0.f,(float)orig.height-1);

        outBoxes.push_back({x1,y1,x2,y2,bestc,best});
    }
    nms(outBoxes, cfg_.nmsTh);

    // seg: [1,1,H,W] → 간단 threshold(0.5)로 binary mask
    int H = cfg_.inputH, W = cfg_.inputW; // ONNX가 동일 해상도라 가정
    cv::Mat daProb(H,W,CV_32F,(void*)da), llProb(H,W,CV_32F,(void*)ll);
    cv::Mat daCh, llCh;  daProb.convertTo(daCh, CV_8U, 255.0);  llProb.convertTo(llCh, CV_8U, 255.0);
    cv::threshold(daCh, daMask, 128, 255, cv::THRESH_BINARY);
    cv::threshold(llCh, llMask, 128, 255, cv::THRESH_BINARY);

    // letterbox 역보정으로 원본 크기 복원
    cv::Mat daFull(orig, CV_8U, cv::Scalar(0)), llFull(orig, CV_8U, cv::Scalar(0));
    cv::Mat canvas(H, W, CV_8U, cv::Scalar(0));
    daMask.copyTo(canvas);
    cv::Mat roi = daFull(cv::Rect(0,0,orig.width,orig.height));
    // 간단히 resize (padding 영역 영향은 경미)
    cv::resize(canvas, daFull, orig, 0,0, cv::INTER_NEAREST);
    cv::resize(llMask,  llFull, orig, 0,0, cv::INTER_NEAREST);
    daMask.swap(daFull); llMask.swap(llFull);

    if(vis){
        *vis = cv::Mat(orig, CV_8UC3, cv::Scalar(0,0,0));
        // 그냥 투명 오버레이
        vis->setTo(cv::Scalar(0,0,0));
    }
}

void YolopTRT::infer(const cv::Mat& bgr,
                     std::vector<Box>& boxes,
                     cv::Mat& daMask, cv::Mat& llMask,
                     cv::Mat* vis){
    float scale; int dx,dy;
    cv::Mat pad = letterbox(bgr, scale, dx, dy);
    std::vector<float> input(3*cfg_.inputH*cfg_.inputW);
    hwc_to_chw(pad, input.data());

    cudaMemcpy(bindings_[inIndex_], input.data(), inSizeBytes_, cudaMemcpyHostToDevice);
    ctx_->enqueueV2(bindings_, 0, nullptr);

    std::vector<float> det(detSizeBytes_/sizeof(float));
    std::vector<float> da (daSizeBytes_ /sizeof(float));
    std::vector<float> ll (llSizeBytes_ /sizeof(float));
    cudaMemcpy(det.data(), bindings_[detIndex_], detSizeBytes_, cudaMemcpyDeviceToHost);
    cudaMemcpy(da.data(),  bindings_[daIndex_],  daSizeBytes_,  cudaMemcpyDeviceToHost);
    cudaMemcpy(ll.data(),  bindings_[llIndex_],  llSizeBytes_,  cudaMemcpyDeviceToHost);

    int detCnt = detSizeBytes_/sizeof(float) / (5+cfg_.numClasses);
    postprocess(bgr.size(), scale, dx, dy, det.data(), detCnt, da.data(), ll.data(), boxes, daMask, llMask, vis);
}

// ── main (단일 이미지 샘플) ─────────────────────────────
int main(int argc, char** argv){
    if(argc<4){
        std::cout << "Usage: " << argv[0] << " <engine.plan> <image> <out.jpg>\n";
        return 0;
    }
    YolopConfig cfg; // 필요시 파라미터 수정
    YolopTRT trt(argv[1], cfg);
    cv::Mat img = cv::imread(argv[2]);
    std::vector<Box> boxes; cv::Mat da,ll,vis;
    trt.infer(img, boxes, da, ll, &vis);

    // 간단 표출
    cv::Mat show = img.clone();
    for(auto& b: boxes)
        cv::rectangle(show, cv::Rect(cv::Point(b.x1,b.y1), cv::Point(b.x2,b.y2)), {0,255,0}, 2);
    cv::imwrite(argv[3], show);
    return 0;
}
