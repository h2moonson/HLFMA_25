#include <yolop_trt/yolop_trt.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

using namespace yolop_trt;

int main(int argc, char** argv){
  if (argc < 3){
    std::cerr << "Usage: " << argv[0] << " <engine.plan> <image_or_dir> [out_dir]\n";
    return 1;
  }
  const std::string plan = argv[1];
  const std::string in   = argv[2];
  const std::string out  = (argc>=4)? argv[3] : "./out";
  std::filesystem::create_directories(out);

  Params P; // 필요하면 인자 파싱해서 P 수정 가능
  YolopTRT y(plan, P);
  if (!y.valid()){ std::cerr << "invalid engine\n"; return 1; }

  auto process_one = [&](const std::string& path){
    cv::Mat img = cv::imread(path);
    if (img.empty()){ std::cerr << "skip " << path << "\n"; return; }
    cv::Mat overlay, da, ll;
    std::vector<Detection> dets;
    y.infer(img, &overlay, &da, &ll, &dets);

    auto stem = std::filesystem::path(path).stem().string();
    cv::imwrite((std::filesystem::path(out)/ (stem + "_overlay.jpg")).string(), overlay);
    cv::imwrite((std::filesystem::path(out)/ (stem + "_da.png")).string(), da);
    cv::imwrite((std::filesystem::path(out)/ (stem + "_ll.png")).string(), ll);
    std::cout << "Saved: " << stem << "_*.{jpg,png}\n";
  };

  if (std::filesystem::is_directory(in)){
    for (auto& p : std::filesystem::directory_iterator(in)){
      if (!p.is_regular_file()) continue;
      auto ext = p.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext==".jpg" || ext==".jpeg" || ext==".png") process_one(p.path().string());
    }
  } else {
    process_one(in);
  }
  return 0;
}
