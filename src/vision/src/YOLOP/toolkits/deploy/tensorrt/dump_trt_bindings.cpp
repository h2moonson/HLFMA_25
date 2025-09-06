#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
  }
} gLogger;

int main(int argc, char** argv){
  if(argc < 2){ std::cout << "Usage: " << argv[0] << " <engine.plan>\n"; return 0; }
  std::ifstream f(argv[1], std::ios::binary);
  if(!f.good()){ std::cerr << "Cannot open engine: " << argv[1] << "\n"; return 1; }
  std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
  auto engine  = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.data(), data.size()));
  if(!engine){ std::cerr << "Failed to deserialize engine\n"; return 2; }

  int n = engine->getNbIOTensors();
  std::cout << "I/O Tensors (" << n << "):\n";
  for(int i=0; i<n; ++i){
    const char* name = engine->getIOTensorName(i);
    auto mode = engine->getTensorIOMode(name);
    auto dims = engine->getTensorShape(name);
    std::cout << "  ["<<i<<"] " << (mode==nvinfer1::TensorIOMode::kINPUT?"INPUT ":"OUTPUT")
              << " name='"<<name<<"' shape=[";
    for(int d=0; d<dims.nbDims; ++d){ std::cout << dims.d[d]; if(d+1<dims.nbDims) std::cout<<"x"; }
    std::cout << "]\n";
  }
  return 0;
}
