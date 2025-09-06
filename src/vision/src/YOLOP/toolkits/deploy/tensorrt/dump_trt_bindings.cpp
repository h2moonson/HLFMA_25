#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace nvinfer1;

struct Logger : public ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << "\n";
  }
} gLogger;

static std::vector<char> readFile(const std::string& path){
  std::ifstream ifs(path, std::ios::binary);
  if(!ifs) throw std::runtime_error("cannot open: " + path);
  return std::vector<char>(std::istreambuf_iterator<char>(ifs), {});
}

int main(int argc, char** argv){
  try{
    std::string plan = (argc>=2)? argv[1] : "../yolop.plan";
    std::vector<char> blob = readFile(plan);
    std::unique_ptr<IRuntime> runtime(createInferRuntime(gLogger));
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(blob.data(), blob.size()));
    int n = engine->getNbIOTensors();
    std::cout << "I/O Tensors ("<<n<<"):\n";
    for(int i=0;i<n;++i){
      const char* name = engine->getIOTensorName(i);
      auto mode = engine->getTensorIOMode(name);
      auto dims = engine->getTensorShape(name);
      std::cout << "  ["<<i<<"] "<<(mode==TensorIOMode::kINPUT ? "INPUT " : "OUTPUT ") << "name='"<<name<<"' shape=[";
      for(int d=0; d<dims.nbDims; ++d){ std::cout<<dims.d[d]; if(d+1<dims.nbDims) std::cout<<"x"; }
      std::cout << "]\n";
    }
    return 0;
  }catch(const std::exception& e){
    std::cerr << e.what() << "\n"; return 1;
  }
}
