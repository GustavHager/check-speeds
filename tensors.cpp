#include <catch2/catch_all.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

using namespace Catch;
using namespace Eigen;

typedef Tensor<float, 3> Tensor3f;
typedef Tensor<float, 2> Tensor2f;

TEST_CASE("NormalizeFeatureTensor", "[Tensor]"){
  Tensor3f t(4, 512, 512);
  array<Index, 1> reductionDim({0});
  array<Index, 3> outShape({1, 512, 512});
  array<Index, 3> bcastShape({4, 1, 1});

  BENCHMARK("Inplace"){
    t = t / t.sum(reductionDim).reshape(outShape).sqrt().broadcast(bcastShape);
    return t;
  };

  BENCHMARK("Inplace+eval"){
    t = t / t.sum(reductionDim).sqrt().eval().reshape(outShape).broadcast(bcastShape);
    return t;
  };

  BENCHMARK("Partial"){
    Tensor2f sum = t.sum(reductionDim).sqrt();
    Tensor3f tNorm = t / sum.reshape(outShape).broadcast(bcastShape);
    return tNorm;
  };

  BENCHMARK("Handwritten"){
    Tensor3f tNorm(4, 512, 512);

    float* inPtr = tNorm.data();
    float* outPtr = tNorm.data();
    size_t numElements = 512*512;
    
    for(size_t i=0; i < numElements; i++){
      float sum = 0;
      
      for(size_t j=0; j < 4; j++){
	sum += std::pow(inPtr[j], 2.f);
      }
      
      for(size_t j=0; j < 4; j++){
	outPtr[j] = inPtr[j] / std::sqrt(sum);
      }
      
      inPtr += 4;
      outPtr += 4;
    }
    return tNorm;
  };

}
