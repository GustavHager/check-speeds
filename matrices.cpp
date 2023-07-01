#include <catch2/catch_all.hpp>
#include <Eigen/Dense>

using namespace Catch;
using namespace Eigen;

TEST_CASE("NormalizeFeatures", "[Matrix]"){
  MatrixXf featureMatrix = MatrixXf::Random(4, 512*512);

  BENCHMARK("Inplace"){
    for(int i=0; i < featureMatrix.cols(); i++){
      featureMatrix.col(i).normalize();
    }
    return featureMatrix;
  };

  BENCHMARK("Not inplace"){
    MatrixXf normalizedFeatures(4, 512*512);
    for(int i=0; i < featureMatrix.cols(); i++){
      normalizedFeatures.col(i) = featureMatrix.col(i).normalized();
    }
    return normalizedFeatures;
  };

  BENCHMARK("Semi Handwritten"){
    MatrixXf out(4, 1024*512);
    for(int i=0; i < featureMatrix.cols(); i++){
      float colsum = featureMatrix.array().sqrt().col(i).sum();
      out.col(i) = featureMatrix.col(i) / colsum;
    }
    return out;
  };

  BENCHMARK("Handwritten"){
	MatrixXf normalizedFeatureMatrix(4, 512*512);
	float* inPtr = featureMatrix.data();
	float* outPtr = normalizedFeatureMatrix.data();
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

	return normalizedFeatureMatrix;
  };
}
