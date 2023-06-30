#include <catch2/catch_all.hpp>
#include <Eigen/Dense>

using namespace Catch;
using namespace Eigen;

TEST_CASE("NormalizeFeatures", "[Matrix]"){
  MatrixXf featureMatrix(32, 1024*512);

  BENCHMARK("Inplace"){
    for(int i=0; i < featureMatrix.cols(); i++){
      featureMatrix.col(i).normalize();
    }
    return featureMatrix;
  };

  BENCHMARK("Not inplace"){
    MatrixXf normalizedFeatures(32, 1024*512);
    for(int i=0; i < featureMatrix.cols(); i++){
      normalizedFeatures.col(i) = featureMatrix.col(i).normalized();
    }
    return normalizedFeatures;
  };

  BENCHMARK("Semi Handwritten"){
    MatrixXf out(32, 1024*512);
    for(int i=0; i < featureMatrix.cols(); i++){
      float colsum = featureMatrix.array().sqrt().col(i).sum();
      out.col(i) = featureMatrix.col(i) / colsum;
    }
    return out;
  };
}
