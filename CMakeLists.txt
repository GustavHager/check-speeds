cmake_minimum_required(VERSION 3.22.1)

project("Compare matrix operations"
  CXX)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Catch2 3 REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(benchmarks
  matrices.cpp
  tensors.cpp)

target_link_libraries(benchmarks PRIVATE
  Eigen3::Eigen
  Catch2::Catch2WithMain
  blas)

target_compile_options(benchmarks PRIVATE
  "-O3"
  "-ffast-math")

target_compile_definitions(benchmarks PRIVATE
  NDEBUG
  EIGEN_USE_BLAS)
  
