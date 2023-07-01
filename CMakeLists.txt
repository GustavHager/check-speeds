cmake_minimum_required(VERSION 3.22.1)

project(Speed-benchmark
	VERSION 1.0
	LANGUAGES CXX)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(Catch2 3 REQUIRED)

add_executable(benchmarks
  matrices.cpp
  tensors.cpp)

target_link_libraries(benchmarks PRIVATE
  Eigen3::Eigen
  Catch2::Catch2WithMain)

target_compile_options(benchmarks PRIVATE
  "-O3"
  "-ffast-math")

target_compile_definitions(benchmarks PRIVATE
  -DEIGEN_NO_ASSERT
  -DNDEBUG
  -DEIGEN_DONT_VECTORIZE
  -DEIGEN_NO_DEBUG
  )
