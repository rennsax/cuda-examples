cmake_minimum_required(VERSION 3.18)
project(matmul LANGUAGES CXX CUDA)

find_package(GTest REQUIRED)

add_library(cumatmul SHARED matmul.cu)

add_executable(test_matmul test_matmul.cpp)
target_link_libraries(test_matmul
  cumatmul
  GTest::gtest GTest::gtest_main
)
