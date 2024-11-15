cmake_minimum_required(VERSION 3.18)
project(ColorToGrayscaleConverter LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

add_library(cucolor2gray gray_scale.cu)

add_executable(color2gray color2gray_main.cpp)
target_link_libraries(color2gray ${OpenCV_LIBS} cucolor2gray)
