cmake_minimum_required(VERSION 3.21)
project(deboggler)

set(CMAKE_CXX_STANDARD 17)
#add_subdirectory("3rdparty/opencv")
#SET("OpenCV_DIR" "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/build/opencv")
find_package(OpenCV REQUIRED)# PATHS ${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(deboggler src/main.cpp android/app/src/main/cpp/ProcessImage.h)
add_executable(neuralnetworktest src/neuralnetwork/main.cpp)
add_executable(solutioner src/boggle_solutioner.cpp)

# linking
target_link_libraries(deboggler ${OpenCV_LIBS})
target_link_libraries(neuralnetworktest ${OpenCV_LIBS})