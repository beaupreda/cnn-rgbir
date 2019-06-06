cmake_minimum_required(VERSION 3.10)
project(rectification)

set(CMAKE_CXX_STANDARD 11)

set(BOOST_ROOT "/store/dabeaq/boost_1_61_0")
set(OpenCV_DIR "/store/dabeaq/opencv/opencv3/share/OpenCV")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(rectification main.cpp)

target_link_libraries(rectification ${OpenCV_LIBS} ${Boost_LIBRARIES})