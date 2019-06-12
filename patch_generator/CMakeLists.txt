cmake_minimum_required(VERSION 3.10)
project(PatchGenerator)

set(CMAKE_CXX_STANDARD 11)

set(BOOST_ROOT "/store/dabeaq/boost_1_61_0")
set(OpenCV_DIR "/store/dabeaq/opencv/opencv3/share/OpenCV")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(yaml-cpp REQUIRED PATHS "/store/dabeaq/yaml")
include_directories(${Boost_INCLUDE_DIRS})

set(src_files src/main.cpp
        include/ImageReader.h
        include/GtReader.h
        include/YamlReader.h
        include/TextReader.h
        include/FileNavigator.h
        include/PatchCreator.h
        include/MapLine.h
        include/MapReader.h
        include/VisualVerificator.h
        include/BinaryWriter.h
        include/MixReader.h
        src/ImageReader.cpp
        src/GtReader.cpp
        src/YamlReader.cpp
        src/TextReader.cpp
        src/FileNavigator.cpp
        src/PatchCreator.cpp
        src/MapLine.cpp
        src/MapReader.cpp
        src/VisualVerificator.cpp
        src/BinaryWriter.cpp
        src/MixReader.cpp)

add_executable(PatchGenerator ${src_files})

target_link_libraries(PatchGenerator ${OpenCV_LIBS} ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES})