
# make sure you are using a fairly recent version of cmake
cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

# these paths are used as hints for finding the dependencies automatically given a predefined environment variable;
# these should not be needed if you installed all your libraries in a system-default location (e.g. /usr/local/)
list(APPEND CMAKE_PREFIX_PATH
    "$ENV{USER_DEVELOP}/cuda-litiv/" "$ENV{USER_DEVELOP}/cuda-latest/" "$ENV{USER_DEVELOP}/cuda/"
    "$ENV{USER_DEVELOP}/opencv/build-litiv/install/" "$ENV{USER_DEVELOP}/opencv/build/install" "$ENV{USER_DEVELOP}/opencv/"
    "$ENV{USER_DEVELOP}/freeglut/build-litiv/install/" "$ENV{USER_DEVELOP}/freeglut/build/install" "$ENV{USER_DEVELOP}/freeglut/"
    "$ENV{USER_DEVELOP}/glfw/build-litiv/install/" "$ENV{USER_DEVELOP}/glfw/build/install" "$ENV{USER_DEVELOP}/glfw/"
    "$ENV{USER_DEVELOP}/glew/build-litiv/install/" "$ENV{USER_DEVELOP}/glew/build/install" "$ENV{USER_DEVELOP}/glew/"
    "$ENV{USER_DEVELOP}/glm/build-litiv/install/" "$ENV{USER_DEVELOP}/glm/build/install" "$ENV{USER_DEVELOP}/glm/"
    "$ENV{USER_DEVELOP}/opengm/build-litiv/install/" "$ENV{USER_DEVELOP}/opengm/build/install" "$ENV{USER_DEVELOP}/opengm/"
    "$ENV{USER_DEVELOP}/eigen3/build-litiv/install/" "$ENV{USER_DEVELOP}/eigen3/build/install/" "$ENV{USER_DEVELOP}/eigen3/"
)

################################################################

# name of the cmake project that we are creating (will be the name of your solution)
project(rectification)
# minimum c++ standard to use (required by the LITIV framework)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# if using the MSVC toolchain, you should compile with MSVC2015 (v140) minimum
if(WIN32)
    if(MSVC AND (MSVC_VERSION LESS 1900))
        message(FATAL_ERROR "MSVC toolchains older than 2015 (v140) are not supported!")
    endif()
endif()
# this line will look for the LITIV framework by searching for the litiv-config.cmake file that is typically
# installed with it when you run `make install` --- by default, it will be located in a place such as /usr/local/
set(litiv_DIR "/store/dabeaq/litiv/build/install")
find_package(litiv REQUIRED)
# here, we create an executable with a simple source file (main.cpp) that prints info about the framework
add_executable(rectification main.cpp)
# we link the global module of the framework (litiv_world) to our executable target (litiv-ext-test)
target_link_libraries(rectification litiv_world)
# finally, we specify the include directories for external dependencies of the framework
target_include_directories(rectification
    PUBLIC
        ${LITIV_EXT_INCLUDE_DIRS}
)

