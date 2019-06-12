# Patch Generator

## Usage
### Dependencies
* [CMake](https://cmake.org/)
* [OpenCV](https://opencv.org/)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)

### Compiling
```
mkdir build
cd build
cmake ../
make -j
```

### Generating Patches
Change line #13 in main.cpp to the location of your config.yml file. Decide the fold number (line #14) and simply execute!
```
./patch_generator
```