# Rectification
Rectifies all images from the St-Charles dataset.

## Usage
### Dependencies
* [CMake](https://cmake.org/)
* [LITIV Computer Vision Framework](https://github.com/plstcharles/litiv)
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
Change line #31 in main.cpp to the location of your config.yml file. Execute the program next!
```
./rectification
```