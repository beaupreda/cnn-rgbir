#include <yaml-cpp/yaml.h>

#include "../include/GtReader.h"

#ifndef PATCHGENERATOR_YAMLREADER_H
#define PATCHGENERATOR_YAMLREADER_H

#define X "x"
#define Y "y"
#define D "d"
#define POINT_LENGTH 6
#define VID04 "vid04/"
#define VID07 "vid07/"
#define VID08 "vid08/"
#define SIZE_VIDEO 6

class YamlReader: public GtReader {
public:
    YamlReader();
    void readFile(const MapLine& mapLine);
    std::string getDispRange(const std::string& filename);
    YAML::Node parse(const std::string& filename);
};

#endif //PATCHGENERATOR_YAMLREADER_H
