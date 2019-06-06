#include "../include/MapLine.h"

MapLine::MapLine() {}

MapLine::MapLine(const std::string& number, const std::string& dataPath, const std::string& absPath,
                 const std::string& pathGt, const std::string& mirror, const std::string& width) {
    this->number = number;
    this->dataPathToImg = dataPath;
    this->absPathToImg = absPath;
    this->pathToGt = pathGt;
    this->mirror = mirror;
    this->width = width;
}

MapLine::MapLine(const MapLine& mapLine) {
    this->number = mapLine.number;
    this->dataPathToImg = mapLine.dataPathToImg;
    this->absPathToImg = mapLine.absPathToImg;
    this->pathToGt = mapLine.pathToGt;
    this->mirror = mapLine.mirror;
    this->width = mapLine.width;
}

const std::string& MapLine::operator[](const MapLine::Index& index) const {
    if (index == NUMBER)
        return this->number;
    if (index == DATASET)
        return this->dataPathToImg;
    if (index == ABSOLUTE)
        return this->absPathToImg;
    if (index == GT)
        return this->pathToGt;
    if (index == MIRROR)
        return this->mirror;
    if (index == WIDTH)
        return this->width;
    return NULL;
}
