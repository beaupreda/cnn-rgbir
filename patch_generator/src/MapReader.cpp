#include "../include/MapReader.h"

MapReader::MapReader() {}

void MapReader::read(const std::string& filename) {
    std::ifstream mapReader(filename);
    std::string number, dataset, absolute, gt, mirror, width;
    while (!mapReader.eof()) {
        mapReader >> number;
        mapReader >> dataset;
        mapReader >> absolute;
        mapReader >> gt;
        mapReader >> mirror;
        mapReader >> width;
        int num = std::stoi(number);
        MapLine line = MapLine(number, dataset, absolute, gt, mirror, width);
        this->mapping.emplace(num, line);
    }
}

std::map<int, MapLine> MapReader::getMapping() const {
    return this->mapping;
}

MapLine MapReader::getMapLine(int index) {
    return this->mapping[index];
}

void MapReader::clearMapping() {
    this->mapping.clear();
}

