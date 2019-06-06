#include "../include/MixReader.h"

MixReader::MixReader() {
    this->textReader = TextReader();
    this->yamlReader = YamlReader();
}

void MixReader::readFile(const MapLine& mapLine) {
    const std::string filename = mapLine[MapLine::GT];
    size_t position = filename.find(GAB_ID);
    if (position != std::string::npos)
        this->textReader.readFile(mapLine);
    position = filename.find(PLSC_ID);
    if (position != std::string::npos)
        this->yamlReader.readFile(mapLine);
}

void MixReader::mergeMaps() {
    for (auto it : this->yamlReader.getImagePoints())
        this->imagePoints[it.first] = it.second;
    for (auto it : this->textReader.getImagePoints())
        this->imagePoints[it.first] = it.second;
}

void MixReader::clearImagePoints() {
    this->imagePoints.clear();
    this->yamlReader.clearImagePoints();
    this->textReader.clearImagePoints();
}
