#include "../include/YamlReader.h"

YamlReader::YamlReader() {}

void YamlReader::readFile(const MapLine& mapLine) {
    const std::string filename = mapLine[MapLine::GT];
    const std::string metadata = getDispRange(filename);
    const std::string number = mapLine[MapLine::NUMBER];
    const int width = std::stoi(mapLine[MapLine::WIDTH]);
    const int isMirror = std::stoi(mapLine[MapLine::MIRROR]);
    std::ifstream dispRangeFile(metadata);
    int minDisp, maxDisp;
    if (dispRangeFile.is_open())
        dispRangeFile >> minDisp >> maxDisp;
    else
        return;
    cv::FileStorage yamlGt(filename, cv::FileStorage::READ);
    cv::FileNode root = yamlGt.root();
    for (cv::FileNodeIterator nIter = root.begin(); nIter != root.end(); ++nIter) {
        if ((*nIter).name().size() == POINT_LENGTH) {
            int x, y, d;
            (*nIter)[X] >> x;
            (*nIter)[Y] >> y;
            (*nIter)[D] >> d;
            cv::Point2i rgbPoint(x, y);
            cv::Point2i lwirPoint(x + d + maxDisp, y);
            if (!isMirror) {
                rgbPoint.x = width - x;
                lwirPoint.x = width - (x + d + maxDisp);
            }
            std::get<RGB>(this->imagePoints[number]).emplace_back(rgbPoint);
            std::get<LWIR>(this->imagePoints[number]).emplace_back(lwirPoint);
        }
    }
}

std::string YamlReader::getDispRange(const std::string& filename) {
    size_t position = filename.find(VID04);
    if (position == std::string::npos) {
        position = filename.find(VID07);
        if (position == std::string::npos) {
            position = filename.find(VID08);
            if (position == std::string::npos) {
                return NULL;
            }
        }
    }
    std::string pathToGt;
    for (size_t i = 0; i < position + SIZE_VIDEO; ++i)
        pathToGt += filename[i];
    return pathToGt + "drange.txt";

}

YAML::Node YamlReader::parse(const std::string& filename) {
    return YAML::LoadFile(filename);
}
