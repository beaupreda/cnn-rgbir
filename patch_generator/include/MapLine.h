#include <string>

#ifndef PATCHGENERATOR_MAPLINE_H
#define PATCHGENERATOR_MAPLINE_H

class MapLine {
public:
    enum Index {NUMBER = 0, DATASET = 1, ABSOLUTE = 2, GT = 3, MIRROR = 4, WIDTH = 5};
    MapLine();
    MapLine(const std::string& number, const std::string& dataPath, const std::string& absPath, const std::string& pathGt,
            const std::string& mirror, const std::string& width);
    MapLine(const MapLine& mapLine);
    const std::string& operator[](const Index& index) const;
private:
    std::string number;
    std::string dataPathToImg;
    std::string absPathToImg;
    std::string pathToGt;
    std::string mirror;
    std::string width;
};

#endif //PATCHGENERATOR_MAPLINE_H
