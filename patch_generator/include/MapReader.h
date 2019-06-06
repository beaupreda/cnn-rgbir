#include <fstream>
#include <map>
#include "MapLine.h"

#ifndef PATCHGENERATOR_MAPREADER_H
#define PATCHGENERATOR_MAPREADER_H

class MapReader {
public:
    MapReader();
    void read(const std::string& filename);
    std::map<int, MapLine> getMapping() const;
    MapLine getMapLine(int index);
    void clearMapping();
private:
    std::map<int, MapLine> mapping;
};

#endif //PATCHGENERATOR_MAPREADER_H
