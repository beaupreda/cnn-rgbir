#include <fstream>
#include <vector>
#include <iostream>

#ifndef PATCHGENERATOR_BINARYWRITER_H
#define PATCHGENERATOR_BINARYWRITER_H

class BinaryWriter {
public:
    BinaryWriter(const std::string& name, const std::string& info);
    void writePointsToFile(const std::vector<std::vector<float>>& pointsToSave);
    void setFilename(const std::string& name, const std::string& info);
private:
    std::string filename;
};

#endif //PATCHGENERATOR_BINARYWRITER_H
