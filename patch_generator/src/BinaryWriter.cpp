#include "../include/BinaryWriter.h"

BinaryWriter::BinaryWriter(const std::string& name, const std::string& info) {
    this->filename = name + info;
}

void BinaryWriter::writePointsToFile(const std::vector<std::vector<float>>& pointsToSave) {
    std::ofstream file(this->filename, std::ofstream::binary | std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << this->filename << std::endl;
        return;
    }
    for (size_t i = 0; i < pointsToSave.size(); ++i) {
        for (size_t j = 0; j < pointsToSave[i].size(); ++j) {
            file.write(reinterpret_cast<const char*>(&pointsToSave[i][j]), sizeof(pointsToSave[i][j]));
        }
    }
    file.close();
}

void BinaryWriter::setFilename(const std::string& name, const std::string& info) {
    this->filename = name + info;
}
