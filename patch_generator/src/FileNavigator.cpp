#include "../include/FileNavigator.h"

FileNavigator::FileNavigator(const std::string& gabFolderLocation, const std::string& plscFolderLocation) {
    this->gabFolderLocation = gabFolderLocation;
    this->plscFolderLocation = plscFolderLocation;
}

void FileNavigator::loadTextFiles() {
    this->textFiles.emplace_back(this->gabFolderLocation + V1P1);
    this->textFiles.emplace_back(this->gabFolderLocation + V1P2);
    this->textFiles.emplace_back(this->gabFolderLocation + V1P3);
    this->textFiles.emplace_back(this->gabFolderLocation + V1P4);
    this->textFiles.emplace_back(this->gabFolderLocation + V2C1P1);
    this->textFiles.emplace_back(this->gabFolderLocation + V2C1P2);
    this->textFiles.emplace_back(this->gabFolderLocation + V2C2P2);
    this->textFiles.emplace_back(this->gabFolderLocation + V2C2P3);
    this->textFiles.emplace_back(this->gabFolderLocation + V2C2P4);
    this->textFiles.emplace_back(this->gabFolderLocation + V3P1);
    this->textFiles.emplace_back(this->gabFolderLocation + V3P2);
    this->textFiles.emplace_back(this->gabFolderLocation + V3P3);
    this->textFiles.emplace_back(this->gabFolderLocation + V3P4);
    this->textFiles.emplace_back(this->gabFolderLocation + V3P5);
}

void FileNavigator::loadYamlFilesLocations() {
    this->yamlFilesLocations.emplace_back(this->plscFolderLocation + VIDEO04);
    this->yamlFilesLocations.emplace_back(this->plscFolderLocation + VIDEO07);
    this->yamlFilesLocations.emplace_back(this->plscFolderLocation + VIDEO08);
}

void FileNavigator::loadYamlFiles(const std::string& directory) {
    boost::filesystem::directory_iterator iterDir(directory);
    boost::filesystem::directory_iterator iterDirEnd;
    while (iterDir != iterDirEnd) {
        if (boost::filesystem::is_regular_file(iterDir->path())) {
            this->yamlFiles.emplace_back(iterDir->path().string());
        }
        ++iterDir;
    }
    std::sort(this->yamlFiles.begin(), this->yamlFiles.end(),
              [](const std::string& str1, const std::string& str2) {
        return str1 < str2;
    });
}

std::vector<std::string> FileNavigator::getTextFiles() const {
    return this->textFiles;
}

std::vector<std::string> FileNavigator::getYamlFilesLocations() const {
    return this->yamlFilesLocations;
}

std::vector<std::string> FileNavigator::getYamlFiles() const {
    return this->yamlFiles;
}
