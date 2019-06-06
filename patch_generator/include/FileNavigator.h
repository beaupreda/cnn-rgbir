#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#ifndef PATCHGENERATOR_FILENAVIGATOR_H
#define PATCHGENERATOR_FILENAVIGATOR_H

#define VIDEO04 "vid04/lwir_gt_disp/"
#define VIDEO07 "/vid07/lwir_gt_disp/"
#define VIDEO08 "/vid08/lwir_gt_disp/"

#define V1P1 "Dataset/vid1/1Person/vid1_1Person.txt"
#define V1P2 "Dataset/vid1/2Person/vid1_2Person.txt"
#define V1P3 "Dataset/vid1/3Person/vid1_3Person.txt"
#define V1P4 "Dataset/vid1/4Person/vid1_4Person.txt"
#define V2C1P1 "Dataset/vid2/cut1/1Person/vid2cut1_1Person.txt"
#define V2C1P2 "Dataset/vid2/cut1/2Person/vid2cut1_2Person.txt"
#define V2C2P2 "Dataset/vid2/cut1/2Person/vid2cut2_2Person.txt"
#define V2C2P3 "Dataset/vid2/cut1/3Person/vid2cut2_3Person.txt"
#define V2C2P4 "Dataset/vid2/cut1/4Person/vid2cut2_4Person.txt"
#define V3P1 "Dataset/vid1/1Person/vid3_1Person.txt"
#define V3P2 "Dataset/vid1/2Person/vid3_2Person.txt"
#define V3P3 "Dataset/vid1/3Person/vid3_3Person.txt"
#define V3P4 "Dataset/vid1/4Person/vid3_4Person.txt"
#define V3P5 "Dataset/vid1/5Person/vid3_5Person.txt"

class FileNavigator {
public:
    FileNavigator(const std::string& gabFolderLocation, const std::string& plscFolderLocation);
    void loadTextFiles();
    void loadYamlFilesLocations();
    void loadYamlFiles(const std::string& directory);
    std::vector<std::string> getTextFiles() const;
    std::vector<std::string> getYamlFilesLocations() const;
    std::vector<std::string> getYamlFiles() const;
private:
    std::string gabFolderLocation;
    std::string plscFolderLocation;
    std::vector<std::string> textFiles;
    std::vector<std::string> yamlFilesLocations;
    std::vector<std::string> yamlFiles;
};

#endif //PATCHGENERATOR_FILENAVIGATOR_H
