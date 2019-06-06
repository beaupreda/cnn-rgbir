#include <utility>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "MapLine.h"

#ifndef PATCHGENERATOR_GTREADER_H
#define PATCHGENERATOR_GTREADER_H

class GtReader {
public:
    enum Position {LEFT = 0, RGB = 0, RIGHT = 1, LWIR = 1};
    GtReader();
    virtual void readFile(const MapLine& mapLine) = 0;
    std::map<std::string, std::pair<std::vector<cv::Point2i>, std::vector<cv::Point2i>>> getImagePoints() const;
    std::vector<cv::Point2i> getLeftPoints(const std::string& imgName);
    std::vector<cv::Point2i> getRgbPoints(const std::string& imgName);
    std::vector<cv::Point2i> getRightPoints(const std::string& imgName);
    std::vector<cv::Point2i> getLwirPoints(const std::string& imgName);
    void clearImagePoints();
protected:
    // image name -> points for RGB (x, y) and LWIR (x - d, y) images
    std::map<std::string, std::pair<std::vector<cv::Point2i>, std::vector<cv::Point2i>>> imagePoints;
};

#endif //PATCHGENERATOR_GTREADER_H
