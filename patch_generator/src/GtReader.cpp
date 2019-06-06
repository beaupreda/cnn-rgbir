#include "../include/GtReader.h"

GtReader::GtReader() {}

std::map<std::string, std::pair<std::vector<cv::Point2i>, std::vector<cv::Point2i>>> GtReader::getImagePoints() const {
    return this->imagePoints;
}

std::vector<cv::Point2i> GtReader::getLeftPoints(const std::string& imgName) {
    return std::get<LEFT>(this->imagePoints[imgName]);
}

std::vector<cv::Point2i> GtReader::getRgbPoints(const std::string& imgName) {
    return std::get<RGB>(this->imagePoints[imgName]);
}

std::vector<cv::Point2i> GtReader::getRightPoints(const std::string& imgName) {
    return std::get<RIGHT>(this->imagePoints[imgName]);
}

std::vector<cv::Point2i> GtReader::getLwirPoints(const std::string& imgName) {
    return std::get<LWIR>(this->imagePoints[imgName]);
}

void GtReader::clearImagePoints() {
    this->imagePoints.clear();
}
