#include <opencv2/imgcodecs/imgcodecs_c.h>
#include "../include/ImageReader.h"

ImageReader::ImageReader() {}

void ImageReader::readImagesFromPath(const std::string& path, const std::string& name, const std::string& ext) {
    const std::string rgbPath = path + RGB_PATH + name + ext;
    const std::string lwirPath = path + LWIR_PATH + name + ext;

    std::get<RGB>(this->imagePair) = cv::imread(rgbPath, CV_LOAD_IMAGE_COLOR);
    std::get<LWIR>(this->imagePair) = cv::imread(lwirPath, CV_LOAD_IMAGE_COLOR);
}

cv::Mat ImageReader::getLeftImg() const {
    return std::get<LEFT>(this->imagePair);
}

cv::Mat ImageReader::getRgbImg() const {
    return std::get<RGB>(this->imagePair);
}

cv::Mat ImageReader::getRightImg() const {
    return std::get<RIGHT>(this->imagePair);
}

cv::Mat ImageReader::getLwirImg() const {
    return std::get<LWIR>(this->imagePair);
}

void ImageReader::setLeftImg(const cv::Mat& leftImg) {
    std::get<LEFT>(this->imagePair) = leftImg;
}

void ImageReader::setRgbImg(const cv::Mat& rgbImg) {
    std::get<RGB>(this->imagePair) = rgbImg;
}

void ImageReader::setRightImg(const cv::Mat& rightImg) {
    std::get<RIGHT>(this->imagePair) = rightImg;
}

void ImageReader::setLwirImg(const cv::Mat& lwirImg) {
    std::get<LWIR>(this->imagePair) = lwirImg;
}

