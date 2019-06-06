#include <utility>
#include <opencv2/opencv.hpp>

#ifndef PATCHGENERATOR_IMAGEREADER_H
#define PATCHGENERATOR_IMAGEREADER_H

#define RGB_PATH "rgb/"
#define LWIR_PATH "lwir/"

class ImageReader {
public:
    enum Position {LEFT = 0, RGB = 0, RIGHT = 1, LWIR = 1};
    ImageReader();
    void readImagesFromPath(const std::string& path, const std::string& name, const std::string& ext);
    cv::Mat getLeftImg() const;
    cv::Mat getRgbImg() const;
    cv::Mat getRightImg() const;
    cv::Mat getLwirImg() const;
    void setLeftImg(const cv::Mat& leftImg);
    void setRgbImg(const cv::Mat& rgbImg);
    void setRightImg(const cv::Mat& rightImg);
    void setLwirImg(const cv::Mat& lwirImg);
private:
    std::pair<cv::Mat, cv::Mat> imagePair;
};

#endif //PATCHGENERATOR_IMAGEREADER_H
