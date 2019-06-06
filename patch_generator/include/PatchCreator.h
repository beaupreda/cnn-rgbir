#include <opencv2/opencv.hpp>

#ifndef PATCHGENERATOR_PATCHCREATOR_H
#define PATCHGENERATOR_PATCHCREATOR_H

class PatchCreator {
public:
    PatchCreator(const cv::Point2i& rgbPoint, const cv::Point2i& lwirPoint, int frameNumber);
    bool checkPatchValidity(int halfWidth, int halfRange, int offset, int width, int height, bool isVertical);
    bool checkRgbPatchValidity(int halfWidth, int halfRange, int offset, int width, int height, bool inverted);
    bool checkLwirPatchValidity(int halfWidth, int halfRange, int offset, int width, int height, bool inverted);
    cv::Point2i getRgbCenter() const;
    cv::Point2i getLwirCenter() const;
    void savePoints(std::vector<std::vector<float>>& validPoints, bool isVertical);
    static void transposeValidPoints(const std::vector<std::vector<float>>& validPoints, std::vector<std::vector<float>>& tValidPoints);
private:
    cv::Point2i rgbCenter;
    cv::Point2i lwirCenter;
    int frameNumber;
};

#endif //PATCHGENERATOR_PATCHCREATOR_H
