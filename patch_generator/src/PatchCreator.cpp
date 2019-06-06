#include "../include/PatchCreator.h"

PatchCreator::PatchCreator(const cv::Point2i& rgbPoint, const cv::Point2i& lwirPoint, int frameNumber) {
    this->rgbCenter = rgbPoint;
    this->lwirCenter = lwirPoint;
    this->frameNumber = frameNumber;
}

bool PatchCreator::checkPatchValidity(int halfWidth, int halfRange, int offset, int width, int height) {
    bool rgbValid = this->rgbCenter.x + halfWidth + halfRange < width &&
                    this->rgbCenter.x - halfWidth - halfRange - offset > 0 &&
                    this->rgbCenter.y + halfWidth < height &&
                    this->rgbCenter.y - halfWidth > 0;
    bool lwirValid = this->lwirCenter.x + halfWidth + halfRange < width &&
                     this->lwirCenter.x - halfWidth - halfRange - offset > 0 &&
                     this->lwirCenter.y + halfWidth < height &&
                     this->lwirCenter.y - halfWidth > 0;
    return rgbValid && lwirValid;
}

cv::Point2i PatchCreator::getRgbCenter() const {
    return this->rgbCenter;
}

cv::Point2i PatchCreator::getLwirCenter() const {
    return this->lwirCenter;
}

void PatchCreator::savePoints(std::vector<std::vector<float>>& validPoints,) {
    float type = 1.0;
    std::vector<float> points = {static_cast<float>(this->frameNumber), type, static_cast<float>(this->rgbCenter.x),
                                 static_cast<float>(this->rgbCenter.y), static_cast<float>(this->lwirCenter.x)};
    validPoints.emplace_back(points);
}

void PatchCreator::transposeValidPoints(const std::vector<std::vector<float>>& validPoints, std::vector<std::vector<float>>& tValidPoints) {
    for (int i = 0; i < validPoints.size(); ++i) {
        for (int j = 0; j < validPoints[i].size(); ++j) {
            tValidPoints[j][i] = validPoints[i][j];
        }
    }
}
