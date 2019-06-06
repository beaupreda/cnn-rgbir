#include <opencv2/opencv.hpp>
#include "ImageReader.h"

#ifndef PATCHGENERATOR_VISUALVERIFICATOR_H
#define PATCHGENERATOR_VISUALVERIFICATOR_H

class VisualVerificator {
public:
    enum INFO {FRAME_NUMBER = 0, NUMBER = 1, RGB_CENTER_X = 2, RGB_CENTER_Y = 3, LWIR_CENTER_X = 4};
    VisualVerificator();
    void showPatches(int halfWidth, int halfRange, const std::vector<float>& patchInfo,
                     const std::string& path, const std::string& extension);
    void showAllPoints(const std::vector<std::vector<float>>& points, const std::string& path, const std::string& extension);
    cv::Rect getRectangleFromCenter(int x, int y, int halfWidth, int halfRange, bool isRgb);
private:
    ImageReader imgReader;
};

#endif //PATCHGENERATOR_VISUALVERIFICATOR_H
