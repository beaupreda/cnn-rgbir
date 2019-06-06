#include "../include/VisualVerificator.h"

VisualVerificator::VisualVerificator() {}

void VisualVerificator::showPatches(int halfWidth, int halfRange, const std::vector<float>& patchInfo,
                                    const std::string& path, const std::string& extension) {
    this->imgReader.readImagesFromPath(path, std::to_string(static_cast<int>(patchInfo[FRAME_NUMBER])), extension);
    cv::Rect rgbRegion = getRectangleFromCenter(static_cast<int>(patchInfo[RGB_CENTER_X]), static_cast<int>(patchInfo[RGB_CENTER_Y]), halfWidth, halfRange, true);
    cv::Rect lwirRegion = getRectangleFromCenter(static_cast<int>(patchInfo[LWIR_CENTER_X]), static_cast<int>(patchInfo[RGB_CENTER_Y]), halfWidth, halfRange, false);
    cv::Mat mergedImg, mergedPatches;
    cv::Mat rgbPatch(this->imgReader.getRgbImg().size(), this->imgReader.getRgbImg().type(), cv::Scalar(0));
    cv::Mat lwirPatch(this->imgReader.getRgbImg().size(), this->imgReader.getRgbImg().type(), cv::Scalar(0));
    this->imgReader.getRgbImg()(rgbRegion).copyTo(rgbPatch(rgbRegion));
    this->imgReader.getLwirImg()(lwirRegion).copyTo(lwirPatch(lwirRegion));
    cv::circle(this->imgReader.getRgbImg(), cv::Point2i(static_cast<int>(patchInfo[RGB_CENTER_X]), static_cast<int>(patchInfo[RGB_CENTER_Y])), 2, cv::Scalar(255, 0, 0), 3);
    cv::circle(this->imgReader.getLwirImg(), cv::Point2i(static_cast<int>(patchInfo[LWIR_CENTER_X]), static_cast<int>(patchInfo[RGB_CENTER_Y])), 2, cv::Scalar(255, 0, 0), 3);
    cv::circle(rgbPatch, cv::Point2i(static_cast<int>(patchInfo[RGB_CENTER_X]), static_cast<int>(patchInfo[RGB_CENTER_Y])), 2, cv::Scalar(255, 0, 0), 3);
    cv::circle(lwirPatch, cv::Point2i(static_cast<int>(patchInfo[LWIR_CENTER_X]), static_cast<int>(patchInfo[RGB_CENTER_Y])), 2, cv::Scalar(255, 0, 0), 3);
    cv::hconcat(rgbPatch, lwirPatch, mergedPatches);
    cv::hconcat(this->imgReader.getRgbImg(), this->imgReader.getLwirImg(), mergedImg);
    std::cout << "Displaying images from: " << path << std::to_string(static_cast<int>(patchInfo[FRAME_NUMBER])) << extension << std::endl;
    cv::namedWindow("Images");
    cv::namedWindow("Patches");
    cv::imshow("Images", mergedImg);
    cv::imshow("Patches", mergedPatches);
    cv::waitKey(0);
}

void VisualVerificator::showAllPoints(const std::vector<std::vector<float>>& points, const std::string& path, const std::string& extension) {
    std::map<int, std::vector<std::pair<cv::Point2i, cv::Point2i>>> pointToImg;
    for (const auto& point : points) {
        int frameNumber = static_cast<int>(point[FRAME_NUMBER]);
        auto rgbPoint = cv::Point2i(static_cast<int>(point[RGB_CENTER_X]), static_cast<int>(point[RGB_CENTER_Y]));
        auto lwirPoint = cv::Point2i(static_cast<int>(point[LWIR_CENTER_X]), static_cast<int>(point[RGB_CENTER_Y]));
        pointToImg[frameNumber].emplace_back(std::make_pair(rgbPoint, lwirPoint));
    }
    for (const auto& iter : pointToImg) {
        this->imgReader.readImagesFromPath(path, std::to_string(iter.first), extension);
        for (const auto& locations : iter.second) {
            cv::circle(this->imgReader.getRgbImg(), locations.first, 2, cv::Scalar(255, 0, 0));
            cv::circle(this->imgReader.getLwirImg(), locations.second, 2, cv::Scalar(255, 0, 0));
        }
        cv::Mat mergedImg;
        cv::hconcat(this->imgReader.getRgbImg(), this->imgReader.getLwirImg(), mergedImg);
        std::cout << iter.first << std::endl;
        cv::namedWindow("Images");
        cv::imshow("Images", mergedImg);
        cv::waitKey(0);
    }
}

cv::Rect VisualVerificator::getRectangleFromCenter(int x, int y, int halfWidth, int halfRange, bool isRgb) {
    if (isRgb) {
        cv::Point2i tl(x - halfWidth, y - halfWidth);
        cv::Point2i br(x + halfWidth, y + halfWidth);
        return cv::Rect(tl, br);
    }
    cv::Point2i tl(x - halfRange - halfWidth, y - halfWidth);
    cv::Point2i br(x + halfRange + halfWidth, y + halfWidth);
    return cv::Rect(tl, br);
}

