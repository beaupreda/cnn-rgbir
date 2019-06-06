#include "../include/TextReader.h"

TextReader::TextReader() {}

void TextReader::readFile(const MapLine& mapLine) {
    const std::string filename = mapLine[MapLine::GT];
    const std::string number = mapLine[MapLine::NUMBER];
    const std::string imagePath = mapLine[MapLine::ABSOLUTE];
    const int isMirror = std::stoi(mapLine[MapLine::MIRROR]);
    const int width = std::stoi(mapLine[MapLine::WIDTH]);
    std::string formattedPath = formatPath(imagePath, isMirror);
    std::ifstream textReader(filename);
    std::string ir, rgb;
    int x, y, d;
    while (!textReader.eof()) {
        textReader >> ir;
        textReader >> rgb;
        formatIrRgb(ir ,rgb);
        textReader >> x;
        textReader >> y;
        textReader >> d;
        if (ir == formattedPath || rgb == formattedPath) {
            cv::Point2i rgbPoint(x - std::abs(d), y);
            cv::Point2i lwirPoint(x, y);
            if (isMirror) {
                rgbPoint.x = width - x + std::abs(d);
                lwirPoint.x = width - x;
            }
            std::get<RGB>(this->imagePoints[number]).emplace_back(rgbPoint);
            std::get<LWIR>(this->imagePoints[number]).emplace_back(lwirPoint);
        }
    }
}

std::string TextReader::formatPath(const std::string& path, const int isMirror) {
    size_t position = path.find(DATASET);
    std::string tempPath;
    if (position != std::string::npos) {
        for (size_t i = position + DATASET_LENGTH; i < path.size(); ++i)
            tempPath += path[i];
    }
    std::string formattedPath;
    position = tempPath.find(VIDEO);
    if (position != std::string::npos) {
        for (size_t i = 0; i < position; ++i)
            formattedPath += tempPath[i];
        if (isMirror) {
            for (size_t i = position + VIDEO_LENGTH; i < tempPath.size() - MIRROR_LENGTH; ++i) {
                formattedPath += tempPath[i];
            }
            formattedPath += JPEG_EXTENSION;
        } else {
            for (size_t i = position + VIDEO_LENGTH; i < tempPath.size(); ++i)
                formattedPath += tempPath[i];
        }
        goto endFunction;
    }
    position = tempPath.find(VIDEO_VARIANT);
    if (position != std::string::npos) {
        for (size_t i = position + VIDEO_VAR_LENGTH; i < tempPath.size(); ++i)
            formattedPath += tempPath[i];
        goto endFunction;
    }
    endFunction:
    for (size_t i = 0; i < formattedPath.size(); ++i)
        if (formattedPath[i] == '/')
            formattedPath.replace(i, 1, "\\\\");
    return formattedPath;
}

void TextReader::formatIrRgb(std::string& ir, std::string& rgb) {
    size_t positionIr = ir.find(FOREGROUND);
    if (positionIr != std::string::npos)
        ir.replace(positionIr, FOREGROUND_LENGTH, "");
    positionIr = ir.find(FOREGROUND);
    if (positionIr != std::string::npos)
        ir.replace(positionIr, FOREGROUND_LENGTH, "");

    size_t positionRgb = rgb.find(FOREGROUND);
    if (positionRgb != std::string::npos)
        rgb.replace(positionRgb, FOREGROUND_LENGTH, "");
    positionRgb = rgb.find(FOREGROUND);
    if (positionRgb != std::string::npos)
        rgb.replace(positionRgb, FOREGROUND_LENGTH, "");
}
