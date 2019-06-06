#include "../include/GtReader.h"

#ifndef PATCHGENERATOR_TEXTREADER_H
#define PATCHGENERATOR_TEXTREADER_H

#define DATASET "Dataset"
#define DATASET_LENGTH 8
#define VIDEO "video"
#define VIDEO_LENGTH 11
#define VIDEO_VARIANT "Video"
#define VIDEO_VAR_LENGTH 12
#define FOREGROUND "Foreground"
#define FOREGROUND_LENGTH 10
#define MIRROR_LENGTH 11
#define JPEG_EXTENSION ".jpg"

class TextReader: public GtReader {
public:
    TextReader();
    void readFile(const MapLine& mapLine);
    std::string formatPath(const std::string& path, const int isMirror);
    void formatIrRgb(std::string& ir, std::string& rgb);
};

#endif //PATCHGENERATOR_TEXTREADER_H
