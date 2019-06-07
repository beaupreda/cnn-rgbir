#include <iostream>

#include "../include/FileNavigator.h"
#include "../include/MapReader.h"
#include "../include/TextReader.h"
#include "../include/YamlReader.h"
#include "../include/PatchCreator.h"
#include "../include/ImageReader.h"
#include "../include/VisualVerificator.h"
#include "../include/BinaryWriter.h"
#include "../include/MixReader.h"

#define PLSC_LOCATION "/store/dabeaq/datasets/litiv/stcharles2018-v04/"
#define GAB_LOCATION "/store/dabeaq/datasets/bilodeauIR/"
#define RECTIFIED_TRAIN_LOCATION "../../dataset3/train/"
#define RECTIFIED_VALIDATION_LOCATION "../../dataset3/validation/"
#define RECTIFIED_TEST_LOCATION "../../dataset3/test/"
#define TRAIN_MAP "../../dataset3/train/map.txt"
#define VALIDATION_MAP "../../dataset3/validation/map.txt"
#define TEST_MAP "../../dataset3/test/map.txt"
#define HALF_WIDTH 1
#define HALF_RANGE 2
#define OFFSET 0
#define PNG_EXTENSION ".png"
#define BIN_EXTENSION ".bin"
#define VERIFICATION false
#define BIN_LOC_TRAIN "../train"
#define BIN_LOC_VALIDATION "../validation"
#define BIN_LOC_TEST "../test"

int main(int argc, char* argv[]) {
    //////////// arguments set-up ////////////
    if (argc != 3)
        std::cerr << "There must be 2 arguments (patch size and half range)" << std::endl;
    const std::string hw = std::string(argv[HALF_WIDTH]);
    const std::string hr = std::string(argv[HALF_RANGE]);
    const int halfWidth = std::stoi(hw);
    const int halfRange = std::stoi(hr);
    
    FileNavigator fn(GAB_LOCATION, PLSC_LOCATION);
    fn.loadTextFiles();
    fn.loadYamlFilesLocations();
    for (const auto& dir : fn.getYamlFilesLocations())
        fn.loadYamlFiles(dir);

    //////////// training ////////////
    std::cout << "Starting patch generation for training data..." << std::endl;
    MapReader mr;
    mr.read(TRAIN_MAP);

    MixReader mixReader;
    for (auto it : mr.getMapping())
        mixReader.readFile(it.second);
    mixReader.mergeMaps();

    std::cout << "Generating patches location for dataset..." << std::endl;
    unsigned int totalPoints = 0;
    ImageReader imgReader;
    std::vector<std::vector<float>> validPoints;
    for (auto it : mixReader.getImagePoints()) {
        imgReader.readImagesFromPath(RECTIFIED_TRAIN_LOCATION, it.first, PNG_EXTENSION);
        totalPoints += 3 * it.second.first.size();
        for (int i = 0; i < it.second.first.size(); ++i) {
            PatchCreator centerPatch(it.second.first[i], it.second.second[i], std::stoi(it.first));
            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                centerPatch.savePoints(validPoints);

            cv::Point2i upperPointRGB(it.second.first[i].x, it.second.first[i].y + 1);
            cv::Point2i upperPointLWIR(it.second.second[i].x, it.second.second[i].y + 1);
            PatchCreator upperPatch(upperPointRGB, upperPointLWIR, std::stoi(it.first));
            if (upperPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                upperPatch.savePoints(validPoints);

            cv::Point2i lowerPointRGB(it.second.first[i].x, it.second.first[i].y - 1);
            cv::Point2i lowerPointLWIR(it.second.second[i].x, it.second.second[i].y - 1);
            PatchCreator lowerPatch(lowerPointRGB, lowerPointLWIR, std::stoi(it.first));
            if (lowerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                lowerPatch.savePoints(validPoints);
        }
    }
    std::cout << std::endl;
    std::cout << "---------- Summary of generated points ----------" << std::endl;
    std::cout << "Number of valid points (train): " << validPoints.size() << std::endl;
    std::cout << "Total number of points (train): " << totalPoints << std::endl;
    std::cout << std::endl;

    //////////// verification ////////////
#if VERIFICATION
    std::cout << "Starting visual verification..." << std::endl;
    VisualVerificator vs;
    vs.showAllPoints(validPoints, RECTIFIED_TRAIN_LOCATION, PNG_EXTENSION);
    for (auto point : validPoints)
        vs.showPatches(halfWidth, halfRange, point, RECTIFIED_TRAIN_LOCATION, PNG_EXTENSION);
#endif

    //////////// save to binary file ////////////
    std::cout << "Saving patches location to binary file..." << std::endl;
    std::cout << std::endl << std::endl;
    BinaryWriter bw(BIN_LOC_TRAIN, BIN_EXTENSION);
    bw.writePointsToFile(validPoints);
    auto train = validPoints;

    //////////// clear variables used for training ////////////
    mr.clearMapping();
    mixReader.clearImagePoints();
    validPoints.clear();

    //////////// validation ////////////
    std::cout << "Starting patch generation for validation data..." << std::endl;
    mr.read(VALIDATION_MAP);

    for (auto itAll : mr.getMapping())
        mixReader.readFile(itAll.second);
    mixReader.mergeMaps();

    std::cout << "Generating patches location for dataset..." << std::endl;
    totalPoints = 0;
    for (auto it : mixReader.getImagePoints()) {
        imgReader.readImagesFromPath(RECTIFIED_VALIDATION_LOCATION, it.first, PNG_EXTENSION);
        totalPoints += 3 * it.second.first.size();
        for (int i = 0; i < it.second.first.size(); ++i) {
            PatchCreator centerPatch(it.second.first[i], it.second.second[i], std::stoi(it.first));

            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                centerPatch.savePoints(validPoints);

            cv::Point2i upperPointRGB(it.second.first[i].x, it.second.first[i].y + 1);
            cv::Point2i upperPointLWIR(it.second.second[i].x, it.second.second[i].y + 1);
            PatchCreator upperPatch(upperPointRGB, upperPointLWIR, std::stoi(it.first));
            if (upperPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                upperPatch.savePoints(validPoints);

            cv::Point2i lowerPointRGB(it.second.first[i].x, it.second.first[i].y - 1);
            cv::Point2i lowerPointLWIR(it.second.second[i].x, it.second.second[i].y - 1);
            PatchCreator lowerPatch(lowerPointRGB, lowerPointLWIR, std::stoi(it.first));
            if (lowerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                lowerPatch.savePoints(validPoints);
        }
    }
    std::cout << std::endl;
    std::cout << "---------- Summary of generated points ----------" << std::endl;
    std::cout << "Number of valid points (validation): " << validPoints.size() << std::endl;
    std::cout << "Total number of points (validation): " << totalPoints << std::endl;
    std::cout << std::endl;

    //////////// verification ////////////
#if VERIFICATION
    std::cout << "Starting visual verification..." << std::endl;
    vs.showAllPoints(validPoints, RECTIFIED_VALIDATION_LOCATION, PNG_EXTENSION);
    for (auto point : validPoints)
        vs.showPatches(halfWidth, halfRange, point, RECTIFIED_VALIDATION_LOCATION, PNG_EXTENSION);
#endif

    //////////// save to binary file ////////////
    std::cout << "Saving patches location to binary file..." << std::endl;
    std::cout << std::endl << std::endl;
    bw.setFilename(BIN_LOC_VALIDATION, BIN_EXTENSION);
    bw.writePointsToFile(validPoints);
    auto val = validPoints;

    //////////// clear variables used for validation ////////////
    mr.clearMapping();
    mixReader.clearImagePoints();
    validPoints.clear();

    //////////// testing ////////////
    std::cout << "Starting patch generation for testing data..." << std::endl;
    mr.read(TEST_MAP);

    for (auto itAll : mr.getMapping())
        mixReader.readFile(itAll.second);
    mixReader.mergeMaps();

    std::cout << "Generating patches location for dataset..." << std::endl;
    totalPoints = 0;
    for (auto it : mixReader.getImagePoints()) {
        imgReader.readImagesFromPath(RECTIFIED_TEST_LOCATION, it.first, PNG_EXTENSION);
        totalPoints += 3 * it.second.first.size();
        for (int i = 0; i < it.second.first.size(); ++i) {
            PatchCreator centerPatch(it.second.first[i], it.second.second[i], std::stoi(it.first));

            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                centerPatch.savePoints(validPoints);

            cv::Point2i upperPointRGB(it.second.first[i].x, it.second.first[i].y + 1);
            cv::Point2i upperPointLWIR(it.second.second[i].x, it.second.second[i].y + 1);
            PatchCreator upperPatch(upperPointRGB, upperPointLWIR, std::stoi(it.first));
            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                centerPatch.savePoints(validPoints);

            cv::Point2i lowerPointRGB(it.second.first[i].x, it.second.first[i].y - 1);
            cv::Point2i lowerPointLWIR(it.second.second[i].x, it.second.second[i].y - 1);
            PatchCreator lowerPatch(lowerPointRGB, lowerPointLWIR, std::stoi(it.first));
            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                centerPatch.savePoints(validPoints);
        }
    }
    std::cout << std::endl;
    std::cout << "---------- Summary of generated points ----------" << std::endl;
    std::cout << "Number of valid points (test): " << validPoints.size() << std::endl;
    std::cout << "Total number of points (test): " << totalPoints << std::endl;
    std::cout << std::endl;

    //////////// verification ////////////
#if VERIFICATION
    std::cout << "Starting visual verification..." << std::endl;
    vs.showAllPoints(validPoints, RECTIFIED_TEST_LOCATION, PNG_EXTENSION);
    for (auto point : validPoints)
        vs.showPatches(halfWidth, halfRange, point, RECTIFIED_TEST_LOCATION, PNG_EXTENSION);
#endif

    //////////// save to binary file ////////////
    std::cout << "Saving patches location to binary file..." << std::endl;
    std::cout << std::endl << std::endl;
    bw.setFilename(BIN_LOC_TEST, BIN_EXTENSION);
    bw.writePointsToFile(validPoints);
    auto testAll = validPoints;

    //////////// clear variables used for validation ////////////
    mr.clearMapping();
    mixReader.clearImagePoints();
    validPoints.clear();
}