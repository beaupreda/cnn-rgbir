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
#define RECTIFIED_TRAIN_LOCATION "../../test_dataset_1_mix_art/train/"
#define RECTIFIED_VALIDATION_LOCATION "../../test_dataset_1_mix_art/validation/"
#define RECTIFIED_TEST_LOCATION "../../test_dataset_1_mix_art/test/"
#define TRAIN_MAP "../../test_dataset_1_mix_art/train/map.txt"
#define VALIDATION_MAP "../../test_dataset_1_mix_art/validation/map.txt"
#define TEST_MAP "../../test_dataset_1_mix_art/test/map.txt"
#define HALF_WIDTH 1
#define HALF_RANGE 2
#define OFFSET 0
#define PNG_EXTENSION ".png"
#define JPEG_EXTENSION ".jpg"
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
    MapReader mrAll;
    mrAll.read(ALL_TRAIN_MAP);

    MixReader mixReader;
    for (auto itAll : mrAll.getMapping())
        mixReader.readFile(itAll.second);
    mixReader.mergeMaps();

    std::cout << "Generating patches location for mix dataset..." << std::endl;
    unsigned int totalPointsAll = 0;
    std::vector<std::vector<float>> validPointsAll;
    for (auto itMix : mixReader.getImagePoints()) {
        imgReader.readImagesFromPath(ALL_RECTIFIED_TRAIN_LOCATION, itMix.first, PNG_EXTENSION);
        totalPointsAll += 3 * itMix.second.first.size();
        for (int i = 0; i < itMix.second.first.size(); ++i) {
            PatchCreator centerPatch(itMix.second.first[i], itMix.second.second[i], std::stoi(itMix.first));
#if NEW_ARCH
            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, VERTICAL))
                centerPatch.savePoints(validPointsAll, VERTICAL);
#else
            if (centerPatch.checkRgbPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, INVERT_RGB_LWIR) &&
                centerPatch.checkLwirPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getLwirImg().cols, imgReader.getLwirImg().rows, INVERT_RGB_LWIR)) {
                centerPatch.savePoints(validPointsAll);
            }
#endif
            cv::Point2i upperPointRGB(itMix.second.first[i].x, itMix.second.first[i].y + 1);
            cv::Point2i upperPointLWIR(itMix.second.second[i].x, itMix.second.second[i].y + 1);
            PatchCreator upperPatch(upperPointRGB, upperPointLWIR, std::stoi(itMix.first));
#if NEW_ARCH
            if (upperPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, VERTICAL))
                upperPatch.savePoints(validPointsAll, VERTICAL);
#else
            if (upperPatch.checkRgbPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, INVERT_RGB_LWIR) &&
                upperPatch.checkLwirPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getLwirImg().cols, imgReader.getLwirImg().rows, INVERT_RGB_LWIR)) {
                upperPatch.savePoints(validPointsAll);
            }
#endif
            cv::Point2i lowerPointRGB(itMix.second.first[i].x, itMix.second.first[i].y - 1);
            cv::Point2i lowerPointLWIR(itMix.second.second[i].x, itMix.second.second[i].y - 1);
            PatchCreator lowerPatch(lowerPointRGB, lowerPointLWIR, std::stoi(itMix.first));
#if NEW_ARCH
            if (lowerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, VERTICAL))
                lowerPatch.savePoints(validPointsAll, VERTICAL);
#else
            if (lowerPatch.checkRgbPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, INVERT_RGB_LWIR) &&
                lowerPatch.checkLwirPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getLwirImg().cols, imgReader.getLwirImg().rows, INVERT_RGB_LWIR)) {
                lowerPatch.savePoints(validPointsAll);
            }
#endif
        }
    }
    std::cout << std::endl;
    std::cout << "---------- Summary of generated points ----------" << std::endl;
    std::cout << "Number of valid points (ALL-train): " << validPointsAll.size() << std::endl;
    std::cout << "Total number of points (ALL-train): " << totalPointsAll << std::endl;
    std::cout << std::endl;

    //////////// verification ////////////
#if VERIFICATION
    std::cout << "Starting visual verification of the points..." << std::endl;
    VisualVerificator vs;
    //vs.showAllPoints(validPointsAll, ALL_RECTIFIED_TRAIN_LOCATION, PNG_EXTENSION, INVERT_RGB_LWIR);
    //vs.showAllPoints(validPointsPlsc, plscTrainRectifiedDataset, PNG_EXTENSION);
    //vs.showAllPoints(validPointsGab, gabTrainRectifiedDataset, JPEG_EXTENSION);
    //for (auto point : validPointsPlsc)
        //vs.showPatches(halfWidth, halfRange, point, plscTrainRectifiedDataset, PNG_EXTENSION);
    //for (auto point : validPointsGab)
        //vs.showPatches(halfWidth, halfRange, point, gabTrainRectifiedDataset, JPEG_EXTENSION);
    for (auto point : validPointsAll)
        vs.showPatches(halfWidth, halfRange, point, ALL_RECTIFIED_TRAIN_LOCATION, PNG_EXTENSION, INVERT_RGB_LWIR, VERTICAL);
#endif

    //////////// save to binary file ////////////
    std::cout << "Saving patches location to binary files..." << std::endl;
    std::cout << std::endl << std::endl;
    BinaryWriter bwAll(ALL_BIN_LOC_TRAIN, "_" + invert + hw + "_" + hr + filetype + "_" + id + BIN_EXTENSION);
    bwAll.writePointsToFile(validPointsAll);
    auto trainAll = validPointsAll;

    //////////// clear variables used for training ////////////
    mrAll.clearMapping();
    mixReader.clearImagePoints();
    validPointsAll.clear();

    //////////// validation ////////////
    std::cout << "Starting patch generation for validation data..." << std::endl;
    mrAll.read(ALL_VAL_MAP);

    for (auto itAll : mrAll.getMapping())
        mixReader.readFile(itAll.second);
    mixReader.mergeMaps();

    std::cout << "Generating patches location for mix dataset..." << std::endl;
    totalPointsAll = 0;
    for (auto itMix : mixReader.getImagePoints()) {
        imgReader.readImagesFromPath(ALL_RECTIFIED_VAL_LOCATION, itMix.first, PNG_EXTENSION);
        totalPointsAll += 3 * itMix.second.first.size();
        for (int i = 0; i < itMix.second.first.size(); ++i) {
            PatchCreator centerPatch(itMix.second.first[i], itMix.second.second[i], std::stoi(itMix.first));
#if NEW_ARCH
            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, VERTICAL))
                centerPatch.savePoints(validPointsAll, VERTICAL);
#else
            if (centerPatch.checkRgbPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, INVERT_RGB_LWIR) &&
                centerPatch.checkLwirPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getLwirImg().cols, imgReader.getLwirImg().rows, INVERT_RGB_LWIR)) {
                centerPatch.savePoints(validPointsAll);
            }
#endif
            cv::Point2i upperPointRGB(itMix.second.first[i].x, itMix.second.first[i].y + 1);
            cv::Point2i upperPointLWIR(itMix.second.second[i].x, itMix.second.second[i].y + 1);
            PatchCreator upperPatch(upperPointRGB, upperPointLWIR, std::stoi(itMix.first));
#if NEW_ARCH
            if (upperPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, VERTICAL))
                upperPatch.savePoints(validPointsAll, VERTICAL);
#else
            if (upperPatch.checkRgbPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, INVERT_RGB_LWIR) &&
                upperPatch.checkLwirPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getLwirImg().cols, imgReader.getLwirImg().rows, INVERT_RGB_LWIR)) {
                upperPatch.savePoints(validPointsAll);
            }
#endif
            cv::Point2i lowerPointRGB(itMix.second.first[i].x, itMix.second.first[i].y - 1);
            cv::Point2i lowerPointLWIR(itMix.second.second[i].x, itMix.second.second[i].y - 1);
            PatchCreator lowerPatch(lowerPointRGB, lowerPointLWIR, std::stoi(itMix.first));
#if NEW_ARCH
            if (lowerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, VERTICAL))
                lowerPatch.savePoints(validPointsAll, VERTICAL);
#else
            if (lowerPatch.checkRgbPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, INVERT_RGB_LWIR) &&
                lowerPatch.checkLwirPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getLwirImg().cols, imgReader.getLwirImg().rows, INVERT_RGB_LWIR)) {
                lowerPatch.savePoints(validPointsAll);
            }
#endif
        }
    }
    std::cout << std::endl;
    std::cout << "---------- Summary of generated points ----------" << std::endl;
    std::cout << "Number of valid points (ALL-validation): " << validPointsAll.size() << std::endl;
    std::cout << "Total number of points (ALL-validation): " << totalPointsAll << std::endl;
    std::cout << std::endl;

    //////////// verification ////////////
#if VERIFICATION
    std::cout << "Starting visual verification of the points..." << std::endl;
    //vs.showAllPoints(validPointsAll, ALL_RECTIFIED_VAL_LOCATION, PNG_EXTENSION);
    //vs.showAllPoints(validPointsPlsc, plscValRectifiedDataset, PNG_EXTENSION);
    //vs.showAllPoints(validPointsGab, gabValRectifiedDataset, JPEG_EXTENSION);
    //for (auto point : validPointsPlsc)
        //vs.showPatches(halfWidth, halfRange, point, plscValRectifiedDataset, PNG_EXTENSION);
    //for (auto point : validPointsGab)
        //vs.showPatches(halfWidth, halfRange, point, gabValRectifiedDataset, JPEG_EXTENSION);
#endif

    //////////// save to binary file ////////////
    std::cout << "Saving patches location to binary files..." << std::endl;
    std::cout << std::endl << std::endl;
    bwAll.setFilename(ALL_BIN_LOC_VAL, "_" + invert + hw + "_" + hr + filetype + "_" + id  + BIN_EXTENSION);
    bwAll.writePointsToFile(validPointsAll);
    auto valAll = validPointsAll;

    //////////// clear variables used for validation ////////////
    mrAll.clearMapping();
    mixReader.clearImagePoints();
    validPointsAll.clear();

    //////////// testing ////////////
    std::cout << "Starting patch generation for testing data..." << std::endl;
    mrAll.read(ALL_TEST_MAP);

    for (auto itAll : mrAll.getMapping())
        mixReader.readFile(itAll.second);
    mixReader.mergeMaps();

    std::cout << "Generating patches location for mix dataset..." << std::endl;
    totalPointsAll = 0;
    for (auto itMix : mixReader.getImagePoints()) {
        imgReader.readImagesFromPath(ALL_RECTIFIED_TEST_LOCATION, itMix.first, PNG_EXTENSION);
        totalPointsAll += 3 * itMix.second.first.size();
        for (int i = 0; i < itMix.second.first.size(); ++i) {
            PatchCreator centerPatch(itMix.second.first[i], itMix.second.second[i], std::stoi(itMix.first));
#if NEW_ARCH
            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, VERTICAL))
                centerPatch.savePoints(validPointsAll, VERTICAL);
#else
            if (centerPatch.checkRgbPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, INVERT_RGB_LWIR) &&
                centerPatch.checkLwirPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getLwirImg().cols, imgReader.getLwirImg().rows, INVERT_RGB_LWIR)) {
                centerPatch.savePoints(validPointsAll);
            }
#endif
            cv::Point2i upperPointRGB(itMix.second.first[i].x, itMix.second.first[i].y + 1);
            cv::Point2i upperPointLWIR(itMix.second.second[i].x, itMix.second.second[i].y + 1);
            PatchCreator upperPatch(upperPointRGB, upperPointLWIR, std::stoi(itMix.first));
#if NEW_ARCH
            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, VERTICAL))
                centerPatch.savePoints(validPointsAll, VERTICAL);
#else
            if (upperPatch.checkRgbPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, INVERT_RGB_LWIR) &&
                upperPatch.checkLwirPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getLwirImg().cols, imgReader.getLwirImg().rows, INVERT_RGB_LWIR)) {
                upperPatch.savePoints(validPointsAll);
            }
#endif
            cv::Point2i lowerPointRGB(itMix.second.first[i].x, itMix.second.first[i].y - 1);
            cv::Point2i lowerPointLWIR(itMix.second.second[i].x, itMix.second.second[i].y - 1);
            PatchCreator lowerPatch(lowerPointRGB, lowerPointLWIR, std::stoi(itMix.first));
#if NEW_ARCH
            if (centerPatch.checkPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, VERTICAL))
                centerPatch.savePoints(validPointsAll, VERTICAL);
#else
            if (lowerPatch.checkRgbPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows, INVERT_RGB_LWIR) &&
                lowerPatch.checkLwirPatchValidity(halfWidth, halfRange, OFFSET, imgReader.getLwirImg().cols, imgReader.getLwirImg().rows, INVERT_RGB_LWIR)) {
                lowerPatch.savePoints(validPointsAll);
            }
#endif
        }
    }
    std::cout << std::endl;
    std::cout << "---------- Summary of generated points ----------" << std::endl;
    std::cout << "Number of valid points (ALL-test): " << validPointsAll.size() << std::endl;
    std::cout << "Total number of points (ALL-test): " << totalPointsAll << std::endl;
    std::cout << std::endl;

    //////////// verification ////////////
#if VERIFICATION
    std::cout << "Starting visual verification of the points..." << std::endl;
    vs.showAllPoints(validPointsGab, gabTestRectifiedDataset, JPEG_EXTENSION, INVERT_RGB_LWIR);
    //for (auto point : validPointsPlsc)
        //vs.showPatches(halfWidth, halfRange, point, plscTestRectifiedDataset, PNG_EXTENSION);
    //for (auto point : validPointsGab)
        //vs.showPatches(halfWidth, halfRange, point, gabTestRectifiedDataset, JPEG_EXTENSION);
#endif

    //////////// save to binary file ////////////
    std::cout << "Saving patches location to binary files..." << std::endl;
    std::cout << std::endl << std::endl;
    bwAll.setFilename(ALL_BIN_LOC_TEST, "_" + invert + hw + "_" + hr + filetype + "_" + id  + BIN_EXTENSION);
    bwAll.writePointsToFile(validPointsAll);
    auto testAll = validPointsAll;

    //////////// clear variables used for validation ////////////
    mrAll.clearMapping();
    mixReader.clearImagePoints();
    validPointsAll.clear();
}