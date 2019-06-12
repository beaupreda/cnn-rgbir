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

#define CONFIG_LOCATION "/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/shared/config.yml"
#define FOLD 1
#define OFFSET 0
#define PNG_EXTENSION ".png"
#define BIN_EXTENSION ".bin"
#define VERIFICATION false
#define BIN_LOC_TRAIN "../train"
#define BIN_LOC_VALIDATION "../validation"
#define BIN_LOC_TEST "../test"

int main(int argc, char* argv[]) {
    //////////// arguments set-up ////////////
    YamlReader yml;
    YAML::Node config = yml.parse(CONFIG_LOCATION);
    const std::string PLSC_LOCATION = config["sc_root"].as<std::string>();
    const std::string GAB_LOCATION = config["litiv_root"].as<std::string>();
    const std::string DATA_ROOT = config["data_root"].as<std::string>();

    std::string rectified_train_location;
    std::string rectified_validation_location;
    std::string rectified_test_location;
    std::string train_map;
    std::string validation_map;
    std::string test_map;
    if (FOLD == config["fold1"]["id"].as<int>()) {
        rectified_train_location = DATA_ROOT + config["fold1"]["dataset"].as<std::string>() + "/train/";
        rectified_validation_location = DATA_ROOT + config["fold1"]["dataset"].as<std::string>() + "/validation/";
        rectified_test_location = DATA_ROOT + config["fold1"]["dataset"].as<std::string>() + "/test/";
        train_map = DATA_ROOT + config["fold1"]["dataset"].as<std::string>() + "/train/map.txt";
        validation_map = DATA_ROOT + config["fold1"]["dataset"].as<std::string>() + "/validation/map.txt";
        test_map = DATA_ROOT + config["fold1"]["dataset"].as<std::string>() + "/test/map.txt";
    } else if (FOLD == config["fold2"]["id"].as<int>()) {
        rectified_train_location = DATA_ROOT + config["fold2"]["dataset"].as<std::string>() + "/train/";
        rectified_validation_location = DATA_ROOT + config["fold2"]["dataset"].as<std::string>() + "/validation/";
        rectified_test_location = DATA_ROOT + config["fold2"]["dataset"].as<std::string>() + "/test/";
        train_map = DATA_ROOT + config["fold2"]["dataset"].as<std::string>() + "/train/map.txt";
        validation_map = DATA_ROOT + config["fold2"]["dataset"].as<std::string>() + "/validation/map.txt";
        test_map = DATA_ROOT + config["fold2"]["dataset"].as<std::string>() + "/test/map.txt";
    } else if (FOLD == config["fold3"]["id"].as<int>()) {
        rectified_train_location = DATA_ROOT + config["fold3"]["dataset"].as<std::string>() + "/train/";
        rectified_validation_location = DATA_ROOT + config["fold3"]["dataset"].as<std::string>() + "/validation/";
        rectified_test_location = DATA_ROOT + config["fold3"]["dataset"].as<std::string>() + "/test/";
        train_map = DATA_ROOT + config["fold3"]["dataset"].as<std::string>() + "/train/map.txt";
        validation_map = DATA_ROOT + config["fold3"]["dataset"].as<std::string>() + "/validation/map.txt";
        test_map = DATA_ROOT + config["fold3"]["dataset"].as<std::string>() + "/test/map.txt";
    } else {
        rectified_train_location = DATA_ROOT + config["custom"]["dataset"].as<std::string>() + "/train/";
        rectified_validation_location = DATA_ROOT + config["custom"]["dataset"].as<std::string>() + "/validation/";
        rectified_test_location = DATA_ROOT + config["custom"]["dataset"].as<std::string>() + "/test/";
        train_map = DATA_ROOT + config["custom"]["dataset"].as<std::string>() + "/train/map.txt";
        validation_map = DATA_ROOT + config["custom"]["dataset"].as<std::string>() + "/validation/map.txt";
        test_map = DATA_ROOT + config["custom"]["dataset"].as<std::string>() + "/test/map.txt";
    }

    const int HALF_WIDTH = config["half_width"].as<int>();
    const int HALF_RANGE = config["half_range"].as<int>();
    
    FileNavigator fn(GAB_LOCATION, PLSC_LOCATION);
    fn.loadTextFiles();
    fn.loadYamlFilesLocations();
    for (const auto& dir : fn.getYamlFilesLocations())
        fn.loadYamlFiles(dir);

    //////////// training ////////////
    std::cout << "Starting patch generation for training data..." << std::endl;
    MapReader mr;
    mr.read(train_map);

    MixReader mixReader;
    for (auto it : mr.getMapping())
        mixReader.readFile(it.second);
    mixReader.mergeMaps();

    std::cout << "Generating patches location for dataset..." << std::endl;
    unsigned int totalPoints = 0;
    ImageReader imgReader;
    std::vector<std::vector<float>> validPoints;
    for (auto it : mixReader.getImagePoints()) {
        imgReader.readImagesFromPath(rectified_train_location, it.first, PNG_EXTENSION);
        totalPoints += 3 * it.second.first.size();
        for (int i = 0; i < it.second.first.size(); ++i) {
            PatchCreator centerPatch(it.second.first[i], it.second.second[i], std::stoi(it.first));
            if (centerPatch.checkPatchValidity(HALF_WIDTH, HALF_RANGE, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                centerPatch.savePoints(validPoints);

            cv::Point2i upperPointRGB(it.second.first[i].x, it.second.first[i].y + 1);
            cv::Point2i upperPointLWIR(it.second.second[i].x, it.second.second[i].y + 1);
            PatchCreator upperPatch(upperPointRGB, upperPointLWIR, std::stoi(it.first));
            if (upperPatch.checkPatchValidity(HALF_WIDTH, HALF_RANGE, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                upperPatch.savePoints(validPoints);

            cv::Point2i lowerPointRGB(it.second.first[i].x, it.second.first[i].y - 1);
            cv::Point2i lowerPointLWIR(it.second.second[i].x, it.second.second[i].y - 1);
            PatchCreator lowerPatch(lowerPointRGB, lowerPointLWIR, std::stoi(it.first));
            if (lowerPatch.checkPatchValidity(HALF_WIDTH, HALF_RANGE, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
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
    vs.showAllPoints(validPoints, rectified_train_location, PNG_EXTENSION);
    for (auto point : validPoints)
        vs.showPatches(HALF_WIDTH, HALF_RANGE, point, rectified_train_location, PNG_EXTENSION);
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
    mr.read(validation_map);

    for (auto itAll : mr.getMapping())
        mixReader.readFile(itAll.second);
    mixReader.mergeMaps();

    std::cout << "Generating patches location for dataset..." << std::endl;
    totalPoints = 0;
    for (auto it : mixReader.getImagePoints()) {
        imgReader.readImagesFromPath(rectified_validation_location, it.first, PNG_EXTENSION);
        totalPoints += 3 * it.second.first.size();
        for (int i = 0; i < it.second.first.size(); ++i) {
            PatchCreator centerPatch(it.second.first[i], it.second.second[i], std::stoi(it.first));

            if (centerPatch.checkPatchValidity(HALF_WIDTH, HALF_RANGE, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                centerPatch.savePoints(validPoints);

            cv::Point2i upperPointRGB(it.second.first[i].x, it.second.first[i].y + 1);
            cv::Point2i upperPointLWIR(it.second.second[i].x, it.second.second[i].y + 1);
            PatchCreator upperPatch(upperPointRGB, upperPointLWIR, std::stoi(it.first));
            if (upperPatch.checkPatchValidity(HALF_WIDTH, HALF_RANGE, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                upperPatch.savePoints(validPoints);

            cv::Point2i lowerPointRGB(it.second.first[i].x, it.second.first[i].y - 1);
            cv::Point2i lowerPointLWIR(it.second.second[i].x, it.second.second[i].y - 1);
            PatchCreator lowerPatch(lowerPointRGB, lowerPointLWIR, std::stoi(it.first));
            if (lowerPatch.checkPatchValidity(HALF_WIDTH, HALF_RANGE, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
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
    vs.showAllPoints(validPoints, rectified_validation_location, PNG_EXTENSION);
    for (auto point : validPoints)
        vs.showPatches(HALF_WIDTH, HALF_RANGE, point, rectified_validation_location, PNG_EXTENSION);
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
    mr.read(test_map);

    for (auto itAll : mr.getMapping())
        mixReader.readFile(itAll.second);
    mixReader.mergeMaps();

    std::cout << "Generating patches location for dataset..." << std::endl;
    totalPoints = 0;
    for (auto it : mixReader.getImagePoints()) {
        imgReader.readImagesFromPath(rectified_test_location, it.first, PNG_EXTENSION);
        totalPoints += 3 * it.second.first.size();
        for (int i = 0; i < it.second.first.size(); ++i) {
            PatchCreator centerPatch(it.second.first[i], it.second.second[i], std::stoi(it.first));

            if (centerPatch.checkPatchValidity(HALF_WIDTH, HALF_RANGE, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                centerPatch.savePoints(validPoints);

            cv::Point2i upperPointRGB(it.second.first[i].x, it.second.first[i].y + 1);
            cv::Point2i upperPointLWIR(it.second.second[i].x, it.second.second[i].y + 1);
            PatchCreator upperPatch(upperPointRGB, upperPointLWIR, std::stoi(it.first));
            if (centerPatch.checkPatchValidity(HALF_WIDTH, HALF_RANGE, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
                centerPatch.savePoints(validPoints);

            cv::Point2i lowerPointRGB(it.second.first[i].x, it.second.first[i].y - 1);
            cv::Point2i lowerPointLWIR(it.second.second[i].x, it.second.second[i].y - 1);
            PatchCreator lowerPatch(lowerPointRGB, lowerPointLWIR, std::stoi(it.first));
            if (centerPatch.checkPatchValidity(HALF_WIDTH, HALF_RANGE, OFFSET, imgReader.getRgbImg().cols, imgReader.getRgbImg().rows))
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
    vs.showAllPoints(validPoints, rectified_test_location, PNG_EXTENSION);
    for (auto point : validPoints)
        vs.showPatches(HALF_WIDTH, HALF_RANGE, point, rectified_test_location, PNG_EXTENSION);
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