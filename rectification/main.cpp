#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <boost/filesystem.hpp>

#define CAM0 "aCamMats0"
#define CAM1 "aCamMats1"
#define DIST_COEFFS0 "aDistCoeffs0"
#define DIST_COEFFS1 "aDistCoeffs1"
#define ROTATION "oRotMat"
#define TRANSLATION "oTranslMat"
#define TARGET_SIZE "oTargetSize"
#define RECTIF_ALPHA "dRectifAlpha"
#define MIN_DISP "mindisp"
#define MAX_DISP "maxdisp"
#define YAML_EXTENSION ".yml"
#define JPEG_EXTENSION ".jpg"
#define PNG_EXTENSION ".png"
#define X "x"
#define Y "y"
#define D "d"
#define DRAW true
#define SHOW_IMAGES true
#define SAVE_IMAGES false

const std::string VIDEO04 = "/store/dabeaq/datasets/stcharles2018-v04/vid04/";
const std::string VIDEO07 = "/store/dabeaq/datasets/stcharles2018-v04/vid07/";
const std::string VIDEO08 = "/store/dabeaq/datasets/stcharles2018-v04/vid08/";
const std::string SAVE_VID04 = "/store/dabeaq/datasets/stcharles2018-v04/rectified_images/vid04/";
const std::string SAVE_VID07 = "/store/dabeaq/datasets/stcharles2018-v04/rectified_images/vid07/";
const std::string SAVE_VID08 = "/store/dabeaq/datasets/stcharles2018-v04/rectified_images/vid08/";
const std::string LWIR_FOLDER = "lwir/";
const std::string RGB_FOLDER = "rgb/";
const std::string LWIR_GT_FOLDER = "lwir_gt_disp/";
const std::string RGB_GT_FOLDER = "rgb_gt_disp/";
const std::string CALIBRATION_FILE = "calibdata.yml";
const std::string DISPARITY_GT_METADATA = "gt_disp_metadata.yml";

enum CAMERAS {RGB = 0, LWIR = 1};

void shift(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Point2f& vDelta, int nFillType=cv::BORDER_CONSTANT,
           const cv::Scalar& vConstantFillValue=cv::Scalar(0,0,0,0)) {
    /*
     *  ORIGINAL SOURCE : http://code.opencv.org/issues/2299
     *
     *  Software License Agreement (BSD License)
     *
     *  Copyright (c) 2012, Willow Garage, Inc.
     *  All rights reserved.
     *
     *  Redistribution and use in source and binary forms, with or without
     *  modification, are permitted provided that the fJPEG_EXTENSIONollowing conditions
     *  are met:
     *
     *   * Redistributions of source code must retain the above copyright
     *     notice, this list of conditions and the following disclaimer.
     *   * Redistributions in binary form must reproduce the above
     *     copyright notice, this list of conditions and the following
     *     disclaimer in the documentation and/or other materials provided
     *     with the distribution.
     *   * Neither the name of Willow Garage, Inc. nor the names of its
     *     contributors may be used to endorse or promote products derived
     *     from this software without specific prior written permission.
     *
     *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
     *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
     *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
     *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
     *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
     *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
     *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
     *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
     *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IJPEG_EXTENSIONN CONTRACT, STRICT
     *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
     *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
     *  POSSIBILITY OF SUCH DAMAGE.
     *
     *  Original Author:  Hilton Bristow
     *  Created: Aug 23, 2012
     *  Modified: Jan 21, 2015
     */
    const cv::Point2i vDeltaInt((int)std::ceil(vDelta.x),(int)std::ceil(vDelta.y));
    const cv::Point2f vDeltaSubPx(std::abs(vDelta.x-vDeltaInt.x),std::abs(vDelta.y-vDeltaInt.y));
    const int nTop = (vDeltaInt.y>0)?vDeltaInt.y:0;     // NOLINT
    const int nBottom = (vDeltaInt.y<0)?-vDeltaInt.y:0; // NOLINT
    const int nLeft = (vDeltaInt.x>0)?vDeltaInt.x:0;    // NOLINT
    const int nRight = (vDeltaInt.x<0)?-vDeltaInt.x:0;  // NOLINT
    cv::Mat oPaddedInput;
    cv::copyMakeBorder(oInput,oPaddedInput,nTop,nBottom,nLeft,nRight,nFillType,vConstantFillValue);
    if(vDeltaSubPx.x>std::numeric_limits<float>::epsilon() || vDeltaSubPx.y>std::numeric_limits<float>::epsilon()) {
        switch(oInput.depth()) {
            case CV_32F: {
                cv::Matx<float,1,2> dx(1-vDeltaSubPx.x,vDeltaSubPx.x); // NOLINT
                cv::Matx<float,2,1> dy(1-vDeltaSubPx.y,vDeltaSubPx.y);
                cv::sepFilter2D(oPaddedInput,oPaddedInput,-1,dx,dy,cv::Point(0,0),0,cv::BORDER_CONSTANT);
                break;
            }
            case CV_64F: {
                cv::Matx<double,1,2> dx(1-vDeltaSubPx.x,vDeltaSubPx.x); // NOLINT
                cv::Matx<double,2,1> dy(1-vDeltaSubPx.y,vDeltaSubPx.y);
                cv::sepFilter2D(oPaddedInput,oPaddedInput,-1,dx,dy,cv::Point(0,0),0,cv::BORDER_CONSTANT);
                break;
            }
            default: {
                cv::Matx<float,1,2> dx(1-vDeltaSubPx.x,vDeltaSubPx.x);
                cv::Matx<float,2,1> dy(1-vDeltaSubPx.y,vDeltaSubPx.y);
                oPaddedInput.convertTo(oPaddedInput,CV_32F);
                cv::sepFilter2D(oPaddedInput,oPaddedInput,CV_32F,dx,dy,cv::Point(0,0),0,cv::BORDER_CONSTANT);
                break;
            }
        }
    }
    const cv::Rect oROI = cv::Rect(std::max(-vDeltaInt.x,0), std::max(-vDeltaInt.y,0),0,0)+oInput.size();
    oOutput = oPaddedInput(oROI);
}

std::string getFilename(const std::string& fullPath, const std::string& VIDEO) {
    size_t found = fullPath.find(VIDEO + LWIR_GT_FOLDER);
    int position = 0;
    if (found != std::string::npos)
        position = static_cast<int>((VIDEO + LWIR_GT_FOLDER).length());
    else {
        found = fullPath.find(VIDEO04 + RGB_FOLDER);
        position = static_cast<int>((VIDEO + RGB_GT_FOLDER).length());
    }
    std::string name;
    for (int i = position; i < fullPath.length(); ++i) {
        name += fullPath[i];
    }
    return name;
}

void loadDirectoryContent(const std::string& directory, std::vector<std::shared_ptr<std::string>>& filenames, const std::string& VIDEO) {
    boost::filesystem::directory_iterator iterDir(directory);
    boost::filesystem::directory_iterator iterDirEnd;
    while (iterDir != iterDirEnd) {
        if (boost::filesystem::is_regular_file(iterDir->path())) {
            auto filename = std::shared_ptr<std::string>(new std::string(getFilename(iterDir->path().string(), VIDEO)));
            filenames.emplace_back(filename);
        }
        ++iterDir;
    }
    std::sort(filenames.begin(), filenames.end(),
              [](std::shared_ptr<std::string> str1, std::shared_ptr<std::string> str2) {
                  return *str1 < *str2;
              });
}

std::string split(const std::string& filename) {
    size_t position = filename.find(YAML_EXTENSION);
    std::string name;
    for (size_t i = 0; i < position; ++i) {
        name += filename[i];
    }
    return name;
}

int main() {
    const std::string VIDEO = VIDEO04;
    const std::string SAVE_VIDEO = SAVE_VID04;

    cv::FileStorage calibration(VIDEO + CALIBRATION_FILE, cv::FileStorage::READ);
    std::array<cv::Mat, 2> camMats, distCoeffs;
    calibration[CAM0] >> camMats[RGB];
    calibration[CAM1] >> camMats[LWIR];
    calibration[DIST_COEFFS0] >> distCoeffs[RGB];
    calibration[DIST_COEFFS1] >> distCoeffs[LWIR];

    cv::Mat rotMat, translMat;
    calibration[ROTATION] >> rotMat;
    calibration[TRANSLATION] >> translMat;

    cv::Size targetSize;
    calibration[TARGET_SIZE] >> targetSize;

    double alphaRectif;
    calibration[RECTIF_ALPHA] >> alphaRectif;

    cv::FileStorage disparityGT(VIDEO + DISPARITY_GT_METADATA, cv::FileStorage::READ);
    int minDisp, maxDisp;
    disparityGT[MIN_DISP] >> minDisp;
    disparityGT[MAX_DISP] >> maxDisp;
    auto shiftValue = static_cast<float>(std::abs(std::abs(minDisp) - std::abs(maxDisp)));

    std::vector<std::shared_ptr<std::string>> gtFilenamesLWIR, gtFilenamesRGB;
    loadDirectoryContent(VIDEO + LWIR_GT_FOLDER, gtFilenamesLWIR, VIDEO);

    std::map<std::string, std::vector<cv::Point3i>> imagePoints;
    for (auto filename : gtFilenamesLWIR) {
        cv::FileStorage gtLWIR(VIDEO + LWIR_GT_FOLDER + *filename, cv::FileStorage::READ);
        cv::FileNode root = gtLWIR.root();
        for (cv::FileNodeIterator it = root.begin(); it != root.end(); ++it) {
            if ((*it).name().size() > 5) {
                int x, y, d;
                (*it)[X] >> x;
                (*it)[Y] >> y;
                (*it)[D] >> d;
                imagePoints[*filename].emplace_back(cv::Point3i(x, y, d));
            }
        }
        std::string nameNoExtension = split(*filename);
        std::string imageName = nameNoExtension + JPEG_EXTENSION;
        cv::Mat imageRGB = cv::imread(VIDEO + RGB_FOLDER + imageName);
        cv::Mat imageLWIR = cv::imread(VIDEO + LWIR_FOLDER + imageName);

        cv::resize(imageRGB, imageRGB, targetSize);
        cv::flip(imageRGB, imageRGB, 1);
        cv::resize(imageLWIR, imageLWIR, targetSize);

        shift(imageRGB.clone(), imageRGB, cv::Point2f(shiftValue, 0.0));
        shift(imageLWIR.clone(), imageLWIR, cv::Point2f(shiftValue, 0.0));

        cv::Mat dispToDepthMap;
        std::array<cv::Mat, 2> rectifRotMat, rectifProjMat;
        cv::stereoRectify(camMats[RGB], distCoeffs[RGB], camMats[LWIR], distCoeffs[LWIR], targetSize, rotMat, translMat,
                          rectifRotMat[RGB], rectifRotMat[LWIR], rectifProjMat[RGB], rectifProjMat[LWIR],
                          dispToDepthMap, 0, alphaRectif, targetSize);

        std::array<std::array<cv::Mat, 2>, 2> rectifMaps;
        cv::initUndistortRectifyMap(camMats[RGB], distCoeffs[RGB], rectifRotMat[RGB], rectifProjMat[RGB], targetSize,
                                    CV_16SC2, rectifMaps[RGB][0], rectifMaps[RGB][1]);
        cv::initUndistortRectifyMap(camMats[LWIR], distCoeffs[LWIR], rectifRotMat[LWIR], rectifProjMat[LWIR], targetSize,
                                    CV_16SC2, rectifMaps[LWIR][0], rectifMaps[LWIR][1]);
        std::array<cv::Mat, 2> rectifiedInputs;
        cv::remap(imageRGB, rectifiedInputs[RGB], rectifMaps[RGB][0], rectifMaps[RGB][1], cv::INTER_LINEAR);
        cv::remap(imageLWIR, rectifiedInputs[LWIR], rectifMaps[LWIR][0], rectifMaps[LWIR][1], cv::INTER_LINEAR);
#if DRAW
        for (auto point : imagePoints[*filename]) {
            cv::circle(rectifiedInputs[RGB], cv::Point2i(point.x + point.z, point.y), 2, cv::Scalar(0, 0, 255)); // point.z = d
            cv::circle(rectifiedInputs[LWIR], cv::Point2i(point.x, point.y), 2, cv::Scalar(0, 0, 255));
            cv::line(rectifiedInputs[RGB], cv::Point2i(0, point.y), cv::Point2i(targetSize.width, point.y), cv::Scalar(255, 0, 0));
            cv::line(rectifiedInputs[LWIR], cv::Point2i(0, point.y), cv::Point2i(targetSize.width, point.y), cv::Scalar(255, 0, 0));
        }
#endif
#if SHOW_IMAGES
        cv::Mat imageMerged;
        cv::hconcat(rectifiedInputs[RGB], rectifiedInputs[LWIR], imageMerged);
        cv::namedWindow("IMG");
        cv::imshow("IMG", imageMerged);
        cv::waitKey(0);
#endif
#if SAVE_IMAGES
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        cv::imwrite(SAVE_VIDEO + RGB_FOLDER + nameNoExtension + PNG_EXTENSION, rectifiedInputs[RGB], compression_params);
        cv::imwrite(SAVE_VIDEO + LWIR_FOLDER + nameNoExtension + PNG_EXTENSION, rectifiedInputs[LWIR], compression_params);
#endif
    }
}