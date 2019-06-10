// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2018 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////////////
//
// This sample demonstrates how to parse and visualize the litiv-stcharles2018
// dataset via the interface provided in the datasets module. Two modes of
// operation are supported in this sample; the first displays the unrectified
// data directly with its foreground segments as overlay, and the second
// displays the rectified data with the registered point pairs used for
// evaluation. If the dataset cannot be located at its default location on the
// system (determined via the CMake variable 'EXTERNAL_DATA_ROOT'), then the
// user will be asked to locate it.
//
///////////////////////////////////////////////////////////////////////////////

#define BATCH_START_IDX 508 // will be used below to skip empty frames at the beginning of each sequence

#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

#include "litiv/datasets.hpp"

// the litiv2018 dataset interface is accessible through the specialized type below
using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::Dataset_LITIV_stcharles2018,lv::NonParallel>;

std::string getName(int number) {
    std::string name = "";
    std::string strNumber = std::to_string(number);
    for (int i = 0; i < (5 - strNumber.size()); ++i)
        name += "0";
    name += strNumber;
    return name;
}

void create_folders(const std::string& saveLocation) {
    const std::vector<std::string> VIDEOS = {"vid04/", "vid07/", "vid08/"};
    const std::vector<std::string> SPECTRUMS = {"rgb/", "lwir/"};

    mkdir(saveLocation.c_str(), 0777);
    for (auto video : VIDEOS) {
        const std::string vidLocation = saveLocation + video;
        mkdir(vidLocation.c_str(), 0777);
        for (auto spectrum : SPECTRUMS) {
            const std::string specLocation = vidLocation + spectrum;
            mkdir(specLocation.c_str(), 0777);
        }
    }
}

int main(int, char**) {
    try {
        const std::string SAVE_VIDEO = "/store/dabeaq/datasets/litiv/stcharles2018-v04/rectified_images_v2/";
        const std::string PNG_EXTENSION = ".png";
        create_folders(SAVE_VIDEO);
        bool bDisplayRectifData = true;
        bool bDisplayDisparityGT = true;
        // first, we ask the user which kind of evaluation mask to visualize (segmentation maps, or disparity corresp maps)
        const std::string sEvalMode = lv::query_user_input("Please select evaluation mode;\n [0 = segmentation]\n [1 = disparity]",{"0","1"},false);
        if(sEvalMode=="0")
            bDisplayDisparityGT = false;
        // next, if visualizing segmentation masks, we will ask the user whether to rectify the images or not
        if(!bDisplayDisparityGT) {
            const std::string sViewMode = lv::query_user_input("Please select view mode;\n [0 = unrectified]\n [1 = rectified]",{"0","1"},false);
            if(sViewMode=="0")
                bDisplayRectifData = false;
        }
        DatasetType::Ptr pDataset;
        // we will now try to load the dataset using the default cmake-provided location for the external data root; unless it was already
        // specified by the user, the importation will fail, and the correct directory will have to be provided at run time.
        // note: the directory passed in should not be the litiv2018 dataset directory itself, but its parent directory.
        do {
            try {
                pDataset = DatasetType::create(
                        "default", // const std::string& sOutputDirName; irrelevant here (not saving anything)
                        false, //bool bSaveOutput; irrelevant here (not saving anything)
                        true, //bool bUseEvaluator; true, as we will be visualizing gt data
                        false, //bool bLoadDepth; irrelevant here (will only visualize visible/infrared pair)
                        bDisplayRectifData, //bool bUndistort; user-defined for visualization
                        bDisplayRectifData, //bool bHorizRectify; user-defined for visualization
                        bDisplayDisparityGT //bool bEvalDisparities; user-defined for visualization
                );
            }
            catch(const lv::Exception& e) {
                // if the parser failed to find the dataset, it will throw an exception containing the following string token
                if(std::string(e.what()).find("top directory")!=std::string::npos) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // used to flush cerr messages before printing
                    const std::string sDirPath = lv::query_user_input("\n\nCould not find litiv2018 dataset directory.\n\nPlease enter the absolute root path where your datasets are located:");
                    if(sDirPath.empty() || sDirPath==" ")
                        lv::datasets::setRootPath(lv::getCurrentWorkDirPath());
                    else {
                        lv::datasets::setRootPath(sDirPath);
                        lvLog_(1,"Set internal data root path to '%s'.\n",lv::datasets::getRootPath().c_str());
                    }
                }
                else
                    throw e;
            }
        } while(!pDataset); // keep trying to load dataset & ask for data root path until it is found
        lv::IDataHandlerPtrArray vpBatches = pDataset->getBatches(false); // returns a list of batches (sequences, in this case) to process
        const size_t nTotBatches = vpBatches.size(); // number of batches (sequences) to process
        lvAssert(nTotBatches>0u); // litiv2018 dataset should not be empty...
        int cnt = 0;
        for(lv::IDataHandlerPtr pBatch : vpBatches) { // process one sequence at a time
            DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch); // cast batch to proper interface for i/o
            lvAssert(oBatch.getInputPacketType()==lv::ImageArrayPacket && oBatch.getOutputPacketType()==lv::ImageArrayPacket); // this means packets should contain sync'd frames
            lvAssert(oBatch.getInputStreamCount()>=2 && oBatch.getInputCount()>=1); // this sample expects a stereo head setup (2 frames per packet)
            lvAssert(oBatch.getIOMappingType()==lv::IndexMapping && oBatch.getGTMappingType()==lv::ElemMapping); // segmentation/disp = 1:1 mapping between inputs and groundtruth
            const size_t nTotPacketCount = oBatch.getInputCount(); // total number of packets (or frame pairs) for this sequence
            lvLog_(1,"\nvisualizing batch '%s'... (%d packets)\n",oBatch.getName().c_str(),nTotPacketCount);
            size_t nCurrIdx = std::min((size_t)BATCH_START_IDX,nTotPacketCount-1); // allow starting off visualization at large index (skips empty frames)
            std::vector<cv::Mat> vInitInput = oBatch.getInputArray(nCurrIdx); // note: matrix content becomes invalid on next getInput call
            lv::DisplayHelperPtr pDisplayHelper = lv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"/../");
            pDisplayHelper->setContinuousUpdates(false); // display will block and wait for a key; space makes it refresh automatically
            pDisplayHelper->setDisplayCursor(false); // not displaying cursor info on displayed images
            const cv::Size oDisplayTileSize(800,600); // displayed tile size
            std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs = {{
                                                                                               std::make_pair(vInitInput[0].clone(),oBatch.getInputStreamName(0)),
                                                                                               std::make_pair(vInitInput[1].clone(),oBatch.getInputStreamName(1))
                                                                                       }};
            lvLog(1,"Parsing metadata...");
            // all sequences in the litiv dataset have some associated metadata; let's load some of it here
            const std::string sGTName = bDisplayDisparityGT?"gt_disp":"gt_masks";
            const std::string sRGBGTDir = oBatch.getDataPath()+"rgb_"+sGTName+"/";
            const std::string sLWIRGTDir = oBatch.getDataPath()+"lwir_"+sGTName+"/";
            std::cout << sGTName << std::endl;
            {
                cv::FileStorage oGTMetadataFS(oBatch.getDataPath()+sGTName+"_metadata.yml",cv::FileStorage::READ);
                if(oGTMetadataFS.isOpened()) {
                    int nPrevPacketCount;
                    oGTMetadataFS["npackets"] >> nPrevPacketCount;
                    lvAssert(nPrevPacketCount==(int)nTotPacketCount);
                    if(bDisplayDisparityGT) {
                        int nMinDisp,nMaxDisp;
                        oGTMetadataFS["mindisp"] >> nMinDisp;
                        oGTMetadataFS["maxdisp"] >> nMaxDisp;
                        lvLog_(1,"min disparity (real) = %d",nMinDisp);
                        lvLog_(1,"max disparity (real) = %d",nMaxDisp);
                        // the 'real' values above might not have been pre-offset to get rid of negatives
                    }
                }
            }
            int nMinDisp=oBatch.getMinDisparity(),nMaxDisp=oBatch.getMaxDisparity(); // this range is the possibly offset value range
            // below is the main loop where we visualize all frame pairs (starting at some offset) in the sequence
            while(nCurrIdx<nTotPacketCount) {
                // this call returns the frame array for the current packet index (with all required preprocessing already completed)
                const std::vector<cv::Mat>& vCurrInputs = oBatch.getInputArray(nCurrIdx);
                // note: the first frame is the RGB frame, and the second one the LWIR frame
                std::array<cv::Mat,2> aInputs;
                for(size_t a=0; a<2u; ++a) {
                    aInputs[a] = vCurrInputs[a].clone(); // we make copies here, as we will draw over the images for our visualizations
                    if(aInputs[a].channels()==1)
                        cv::cvtColor(aInputs[a],aInputs[a],cv::COLOR_GRAY2BGR); // convert everything to RGB so drawing is easier/prettier
                }
                // we now query for the groundtruth data associated with the current packet index
                const std::vector<cv::Mat>& vCurrGT = oBatch.getGTArray(nCurrIdx);
                lvAssert_(vCurrGT.size()>=2,"array size mistmatch");
                std::cout << vCurrGT[0].size();// the array size will always be fixed, but the matrices may be empty
                for(size_t a=0; a<2u; ++a) {
                    if(bDisplayDisparityGT) {
                        // if displaying disparity groundtruth, we will draw corresponding points on both frames with a depth-scaled color
                        cv::Mat oDispMask = vCurrGT[a].clone();
                        if(!oDispMask.empty() && cv::countNonZero(oDispMask!=lv::ILITIVStCharles2018Dataset::s_nDontCareDispLabel)>0) {
                            std::string nameNoExtension = getName(nCurrIdx);
                            std::vector<int> compression_params;
                            compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                            compression_params.push_back(9);
                            cv::imwrite(SAVE_VIDEO + oBatch.getName() + "/rgb/" + nameNoExtension + PNG_EXTENSION,
                                        vCurrInputs[0], compression_params);
                            cv::imwrite(SAVE_VIDEO + oBatch.getName() + "/lwir/" + nameNoExtension + PNG_EXTENSION,
                                        vCurrInputs[1], compression_params);
                            cv::Mat oColorMap(oDispMask.size(),CV_8UC3,cv::Scalar_<uchar>::all(0)),oLUMap(oDispMask.size(),CV_8UC1,cv::Scalar_<uchar>::all(0));
                            for(int nRowIdx=0; nRowIdx<oDispMask.rows; ++nRowIdx) {
                                for(int nColIdx=0; nColIdx<oDispMask.cols; ++nColIdx) {
                                    const int nCurrDisp = (int)oDispMask.at<uchar>(nRowIdx,nColIdx);
                                    if(nCurrDisp!=lv::ILITIVStCharles2018Dataset::s_nDontCareDispLabel) {
                                        const cv::Vec3b vColor = lv::getBGRFromHSL(((float(std::min(std::max(nCurrDisp,nMinDisp),nMaxDisp))-nMinDisp)/(nMaxDisp-nMinDisp))*240,1.0f,0.5f);
                                        oColorMap.at<cv::Vec3b>(nRowIdx,nColIdx) = vColor;
                                        oLUMap.at<uchar>(nRowIdx,nColIdx) = 255;
                                    }
                                }
                            }
                            cv::dilate(oColorMap,oColorMap,cv::Mat(),cv::Point(-1,-1),2);
                            cv::dilate(oLUMap,oLUMap,cv::Mat(),cv::Point(-1,-1),2);
                            oColorMap.copyTo(aInputs[a],oLUMap!=0);
                        }
                        else
                            lv::putText(aInputs[a],"missing gt disp mask",cv::Scalar_<uchar>(0,0,255),true);
                    }
                    else {
                        // if displaying segmentation groundtruth, we will simply draw the masks (in red at 50% opacity) on top of the original images
                        cv::Mat oSegmMask = vCurrGT[a].clone();
                        if(!oSegmMask.empty() && cv::countNonZero(oSegmMask==DATASETUTILS_POSITIVE_VAL)>0) {
                            cv::cvtColor(oSegmMask==DATASETUTILS_POSITIVE_VAL,oSegmMask,cv::COLOR_GRAY2BGR);
                            oSegmMask &= cv::Scalar_<uchar>(0,0,255);
                            cv::addWeighted(aInputs[a],0.5,oSegmMask,0.5,0.0,aInputs[a]);
                        }
                        else
                            lv::putText(aInputs[a],"missing gt segm mask",cv::Scalar_<uchar>(0,0,255),true);
                    }
                }
                // here, we update the display matrices with the new images
                for(size_t a=0u; a<aInputs.size(); ++a)
                    vvDisplayPairs[0][a].first = aInputs[a];
                lvLog_(1,"\t[packet %d/%d]",nCurrIdx+1,nTotPacketCount);
                // pass the new matrices to the display helper, which will handle i/o and tiling
                pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
                int nKey = pDisplayHelper->waitKey();
                ++nCurrIdx;
                if(nKey=='q')
                    break;
            }
            lvLog(1,"... batch done.\n");
            cnt++;
        }
    }
    catch(const lv::Exception&) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught lv::Exception (check stderr)\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(const cv::Exception&) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception (check stderr)\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
    return 0;
}