# Siamese CNNs for RGB-LWIR Disparity Estimation
This repository contains all the code to reproduce the experiments in our paper [Siamese CNNs for RGB-LWIR Disparity Estimation](http://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Beaupre_Siamese_CNNs_for_RGB-LWIR_Disparity_Estimation_CVPRW_2019_paper.pdf). It is separated into three modules.

* Patch Generator: Generates the disparity locations (center of patches) for training, validation and testing set.
* Rectification: Rectifies images of the St-Charles dataset.
* Stereo: Generates dataset based on the chosen fold and contains scripts to train and test the model.
* Shared: Contains configuration file which all modules use.

## Citation
```
@InProceedings{Beaupre_2019_CVPR_Workshops,
author = {Beaupre, David-Alexandre and Bilodeau, Guillaume-Alexandre},
title = {Siamese CNNs for RGB-LWIR Disparity Estimation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```

## Usage
Please refer to README files in each module for more details. Make sure to change paths and other variables in the [config file](./shared/config.yml) to your own values.

* [Patch Generator](./patch_generator/README.md)
* [Rectification](./rectification/README.md)
* [Stereo](./stereo/README.md)

### Datasets
Simply put both datasets in a folder named "litiv".

* [LITIV dataset](https://share.polymtl.ca/alfresco/service/api/path/content;cm:content/workspace/SpacesStore/Company%20Home/Sites/litiv-web/documentLibrary/Datasets/BilodeauetAlInfraredDataset.zip?a=true&guest=true)
* [St-Charles dataset]()
