# Siamese CNNs for RGB-LWIR Disparity Estimation
This repository contains all the code to reproduce the experiments in our paper. It is separated into three modules.

* Patch Generator: Generates the disparity locations (center of patches) for training, validation and testing set.
* Rectification: Rectifies images of the St-Charles dataset.
* Stereo: Generates dataset based on the chosen fold and contains scripts to train and test the model.

## Citation
Coming soon!

## Usage
Please refer to README files in each module for more details.

* [Patch Generator](./patch_generator/README.md)
* [Rectification](./rectification/README.md)
* [Stereo](./stereo/README.md)

### Dataset
* [LITIV dataset](https://share.polymtl.ca/alfresco/service/api/path/content;cm:content/workspace/SpacesStore/Company%20Home/Sites/litiv-web/documentLibrary/Datasets/BilodeauetAlInfraredDataset.zip?a=true&guest=true)
* [St-Charles dataset]()