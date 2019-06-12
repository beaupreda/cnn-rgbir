# Stereo
Trains and test the proposed model. Generates the split dataset (train-validation-test). The pretrain folder contains the weights and batch normalization statistics for every fold in our paper ([parameters](./pretrain/parameters) folder). The patch locations for the same folds are in the [disparity_locations](./pretrain/disparity_locations) folder.

## Usage
### Dependencies
* [Torch](http://torch.ch/)
* [PyYAML](https://pypi.org/project/PyYAML/)
* [Pillow](https://pypi.org/project/Pillow/)

### Dataset
Generate the dataset according to the folds in the paper (1, 2 or 3). For your own data, you can put any number there.
```
python3 dataset.py --fold [FOLD_NB] --config [PATH TO CONFIG FILE]
```

### Training
Set up all parameters in the [config](../shared/config.yml) file and simply call the python training script.
```
python3 training.py
```
### Testing
Set up all parameters in the [config](../shared/config.yml) file and simply call the python testing script.
```
python3 testing.py
```