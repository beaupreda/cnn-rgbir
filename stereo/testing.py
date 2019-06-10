'''
script used to execute multiple runs of testing
simply calls corresponding lua script with chosen arguments

David-Alexandre Beaupre
'''

import os
import subprocess


def prepare_test(gpuid=None,
                 test_nb=None,
                 data_root=None,
                 util_root=None,
                 tb=None,
                 psz=None,
                 half_range=None,
                 fold=None,
                 weights=None,
                 bn=None):
    params = locals()
    call = []
    call.append('th')
    call.append('test.lua')
    for key in params.keys():
        if params[key] is not None:
            call.append('--' + key)
            call.append(params[key])
    return call


# path to dataset
data_root = '/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/'
fold1 = os.path.join(data_root, 'dataset1')
fold2 = os.path.join(data_root, 'dataset2')
fold3 = os.path.join(data_root, 'dataset3')
# path to folder containing .bin files (either disparity_locations in pretrain of patch_generator in stereo)
util_root = '/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/stereo/pretrain/disparity_locations'
gpuid = '0'
tb = '100'
patch_size = '18'
half_range = '60'


test_fold1 = prepare_test(gpuid=gpuid,
                          test_nb='106',
                          data_root=fold1,
                          util_root=util_root,
                          tb=tb,
                          psz=patch_size,
                          half_range=half_range,
                          fold='1',
                          weights='/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/stereo/pretrain/parameters/weights1.t7',
                          bn='/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/stereo/pretrain/parameters/bn1.t7')

test_fold2 = prepare_test(gpuid=gpuid,
                          test_nb='178',
                          data_root=fold2,
                          util_root=util_root,
                          tb=tb,
                          psz=patch_size,
                          half_range=half_range,
                          fold='2',
                          weights='/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/stereo/pretrain/parameters/weights2.t7',
                          bn='/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/stereo/pretrain/parameters/bn2.t7')

test_fold3 = prepare_test(gpuid=gpuid,
                          test_nb='134',
                          data_root=fold3,
                          util_root=util_root,
                          tb=tb,
                          psz=patch_size,
                          half_range=half_range,
                          fold='3',
                          weights='/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/stereo/pretrain/parameters/weights3.t7',
                          bn='/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/stereo/pretrain/parameters/bn3.t7')

subprocess.call(test_fold1)
subprocess.call(test_fold2)
subprocess.call(test_fold3)
