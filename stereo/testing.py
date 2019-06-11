'''
script used to execute multiple runs of testing
simply calls corresponding lua script with chosen arguments

David-Alexandre Beaupre
'''

import os
import subprocess
from utils import InputParser, YamlReader


def prepare_test(gpu_id=None,
                 test_nb=None,
                 data_root=None,
                 testing=None,
                 tb=None,
                 psz=None,
                 half_range=None,
                 fold=None,
                 weights=None,
                 bn=None):
    params = locals()
    call = list()
    call.append('th')
    call.append('test.lua')
    for key in params.keys():
        if params[key] is not None:
            call.append('--' + key)
            call.append(params[key])
    return call


def test():
    input_parser = InputParser()
    input_parser.add_arguments('--fold', '1', 'Fold to test data')
    input_parser.add_arguments('--config', '/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/shared/config.yml', 'Path to the configuration file')
    args = input_parser.get_arguments()

    yml = YamlReader(args.config)
    config = yml.parse()

    data_root = config['output_dataset']
    if int(args.fold) == config['fold1']['id']:
        data_root = os.path.join(data_root, config['fold1']['dataset'])
        weights_file = config['fold1']['weights']
        bn_file = config['fold1']['bn']
        test_file = config['fold1']['test']
        test_nb = str(config['fold1']['test_nb'])
    elif int(args.fold) == config['fold2']['id']:
        data_root = os.path.join(data_root, config['fold2']['dataset'])
        weights_file = config['fold2']['weights']
        bn_file = config['fold2']['bn']
        test_file = config['fold2']['test']
        test_nb = str(config['fold2']['test_nb'])
    elif int(args.fold) == config['fold3']['id']:
        data_root = os.path.join(data_root, config['fold3']['dataset'])
        weights_file = config['fold3']['weights']
        bn_file = config['fold3']['bn']
        test_file = config['fold3']['test']
        test_nb = str(config['fold3']['test_nb'])
    else:
        data_root = os.path.join(data_root, config['custom']['dataset'])
        weights_file = config['custom']['weights']
        bn_file = config['custom']['bn']
        test_file = config['custom']['test']
        test_nb = str(config['custom']['test_nb'])

    disp_root = config['disp_root']
    param_root = None
    if int(args.fold) == 1 or int(args.fold) == 2 or int(args.fold) == 3:
        disp_root = config['pretrain_disp_root']
        param_root = config['pretrain_param_root']

    gpu_id = str(config['gpu_id'])
    tb = str(config['tb'])
    patch_size = str(config['half_width'])
    half_range = str(config['half_range'])
    testing = os.path.join(disp_root, test_file)
    weights = os.path.join(param_root, weights_file)
    bn = os.path.join(param_root, bn_file)

    run = prepare_test(gpu_id=gpu_id,
                       test_nb=test_nb,
                       data_root=data_root,
                       testing=testing,
                       tb=tb,
                       psz=patch_size,
                       half_range=half_range,
                       fold=args.fold,
                       weights=weights,
                       bn=bn)

    subprocess.call(run)


if __name__ == '__main__':
    test()
