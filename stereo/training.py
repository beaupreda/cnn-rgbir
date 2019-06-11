'''
script used to execute multiple runs of training
simply calls corresponding lua script with chosen arguments

David-Alexandre Beaupre
'''

import os
import subprocess
from utils import InputParser, YamlReader


def prepare_train(batchSize=None,
                  gpu_id=None,
                  train_nb=None,
                  validation_nb=None,
                  learningRate=None,
                  learningRateDecay=None,
                  weightDecay=None,
                  momentum=None,
                  model=None,
                  epoch_step=None,
                  weight_epoch=None,
                  max_epoch=None,
                  iter_per_epoch=None,
                  data_root=None,
                  training=None,
                  validation=None,
                  tb=None,
                  validation_points=None,
                  opt_method=None,
                  showCurve=None,
                  psz=None,
                  half_range=None,
                  fold=None):
    params = locals()
    call = list()
    call.append('th')
    call.append('train.lua')
    for key in params.keys():
        if params[key] is not None:
            call.append('--' + key)
            call.append(params[key])
    return call


def train():
    input_parser = InputParser()
    input_parser.add_arguments('--fold', '1', 'Fold to test data')
    input_parser.add_arguments('--config', '/home/travail/dabeaq/litiv/masters/pbvs2019/cnn-rgbir/shared/config.yml', 'Path to the configuration file')
    args = input_parser.get_arguments()

    yml = YamlReader(args.config)
    config = yml.parse()

    data_root = config['output_dataset']
    if int(args.fold) == config['fold1']['id']:
        data_root = os.path.join(data_root, config['fold1']['dataset'])
        train_file = config['fold1']['train']
        validation_file = config['fold1']['validation']
        train_nb = str(config['fold1']['train_nb'])
        validation_nb = str(config['fold1']['validation_nb'])
    elif int(args.fold) == config['fold2']['id']:
        data_root = os.path.join(data_root, config['fold2']['dataset'])
        train_file = config['fold2']['train']
        validation_file = config['fold2']['validation']
        train_nb = str(config['fold2']['train_nb'])
        validation_nb = str(config['fold2']['validation_nb'])
    elif int(args.fold) == config['fold3']['id']:
        data_root = os.path.join(data_root, config['fold3']['dataset'])
        train_file = config['fold3']['train']
        validation_file = config['fold3']['validation']
        train_nb = str(config['fold3']['train_nb'])
        validation_nb = str(config['fold3']['validation_nb'])
    else:
        data_root = os.path.join(data_root, config['custom']['dataset'])
        train_file = config['custom']['train']
        validation_file = config['custom']['validation']
        train_nb = str(config['custom']['train_nb'])
        validation_nb = str(config['custom']['validation_nb'])

    disp_root = config['disp_root']
    if int(args.fold) == 1 or int(args.fold) == 2 or int(args.fold) == 3:
        disp_root = config['pretrain_disp_root']
    training = os.path.join(disp_root, train_file)
    validation = os.path.join(disp_root, validation_file)

    gpu_id = str(config['gpu_id'])
    learning_rate = str(config['learning_rate'])
    lr_weight_decay = str(config['decay'])
    weight_decay = str(config['weight_decay'])
    momentum = str(config['momentum'])
    epoch_step = str(config['epoch_step'])
    weight_epoch = str(config['weight_epoch'])
    max_epoch = str(config['max_epoch'])
    iter_per_epoch = str(config['iter'])
    validation_points = str(config['validation_points'])
    optimizer = config['opt']
    tb = str(config['tb'])
    batch_size = str(config['bs'])
    half_width = str(config['half_width'])
    half_range = str(config['half_range'])

    run = prepare_train(batchSize=batch_size,
                        gpu_id=gpu_id,
                        train_nb=train_nb,
                        validation_nb=validation_nb,
                        training=training,
                        validation=validation,
                        learningRate=learning_rate,
                        learningRateDecay=lr_weight_decay,
                        weightDecay=weight_decay,
                        momentum=momentum,
                        epoch_step=epoch_step,
                        weight_epoch=weight_epoch,
                        max_epoch=max_epoch,
                        iter_per_epoch=iter_per_epoch,
                        data_root=data_root,
                        tb=tb,
                        validation_points=validation_points,
                        opt_method=optimizer,
                        psz=half_width,
                        half_range=half_range,
                        fold=args.fold)

    subprocess.call(run)


if __name__ == '__main__':
    train()
