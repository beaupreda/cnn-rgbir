'''
script used to execute multiple runs of training
simply calls corresponding lua script with chosen arguments

David-Alexandre Beaupre
'''


import subprocess


def prepare_call_new_arch(batchSize=None,
                          gpuid=None,
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
                          util_root=None,
                          tb=None,
                          validation_points=None,
                          opt_method=None,
                          showCurve=None,
                          psz=None,
                          half_range=None,
                          fold=None):
    params = locals()
    call = []
    call.append('th')
    call.append('train.lua')
    for key in params.keys():
        if params[key] is not None:
            call.append('--' + key)
            call.append(params[key])
    return call


lua_run = prepare_call_new_arch(batchSize='64',
                                gpuid='0',
                                train_nb='5456',
                                validation_nb='60',
                                learningRate='0.001',
                                learningRateDecay='1e-7',
                                weightDecay='0.0005',
                                momentum='0.9',
                                epoch_step='40',
                                weight_epoch='10',
                                max_epoch='200',
                                iter_per_epoch='200',
                                data_root='/home/travail/dabeaq/litiv/stereo/test_dataset_1_mix',
                                util_root='/home/travail/dabeaq/litiv/stereo/patch_generator',
                                tb='100',
                                validation_points='9000',
                                opt_method='adam',
                                psz='18',
                                half_range='60',
                                fold='1')

subprocess.call(lua_run)
