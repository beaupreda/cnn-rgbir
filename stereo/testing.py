'''
script used to execute multiple runs of testing
simply calls corresponding lua script with chosen arguments

David-Alexandre Beaupre
'''


import subprocess


def prepare_test(gpuid=None,
                 test_nb=None,
                 data_root=None,
                 util_root=None,
                 tb=None,
                 psz=None,
                 half_range=None,
                 folder_name=None,
                 fold=None):
    params = locals()
    call = []
    call.append('th')
    call.append('test.lua')
    for key in params.keys():
        if params[key] is not None:
            call.append('--' + key)
            call.append(params[key])
    return call


test_fold1 = prepare_test(gpuid='0',
                          test_nb='106',
                          data_root='/home/travail/dabeaq/litiv/stereo/test_dataset_1_mix_art',
                          util_root='/home/travail/dabeaq/litiv/stereo/patch_generator',
                          tb='100',
                          psz='18',
                          half_range='60',
                          folder_name='/home/travail/dabeaq/litiv/stereo/eff_dl_stereo_match/logs/run_2018y_9m_4d_16h_52min_53sec'
                          fold='1')

test_fold2 = prepare_test(gpuid='0',
                          test_num='178',
                          data_root='/home/travail/dabeaq/litiv/stereo/test_dataset_2_mix',
                          util_root='/home/travail/dabeaq/litiv/stereo/patch_generator',
                          tb='100',
                          psz='18',
                          half_range='60',
                          folder_name='/home/travail/dabeaq/litiv/stereo/eff_dl_stereo_match/logs/run_2018y_9m_10d_15h_27min_52sec',
                          fold='2')

test_fold3 = prepare_test(gpuid='0',
                          test_num='134',
                          data_root='/home/travail/dabeaq/litiv/stereo/test_dataset_3_mix',
                          util_root='/home/travail/dabeaq/litiv/stereo/patch_generator',
                          tb='100',
                          psz='18',
                          half_range='60',
                          folder_name='/home/travail/dabeaq/litiv/stereo/eff_dl_stereo_match/logs/run_2018y_9m_11d_16h_46min_5sec',
                          fold='3')

subprocess.call(test_fold1)
subprocess.call(test_fold2)
subprocess.call(test_fold3)
