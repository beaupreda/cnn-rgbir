import subprocess

def prepare_call(data_version=None,
                batchSize=None,
                gpuid=None,
                tr_num=None,
                val_num=None,
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
                num_val_loc=None,
                opt_method=None,
                type=None,
                invert=None,
                showCurve=None,
                psz=None,
                half_range=None,
                fine_tune=None,
                model_param=None,
                bn_meanstd=None
                ):
    params = locals()
    call = []
    call.append('th')
    call.append('train_match.lua')
    for key in params.keys():
        if params[key] is not None:
            call.append('--' + key)
            call.append(params[key])
    return call

def prepare_call_new_arch(data_version=None,
                          batchSize=None,
                          gpuid=None,
                          tr_num=None,
                          val_num=None,
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
                          num_val_loc=None,
                          opt_method=None,
                          type=None,
                          invert=None,
                          full_eval=None,
                          share=None,
                          showCurve=None,
                          psz=None,
                          half_range=None,
                          fine_tune=None,
                          model_param=None,
                          bn_meanstd=None):
    params = locals()
    call = []
    call.append('th')
    call.append('train_match_two_networks.lua')
    for key in params.keys():
        if params[key] is not None:
            call.append('--' + key)
            call.append(params[key])
    return call

def pytorch_test(data_version=None,
                          batchSize=None,
                          gpuid=None,
                          tr_num=None,
                          val_num=None,
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
                          num_val_loc=None,
                          opt_method=None,
                          type=None,
                          invert=None,
                          full_eval=None,
                          share=None,
                          showCurve=None,
                          psz=None,
                          half_range=None,
                          fine_tune=None,
                          model_param=None,
                          bn_meanstd=None):
    params = locals()
    call = []
    call.append('th')
    call.append('compare_pytorch.lua')
    for key in params.keys():
        if params[key] is not None:
            call.append('--' + key)
            call.append(params[key])
    return call

def prepare_test(data_version=None,
                          gpuid=None,
                          test_num=None,
                          model=None,
                          data_root=None,
                          util_root=None,
                          tb=None,
                          type=None,
                          invert=None,
                          psz=None,
                          half_range=None,
                          folder_name=None,
                          dual_net=None,
                          share=None):
    params = locals()
    call = []
    call.append('th')
    call.append('test_two_networks.lua')
    for key in params.keys():
        if params[key] is not None:
            call.append('--' + key)
            call.append(params[key])
    return call

test_run1 = prepare_test(data_version='all',
                         gpuid='0',
                         test_num='106',
                         model='dot_win37_dep6',
                         type='1',
                         data_root='/home/travail/dabeaq/litiv/stereo/test_dataset_1_mix_art',
                         util_root='/home/travail/dabeaq/litiv/stereo/patch_generator',
                         tb='100',
                         invert='0',
                         psz='18',
                         half_range='60',
                         folder_name='/home/travail/dabeaq/litiv/stereo/eff_dl_stereo_match/logs/run_2018y_9m_4d_16h_52min_53sec',
                         dual_net='1',
                         share='0')

test_run2 = prepare_test(data_version='all',
                         gpuid='0',
                         test_num='178',
                         model='dot_win37_dep6',
                         type='2',
                         data_root='/home/travail/dabeaq/litiv/stereo/test_dataset_2_mix',
                         util_root='/home/travail/dabeaq/litiv/stereo/patch_generator',
                         tb='100',
                         invert='0',
                         psz='18',
                         half_range='60',
                         folder_name='/home/travail/dabeaq/litiv/stereo/eff_dl_stereo_match/logs/run_2018y_9m_10d_15h_27min_52sec',
                         dual_net='1',
                         share='0')

test_run3 = prepare_test(data_version='all',
                         gpuid='0',
                         test_num='134',
                         model='dot_win37_dep6',
                         type='3',
                         data_root='/home/travail/dabeaq/litiv/stereo/test_dataset_3_mix',
                         util_root='/home/travail/dabeaq/litiv/stereo/patch_generator',
                         tb='100',
                         invert='0',
                         psz='18',
                         half_range='60',
                         folder_name='/home/travail/dabeaq/litiv/stereo/eff_dl_stereo_match/logs/run_2018y_9m_11d_16h_46min_5sec',
                         dual_net='1',
                         share='0')


lua_run1 = prepare_call_new_arch(data_version='all',
            invert='0',
            type='1',
            batchSize='64',
            gpuid='0',
            tr_num='5456',
            val_num='60',
            learningRate='0.001',
            learningRateDecay='1e-7',
            weightDecay='0.0005',
            momentum='0.9',
            model='dot_win37_dep6',
            epoch_step='40',
            weight_epoch='10',
            max_epoch='200',
            iter_per_epoch='200',
            data_root='/home/travail/dabeaq/litiv/stereo/test_dataset_1_mix',
            util_root='/home/travail/dabeaq/litiv/stereo/patch_generator',
            tb='100',
            num_val_loc='9000',
            opt_method='adam',
            psz='37',
            half_range='60')

lua_run_test = pytorch_test(data_version='all',
            invert='0',
            type='1',
            batchSize='64',
            gpuid='0',
            tr_num='5456',
            val_num='60',
            learningRate='0.001',
            learningRateDecay='1e-7',
            weightDecay='0.0005',
            momentum='0.9',
            model='dot_win37_dep6',
            epoch_step='40',
            weight_epoch='10',
            max_epoch='200',
            iter_per_epoch='200',
            data_root='/home/travail/dabeaq/litiv/stereo/test_dataset_1_mix',
            util_root='/home/travail/dabeaq/litiv/stereo/patch_generator',
            tb='100',
            num_val_loc='9000',
            opt_method='adam',
            psz='37',
            half_range='60')


#subprocess.call(lua_run1)
#subprocess.call(lua_run2)
#subprocess.call(lua_run3)

subprocess.call(lua_run_test)
#subprocess.call(test_run2)
#subprocess.call(test_run3)
