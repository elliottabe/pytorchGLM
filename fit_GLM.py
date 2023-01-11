import warnings 
warnings.filterwarnings('ignore')
import logging

import ray
from ray import tune
from ray import air

import torch
torch.backends.cudnn.benchmark = True

import pytorchGLM as pglm
from pytorchGLM.main.training import train_network

if __name__ == '__main__': 
    ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
        )
    # Input arguments
    args = pglm.arg_parser()
    device = torch.device("cuda:{}".format(pglm.get_freer_gpu()) if torch.cuda.is_available() else "cpu")
    print('Device:',device)
    ModRun = [int(i) for i in args['ModRun'].split(',')] #[0,1,2,3,4] #-1,
    Kfold = args['Kfold']

    for ModelRun in ModRun:
        if ModelRun == -1: # train shifter
            Nepochs_saved = args['Nepochs']
            args['train_shifter']  = True
            args['Nepochs']        = 5000
            ModelID                = 1
            params, file_dict, exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
            params['lag_list']     = [0]
            params['nt_glm_lag']   = len(params['lag_list'])
        elif ModelRun == 0: # pos only
            args['train_shifter']  = False
            ModelID                = 0
            params, file_dict, exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
        elif ModelRun == 1: # vis only
            args['train_shifter']  = False
            ModelID                = 1
            params, file_dict, exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
        elif ModelRun == 2: # add fit
            args['train_shifter']  = False
            # args['NoL1']           = False
            ModelID                = 2
            params, file_dict, exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
        elif ModelRun == 3: # mul. fit
            args['train_shifter']  = False
            # args['NoL1']           = False
            ModelID                = 3
            params, file_dict, exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
        elif ModelRun == 4: # head-fixed
            args['train_shifter']  = False
            args['free_move']      = False
            # args['NoL1']           = False
            ModelID                = 4
            params, file_dict, exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)


        data = pglm.load_aligned_data(file_dict, params, reprocess=False)
        params = pglm.get_modeltype(params)
        train_dataset, test_dataset, network_config = pglm.load_datasets(file_dict,params,single_trial=False)

        sync_config = tune.SyncConfig()  # the default mode is to use use rsync
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_network,params=params,train_dataset=train_dataset,test_dataset=test_dataset),
                resources={"cpu": 2, "gpu": .5}),
            tune_config=tune.TuneConfig(metric="avg_loss",mode="min",),
            param_space=network_config,
            run_config=air.RunConfig(local_dir=params['save_model'], name="NetworkAnalysis",sync_config=sync_config,verbose=2)
        )
        results = tuner.fit()

        best_result = results.get_best_result("avg_loss", "min")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(best_result.metrics["avg_loss"]))
        df = results.get_dataframe()
        best_network = list(params['save_model'].glob('*{}.pt'.format(best_result.metrics['trial_id'])))[0]
        pglm.h5store(params['save_model'] / 'NetworkAnalysis/experiment_data.h5', df, **{'best_network':best_network})
    
