import warnings 
warnings.filterwarnings('ignore')
import logging

import ray
from ray import tune
from ray import air
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

import torch
torch.backends.cudnn.benchmark = True

import pytorchGLM as pglm

if __name__ == '__main__': 
    # Input arguments
    args = pglm.arg_parser()
    if args['load_ray']:
        ray.init(ignore_reinit_error=True,include_dashboard=True)
    device = torch.device("cuda:{}".format(pglm.get_freer_gpu()) if torch.cuda.is_available() else "cpu")
    print('Device:',device)
    ModRun = [int(i) for i in args['ModRun'].split(',')] #[0,1,2,3,4] #-1,
    Kfold = args['Kfold']

    for ModelRun in ModRun:
        if ModelRun == -1: # train shifter
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
            # args['NoL1']         = False
            ModelID                = 2
            params, file_dict, exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
            ##### Grab Best Vis Network Name #####
            exp_filename = list((params['save_model_Vis'] / ('NetworkAnalysis/')).glob('*experiment_data.h5'))[-1]
            _,metadata= pglm.h5load(exp_filename)
            params['best_vis_network'] = params['save_model_Vis']/metadata['best_network'].name
        elif ModelRun == 3: # mul. fit
            args['train_shifter']  = False
            # args['NoL1']         = False
            ModelID                = 3
            params, file_dict, exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
            ##### Grab Best Vis Network Name #####
            exp_filename = list((params['save_model_Vis'] / ('NetworkAnalysis/')).glob('*experiment_data.h5'))[-1]
            _,metadata= pglm.h5load(exp_filename)
            params['best_vis_network'] = params['save_model_Vis']/metadata['best_network'].name
        elif ModelRun == 4: # head-fixed
            args['train_shifter']  = False
            args['free_move']      = False
            ModelID                = 4
            params, file_dict, exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)


        data = pglm.load_aligned_data(file_dict, params, reprocess=False)
        params = pglm.get_modeltype(params)
        datasets, network_config, initial_params = pglm.load_datasets(file_dict,params)
        algo = HyperOptSearch(points_to_evaluate=initial_params)
        algo = ConcurrencyLimiter(algo, max_concurrent=4)
        num_samples = args['num_samples']
        sync_config = tune.SyncConfig()  # the default mode is to use use rsync
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(pglm.train_network,**datasets,params=params),
                resources={"cpu":args['cpus_per_task'], "gpu": args['gpus_per_task']}),
            tune_config=tune.TuneConfig(metric="avg_loss",mode="min",search_alg=algo,num_samples=num_samples),
            param_space=network_config,
            run_config=air.RunConfig(local_dir=params['save_model'], name="NetworkAnalysis",sync_config=sync_config,verbose=2)
        )
        results = tuner.fit()

        best_result = results.get_best_result("avg_loss", "min")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(best_result.metrics["avg_loss"]))
        df = results.get_dataframe()
        best_network = list(params['save_model'].glob('*{}.pt'.format(best_result.metrics['trial_id'])))[0]
        exp_filename = '_'.join([params['model_type'],params['data_name_fm']]) + 'experiment_data.h5'
        pglm.h5store(params['save_model'] / ('NetworkAnalysis/{}'.format(exp_filename)), df, **{'best_network':best_network,'trial_id':best_result.metrics['trial_id']})

        ##### Evaluate hyperparameter search #####
        pglm.evaluate_networks(best_network,best_result.config,params,datasets['xte'],datasets['xte_pos'],datasets['yte'])

        ##### If traning shifter evaluate #####
        if ModelRun == -1:
            pglm.evaluate_shifter(best_network,params)