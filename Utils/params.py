import ray
import argparse
import yaml
import warnings 
warnings.filterwarnings('ignore')

import numpy as np

from ray import tune
from pathlib import Path

from Utils.utils import str_to_bool


def arg_parser(jupyter=False):
    parser = argparse.ArgumentParser(description=__doc__)
    ##### Directory Parameters #####
    parser.add_argument('--date_ani',           type=str, default='070921/J553RT') #'020422/J577RT' '122021/J581RT')# '020422/J577RT')#
    parser.add_argument('--base_dir',           type=str, default='~/Research/SensoryMotorPred_Data/Testing')
    parser.add_argument('--fig_dir',            type=str, default='~/Research/SensoryMotorPred_Data/FigTesting')
    parser.add_argument('--data_dir',           type=str, default='~/Goeppert/nlab-nas/Dylan/freely_moving_ephys/ephys_recordings/')
    ##### Simulation Parameters ##### 
    parser.add_argument('--model_dt',           type=float, default=0.05)
    parser.add_argument('--ds_vid',             type=int, default=4)
    parser.add_argument('--Kfold',              type=int, default=0)
    parser.add_argument('--ModRun',             type=str,default='1')
    parser.add_argument('--Nepochs',            type=int, default=10)
    parser.add_argument('--load_ray',           type=str_to_bool, default=False)
    ##### Model Paremeters #####    
    parser.add_argument('--do_norm',            type=str_to_bool, default=True)
    parser.add_argument('--crop_input',         type=str_to_bool, default=True)
    parser.add_argument('--free_move',          type=str_to_bool, default=True)
    parser.add_argument('--thresh_cells',       type=str_to_bool, default=True)
    parser.add_argument('--fm_dark',            type=str_to_bool, default=False)
    parser.add_argument('--NoL1',               type=str_to_bool, default=False)
    parser.add_argument('--NoL2',               type=str_to_bool, default=False)
    parser.add_argument('--NoShifter',          type=str_to_bool, default=False)
    parser.add_argument('--do_shuffle',         type=str_to_bool, default=False)
    parser.add_argument('--use_spdpup',         type=str_to_bool, default=False)
    parser.add_argument('--only_spdpup',        type=str_to_bool, default=False)
    parser.add_argument('--train_shifter',      type=str_to_bool, default=False)
    parser.add_argument('--shifter_5050',       type=str_to_bool, default=False)
    parser.add_argument('--shifter_5050_run',   type=str_to_bool, default=False)
    parser.add_argument('--EyeHead_only',       type=str_to_bool, default=False)
    parser.add_argument('--EyeHead_only_run',   type=str_to_bool, default=False)
    parser.add_argument('--SimRF',              type=str_to_bool, default=False)

    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    return vars(args)

def load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False):
    """ Set up params dictionary for loading data and model info.

    Args:
        args (dict): arguments from argparse
        ModelID (int): Model Idenfiter
        exp_dir_name (str, optional): _description_. Defaults to None.
        nKfold (int, optional): _description_. Defaults to 0.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        params (dict): dictionary of parameters
        exp (obj): Test_tube object for organizing files and tensorboard
    """
    from test_tube import Experiment
    ##### Check Stimulus Condition #####
    free_move = args['free_move']
    if free_move:
        if args['fm_dark']:
            fm_dir = 'fm1_dark'
        else:
            fm_dir = 'fm1'
        stim_cond = fm_dir
    else:
        stim_cond = 'hf1_wn' 

    ##### Create directories and paths #####
    date_ani = args['date_ani']
    date_ani2 = '_'.join(date_ani.split('/'))
    data_dir = Path(args['data_dir']).expanduser() / date_ani / stim_cond 
    base_dir = Path(args['base_dir']).expanduser()
    save_dir_fm = base_dir / date_ani / fm_dir
    save_dir_hf = base_dir / date_ani / 'hf1_wn'
    save_dir_fm.mkdir(parents=True, exist_ok=True)
    save_dir_hf.mkdir(parents=True, exist_ok=True)
    save_dir = (base_dir / date_ani / stim_cond)
    save_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (Path(args['fig_dir']).expanduser()/date_ani/stim_cond)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ##### Set up exp name #####
    if exp_dir_name is None: 
        if args['shifter_5050']:
            exp_dir_name = 'shifter5050'
        elif args['EyeHead_only']:
            exp_dir_name = 'EyeHead_only'
        elif args['only_spdpup']:
            exp_dir_name = 'OnlySpdPupil'
        elif args['crop_input']:
            exp_dir_name = 'CropInputs'
        else:
            exp_dir_name = 'RevisionSims'
            
    exp = Experiment(name='ModelID{}'.format(ModelID),
                        save_dir=save_dir / exp_dir_name, #'Shifter_TrTe_testing2', #'GLM_Network',#
                        debug=debug,
                        version=nKfold)

    save_model = exp.save_dir / exp.name / 'version_{}'.format(nKfold)
    save_model_Vis = exp.save_dir / 'ModelID1' /'version_{}'.format(nKfold)
    save_dir_fm_exp = save_dir_fm / exp.save_dir.name
    save_dir_fm_exp.mkdir(parents=True, exist_ok=True)
    save_dir_hf_exp = save_dir_hf / exp.save_dir.name
    save_dir_hf_exp.mkdir(parents=True, exist_ok=True)

    params = {
        ##### Data Parameters #####
        'data_dir':                 data_dir,
        'base_dir':                 base_dir,
        'exp_name_base':            base_dir.name,
        'free_move':                free_move,
        'fm_dir':                   fm_dir,
        'stim_cond':                stim_cond,
        'save_dir':                 save_dir,
        'save_dir_fm':              save_dir_fm,
        'save_dir_hf':              save_dir_hf,
        'save_dir_fm_exp':          save_dir_fm / exp.save_dir.name,
        'save_dir_hf_exp':          save_dir_hf / exp.save_dir.name,
        'exp_name':                 exp.save_dir.name,
        'fig_dir':                  fig_dir,
        'save_model':               save_model,
        'save_model_Vis':           save_model_Vis,
        'date_ani2':                date_ani2,
        'model_dt':                 args['model_dt'],
        'quantiles':                [.05,.95],
        'thresh_cells':             args['thresh_cells'], 
        ##### Model Parameters #####
        'lag_list':                 [0], #[-2,-1,0,1,2], # List of which timesteps to include in model fit
        'Nepochs':                  args['Nepochs'],
        'do_shuffle':               args['do_shuffle'],
        'do_norm':                  args['do_norm'],
        'train_shifter':            args['train_shifter'],
        'ModelID':                  ModelID,
        'load_Vis' :                True if ModelID>1 else False,
        'LinMix':                   True if ModelID==2 else False,
        'Kfold':                    args['Kfold'],
        'NoL1':                     args['NoL1'],
        'NoL2':                     args['NoL2'],
        'position_vars':            ['th','phi','pitch','roll','speed','eyerad'], # Which variables to use for position fits
        'use_spdpup':               args['use_spdpup'],
        'only_spdpup':              args['only_spdpup'],
        'EyeHead_only':             args['EyeHead_only'],
        'EyeHead_only_run':         args['EyeHead_only_run'],
        'SimRF':                    args['SimRF'],
        'NoShifter':                args['NoShifter'],
        'downsamp_vid':             args['ds_vid'],
        'shifter_5050':             args['shifter_5050'],
        'bin_length':               40,
        'shifter_train_size':       .9,
        'shift_hidden':             20,
        'shifter_5050_run':         args['shifter_5050_run'],
        'crop_input':               5 if args['crop_input']==True else 0,
    }

    params['nt_glm_lag']=len(params['lag_list']) # number of timesteps for model fits
    params['data_name'] = '_'.join([params['date_ani2'],params['stim_cond']])
    
    ##### Saves yaml of parameters #####
    if debug==False:
        params2=params.copy()
        for key in params2.keys():
            if isinstance(params2[key], Path):
                params2[key]=params2[key].as_posix()

        pfile_path = save_model / 'model_params.yaml'
        with open(pfile_path, 'w') as file:
            doc = yaml.dump(params2, file, sort_keys=True)

    return params, exp



def make_network_config(params,single_trial=False):
    """ Create Network Config dictionary for hyperparameter search

    Args:
        params (dict): key parameters dictionary
        single_trial (bool): boolean for if testing single trial. Default=False
        device (string): cuda

    Returns:
        network_config (dict): dictionary with hyperparameters
    """
    network_config = {}
    network_config['in_features']   = params['nk']
    network_config['Ncells']        = params['Ncells']
    network_config['shift_in']      = params['shift_in']
    network_config['shift_hidden']  = params['shift_hidden']
    network_config['shift_out']     = params['shift_out']
    network_config['LinMix']        = params['LinMix']
    network_config['pos_features']  = params['pos_features']
    network_config['lr_shift']      = 1e-2
    network_config['lr_w']          = 1e-3
    network_config['lr_b']          = 1e-3
    network_config['lr_m']          = 1e-3
    if params['NoL1']:
        network_config['L1_alpha']  = None
        network_config['L1_alpham'] = None
    else:
        network_config['L1_alpha']  = .0001
        network_config['L1_alpham'] = None

    if params['NoL2']:
        network_config['L2_lambda']   = 0
        network_config['L2_lambda_m'] = 0
    else:
        if single_trial:
            network_config['L2_lambda_m'] = np.logspace(-2, 3, 20)[1]
            network_config['L2_lambda']   = np.logspace(-2, 3, 20)[1]
        else:
            network_config['L2_lambda']   = tune.grid_search(np.logspace(-2, 3, 2))
            network_config['L2_lambda_m'] = tune.grid_search(np.logspace(-2, 3, 2))
    return network_config
