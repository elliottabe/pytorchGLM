import argparse
import itertools
import logging
import sys
import yaml
from pathlib import Path

import numpy as np
import ray
import torch
import torchvision.utils as vutils
import torch.optim as optim
from test_tube import Experiment
from tqdm.auto import tqdm

sys.path.append(str(Path('.').absolute()))
import io_dict_to_hdf5 as ioh5
from format_data import *
from models import *
from utils import *

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def arg_parser(jupyter=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--free_move', type=str_to_bool, default=True)
    parser.add_argument('--prey_cap', type=str_to_bool, default=False)
    parser.add_argument('--date_ani', type=str, default='102821/J570LT')#'070921/J553RT')
    parser.add_argument('--save_dir', type=str, default='~/Research/SensoryMotorPred_Data/data/')
    parser.add_argument('--fig_dir', type=str, default='~/Research/SensoryMotorPred_Data/Figures')
    parser.add_argument('--data_dir', type=str, default='~/Goeppert/freely_moving_ephys/ephys_recordings/')
    parser.add_argument('--MovModel', type=int, default=1)
    parser.add_argument('--load_ray', type=str_to_bool, default=False)
    parser.add_argument('--LinMix', type=str_to_bool, default=False)
    parser.add_argument('--NoL1', type=str_to_bool, default=False)
    parser.add_argument('--NoShifter', type=str_to_bool, default=False)
    parser.add_argument('--do_norm', type=str_to_bool, default=True)
    parser.add_argument('--do_shuffle', type=str_to_bool, default=False)
    parser.add_argument('--train_shifter', type=str_to_bool, default=False)
    parser.add_argument('--thresh_cells', type=str_to_bool, default=True)
    parser.add_argument('--SimRF', type=str_to_bool, default=False)
    parser.add_argument('--NKfold', type=int, default=1)
    parser.add_argument('--ModRun', type=str,default='0,1,2,3,4')
    parser.add_argument('--shiftn', type=int, default=12)
    parser.add_argument('--Nepochs', type=int, default=10000)
    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    return vars(args)


def get_model(input_size, output_size, meanbias, MovModel, device, l, a, params, NepochVis=10000, best_shifter_Nepochs=2000, Kfold=0, **kwargs):

    l1 = LinVisNetwork(input_size,output_size,
                        reg_alph=params['alphas'][a],reg_alphm=params['alphas_m'][a],move_features=params['move_features'],
                        train_shifter=params['train_shifter'], shift_hidden=50,
                        LinMix=params['LinMix'], device=device,).to(device)
    if (params['train_shifter']==False) & (params['MovModel']!=0) & (params['NoShifter']==False) & (params['SimRF']==False):
        state_dict = l1.state_dict()
        best_shift = 'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:01d}.pth'.format('Pytorch_BestShift',int(params['model_dt']*1000), 1, 1, best_shifter_Nepochs, Kfold)
        checkpoint = torch.load(params['save_dir']/best_shift)
        for key in state_dict.keys():
            if 'posNN' not in key:
                if 'weight' in key:
                    state_dict[key] = checkpoint['model_state_dict'][key].repeat(1,params['nt_glm_lag'])
                else:
                    state_dict[key] = checkpoint['model_state_dict'][key]
        l1.load_state_dict(state_dict)
    elif (params['NoShifter']==True) :
        pass
    elif (params['SimRF']==True):
        SimRF_file = params['save_dir'].parent.parent.parent/'121521/SimRF/fm1/SimRF_withL1_dt050_T01_Model1_NB10000_Kfold00_best.h5'
        SimRF_data = ioh5.load(SimRF_file)
        l1.Cell_NN[0].weight.data = torch.from_numpy(SimRF_data['sta'].astype(np.float32).T).to(device)
        l1.Cell_NN[0].bias.data = torch.from_numpy(SimRF_data['bias_sim'].astype(np.float32)).to(device)
        # pass
    else:
        if meanbias is not None:
            state_dict = l1.state_dict()
            state_dict['Cell_NN.0.bias']=meanbias
            l1.load_state_dict(state_dict)
    if MovModel == 0: 
        optimizer = optim.Adam(params=[{'params': [l1.Cell_NN[0].weight],'lr':params['lr_w'][1]}, 
                                        {'params': [l1.Cell_NN[0].bias],'lr':params['lr_b'][1]},])
    elif MovModel == 1:
        if params['train_shifter']:
            optimizer = optim.Adam(params=[{'params': [l1.Cell_NN[0].weight],'lr':params['lr_w'][1],'weight_decay':params['lambdas'][l]}, 
                                           {'params': [l1.Cell_NN[0].bias],'lr':params['lr_b'][1]},
                                           {'params': list(l1.shifter_nn.parameters()),'lr': params['lr_shift'][1],'weight_decay':.0001}])
        else:
            optimizer = optim.Adam(params=[{'params': [l1.Cell_NN[0].weight],'lr':params['lr_w'][1],'weight_decay':params['lambdas'][l]}, 
                                           {'params': [l1.Cell_NN[0].bias],'lr':params['lr_b'][1]},])
    else:
        if params['NoL1']:
            model_type = 'Pytorch_Vis_NoL1'
        else:
            model_type = 'Pytorch_Vis_withL1'
        GLM_LinVis = ioh5.load(params['save_model_Vis']/'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}_best.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], 1, NepochVis, Kfold))
        state_dict = l1.state_dict()
        for key in state_dict.keys():
            if 'posNN' not in key:
                state_dict[key] = torch.from_numpy(GLM_LinVis[key].astype(np.float32))
        l1.load_state_dict(state_dict)
        optimizer = optim.Adam(params=[{'params': [param for name, param in l1.posNN.named_parameters() if 'weight' in name],'lr':params['lr_m'][1]},
                                       {'params': [param for name, param in l1.posNN.named_parameters() if 'bias' in name],'lr':params['lr_b'][1]},])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(params['Nepochs']/4))
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9999)
    # scheduler = None
    return l1, optimizer, scheduler

def train_model(xtr,xte,xtrm,xtem,shift_in_tr,shift_in_te,ytr,yte,Nepochs,l1,optimizer,scheduler=None,pbar=None,track_all=False):
    vloss_trace = np.zeros((Nepochs, ytr.shape[-1]), dtype=np.float32)
    tloss_trace = np.zeros((Nepochs, ytr.shape[-1]), dtype=np.float32)
    Epoch_GLM = {}
    if track_all:
        for name, p in l1.named_parameters():
            Epoch_GLM[name] = np.zeros((Nepochs,) + p.shape, dtype=np.float32)

    if pbar is None:
        pbar = pbar2 = tqdm(np.arange(Nepochs))
    else:
        pbar2 = np.arange(Nepochs)
    for batchn in pbar2:
        out = l1(xtr, xtrm, shift_in_tr)
        loss = l1.loss(out, ytr)
        pred = l1(xte, xtem, shift_in_te)
        val_loss = l1.loss(pred, yte)
        vloss_trace[batchn] = val_loss.clone().cpu().detach().numpy()
        tloss_trace[batchn] = loss.clone().cpu().detach().numpy()
        pbar.set_description('Loss: {:.03f}'.format(np.nanmean(val_loss.clone().cpu().detach().numpy())))
        pbar.refresh()
        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if track_all:
            for name, p in l1.named_parameters():
                Epoch_GLM[name][batchn] = p.clone().cpu().detach().numpy()
    return vloss_trace, tloss_trace, l1, optimizer, scheduler, Epoch_GLM

def load_GLM_data(data, params, train_idx, test_idx):

    if params['free_move']:
        move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis],data['train_roll'][:, np.newaxis], data['train_pitch'][:, np.newaxis]))
        move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis],data['test_roll'][:, np.newaxis], data['test_pitch'][:, np.newaxis]))
        model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis],data['model_roll'][:, np.newaxis], data['model_pitch'][:, np.newaxis]))
    else:
        move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis], np.zeros(data['train_phi'].shape)[:, np.newaxis], data['train_pitch'][:, np.newaxis]))
        move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis], np.zeros(data['test_phi'].shape)[:, np.newaxis], data['test_pitch'][:, np.newaxis]))
        model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis], np.zeros(data['model_phi'].shape)[:, np.newaxis], data['model_pitch'][:, np.newaxis]))

    ##### Save dimension #####    
    params['nks'] = np.shape(data['train_vid'])[1:]
    params['nk'] = params['nks'][0]*params['nks'][1]*params['nt_glm_lag']
    ncells = data['model_nsp'].shape[-1]
    # Reshape data (video) into (T*n)xN array
    if params['train_shifter']:
        rolled_vid = np.hstack([np.roll(data['model_vid_sm'], nframes, axis=0) for nframes in params['lag_list']])
        move_quantiles = np.quantile(model_move,params['quantiles'],axis=0)
        train_range = np.all(((move_train>move_quantiles[0]) & (move_train<move_quantiles[1])),axis=1)
        test_range = np.all(((move_test>move_quantiles[0]) & (move_test<move_quantiles[1])),axis=1)
        x_train = rolled_vid[train_idx].reshape((len(train_idx), 1)+params['nks']).astype(np.float32)[train_range]
        x_test = rolled_vid[test_idx].reshape((len(test_idx), 1)+params['nks']).astype(np.float32)[test_range]
        move_train = move_train[train_range]
        move_test = move_test[test_range]
        shift_in_tr = torch.from_numpy(move_train[:, (0, 1, 3)].astype(np.float32)).to(device)
        shift_in_te = torch.from_numpy(move_test[:, (0, 1, 3)].astype(np.float32)).to(device)
        ytr = torch.from_numpy(data['train_nsp'][train_range].astype(np.float32)).to(device)
        yte = torch.from_numpy(data['test_nsp'][test_range].astype(np.float32)).to(device)
        data['train_nsp']=data['train_nsp'][train_range]
        data['test_nsp']=data['test_nsp'][test_range]
    elif params['NoShifter']:
        rolled_vid = np.hstack([np.roll(data['model_vid_sm'], nframes, axis=0) for nframes in params['lag_list']])
        x_train = rolled_vid[train_idx].reshape(len(train_idx), -1).astype(np.float32)
        x_test = rolled_vid[test_idx].reshape(len(test_idx), -1).astype(np.float32)
        shift_in_tr = None
        shift_in_te = None
        ytr = torch.from_numpy(data['train_nsp'].astype(np.float32)).to(device)
        yte = torch.from_numpy(data['test_nsp'].astype(np.float32)).to(device)
    else:
        model_vid_sm_shift = ioh5.load(params['save_dir']/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(int(params['model_dt']*1000), 1))['model_vid_sm_shift']  # [:,5:-5,5:-5]
        params['nks'] = np.shape(model_vid_sm_shift)[1:]
        params['nk'] = params['nks'][0]*params['nks'][1]*params['nt_glm_lag']
        rolled_vid = np.hstack([np.roll(model_vid_sm_shift, nframes, axis=0) for nframes in params['lag_list']])  
        x_train = rolled_vid[train_idx].reshape(len(train_idx), -1).astype(np.float32)
        x_test = rolled_vid[test_idx].reshape(len(test_idx), -1).astype(np.float32)
        shift_in_tr = None
        shift_in_te = None
        ytr = torch.from_numpy(data['train_nsp'].astype(np.float32)).to(device)
        yte = torch.from_numpy(data['test_nsp'].astype(np.float32)).to(device)


    if params['MovModel'] == 0:
        model_type = 'Pytorch_Mot'
    elif params['MovModel'] == 1:
        model_type = 'Pytorch_Vis'
    elif params['MovModel'] == 2:
        model_type = 'Pytorch_Add'
    elif params['MovModel'] == 3:
        model_type = 'Pytorch_Mul'

    if params['train_shifter']:
        params['save_model_shift'] = params['save_dir'] / 'models/Shifter'
        params['save_model_shift'].mkdir(parents=True, exist_ok=True)
        params['NoL1'] = True
        params['do_norm']=True
        model_type = model_type + 'Shifter'
    elif params['NoShifter']:
        model_type = model_type + 'NoShifter'

    if params['NoL1']:
        model_type = model_type + '_NoL1'
    else:
        model_type = model_type + '_withL1'
    
    if params['SimRF']:
        SimRF_file = params['save_dir'].parent.parent.parent/'121521/SimRF/fm1/SimRF_withL1_dt050_T01_Model1_NB10000_Kfold00_best.h5'
        SimRF_data = ioh5.load(SimRF_file)
        model_type = model_type + '_SimRF'
        ytr = torch.from_numpy(SimRF_data['ytr'].astype(np.float32)).to(device)
        yte = torch.from_numpy(SimRF_data['yte'].astype(np.float32)).to(device)
        params['save_model'] = params['save_model'] / 'SimRF'
        params['save_model'].mkdir(parents=True, exist_ok=True)
        meanbias = torch.from_numpy(SimRF_data['bias_sim'].astype(np.float32)).to(device)
    else:
        meanbias = torch.mean(torch.tensor(data['model_nsp'], dtype=torch.float32), axis=0)
    input_size = params['nk']
    output_size = ytr.shape[1]
    params['lr_shift'] = [1e-3,1e-2]
    params['Ncells'] = ytr.shape[-1]
    # Reshape data (video) into (T*n)xN array
    if params['MovModel'] == 0:
        mx_train = move_train.copy()
        mx_test = move_test.copy()
        xtr = torch.from_numpy(mx_train.astype(np.float32)).to(device)
        xte = torch.from_numpy(mx_test.astype(np.float32)).to(device)
        xtrm = None
        xtem = None
        params['nk'] = xtr.shape[-1]
        params['move_features'] = None 
        params['lambdas'] = np.array([None])
        params['alphas'] = np.array([None])
        params['lambdas_m'] = np.array([None]) 
        params['alphas_m'] = np.array([None]) 
        params['nlam'] = len(params['lambdas'])
        params['nalph'] = len(params['alphas'])
        params['lr_w'] = [1e-6, 1e-3]
        params['lr_b'] = [1e-6, 1e-3]
        input_size = xtr.shape[-1]
    elif params['MovModel'] == 1:
        xtr = torch.from_numpy(x_train).to(device)
        xte = torch.from_numpy(x_test).to(device)
        xtrm = None
        xtem = None
        params['move_features'] = None
        sta_init = torch.zeros((output_size, xtr.shape[-1]))
        params['alphas'] = np.array([.0001 if params['NoL1']==False else None])
        params['lambdas'] = np.hstack((0, np.logspace(-2, 3, 20)))
        params['nlam'] = len(params['lambdas'])
        params['nalph'] = len(params['alphas'])
        params['lambdas_m'] = np.array(params['nlam']*[None])
        params['alphas_m'] = np.array(params['nalph']*[None])
        params['lr_w'] = [1e-5, 1e-3]
        params['lr_b'] = [1e-5, 5e-3]
    else:
        xtr = torch.from_numpy(x_train.astype(np.float32)).to(device)
        xte = torch.from_numpy(x_test.astype(np.float32)).to(device)
        xtrm = torch.from_numpy(move_train.astype(np.float32)).to(device)
        xtem = torch.from_numpy(move_test.astype(np.float32)).to(device)
        params['move_features'] = xtrm.shape[-1]
        sta_init = torch.zeros((output_size, xtr.shape[-1]))
        params['alphas'] = np.array([None])
        params['lambdas'] =  np.hstack((0, np.logspace(-2, 3, 20)))
        params['nalph'] = len(params['alphas'])
        params['alphas_m'] = np.array(params['nalph']*[None])
        params['lambdas_m'] = np.array([0]) # np.hstack((0, np.logspace(-5, 6, 40)))
        params['nlam'] = len(params['lambdas_m'])
        params['lr_w'] = [1e-5, 1e-3]
        params['lr_m'] = [1e-6, 1e-3]
        params['lr_b'] = [1e-6, 1e-3]

    return params, xtr, xtrm, xte, xtem, ytr, yte, shift_in_tr, shift_in_te, input_size, output_size, model_type, meanbias, model_move

def load_params(MovModel,Kfolds:int,args,file_dict=None,debug=False):

    ##### '070921/J553RT' '102621/J558NC' '110421/J569LT' #####
    free_move = args['free_move']
    fm_dir = 'fm1' if args['prey_cap']==False else 'fm1_prey'
    if args['free_move']:
        stim_type = fm_dir
    else:
        stim_type = 'hf1_wn' 
    date_ani = args['date_ani']
    date_ani2 = '_'.join(date_ani.split('/'))
    data_dir = Path(args['data_dir']).expanduser() / date_ani / stim_type 
    save_dir = (Path(args['save_dir']).expanduser() / date_ani / stim_type)
    base_dir = (Path(args['save_dir']).expanduser() / date_ani)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_fm = save_dir.parent / fm_dir
    save_dir_hf = save_dir.parent / 'hf1_wn'
    save_dir_fm.mkdir(parents=True, exist_ok=True)
    save_dir_hf.mkdir(parents=True, exist_ok=True)
    fig_dir = (Path(args['fig_dir']).expanduser()/'Encoding'/date_ani/stim_type)
    fig_dir.mkdir(parents=True, exist_ok=True)

    exp = Experiment(name='MovModel{}'.format(MovModel),
                     save_dir=save_dir / 'GLM_Network',
                     debug=debug,
                     version=Kfolds)
    save_model = exp.save_dir / exp.name / 'version_{}'.format(Kfolds)
    save_model_Vis = exp.save_dir / 'MovModel1' /'version_{}'.format(Kfolds)

    params = {
        'Nepochs': args['Nepochs'],
        'model_dt': .05,
        'do_shuffle': args['do_shuffle'],
        'do_norm': args['do_norm'],
        'do_worldcam_correction': False,
        'lag_list': [-2,-1,0,1,2],
        'free_move': free_move,
        'stim_type': stim_type,
        'base_dir': base_dir,
        'save_dir': save_dir,
        'save_dir_fm': save_dir_fm,
        'save_dir_hf': save_dir_hf,
        'data_dir': data_dir,
        'fig_dir': fig_dir,
        'save_model': save_model,
        'save_model_Vis': save_model_Vis,
        'shiftn': args['shiftn'], 
        'train_shifter': args['train_shifter'],
        'MovModel': MovModel,
        'load_Vis' : True if MovModel>1 else False,
        'LinMix': True if MovModel==2 else False,
        'NKfold': args['NKfold'],
        'NoL1': args['NoL1'],
        'SimRF': args['SimRF'],
        'date_ani2': date_ani2,
        'bin_length': 40,
        'NoShifter': args['NoShifter'],
        'quantiles': [.05,.95],
        'thresh_cells': args['thresh_cells'],
    }

    params['nt_glm_lag']=len(params['lag_list'])
    if params['do_worldcam_correction']:
        params['WC_type'] = 'C'
    else:
        params['WC_type'] = 'UC'

    if file_dict is None:
        file_dict = {'cell': 0,
                    'drop_slow_frames': True,
                    'ephys': list(data_dir.glob('*ephys_merge.json'))[0].as_posix(),
                    'ephys_bin': list(data_dir.glob('*Ephys.bin'))[0].as_posix(),
                    'eye': list(data_dir.glob('*REYE.nc'))[0].as_posix(),
                    'imu': list(data_dir.glob('*imu.nc'))[0].as_posix() if stim_type == fm_dir else None,
                    'mapping_json': Path('~/Research/Github/FreelyMovingEphys/probes/channel_maps.json').expanduser(),
                    'mp4': True,
                    'name': date_ani2 + '_control_Rig2_' + stim_type,  # 070921_J553RT
                    'probe_name': 'DB_P128-6',
                    'save': data_dir.as_posix(),
                    'speed': list(data_dir.glob('*speed.nc'))[0].as_posix() if stim_type == 'hf1_wn' else None,
                    'stim_type': 'light',
                    'top': list(data_dir.glob('*TOP1.nc'))[0].as_posix() if stim_type == fm_dir else None,
                    'world': list(data_dir.glob('*world.nc'))[0].as_posix(), }
    if debug==False:
        params2=params.copy()
        for key in params2.keys():
            if isinstance(params2[key], Path):
                params2[key]=params2[key].as_posix()

        pfile_path = save_model / 'model_params.yaml'
        with open(pfile_path, 'w') as file:
            doc = yaml.dump(params2, file, sort_keys=True)

    return params, file_dict, exp


if __name__ == '__main__':
    args = arg_parser()
    if args['load_ray']:
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
        )
    ModRun = [int(i) for i in args['ModRun'].split(',')] #[0,1,2,3,4] #-1,
    # ModRun = [1]
    for Kfold in np.arange(args['NKfold']):
        for ModelRun in ModRun:
            if ModelRun == -1:
                args['train_shifter']=True
                # args['Nepochs'] = 2000
                params, file_dict, exp = load_params(1,Kfold,args)
            elif ModelRun == 0:
                args['train_shifter']=False
                # args['Nepochs']= 5000
                params, file_dict, exp = load_params(0,Kfold,args)
            elif ModelRun == 1:
                args['train_shifter']=False
                # args['Nepochs'] = 5000
                # args['NoL1'] = False
                params, file_dict, exp = load_params(1,Kfold,args,file_dict={})
            elif ModelRun == 2:
                args['train_shifter']=False
                # args['Nepochs'] = 5000
                args['NoL1'] = False
                params, file_dict, exp = load_params(2,Kfold,args)
            elif ModelRun == 3:
                args['train_shifter']=False
                # args['Nepochs'] = 5000
                args['NoL1'] = False
                params, file_dict, exp = load_params(3,Kfold,args)
            elif ModelRun == 4:
                args['train_shifter']=False
                args['free_move']=False
                args['NoL1'] = False
                # args['Nepochs'] = 5000
                params, file_dict, exp = load_params(1,Kfold,args)
            VisNepochs = args['Nepochs']
            data, train_idx_list, test_idx_list = load_train_test(file_dict, **params)
            train_idx = train_idx_list[Kfold]
            test_idx = test_idx_list[Kfold]
            data = load_Kfold_data(data,train_idx,test_idx,params)
            # locals().update(data)

            params, xtr, xtrm, xte, xtem, ytr, yte, shift_in_tr, shift_in_te, input_size, output_size, model_type, meanbias, model_move = load_GLM_data(data, params, train_idx, test_idx)
            print('Model: {}, LinMix: {}, move_features: {}, Ncells: {}, train_shifter: {}'.format(params['MovModel'],params['LinMix'],params['move_features'],params['Ncells'],params['train_shifter']))
            
            GLM_CV = {}
            GLM_CV['loss_regcv']      = np.zeros((params['nalph'], params['nlam'], output_size))
            GLM_CV['pred_cv']         = np.zeros((params['nalph'], params['nlam'], output_size, xte.shape[0]), dtype=np.float32)
            GLM_CV['out_cv']          = np.zeros((params['nalph'], params['nlam'], output_size, xtr.shape[0]), dtype=np.float32)
            GLM_CV['tloss_trace_all'] = np.zeros((params['nalph'], params['nlam'], output_size, params['Nepochs']), dtype=np.float32)
            GLM_CV['vloss_trace_all'] = np.zeros((params['nalph'], params['nlam'], output_size, params['Nepochs']), dtype=np.float32)
            GLM_CV['r2_test']         = np.zeros((params['nalph'], params['nlam'], output_size))

            for a, reg_alph in enumerate(params['alphas']):
                if params['MovModel'] == 1:
                    pbar = tqdm(range(len(params['lambdas'])))
                elif (params['MovModel'] != 1):
                    pbar = tqdm(range(len(params['lambdas_m'])))
                
                for reg_lam, l in enumerate(pbar):
                    l1,optimizer,scheduler = get_model(input_size, output_size, meanbias, params['MovModel'], device, l, a, params,NepochVis=VisNepochs)
                    if (a==0) & (l==0):
                        for name, p in l1.named_parameters():
                            GLM_CV[name] = np.zeros((params['nalph'],params['nlam'],) + p.shape, dtype=np.float32)
                    # elif (l!=0) & (params['MovModel']==1) & (params['train_shifter']==False):
                    #     l1.load_state_dict(state_dict)

                    vloss_trace, tloss_trace, l1, optimizer, scheduler, Epoch_GLM = train_model(xtr,xte,xtrm,xtem,shift_in_tr,shift_in_te,ytr,yte,params['Nepochs'],l1,optimizer,scheduler,pbar=None)
                    model_name = 'GLM_{}_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_alph{}_lam{}_Kfold{}.pth'.format(model_type, params['WC_type'], int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'], a, l, Kfold)

                    for name, p in l1.named_parameters():
                        GLM_CV[name][a,l] = p.clone().cpu().detach().numpy()

                    # if (l == 0) & (params['MovModel']==1):
                    #     state_dict = l1.state_dict()

                    GLM_CV['tloss_trace_all'][a, l] = tloss_trace.T
                    GLM_CV['vloss_trace_all'][a, l] = vloss_trace.T
                    
                    pred = l1(xte, xtem, shift_in_te)
                    GLM_CV['loss_regcv'][a, l] = l1.loss(pred, yte).cpu().detach().numpy()
                    GLM_CV['pred_cv'][a, l] = pred.detach().cpu().numpy().squeeze().T
                    out = l1(xtr, xtrm, shift_in_tr)
                    GLM_CV['out_cv'][a, l] = out.detach().cpu().numpy().squeeze().T

                    sp_smooth = np.apply_along_axis(lambda m: np.convolve(m, np.ones(params['bin_length']), mode='same')/(params['bin_length'] * params['model_dt']), axis=0, arr=data['test_nsp'])[params['bin_length']:-params['bin_length']]
                    pred_smooth = np.apply_along_axis(lambda m: np.convolve(m, np.ones(params['bin_length']), mode='same')/(params['bin_length'] * params['model_dt']), axis=1, arr=GLM_CV['pred_cv'][a, l])[:,params['bin_length']:-params['bin_length']].T
                    GLM_CV['r2_test'][a,l] = np.array([(np.corrcoef(sp_smooth[:,celln],pred_smooth[:,celln])[0, 1])**2 for celln in range(params['Ncells'])])
                    
                    if params['train_shifter']:
                        torch.save({'reg_alph': reg_alph, 'reg_lam':l, 'model_state_dict': l1.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, (params['save_model_shift'] / model_name))
                    
            if (params['MovModel'] ==1):
                GLM_CV['loss_regcv'][:, 0, :] = np.nan

            malph, mlam, cellnum = np.where(GLM_CV['loss_regcv'] == np.nanmin(GLM_CV['loss_regcv'], axis=(0, 1), keepdims=True))
            cellnum, m_cinds = np.unique(cellnum, return_index=True)
            malph = malph[m_cinds]
            mlam = mlam[m_cinds]
            sortinds = cellnum.argsort()
            cellnum = cellnum[sortinds]
            malph = malph[sortinds]
            mlam = mlam[sortinds]

            GLM_Data = {}
            for key in GLM_CV.keys():
                if 'shifter' in key:
                    GLM_Data[key] = GLM_CV[key][malph,mlam]    
                else:
                    GLM_Data[key] = GLM_CV[key][malph,mlam,cellnum]
            GLM_Data['loss_regcv']=GLM_CV['loss_regcv']
            if params['do_shuffle']:
                save_datafile_all = params['save_model']/'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}_shuffled_all.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],Kfold)
                save_datafile_best = params['save_model']/'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}_shuffled_best.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],Kfold)
            else:
                save_datafile_all = params['save_model']/'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}_all.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],Kfold)
                save_datafile_best = params['save_model']/'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}_best.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],Kfold)
            ioh5.save(save_datafile_all, GLM_CV)
            ioh5.save(save_datafile_best, GLM_Data)
            print(save_datafile_all)

            if params['train_shifter']:
                best_shifter = np.nanargmin(np.nanmean(GLM_CV['loss_regcv'][0],axis=-1))
                Best_RF={}
                for name, p in l1.named_parameters():
                    if 'shifter' not in name:
                        Best_RF[name] = torch.from_numpy(GLM_Data[name])
                best_shift = 'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:01d}.pth'.format('Pytorch_BestShift',int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],Kfold)

                torch.save({'model_state_dict': Best_RF}, (params['save_dir_fm'] / best_shift))
                torch.save({'model_state_dict': Best_RF}, (params['save_dir_hf'] / best_shift))
                a = 0 
                # save_model2 = save_model/'Shifter'
                model_vid_sm_shift2 = {}
                pdf_name = params['fig_dir']/ 'VisMov_{}_dt{:03d}_Lags{:02d}_MovModel{:d}_CellSummary.pdf'.format('Pytorch_VisMov_AddMul_NoL1_Shifter',int(params['model_dt']*1000),params['nt_glm_lag'], params['MovModel'])
                with PdfPages(pdf_name) as pdf:
                    for l,lam in enumerate(tqdm(params['lambdas'])):
                        model_name = 'GLM_{}_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_alph{}_lam{}_Kfold{:01d}.pth'.format(model_type,'UC',int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],a,l,Kfold)
                        # model_name = 'GLMShifter_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_pretrain_xyz.pth'.format(WC_type,int(params['model_dt']*1000), params['nt_glm_lag'] MovModel, Nbatches)
                        checkpoint = torch.load(params['save_model_shift']/model_name)
                        l1.load_state_dict(checkpoint['model_state_dict'])
                        l1.cpu()
                        
                        ##### Sweep -40 to 40 degrees
                        FM_move_avg = np.load(params['save_dir_fm']/'FM_MovAvg_dt{:03d}.npy'.format(int(params['model_dt']*1000)))
                        th_range = 40/FM_move_avg[1,0]
                        phi_range = 40/FM_move_avg[1,1]
                        pitch_range = 40/FM_move_avg[1,2]
                        ang_sweepx,ang_sweepy,ang_sweepz = np.meshgrid(np.linspace(-th_range,th_range,81),np.linspace(-phi_range,phi_range,81),np.linspace(-pitch_range,pitch_range,81),sparse=False,indexing='ij')
                        # ang_sweepx,ang_sweepy,ang_sweepz = np.meshgrid(np.arange(-40,40),np.arange(-40,40),np.arange(-40,40),sparse=False,indexing='ij')
                        shift_mat = np.zeros((3,) + ang_sweepx.shape)
                        for i in range(ang_sweepx.shape[0]):
                            for j in range(ang_sweepy.shape[1]):
                                ang_sweep = torch.from_numpy(np.vstack((ang_sweepx[i,j,:],ang_sweepy[i,j,:],ang_sweepz[i,j,:])).astype(np.float32).T)
                                shift_vec = l1.shifter_nn(ang_sweep).detach().cpu().numpy()
                                shift_mat[0,i,j] = shift_vec[:,0] #np.clip(shift_vec[:,0],a_min=-15,a_max=15)
                                shift_mat[1,i,j] = shift_vec[:,1] #np.clip(shift_vec[:,1],a_min=-15,a_max=15)
                                shift_mat[2,i,j] = shift_vec[:,2] #np.clip(shift_vec[:,2],a_min=-30,a_max=30)

                            
                        fig, ax = plt.subplots(1,4,figsize=(20,5))
                        shift_titles = [r'$dx$',r'$dy$',r'$d\alpha,\phi=0$',r'$d\alpha,\theta=0$']
                        ticks=np.arange(0,90,20)
                        ticklabels=np.arange(-40,50,20)
                        shift_matshow=np.stack((shift_mat[0,:,:,40].T,shift_mat[1,:,:,40].T,shift_mat[2,:,40,:].T,shift_mat[2,40,:,:].T))
                        crange_list = np.stack((np.max(np.abs(shift_mat[:2])),np.max(np.abs(shift_mat[:2])), np.max(np.abs(shift_mat[2])), np.max(np.abs(shift_mat[2]))))
                        for n in range(4):
                            im1=ax[n].imshow(shift_matshow[n],vmin=-crange_list[n], vmax=crange_list[n], origin='lower', cmap='RdBu_r')
                            cbar1 = add_colorbar(im1)
                            ax[n].set_xticks(ticks)
                            ax[n].set_xticklabels(ticklabels)
                            ax[n].set_yticks(ticks)
                            ax[n].set_yticklabels(ticklabels)
                            ax[n].set_xlabel(r'$\theta$')
                            ax[n].set_ylabel(r'$\phi$')
                            ax[n].set_title(shift_titles[n])

                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                ##### Save FM Shifted World Cam #####
                model_name = 'GLM_{}_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_alph{}_lam{}_Kfold{:01d}.pth'.format(model_type,'UC',int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],a,best_shifter,Kfold)
                checkpoint = torch.load(params['save_model_shift']/model_name)
                l1.load_state_dict(checkpoint['model_state_dict'])
                l1.cpu()
                shift_out = l1.shifter_nn(torch.from_numpy(model_move[:,(0,1,3)].astype(np.float32)))
                shift = Affine(angle=shift_out[:,-1],translation=shift_out[:,:2])
                model_vid_sm_shift = shift(torch.from_numpy(data['model_vid_sm'][:,np.newaxis].astype(np.float32))).detach().cpu().numpy().squeeze()
                model_vid_sm_shift2['model_vid_sm_shift'] = model_vid_sm_shift
                ioh5.save(params['save_dir_fm']/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(int(params['model_dt']*1000), params['MovModel']),model_vid_sm_shift2)


                ##### Save HF Shifted World Cam #####
                args['free_move'] = False
                args['train_shifter']=True
                args['Nepochs'] = 2000
                params, file_dict, exp = load_params(1,Kfold,args)
                VisNepochs = args['Nepochs']
                data, train_idx_list, test_idx_list = load_train_test(file_dict, **params)
                train_idx = train_idx_list[Kfold]
                test_idx = test_idx_list[Kfold]
                data = load_Kfold_data(data,train_idx,test_idx,params)
                params, xtr, xtrm, xte, xtem, ytr, yte, shift_in_tr, shift_in_te, input_size, output_size, model_type, meanbias, model_move = load_GLM_data(data, params, train_idx, test_idx)
                shift_out = l1.shifter_nn(torch.from_numpy(model_move[:,(0,1,3)].astype(np.float32)))
                shift = Affine(angle=shift_out[:,-1],translation=shift_out[:,:2])
                model_vid_sm_shift = shift(torch.from_numpy(data['model_vid_sm'][:,np.newaxis].astype(np.float32))).detach().cpu().numpy().squeeze()
                model_vid_sm_shift2['model_vid_sm_shift'] = model_vid_sm_shift
                ioh5.save(params['save_dir_hf']/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(int(params['model_dt']*1000), params['MovModel']),model_vid_sm_shift2)
                args['free_move'] = True
