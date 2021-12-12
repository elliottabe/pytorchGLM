import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import ray
import torch
import torchvision.utils as vutils
import torch.optim as optim
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
    parser.add_argument('--date_ani', type=str, default='070921/J553RT')
    parser.add_argument('--save_dir', type=str, default='~/Research/SensoryMotorPred_Data/data/')
    parser.add_argument('--fig_dir', type=str, default='~/Research/SensoryMotorPred_Data/Figures')
    parser.add_argument('--data_dir', type=str, default='~/Goeppert/freely_moving_ephys/ephys_recordings/')
    parser.add_argument('--MovModel', type=int, default=1)
    parser.add_argument('--load_ray', type=str_to_bool, default=False)
    parser.add_argument('--LinMix', type=str_to_bool, default=False)
    parser.add_argument('--LinNonLinMix', type=str_to_bool, default=False)
    parser.add_argument('--NonLinLayer', type=str_to_bool, default=False)
    parser.add_argument('--NoL1', type=str_to_bool, default=False)
    parser.add_argument('--do_norm', type=str_to_bool, default=False)
    parser.add_argument('--train_shifter', type=str_to_bool, default=False)
    parser.add_argument('--NKfold', type=int, default=1)
    parser.add_argument('--shiftn', type=int, default=7)
    parser.add_argument('--Nepochs', type=int, default=12000)
    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    return vars(args)


def train_model(xtr,xte,xtrm,xtem,shift_in_tr,shift_in_te,ytr,yte,Nepochs,l1,optimizer,scheduler=None,pbar=None,w_move_traces=None,biasm_traces=None):
    vloss_trace = np.zeros((Nepochs, ytr.shape[-1]), dtype=np.float32)
    tloss_trace = np.zeros((Nepochs, ytr.shape[-1]), dtype=np.float32)
    if pbar is None:
        pbar = pbar2 = tqdm(np.arange(Nepochs))
    else:
        pbar2 = np.arange(Nepochs)
    for batchn in pbar2:
            out = l1(xtr, xtrm,shift_in_tr)
            loss = l1.loss(out, ytr)
            pred = l1(xte, xtem,shift_in_te)
            val_loss = l1.loss(pred, yte)
            vloss_trace[batchn] = val_loss.clone().cpu().detach().numpy()
            tloss_trace[batchn] = loss.clone().cpu().detach().numpy()
            pbar.set_description('Loss: {:.03f}'.format(np.mean(val_loss.clone().cpu().detach().numpy())))
            pbar.refresh()
            optimizer.zero_grad()
            mask = torch.zeros_like(loss)
            mask[126] = 1
            loss.backward(mask)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if w_move_traces is not None:
                w_move_traces[batchn] = l1.move_weights.clone().detach().cpu().numpy()
            if biasm_traces is not None:
                biasm_traces[batchn] = l1.bias_m.clone().detach().cpu().numpy()
    return vloss_trace, tloss_trace, w_move_traces, biasm_traces

def load_GLM_data(data, params, train_idx, test_idx):
    avgfm_eye = np.load(params['save_dir_fm']/'FM_AvgEye_dt{:03d}.npy'.format(int(params['model_dt']*1000)))
    if params['free_move']:
        move_train = np.hstack((data['train_th'][:, np.newaxis]-avgfm_eye[0], data['train_phi'][:, np.newaxis]-avgfm_eye[1],data['train_roll'][:, np.newaxis], data['train_pitch'][:, np.newaxis]))
        move_test = np.hstack((data['test_th'][:, np.newaxis]-avgfm_eye[0], data['test_phi'][:, np.newaxis]-avgfm_eye[1],data['test_roll'][:, np.newaxis], data['test_pitch'][:, np.newaxis]))
        model_move = np.hstack((data['model_th'][:, np.newaxis]-avgfm_eye[0], data['model_phi'][:, np.newaxis]-avgfm_eye[1],data['model_roll'][:, np.newaxis], data['model_pitch'][:, np.newaxis]))
    else:
        move_train = np.hstack((data['train_th'][:, np.newaxis]-avgfm_eye[0], data['train_phi'][:, np.newaxis]-avgfm_eye[1], np.zeros(data['train_phi'].shape)[:, np.newaxis], np.zeros(data['train_phi'].shape)[:, np.newaxis]))
        move_test = np.hstack((data['test_th'][:, np.newaxis]-avgfm_eye[0], data['test_phi'][:, np.newaxis]-avgfm_eye[1], np.zeros(data['test_phi'].shape)[:, np.newaxis], np.zeros(data['test_phi'].shape)[:, np.newaxis]))
        model_move = np.hstack((data['model_th'][:, np.newaxis]-avgfm_eye[0], data['model_phi'][:, np.newaxis]-avgfm_eye[1], np.zeros(data['model_phi'].shape)[:, np.newaxis], np.zeros(data['model_phi'].shape)[:, np.newaxis]))

    ##### Save dimension #####    
    params['nks'] = np.shape(data['train_vid'])[1:]
    params['nk'] = params['nks'][0]*params['nks'][1]*params['nt_glm_lag']
    ncells = data['model_nsp'].shape[-1]
    # Reshape data (video) into (T*n)xN array
    if params['train_shifter']:
        rolled_vid = np.hstack([np.roll(data['model_vid_sm'], nframes, axis=0) for nframes in params['lag_list']])  
        x_train = rolled_vid[train_idx] #.reshape(len(train_idx), -1)
        x_test = rolled_vid[test_idx] #.reshape(len(test_idx), -1)
        shift_in_tr = torch.from_numpy(move_train[:, (0, 1, 3)].astype(np.float32)).to(device)
        shift_in_te = torch.from_numpy(move_test[:, (0, 1, 3)].astype(np.float32)).to(device)
    else:    
        model_vid_sm_shift = ioh5.load(params['save_dir']/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(int(params['model_dt']*1000), 1))['model_vid_sm_shift{}'.format(params['shiftn'])]  # [:,5:-5,5:-5]
        nks = np.shape(model_vid_sm_shift)[1:]
        nk = nks[0]*nks[1]*params['nt_glm_lag']
        rolled_vid = np.hstack([np.roll(model_vid_sm_shift, nframes, axis=0) for nframes in params['lag_list']])  
        x_train = rolled_vid[train_idx].reshape(len(train_idx), -1)
        x_test = rolled_vid[test_idx].reshape(len(test_idx), -1)
        shift_in_tr = None
        shift_in_te = None


    LinMix = params['LinMix']
    if params['MovModel'] == 0:
        model_type = 'Pytorch_Mot'
    elif params['MovModel'] == 1:
        model_type = 'Pytorch_NonLinVis'
    elif params['MovModel'] == 2:
        if LinMix:
            model_type = 'Pytorch_NonLinVis_Add'
        else:
            model_type = 'Pytorch_NonLinVis_Mul'
    if params['NoL1']:
        model_type = model_type + '_NoL1'
    
    ytr = torch.from_numpy(data['train_nsp'].astype(np.float32)).to(device)
    yte = torch.from_numpy(data['test_nsp'].astype(np.float32)).to(device)

    input_size = nk
    output_size = ytr.shape[1]

    # Reshape data (video) into (T*n)xN array
    if params['MovModel'] == 0:
        mx_train = move_train.copy()
        mx_test = move_test.copy()
        xtr = torch.from_numpy(mx_train.astype(np.float32)).to(device)
        xte = torch.from_numpy(mx_test.astype(np.float32)).to(device)
        xtrm = None
        xtem = None
        nk = xtr.shape[-1]
        params['move_features'] = None 
        params['lambdas'] = np.array([None])
        params['alphas'] = np.array([None])
        params['lambdas_m'] = np.array([None]) 
        params['alphas_m'] = np.array([None]) 
        params['nlam'] = len(params['lambdas'])
        params['nalph'] = len(params['alphas'])
    elif params['MovModel'] == 1:
        xtr = torch.from_numpy(x_train).to(device)
        xte = torch.from_numpy(x_test).to(device)
        xtrm = None
        xtem = None
        params['move_features'] = None
        params['alphas'] = np.array([.009 if params['NoL1']==False else None])
        params['lambdas'] = np.logspace(-3, 3.25, 20) #np.hstack((np.logspace(-3, 3.25, 10)[0], np.logspace(-3, 3.25, 20)))
        params['nlam'] = len(params['lambdas'])
        params['nalph'] = len(params['alphas'])
        params['lambdas_m'] = np.array(params['nlam']*[None])
        params['alphas_m'] = np.array(params['nalph']*[None])
        params['lr_w'] = [1e-5, 1e-3]
        params['lr_b'] = [1e-5, 1e-2]
    elif params['MovModel'] == 2:
        xtr = torch.from_numpy(x_train.astype(np.float32)).to(device)
        xte = torch.from_numpy(x_test.astype(np.float32)).to(device)
        xtrm = torch.from_numpy(move_train.astype(np.float32)).to(device)
        xtem = torch.from_numpy(move_test.astype(np.float32)).to(device)
        params['move_features'] = xtrm.shape[-1]
        params['alphas'] = np.array([None])
        params['lambdas'] = np.logspace(-3, 3.25, 20)#  np.hstack((np.logspace(-3, 3.25, 10)[0], np.logspace(-3, 3.25, 20)))
        params['nalph'] = len(params['alphas'])
        params['nlam'] = len(params['lambdas'])
        params['alphas_m'] = np.array(params['nalph']*[None])
        params['lambdas_m'] = np.array(params['nlam']*[0]) # np.hstack((0, np.logspace(-5, 6, 40)))
        params['lr_w'] = [1e-5, 1e-3]
        params['lr_m'] = [1e-6, 1e-3]
        params['lr_b'] = [1e-6, 1e-2]


    hidden_size = 2
    params['lr_shift'] = [1e-3,1e-1]
    meanbias = torch.mean(torch.tensor(data['model_nsp'], dtype=torch.float32), axis=0)
    params['Nepochs'] = params['Nepochs']
    params['LinMix'] = LinMix
    params['Ncells'] = ytr.shape[-1]

    return params, xtr, xtrm, xte, xtem, ytr, yte, shift_in_tr, shift_in_te, input_size, hidden_size, output_size, model_type, meanbias


if __name__ == '__main__':
    args = arg_parser()
    if args['load_ray']:
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
        )
    MovModel = args['MovModel']
    free_move = args['free_move']
    free_move = True
    if free_move:
        stim_type = 'fm1'
    else:
        stim_type = 'hf1_wn'  # 'fm1' #
    # 128: 070921/J553RT
    # '110421/J569LT'# #'062921/G6HCK1ALTRN' '102621/J558NC'  # '110421/J569LT' #
    # date_ani = '110421/J569LT'# args['date_ani']
    # date_ani = '102621/J558NC' 
    date_ani = '070921/J553RT' 
    date_ani2 = '_'.join(date_ani.split('/'))
    data_dir = Path(args['data_dir']).expanduser() / date_ani / stim_type 
    save_dir = (Path(args['save_dir']).expanduser() / date_ani/ stim_type)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_fm = save_dir.parent / 'fm1'
    save_dir_hf = save_dir.parent / 'hf1_wn'
    fig_dir = (Path(args['fig_dir']).expanduser()/'Encoding'/date_ani/stim_type)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_model = (save_dir/ 'models' / '{}'.format(MovModel))
    save_model.mkdir(parents=True, exist_ok=True)

    file_dict = {'cell': 0,
                'drop_slow_frames': True,
                'ephys': list(data_dir.glob('*ephys_merge.json'))[0].as_posix(),
                'ephys_bin': list(data_dir.glob('*Ephys.bin'))[0].as_posix(),
                'eye': list(data_dir.glob('*REYE.nc'))[0].as_posix(),
                'imu': list(data_dir.glob('*imu.nc'))[0].as_posix() if stim_type == 'fm1' else None,
                'mapping_json': Path('~/Research/Github/FreelyMovingEphys/probes/channel_maps.json').expanduser(),
                'mp4': True,
                'name': date_ani2 + '_control_Rig2_' + stim_type,  # 070921_J553RT
                'probe_name': 'DB_P128-6',
                'save': data_dir.as_posix(),
                'speed': list(data_dir.glob('*speed.nc'))[0].as_posix() if stim_type == 'hf1_wn' else None,
                'stim_type': 'light',
                'top': list(data_dir.glob('*TOP1.nc'))[0].as_posix() if stim_type == 'fm1' else None,
                'world': list(data_dir.glob('*world.nc'))[0].as_posix(), }
    params = {
        'Nepochs': args['Nepochs'],
        'model_dt': .05,
        'do_shuffle': False,
        'do_norm': False,
        'do_worldcam_correction': False,
        'lag_list': np.array([-2, -1, 0, 1, 2]),
        'free_move': free_move,
        'save_dir': save_dir,
        'save_dir_fm': save_dir_fm,
        'save_dir_hf': save_dir_hf,
        'data_dir': data_dir,
        'fig_dir': fig_dir,
        'save_model': save_model,
        'shiftn': 7, 
        'train_shifter': False,
        'MovModel': MovModel,
        'load_Vis' : True if MovModel==2 else False,
        'LinMix': False if args['LinNonLinMix']==True else args['LinMix'],
        'LinNonLinMix': args['LinNonLinMix'],
        'NKfold': args['NKfold'],
        'NoL1': args['NoL1'],
        'hidden_size': 128,
    }
    params['nt_glm_lag']=len(params['lag_list'])
    Nepochs = params['Nepochs']
    if params['do_worldcam_correction']:
        params['WC_type'] = 'C'
    else:
        params['WC_type'] = 'UC'

    data, train_idx_list, test_idx_list = load_train_test(file_dict, **params)


    # Initialize movement combinations
    titles = np.array(['Theta', 'Phi', 'Roll', 'Pitch'])
    titles_all = []
    for n in range(1, len(titles)+1):
        perms = np.array(list(itertools.combinations(np.arange(len(titles)), n)))
        for ind in range(perms.shape[0]):
            titles_all.append('_'.join([t for t in titles[perms[ind]]]))


    print(params)
    a=0
    for Kfold in np.arange(args['NKfold']):
        data, train_idx_list, test_idx_list = load_train_test(file_dict, **params)

        ##### Set Train Test Splits #####
        train_idx = train_idx_list[Kfold]
        test_idx = test_idx_list[Kfold]
        data = load_Kfold_data(data,train_idx,test_idx,params)
        locals().update(data)

        params, xtr, xtrm, xte, xtem, ytr, yte, shift_in_tr, shift_in_te, input_size, hidden_size, output_size, model_type, meanbias = load_GLM_data(data, params, train_idx, test_idx)
        print('Model: {}, move_features: {}'.format(MovModel, params['move_features']))

        loss_regcv = np.zeros((params['nalph'], params['nlam'], output_size))
        pred_cv = np.zeros((params['nalph'], params['nlam'], output_size, xte.shape[0]), dtype=np.float32)
        out_cv = np.zeros((params['nalph'], params['nlam'], output_size, xtr.shape[0]), dtype=np.float32)
        tloss_trace_all = np.zeros((params['nalph'], params['nlam'], output_size,params['Nepochs']), dtype=np.float32)
        vloss_trace_all = np.zeros((params['nalph'], params['nlam'], output_size,params['Nepochs']), dtype=np.float32)

        # pbar = tqdm(params['lambdas_m'])
        l1 = VisNetwork(input_size, hidden_size, output_size, reg_alph=params['alphas'][a],
                        move_features=params['move_features'], train_shifter=False, 
                        LinMix=params['LinMix'], device=device,).to(device)
        # torch.nn.init.zeros_(l1.visNN.Layer0.weight)
        state_dict = l1.state_dict()
        l1_params_cv = {}
        for name, NN_param in l1.named_parameters():
            l1_params_cv[name] = []
        # visNN_traces = l1_params_cv.copy()
        pbar = tqdm(params['lambdas'])
        for l, reg_lam in enumerate(pbar):
            l1 = VisNetwork(input_size, hidden_size, output_size, reg_alph=params['alphas'][a],
                            move_features=params['move_features'], train_shifter=False, 
                            LinMix=params['LinMix'], device=device,).to(device)
            # if (l == 0) & (MovModel==1):
            #     torch.nn.init.zeros_(l1.visNN.Layer0.weight)
            # state_dict['visNN.Layer1.bias'] = meanbias
            l1.load_state_dict(state_dict)
            if MovModel == 1:
                optimizer = optim.Adam(params=[{'params': [p],'lr':params['lr_w'][0],'weight_decay':params['lambdas'][l]} for name, p in l1.visNN.named_parameters() if '0.weight' in name] + \
                                                [{'params': [p],'lr':params['lr_w'][1]}  for name, p in l1.visNN.named_parameters() if '2.weight' in name] + \
                                                [{'params': [p],'lr':params['lr_b'][0]} for name, p in l1.visNN.named_parameters() if 'bias' in name])
            elif MovModel == 2:
                
                optimizer = optim.Adam(params=[{'params': [param for name,param in l1.posNN.named_parameters() if 'weight' in name],'lr':params['lr_w'][1],'weight_decay':params['lambdas'][l]},
                                                {'params': [param for name,param in l1.posNN.named_parameters() if 'bias' in name],'lr':params['lr_b'][1]},])

            model_name = 'GLM_{}_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_alph{}_lam{}_Kfold{}.pth'.format(model_type, params['WC_type'], int(params['model_dt']*1000), params['nt_glm_lag'], MovModel, params['Nepochs'], a, l, Kfold)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(params['Nepochs']/10))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.995)
            vloss_trace = np.zeros((ytr.shape[-1], params['Nepochs']), dtype=np.float32)
            tloss_trace = np.zeros((ytr.shape[-1], params['Nepochs']), dtype=np.float32)

            # if pbar is None:
            #     pbar = pbar2 = tqdm(np.arange(Nepochs))
            # else:
            pbar2 = tqdm(np.arange(params['Nepochs']))
            for batchn in pbar2:
                out = l1(xtr, xtrm, shift_in_tr)
                loss = l1.loss(out, ytr)
                pred = l1(xte, xtem, shift_in_te)
                val_loss = l1.loss(pred, yte)
                vloss_trace[:,batchn] = val_loss.clone().cpu().detach().numpy()
                tloss_trace[:,batchn] = loss.clone().cpu().detach().numpy()
                pbar2.set_description('Loss: {:.03f}'.format(np.mean(val_loss.clone().cpu().detach().numpy())))
                pbar2.refresh()
                optimizer.zero_grad()
                loss.backward(torch.ones_like(loss))
                optimizer.step()
                scheduler.step()
                # for name, NN_param in l1.named_parameters():
                #     visNN_traces[name].append(NN_param.clone().detach().cpu().numpy())

            if (l == 0) & (MovModel==1):
                state_dict = l1.state_dict()
            tloss_trace_all[a, l] = tloss_trace
            vloss_trace_all[a, l] = vloss_trace
            for name, NN_p in l1.named_parameters():
                l1_params_cv[name].append(NN_p.clone().detach().cpu().numpy())
            pred = l1(xte, xtem)
            loss_regcv[a, l] = l1.loss(pred, yte).cpu().detach().numpy()
            pred_cv[a, l] = pred.detach().cpu().numpy().squeeze().T
            out = l1(xtr, xtrm)
            out_cv[a, l] = out.detach().cpu().numpy().squeeze().T



            torch.save({'model_state_dict': l1.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,}, save_model / model_name)

        for name in l1_params_cv.keys():
            l1_params_cv[name] = np.stack(l1_params_cv[name])


        if (MovModel == 1):
            loss_regcv[:, 0, :] = np.nan
        malph, mlam, cellnum = np.where(loss_regcv == np.nanmin(loss_regcv, axis=(0, 1), keepdims=True))
        cellnum, m_cinds = np.unique(cellnum, return_index=True)
        malph = malph[m_cinds]
        mlam = mlam[m_cinds]
        sortinds = cellnum.argsort()
        cellnum = cellnum[sortinds]
        malph = malph[sortinds]
        mlam = mlam[sortinds]

        visNN_best = {}
        posNN_best = {}
        for name in l1_params_cv.keys():
            visNN_best[name] = l1_params_cv[name][mlam]
                # posNN_best[name] = l1_params_cv[name][mlam]
                
        pred_test = pred_cv[malph, mlam, cellnum]
        pred_train = out_cv[malph, mlam, cellnum]
        tloss_trace_all2 = tloss_trace_all[malph, mlam, cellnum]
        vloss_trace_all2 = vloss_trace_all[malph, mlam, cellnum]

        output_size = len(cellnum)
        bin_length = 40
        r2_test = np.zeros((output_size))
        for celln in range(output_size):
            sp_smooth = ((np.convolve(data['test_nsp'][:, celln], np.ones(bin_length), 'same')) / (bin_length * params['model_dt']))[bin_length:-bin_length]
            pred_smooth = ((np.convolve(pred_test[celln], np.ones(bin_length), 'same')) / (bin_length * params['model_dt']))[bin_length:-bin_length]
            r2_test[celln] = (np.corrcoef(sp_smooth, pred_smooth)[0, 1])**2


        
        if MovModel == 0:
            GLM_Data = {'r2_test': r2_test,
                        'test_nsp': data['test_nsp'],
                        'pred_test': pred_test,
                        'pred_train': pred_train,
                        'tloss_trace_all': tloss_trace_all2,
                        'vloss_trace_all': vloss_trace_all2,
                        'loss_regcv': loss_regcv,
                        'visNN': visNN_best,
                        'posNN': posNN_best,
                        }
        elif MovModel == 1:
            GLM_Data = {'r2_test': r2_test,
                        'test_nsp': data['test_nsp'],
                        'pred_train': pred_train,
                        'pred_test': pred_test,
                        'tloss_trace_all': tloss_trace_all2,
                        'vloss_trace_all': vloss_trace_all2,
                        'loss_regcv': loss_regcv,
                        'visNN': visNN_best,
                        }
        else:
            GLM_Data = {'r2_test': r2_test,
                        'test_nsp': data['test_nsp'],
                        'pred_test': pred_test,
                        'pred_train': pred_train,
                        'tloss_trace_all': tloss_trace_all2,
                        'vloss_trace_all': vloss_trace_all2,
                        'loss_regcv': loss_regcv,
                        'visNN': visNN_best,
                        'posNN': posNN_best,
                        }
        if params['do_shuffle']:
            save_datafile = save_dir/'GLM_{}_VisMov_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}_shuffled.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], MovModel, params['Nepochs'],Kfold)
        else:
            save_datafile = save_dir/'GLM_{}_VisMov_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], MovModel, params['Nepochs'], Kfold)
        ioh5.save(save_datafile, GLM_Data)
