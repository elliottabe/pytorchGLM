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
    parser.add_argument('--Nepochs', type=int, default=5000)
    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    return vars(args)

def get_model(input_size, output_size, sta_init, meanbias, MovModel, device, l, a, params, NepochVis=12000, Kfold=0,**kwargs):

    if params['load_Vis']:
        GLM_Vis = ioh5.load(params['save_dir']/'GLM_{}_VisMov_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}.h5'.format('Pytorch_Vis',int(params['model_dt']*1000), params['nt_glm_lag'], 1, NepochVis,Kfold))
        sta_init=torch.from_numpy(GLM_Vis['sta_all'].reshape(GLM_Vis['test_nsp'].shape[-1],-1).astype(np.float32))
        meanbias=torch.from_numpy(GLM_Vis['bias'].astype(np.float32))

    l1 = PoissonGLM_AddMult(input_size,output_size,
                            reg_alph=params['alphas'][a],reg_alphm=params['alphas_m'][a],move_features=params['move_features'],
                            meanfr=meanbias,init_sta=sta_init,
                            train_shifter=params['train_shifter'],
                            LinMix=params['LinMix'], NonLinLayer=params['NonLinLayer'], 
                            LinNonLinMix=params['LinNonLinMix'],
                            device=device,).to(device)
    if MovModel == 0: 
        optimizer = optim.Adam(params=[{'params': [l1.weight],'lr': params['lr_w'][1],},
                                       {'params': [l1.bias],'lr':params['lr_b'][1]},])
    elif MovModel == 1:
        if params['train_shifter']:
            optimizer = optim.Adam(params=[{'params': [l1.weight],'lr':params['lr_w'][1],'weight_decay':params['lambdas'][l]},
                                        {'params': [l1.bias],'lr':params['lr_b'][1]},
                                        {'params': list(l1.shifter_nn.parameters()),'lr': params['lr_shift'][1],'weight_decay':.0001}])
        else:
            optimizer = optim.Adam(params=[{'params': [l1.weight],'lr':params['lr_w'][1],'weight_decay':params['lambdas'][l]},
                                {'params': [l1.bias],'lr':params['lr_b'][1]},])
    else:            
        if params['load_Vis']:
            if params['NonLinLayer']: 
                # optimizer = optim.Adam(params=[{'params': [l1.move_weights],'lr':params['lr_m'][1], 'weight_decay': params['lambdas_m'][l]},
                #                             {'params': [l1.bias_m],'lr':params['lr_b'][1]},]) 
                # optimizer = optim.Adam(params=[{'params': list(l1.NonLinMixLayer.parameters()),'lr':params['lr_m'][1], 'weight_decay': params['lambdas_m'][l]},
                #                             # {'params': [l1.bias_m],'lr':params['lr_b'][1]},
                #                             ]) 
                optimizer = optim.Adam(params=[{'params': [l1.NonLinMixLayer[0].weight], 'lr': params['lr_m'][1], 'weight_decay': params['lambdas_m'][l]},
                                            {'params': [l1.NonLinMixLayer[0].bias],'lr':params['lr_b'][1]},
                                            {'params': [l1.NonLinMixLayer[2].weight], 'lr': params['lr_m'][1], 'weight_decay': params['lambdas_m'][l]},
                                            {'params': [l1.NonLinMixLayer[2].bias],'lr':params['lr_b'][1]},
                                            ])
            elif params['LinNonLinMix']:
                optimizer = optim.Adam(params=[{'params': [l1.moveW_mul],'lr':params['lr_m'][1], 'weight_decay': params['lambdas_m'][l]},
                                                {'params': [l1.moveW_add],'lr':params['lr_m'][1], 'weight_decay': params['lambdas_m'][l]},
                                                {'params': [l1.biasm_mul],'lr':params['lr_b'][1]},
                                                {'params': [l1.biasm_add],'lr':params['lr_b'][1]},
                                                {'params': [l1.gamma], 'lr':1e-3}])     
            else:
                optimizer = optim.Adam(params=[{'params': [l1.move_weights],'lr':params['lr_m'][1], 'weight_decay': params['lambdas_m'][l]},
                                                {'params': [l1.bias_m],'lr':params['lr_b'][1]},]) 
        else:
            if params['LinMix']:
                optimizer = optim.Adam(params=[{'params': [l1.weight],'lr':params['lr_w'][1],'weight_decay':params['lambdas'][l]},
                                            {'params': [l1.bias],'lr':params['lr_b'][1]},
                                            {'params': [l1.move_weights],'lr':params['lr_m'][1], 'weight_decay': params['lambdas_m'][l]},
                                            {'params': [l1.bias_m], 'lr':params['lr_b'][1]}, ])
            else:
                optimizer = optim.Adam(params=[{'params': [l1.weight],'lr':params['lr_w'][1],'weight_decay':params['lambdas'][l]},
                                                {'params': [l1.bias],'lr':params['lr_b'][1]},
                                                {'params': [l1.move_weights],'lr':params['lr_m'][1], 'weight_decay': params['lambdas_m'][l]},
                                                {'params': [l1.bias_m],'lr':params['lr_b'][1]},]) 

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(params['Nepochs']/4))
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9999)
    # scheduler = None
    return l1, optimizer, scheduler


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


if __name__ == '__main__':
    args = arg_parser()
    if args['load_ray']:
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
        )
    MovModel = args['MovModel']
    free_move = args['free_move']
    fm_dir = 'fm1'
    # fm_dir = 'fm1_prey'
    if free_move:
        stim_type = fm_dir
    else:
        stim_type = 'hf1_wn'  # 'fm1' #
    # 012821/EE8P6LT
    # 128: 070921/J553RT
    # '110421/J569LT'# #'062921/G6HCK1ALTRN' '102621/J558NC' '110421/J569LT' #' 110421/J569LT' #
    date_ani = args['date_ani'] 
    date_ani2 = '_'.join(date_ani.split('/'))
    data_dir = Path(args['data_dir']).expanduser() / date_ani / stim_type 
    save_dir = (Path(args['save_dir']).expanduser() / date_ani/ stim_type)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_fm = save_dir.parent / fm_dir
    save_dir_hf = save_dir.parent / 'hf1_wn'
    fig_dir = (Path(args['fig_dir']).expanduser()/'Encoding'/date_ani/stim_type)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_model = (save_dir/ 'models' / '{}'.format(MovModel))
    save_model.mkdir(parents=True, exist_ok=True)

    # file_dict={}
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

    params = {
        'Nepochs': args['Nepochs'],
        'model_dt': .05,
        'do_shuffle': False,
        'do_norm': args['do_norm'],
        'lag_list': np.array([0]) if args['train_shifter'] else np.array([-2, -1, 0, 1, 2]),
        'free_move': free_move,
        'save_dir': save_dir,
        'save_dir_fm': save_dir_fm,
        'save_dir_hf': save_dir_hf,
        'data_dir': data_dir,
        'fig_dir': fig_dir,
        'save_model': save_model,
        'shiftn': args['shiftn'], # 7, 
        'train_shifter': args['train_shifter'],
        'MovModel': MovModel,
        'load_Vis' : True if MovModel==2 else False,
        'LinMix': False if args['LinNonLinMix']==True else args['LinMix'],
        'LinNonLinMix': args['LinNonLinMix'],
        'NKfold': args['NKfold'],
        'quantiles': [.05,.95],
    }
    params['nt_glm_lag']=len(params['lag_list'])
    Nepochs = params['Nepochs']

    params['WC_type'] = 'UC'


    LinMix = params['LinMix']
    LinNonLinMix = params['LinNonLinMix']
    if MovModel == 0:
        model_type = 'Pytorch_Mot'
    elif MovModel == 1:
        model_type = 'Pytorch_Vis'
    elif MovModel == 2:
        if LinMix:
            model_type = 'Pytorch_Add'
        elif LinNonLinMix:
            model_type = 'Pytorch_AddMul'
        else:
            model_type = 'Pytorch_Mul'

    if params['train_shifter']:
        save_model = save_model.parent / 'Shifter'
        save_model.mkdir(parents=True, exist_ok=True)
        args['NoL1'] = True
        model_type = model_type + 'Shifter'

    if args['NoL1']:
        model_type = model_type + '_NoL1'
    print(model_type)
    # Initialize movement combinations
    titles = np.array(['Theta', 'Phi', 'Roll', 'Pitch'])
    titles_all = []
    for n in range(1, len(titles)+1):
        perms = np.array(list(itertools.combinations(np.arange(len(titles)), n)))
        for ind in range(perms.shape[0]):
            titles_all.append('_'.join([t for t in titles[perms[ind]]]))


    print(params)
    
    for Kfold in np.arange(args['NKfold']):
        data, train_idx_list, test_idx_list = load_train_test(file_dict, **params)

        ##### Set up Train/Test Splits #####
        train_idx = train_idx_list[Kfold]
        test_idx = test_idx_list[Kfold]
        data = load_Kfold_data(data,train_idx,test_idx,params)
        locals().update(data)

        # avgfm_eye = np.load(save_dir_fm/'FM_AvgEye_dt{:03d}.npy'.format(int(params['model_dt']*1000)))
        if free_move:
            move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis],data['train_roll'][:, np.newaxis], data['train_pitch'][:, np.newaxis]))
            move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis],data['test_roll'][:, np.newaxis], data['test_pitch'][:, np.newaxis]))
            model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis],data['model_roll'][:, np.newaxis], data['model_pitch'][:, np.newaxis]))
        else:
            move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis], np.zeros(data['train_phi'].shape)[:, np.newaxis], np.zeros(data['train_phi'].shape)[:, np.newaxis]))
            move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis], np.zeros(data['test_phi'].shape)[:, np.newaxis], np.zeros(data['test_phi'].shape)[:, np.newaxis]))
            model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis], np.zeros(data['model_phi'].shape)[:, np.newaxis], np.zeros(data['model_phi'].shape)[:, np.newaxis]))



        ##### Start GLM Parallel Processing #####
        nks = np.shape(data['train_vid'])[1:]
        nk = nks[0]*nks[1]*params['nt_glm_lag']
        perms = np.array(list(itertools.combinations(np.arange(len(titles)), 4)))
        # Reshape data (video) into (T*n)xN array
        if params['train_shifter']:
            rolled_vid = np.hstack([np.roll(data['model_vid_sm'], nframes, axis=0) for nframes in params['lag_list']])
            move_quantiles = np.quantile(model_move,params['quantiles'],axis=0)
            train_range = np.all(((move_train>move_quantiles[0]) & (move_train<move_quantiles[1])),axis=1)
            test_range = np.all(((move_test>move_quantiles[0]) & (move_test<move_quantiles[1])),axis=1)
            x_train = rolled_vid[train_idx].reshape((len(train_idx), 1)+nks).astype(np.float32)[train_range]
            x_test = rolled_vid[test_idx].reshape((len(test_idx), 1)+nks).astype(np.float32)[test_range]
            move_train = move_train[train_range]
            move_test = move_test[test_range]
            shift_in_tr = torch.from_numpy(move_train[:, (0, 1, 3)].astype(np.float32)).to(device)
            shift_in_te = torch.from_numpy(move_test[:, (0, 1, 3)].astype(np.float32)).to(device)
            ytr = torch.from_numpy(data['train_nsp'][train_range].astype(np.float32)).to(device)
            yte = torch.from_numpy(data['test_nsp'][test_range].astype(np.float32)).to(device)
            data['train_nsp']=data['train_nsp'][train_range]
            data['test_nsp']=data['test_nsp'][test_range]
        else:
            model_vid_sm_shift = ioh5.load(save_dir/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(int(params['model_dt']*1000), 1))['model_vid_sm_shift{}'.format(params['shiftn'])]  # [:,5:-5,5:-5]
            nks = np.shape(model_vid_sm_shift)[1:]
            nk = nks[0]*nks[1]*params['nt_glm_lag']
            rolled_vid = np.hstack([np.roll(model_vid_sm_shift, nframes, axis=0) for nframes in params['lag_list']])  
            x_train = rolled_vid[train_idx].reshape(len(train_idx), -1)
            x_test = rolled_vid[test_idx].reshape(len(test_idx), -1)
            shift_in_tr = None
            shift_in_te = None
            ytr = torch.from_numpy(data['train_nsp'].astype(np.float32)).to(device)
            yte = torch.from_numpy(data['test_nsp'].astype(np.float32)).to(device)

        input_size = nk
        output_size = ytr.shape[1]

        # Reshape data (video) into (T*n)xN array
        if MovModel == 0:
            mx_train = move_train[:, perms[ind]]
            mx_test = move_test[:, perms[ind]]
            xtr = torch.from_numpy(mx_train.astype(np.float32)).to(device)
            xte = torch.from_numpy(mx_test.astype(np.float32)).to(device)
            xtrm = None
            xtem = None
            nk = xtr.shape[-1]
            input_size = nk
            params['move_features'] = None 
            sta_init = torch.zeros((output_size, xtr.shape[-1]))
            params['lambdas'] = np.array([None])
            params['alphas'] = np.array([None])
            params['lambdas_m'] = np.array([None]) 
            params['alphas_m'] = np.array([None]) 
            nlam = len(params['lambdas'])
            nalph = len(params['alphas'])
            params['alphas_m'] = np.array(nalph*[None])
            params['lambdas_m'] = np.array([0])
            w_move_cv = np.zeros((nalph, nlam, output_size, nk))
            w_move_traces_all = np.zeros((nalph, nlam, params['Nepochs'], output_size, input_size))
            params['lr_w'] = [1e-5, 1e-3]
            params['lr_m'] = [1e-6, 1e-2]
            params['lr_b'] = [1e-6, 1e-2]
        elif MovModel == 1:
            xtr = torch.from_numpy(x_train).to(device)
            xte = torch.from_numpy(x_test).to(device)
            xtrm = None
            xtem = None
            params['move_features'] = None
            sta_init = torch.zeros((output_size, xtr.shape[-1]))
            params['alphas'] = np.array([.009 if args['NoL1']==False else None])
            params['lambdas'] = np.hstack((np.logspace(-3, 3.25, 20)[0], np.logspace(-3, 4, 40)))
            nlam = len(params['lambdas'])
            nalph = len(params['alphas'])
            params['lambdas_m'] = np.array(nlam*[None])
            params['alphas_m'] = np.array(nalph*[None])
            params['lr_w'] = [1e-5, 1e-3]
            params['lr_b'] = [1e-5, 5e-3]
        elif MovModel == 2:
            xtr = torch.from_numpy(x_train.astype(np.float32)).to(device)
            xte = torch.from_numpy(x_test.astype(np.float32)).to(device)
            xtrm = torch.from_numpy(move_train[:, perms[ind]].astype(np.float32)).to(device)
            xtem = torch.from_numpy(move_test[:, perms[ind]].astype(np.float32)).to(device)
            params['move_features'] = xtrm.shape[-1]
            sta_init = torch.zeros((output_size, xtr.shape[-1]))
            params['alphas'] = np.array([None])
            params['lambdas'] =  np.hstack((np.logspace(-3, 3.25, 10)[0], np.logspace(-3, 4, 40)))
            nalph = len(params['alphas'])
            params['alphas_m'] = np.array(nalph*[None])
            params['lambdas_m'] = np.array([0]) # np.hstack((0, np.logspace(-5, 6, 40)))
            nlam = len(params['lambdas_m'])
            if LinNonLinMix:
                w_move_cv_add = np.zeros((nalph, nlam, output_size, params['move_features']))
                w_move_cv_mul = np.zeros((nalph, nlam, output_size, params['move_features']))
            else:
                w_move_cv = np.zeros((nalph, nlam, output_size, params['move_features']))

            params['lr_w'] = [1e-5, 1e-3]
            params['lr_m'] = [1e-6, 5e-3]
            params['lr_b'] = [1e-6, 1e-2]

        params['lr_shift'] = [1e-3,1e-1]
        meanbias = torch.mean(torch.tensor(data['model_nsp'], dtype=torch.float32), axis=0)
        params['Nepochs'] = params['Nepochs']
        params['LinMix'] = LinMix
        params['NonLinLayer'] = args['NonLinLayer']
        params['Ncells'] = ytr.shape[-1]
        print('Model: {}, move_features: {}'.format(MovModel, params['move_features']))

        msetrain = np.zeros((nalph, nlam, output_size))
        loss_regcv = np.zeros((nalph, nlam, output_size))
        pred_cv = np.zeros((x_test.shape[0], nalph, nlam, output_size), dtype=np.float32)
        out_cv = np.zeros((x_train.shape[0], nalph, nlam, output_size), dtype=np.float32)
        w_cv = np.zeros((input_size, nalph, nlam, output_size), dtype=np.float32)
        bias_cv = np.zeros((nalph, nlam, output_size), dtype=np.float32)
        if LinNonLinMix:
            biasm_cv_add = np.zeros((nalph, nlam, output_size), dtype=np.float32)
            biasm_cv_mul = np.zeros((nalph, nlam, output_size), dtype=np.float32)
        else:    
            biasm_cv = np.zeros((nalph, nlam, output_size), dtype=np.float32)
        tloss_trace_all = np.zeros((nalph, nlam, Nepochs, output_size), dtype=np.float32)
        vloss_trace_all = np.zeros((nalph, nlam, Nepochs, output_size), dtype=np.float32)
        bias_traces_all = np.zeros((nalph, nlam, Nepochs, output_size), dtype=np.float32)
        if LinNonLinMix:
            gamma_trace_cv = np.zeros((nalph, nlam, Nepochs,output_size), dtype=np.float32)
        start = time.time()
        for a, reg_alph in enumerate(tqdm(params['alphas'],leave=True)):
            sta_init = torch.zeros((output_size,input_size))
            if MovModel == 1:
                pbar = tqdm(params['lambdas'])
            elif (MovModel != 1): 
                pbar = tqdm(params['lambdas_m'])
            for l, reg_lam in enumerate(pbar):
                l1,optimizer,scheduler = get_model(input_size, output_size, sta_init, meanbias, MovModel, device, l, a, params, Kfold=Kfold)
                model_name = 'GLM_{}_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_alph{}_lam{}_Kfold{}.pth'.format(model_type, params['WC_type'], int(params['model_dt']*1000), params['nt_glm_lag'], MovModel, params['Nepochs'], a, l, Kfold)

                vloss_trace = np.zeros((Nepochs, output_size), dtype=np.float32)
                tloss_trace = np.zeros((Nepochs, output_size), dtype=np.float32)
                if LinNonLinMix:
                    gamma_trace = np.zeros((Nepochs,output_size),dtype=np.float32)
                for batchn in np.arange(Nepochs):
                    out = l1(xtr,xtrm,shift_in_tr)
                    loss = l1.loss(out, ytr)
                    pred = l1(xte, xtem,shift_in_te)
                    val_loss = l1.loss(pred, yte)
                    vloss_trace[batchn] = val_loss.clone().cpu().detach().numpy()
                    tloss_trace[batchn] = loss.clone().cpu().detach().numpy()
                    bias_traces_all[a, l, batchn] = l1.bias.clone().cpu().detach().numpy()
                    if LinNonLinMix:
                        gamma_trace[batchn] = l1.gamma.clone().cpu().detach().numpy()
                    pbar.set_description('Loss: {:.03f}'.format(np.mean(val_loss.clone().cpu().detach().numpy())))
                    pbar.refresh()
                    optimizer.zero_grad()
                    loss.backward(torch.ones_like(loss))
                    optimizer.step()
                    scheduler.step()
                tloss_trace_all[a, l] = tloss_trace
                vloss_trace_all[a, l] = vloss_trace
                bias_cv[a, l] = l1.bias.clone().cpu().detach().numpy()
                if MovModel != 0:
                    w_cv[:, a, l] = l1.weight.clone().cpu().detach().numpy().T 
                if MovModel == 0:
                    w_move_cv[a, l] = l1.weight.clone().cpu().detach().numpy()  
                elif MovModel != 1:
                    if LinNonLinMix:
                        gamma_trace_cv[a,l] = gamma_trace
                        w_move_cv_add[a,l] = l1.moveW_add.clone().cpu().detach().numpy()
                        w_move_cv_mul[a,l] = l1.moveW_mul.clone().cpu().detach().numpy()
                        biasm_cv_add[a,l] = l1.biasm_add.clone().cpu().detach().numpy()
                        biasm_cv_mul[a,l] = l1.biasm_mul.clone().cpu().detach().numpy()
                    else:
                        biasm_cv[a,l] = l1.bias_m.clone().cpu().detach().numpy()
                        w_move_cv[a, l] = l1.move_weights.clone().cpu().detach().numpy()  
                pred = l1(xte, xtem, shift_in_te)
                loss_regcv[a, l] = l1.loss(pred, yte).cpu().detach().numpy()

                pred_cv[:, a, l] = pred.detach().cpu().numpy().squeeze()
                out = l1(xtr, xtrm, shift_in_tr)
                out_cv[:, a, l] = out.detach().cpu().numpy().squeeze()

                if (a == 0) & (l == 0) & (MovModel==1):
                    sta_init = l1.weight.clone().detach()
                torch.save({'reg_alph': reg_alph,'reg_lam': reg_lam,'model_state_dict': l1.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,}, save_model / model_name)
        print('GLM: ', time.time()-start)


        if MovModel == 1:
            loss_regcv[:, 0, :] = np.nan
        malph, mlam, cellnum = np.where(loss_regcv == np.nanmin(loss_regcv, axis=(0, 1), keepdims=True))
        cellnum, m_cinds = np.unique(cellnum, return_index=True)
        malph = malph[m_cinds]
        mlam = mlam[m_cinds]
        sortinds = cellnum.argsort()
        cellnum = cellnum[sortinds]
        malph = malph[sortinds]
        mlam = mlam[sortinds]
        pred_test = pred_cv[:, malph, mlam, cellnum]
        pred_train = out_cv[:, malph, mlam, cellnum]
        bias = bias_cv[malph, mlam, cellnum]
        tloss_trace_all2 = tloss_trace_all[malph, mlam, :, cellnum]
        vloss_trace_all2 = vloss_trace_all[malph, mlam, :, cellnum]
        bias_traces = bias_traces_all[malph, mlam, :, cellnum]

        output_size = len(cellnum)
        if MovModel != 0:
            sta_all = w_cv[:, malph, mlam, cellnum].T.reshape((len(cellnum), params['nt_glm_lag'],)+nks)
        if MovModel != 1:
            if LinNonLinMix:
                w_move_add = w_move_cv_add[malph, mlam, cellnum]
                biasm_add = biasm_cv_add[malph,mlam,cellnum]
                w_move_mul = w_move_cv_mul[malph, mlam, cellnum]
                biasm_mul = biasm_cv_mul[malph,mlam,cellnum]
                gamma_trace_all = gamma_trace_cv[malph,mlam,:,cellnum]
            else:    
                w_move = w_move_cv[malph, mlam, cellnum]
                biasm = biasm_cv[malph,mlam,cellnum]
        bin_length = 40
        r2_test = np.zeros((output_size))
        for celln in range(output_size):
            sp_smooth = ((np.convolve(data['test_nsp'][:, celln], np.ones(bin_length), 'same')) / (bin_length * params['model_dt']))[bin_length:-bin_length]
            pred_smooth = ((np.convolve(pred_test[:, celln], np.ones(bin_length), 'same')) / (bin_length * params['model_dt']))[bin_length:-bin_length]
            r2_test[celln] = (np.corrcoef(sp_smooth, pred_smooth)[0, 1])**2

        if MovModel == 0:
            GLM_Data = {'r2_test': r2_test,
                        'test_nsp': data['test_nsp'],
                        'pred_test': pred_test,
                        'bias': bias,
                        'tloss_trace_all': tloss_trace_all2,
                        'vloss_trace_all': vloss_trace_all2,
                        'loss_regcv': loss_regcv,
                        'pred_train': pred_train,
                        'w_move': w_move}
        elif MovModel == 1:
            GLM_Data = {'r2_test': r2_test,
                        'sta_all': sta_all,
                        'test_nsp': data['test_nsp'],
                        'pred_test': pred_test,
                        'bias': bias,
                        'tloss_trace_all': tloss_trace_all2,
                        'vloss_trace_all': vloss_trace_all2,
                        'loss_regcv': loss_regcv,
                        'pred_train': pred_train,
                        }
        else:
            if LinNonLinMix:
                GLM_Data = {'r2_test': r2_test,
                            'sta_all': sta_all,
                            'test_nsp': data['test_nsp'],
                            'pred_test': pred_test,
                            'bias': bias,
                            'tloss_trace_all': tloss_trace_all2,
                            'vloss_trace_all': vloss_trace_all2,
                            'loss_regcv': loss_regcv,
                            'pred_train': pred_train,
                            'w_move_add': w_move_add,
                            'biasm_add': biasm_add,
                            'w_move_mul': w_move_mul,
                            'biasm_mul': biasm_mul,
                            'gamma_trace_all': gamma_trace_all,
                            }
            else:
                GLM_Data = {'r2_test': r2_test,
                            'sta_all': sta_all,
                            'test_nsp': data['test_nsp'],
                            'pred_test': pred_test,
                            'bias': bias,
                            'biasm': biasm,
                            'tloss_trace_all': tloss_trace_all2,
                            'vloss_trace_all': vloss_trace_all2,
                            'loss_regcv': loss_regcv,
                            'pred_train': pred_train,
                            'w_move': w_move}
                            
        if params['do_shuffle']:
            save_datafile = save_dir/'GLM_{}_VisMov_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}_shuffled.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], MovModel, Nepochs,Kfold)
        else:
            save_datafile = save_dir/'GLM_{}_VisMov_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], MovModel, Nepochs,Kfold)
        ioh5.save(save_datafile, GLM_Data)
        print(save_datafile)
