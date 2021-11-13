import argparse
import gc
import itertools
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr
from kornia.geometry.transform import Affine
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
from scipy.ndimage import shift as imshift
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import medfilt
from scipy.stats import binned_statistic
from sklearn import linear_model as lm
from sklearn.metrics import mean_poisson_deviance, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms
from tqdm.auto import tqdm

sys.path.append(str(Path('.').absolute()))
import io_dict_to_hdf5 as ioh5
from format_data import *
from models import *
from utils import *


torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ray.init(
    ignore_reinit_error=True,
    logging_level=logging.ERROR,
)


def arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--free_move', type=bool, default=True)
    parser.add_argument('--date_ani', type=str, default='070921/J553RT')
    parser.add_argument('--source_path', type=str,
                        default='~/Research/PredAudio/')
    parser.add_argument('--outputmode', type=str, default='prediction',
                        help='Chose a loss function: error, prediction')


    args = parser.parse_args()
    return args


def get_model(input_size, output_size, MovModel, move_features, meanbias, sta_init, device, LinNetwork,lr_w,lr_b,lr_m):
    if MovModel == 0:
        l1 = PoissonGLM_VM_staticreg(input_size, output_size, reg_lam=None, reg_alph=None, move_features=move_features,
                                        meanfr=meanbias, init_sta=sta_init, device=device, LinNetwork=LinNetwork).to(device)
        optimizer = optim.Adam(params=[{'params': [l1.weight], 'lr': 1e-3, },
                                        {'params': [l1.bias], 'lr':lr_b[1]}, ], lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(Nbatches/4))
    elif MovModel == 1:
        l1 = PoissonGLM_VM_staticreg(input_size, output_size, reg_lam=None, reg_alph=reg_alph, move_features=move_features,
                                        meanfr=meanbias, init_sta=sta_init, device=device, LinNetwork=LinNetwork).to(device)
        optimizer = optim.Adam(params=[{'params': [l1.weight], 'lr':lr_w[1], 'weight_decay':lambdas[l]},
                                        {'params': [l1.bias], 'lr':lr_b[1]}, ], lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(Nbatches/4))
    else:
        l1 = PoissonGLM_VM_staticreg(input_size, output_size, reg_lam=None, reg_alph=reg_alph, reg_alphm=alphas_m[
            a], move_features=move_features, meanfr=meanbias, init_sta=sta_init, device=device, LinNetwork=LinNetwork).to(device)
        optimizer = optim.Adam(params=[{'params': [l1.weight], 'lr':lr_w[1], 'weight_decay':lambdas[l]},
                                        {'params': [l1.bias], 'lr':lr_b[1]},
            {'params': [l1.move_weights], 'lr':lr_m[1], 'weight_decay': lambdas_m[l]},
            {'params': [l1.bias_m], 'lr':lr_b[1]}, ])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(Nbatches/4))


if __name__ == '__main__':
    args = arg_parser()


    free_move = args['free_move']
    if free_move:
        stim_type = 'fm1'
    else:
        stim_type = 'hf1_wn'  # 'fm1' #
    # 012821/EE8P6LT
    # 128: 070921/J553RT
    # '110421/J569LT'# #'062921/G6HCK1ALTRN' '102621/J558NC' '110421/J569LT' #' 110421/J569LT' #
    date_ani = args['date_ani'] #'070921/J553RT'
    date_ani2 = '_'.join(date_ani.split('/'))
    data_dir = Path(
        '~/Goeppert/freely_moving_ephys/ephys_recordings/').expanduser() / date_ani / stim_type
    save_dir = check_path(Path(
        '~/Research/SensoryMotorPred_Data/data/').expanduser() / date_ani, stim_type)
    FigPath = check_path(
        Path('~/Research/SensoryMotorPred_Data').expanduser(), 'Figures/Encoding')
    FigPath = check_path(FigPath/date_ani, stim_type)
    save_model = check_path(save_dir, 'models')

    file_dict = {'cell': 0,
                'drop_slow_frames': True,
                'ephys': list(data_dir.glob('*ephys_merge.json'))[0].as_posix(),
                'ephys_bin': list(data_dir.glob('*Ephys.bin'))[0].as_posix(),
                'eye': list(data_dir.glob('*REYE.nc'))[0].as_posix(),
                'imu': list(data_dir.glob('*imu.nc'))[0].as_posix() if stim_type == 'fm1' else None,
                'mapping_json': '/home/seuss/Research/Github/FreelyMovingEphys/probes/channel_maps.json',
                'mp4': True,
                'name': date_ani2 + '_control_Rig2_'+stim_type,  # 070921_J553RT
                'probe_name': 'DB_P128-6',
                'save': data_dir.as_posix(),
                'speed': list(data_dir.glob('*speed.nc'))[0].as_posix() if stim_type == 'hf1_wn' else None,
                'stim_type': 'light',
                'top': list(data_dir.glob('*TOP1.nc'))[0].as_posix() if stim_type == 'fm1' else None,
                'world': list(data_dir.glob('*world.nc'))[0].as_posix(), }


    model_dt = .05
    do_shuffle = False
    do_norm = False
    do_worldcam_correction = False
    if do_worldcam_correction:
        WC_type = 'C'
    else:
        WC_type = 'UC'
    data, train_idx, test_idx = load_train_test(file_dict, save_dir, model_dt=model_dt, do_shuffle=do_shuffle, do_norm=do_norm,
                                                free_move=free_move, has_imu=free_move, has_mouse=False, do_worldcam_correction=do_worldcam_correction, reprocess=False)
    locals().update(data)

    Nbatches = 5000

    lag_list = np.array([-2, -1, 0, 1, 2])
    nt_glm_lag = len(lag_list)
    print(lag_list, 1000*lag_list*model_dt)
    do_shuffle = False
    # LinNetwork = False
    GLM_All_Data = {}
    for LinNetwork in [True, False]:
        for MovModel in [1, 2]:
            if LinNetwork:
                model_type = 'Pytorch_Lin'
            else:
                model_type = 'Pytorch_Nonlin'
            # Load Data
            data, train_idx, test_idx = load_train_test(file_dict, save_dir, model_dt=model_dt, do_shuffle=do_shuffle, do_norm=False,
                                                        free_move=free_move, has_imu=free_move, has_mouse=False, do_worldcam_correction=do_worldcam_correction)
            locals().update(data)


            # Initialize movement combinations
            titles = np.array(['Theta', 'Phi', 'Roll', 'Pitch']
                            )  # 'dg_p','dg_n' 'roll','pitch'
            titles_all = []
            for n in range(1, len(titles)+1):
                perms = np.array(
                    list(itertools.combinations(np.arange(len(titles)), n)))
                for ind in range(perms.shape[0]):
                    titles_all.append('_'.join([t for t in titles[perms[ind]]]))
            if free_move:
                move_train = np.hstack((train_th[:, np.newaxis], train_phi[:, np.newaxis],
                                    train_roll[:, np.newaxis], medfilt(train_pitch, 3)[:, np.newaxis]))
                move_test = np.hstack((test_th[:, np.newaxis], test_phi[:, np.newaxis],
                                    test_roll[:, np.newaxis], medfilt(test_pitch, 3)[:, np.newaxis]))
                model_move = np.hstack((model_th[:, np.newaxis], model_phi[:, np.newaxis],
                                    model_roll[:, np.newaxis], medfilt(model_pitch, 3)[:, np.newaxis]))
                shift_in_tr = torch.from_numpy(
                    move_train[:, (0, 1, 3)].astype(np.float32)).to(device)
                shift_in_te = torch.from_numpy(
                    move_test[:, (0, 1, 3)].astype(np.float32)).to(device)
                # model_move = model_move - np.mean(model_move,axis=0)
                # move_test = move_test - np.mean(move_test,axis=0)
                # move_train = move_train - np.mean(move_train,axis=0)
            else:
                move_train = np.hstack((train_th[:, np.newaxis], train_phi[:, np.newaxis], np.zeros(
                    train_phi.shape)[:, np.newaxis], np.zeros(train_phi.shape)[:, np.newaxis]))
                move_test = np.hstack((test_th[:, np.newaxis], test_phi[:, np.newaxis], np.zeros(
                    test_phi.shape)[:, np.newaxis], np.zeros(test_phi.shape)[:, np.newaxis]))
                model_move = np.hstack((model_th[:, np.newaxis], model_phi[:, np.newaxis], np.zeros(
                    model_phi.shape)[:, np.newaxis], np.zeros(model_phi.shape)[:, np.newaxis]))
                shift_in_tr = torch.from_numpy(
                    move_train[:, (0, 1, 3)].astype(np.float32)).to(device)
                shift_in_te = torch.from_numpy(
                    move_test[:, (0, 1, 3)].astype(np.float32)).to(device)

            ##### Start GLM Parallel Processing #####
            nks = np.shape(train_vid)[1:]
            nk = nks[0]*nks[1]*nt_glm_lag
            n = 4
            ind = 0
            perms = np.array(
                list(itertools.combinations(np.arange(len(titles)), n)))
            ##### Start GLM Parallel Processing #####
            # Reshape data (video) into (T*n)xN array
            if do_worldcam_correction:
                rolled_vid = np.hstack(
                    [np.roll(model_vid_sm, nframes, axis=0) for nframes in lag_list])  # nt_glm_lag
            else:
                shiftn = 7
                model_vid_sm_shift = ioh5.load(save_dir/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(
                    int(model_dt*1000), 1))['model_vid_sm_shift{}'.format(shiftn)]  # [:,5:-5,5:-5]
                nks = np.shape(model_vid_sm_shift)[1:]
                nk = nks[0]*nks[1]*nt_glm_lag
                rolled_vid = np.hstack([np.roll(
                    model_vid_sm_shift, nframes, axis=0) for nframes in lag_list])  # nt_glm_lag
            rolled_vid_flat = rolled_vid.reshape(rolled_vid.shape[0], -1)
            x_train = rolled_vid[train_idx].reshape(len(train_idx), -1)
            x_test = rolled_vid[test_idx].reshape(len(test_idx), -1)

            ytr = torch.from_numpy(train_nsp.astype(np.float32)).to(device)
            yte = torch.from_numpy(test_nsp.astype(np.float32)).to(device)

            input_size = nk
            output_size = ytr.shape[1]

            # MovModel = 2
            # Reshape data (video) into (T*n)xN array
            if MovModel == 0:
                mx_train = move_train[:, perms[ind]]
                mx_test = move_test[:, perms[ind]]
                xtr = torch.from_numpy(mx_train.astype(np.float32)).to(device)
                xte = torch.from_numpy(mx_test.astype(np.float32)).to(device)
                move_features = None  # mx_train.shape[-1]
                nk = 0
                xtrm = None
                xtem = None
                sta_init = None
                lambdas = [0]  # (2**(np.arange(0,10)))
                nlam = len(lambdas)
                alphas = [None]  # np.array([.005,.01,.02]) #np.arange(.01,.5,.05)
                nalph = len(alphas)
                w_move_traces_all = np.zeros(
                    (nalph, nlam, Nbatches, output_size, input_size))
            elif MovModel == 1:
                x_train_m1 = (rolled_vid[train_idx].reshape(
                    len(train_idx), -1)).astype(np.float32)
                x_test_m1 = (rolled_vid[test_idx].reshape(
                    len(test_idx), -1)).astype(np.float32)
                xtr = torch.from_numpy(x_train_m1).to(device)
                xte = torch.from_numpy(x_test_m1).to(device)
                move_features = None
                xtrm = None
                xtem = None
                # np.hstack((np.logspace(-2,3, 10)[0],np.logspace(-2,3, 10)))
                lambdas = np.hstack(
                    (np.logspace(-3, 3.25, 10)[0], np.logspace(-2, 3.25, 20)))
                nlam = len(lambdas)
                alphas = np.array([None])  # .01])
                nalph = len(alphas)
                sta_init = torch.zeros((output_size, xtr.shape[-1]))
            elif MovModel == 2:
                xtrm = torch.from_numpy(
                    move_train[:, perms[ind]].astype(np.float32)).to(device)
                xtem = torch.from_numpy(
                    move_test[:, perms[ind]].astype(np.float32)).to(device)
                xtr = torch.from_numpy(x_train.astype(np.float32)).to(device)
                xte = torch.from_numpy(x_test.astype(np.float32)).to(device)
                move_features = xtrm.shape[-1]
                lambdas = np.hstack(
                    (np.logspace(-3, 3.25, 10)[0], np.logspace(-2, 3.25, 20)))
                # np.hstack((np.logspace(-2,3, 10)[0]/10,np.logspace(-2,3, 11)/10))
                lambdas_m = .0001*np.ones(len(lambdas))
                nlam = len(lambdas)
                alphas = np.array([None])  # np.array([.005])
                alphas_m = np.array([None])  # .01])
                nalph = len(alphas)
                sta_init = torch.zeros((output_size, xtr.shape[-1]))
                w_move_cv = np.zeros((nalph, nlam, output_size, move_features))
                w_move_traces_all = np.zeros(
                    (nalph, nlam, Nbatches, output_size, move_features))

            if LinNetwork:
                #             meanbias = torch.log(torch.exp(torch.mean(torch.tensor(model_nsp,dtype=torch.float32),axis=0))-1)
                meanbias = torch.mean(torch.tensor(model_nsp, dtype=torch.float32), axis=0)
            else:
                meanbias = torch.log(torch.mean(torch.tensor(model_nsp, dtype=torch.float32), axis=0))

            print('Model: {}, move_features: {}'.format(MovModel, move_features))

            msetrain = np.zeros((nalph, nlam, output_size))
            msetest = np.zeros((nalph, nlam, output_size))
            pred_cv = np.zeros((x_test.shape[0], nalph, nlam, output_size), dtype=np.float32)
            out_cv = np.zeros((x_train.shape[0], nalph, nlam, output_size), dtype=np.float32)
            w_cv = np.zeros((x_train.shape[-1], nalph, nlam, output_size), dtype=np.float32)
            bias_cv = np.zeros((nalph, nlam, output_size), dtype=np.float32)
            tloss_trace_all = np.zeros((nalph, nlam, Nbatches, output_size), dtype=np.float32)
            vloss_trace_all = np.zeros((nalph, nlam, Nbatches, output_size), dtype=np.float32)
            bias_traces_all = np.zeros((nalph, nlam, Nbatches, output_size), dtype=np.float32)

            lr_w = [1e-5, 1e-3]
            lr_b = [1e-5, 5e-3]
            lr_m = [1e-5, 1e-3]
            start = time.time()
            for a, reg_alph in enumerate(tqdm(alphas)):
                sta_init = torch.zeros((output_size, xtr.shape[-1]))
    #             meanbias = torch.mean(torch.tensor(model_nsp,dtype=torch.float32),axis=0)
                pbar = tqdm(lambdas)
                for l, reg_lam in enumerate(pbar):
                    if MovModel == 0:
                        l1 = PoissonGLM_VM_staticreg(input_size, output_size, reg_lam=None, reg_alph=None, move_features=move_features,
                                                    meanfr=meanbias, init_sta=sta_init, device=device, LinNetwork=LinNetwork).to(device)
                        optimizer = optim.Adam(params=[{'params': [l1.weight], 'lr': 1e-3, },
                                                        {'params': [l1.bias], 'lr':lr_b[1]}, ], lr=5e-5)
    #                     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[lr_m[0],lr_b[0]], max_lr=[lr_m[1],lr_b[1]], cycle_momentum=False)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(Nbatches/4))
                    elif MovModel == 1:
                        l1 = PoissonGLM_VM_staticreg(input_size, output_size, reg_lam=None, reg_alph=reg_alph, move_features=move_features,
                                                    meanfr=meanbias, init_sta=sta_init, device=device, LinNetwork=LinNetwork).to(device)
                        optimizer = optim.Adam(params=[{'params': [l1.weight], 'lr':lr_w[1], 'weight_decay':lambdas[l]},
                                                    {'params': [l1.bias], 'lr':lr_b[1]}, ], lr=5e-5)
    #                     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[lr_w[0],lr_b[0]], max_lr=[lr_w[1],lr_b[1]], cycle_momentum=False)
                        scheduler = torch.optim.lr_scheduler.StepLR(
                            optimizer, step_size=int(Nbatches/4))
                    else:
                        l1 = PoissonGLM_VM_staticreg(input_size, output_size, reg_lam=None, reg_alph=reg_alph, reg_alphm=alphas_m[
                                                    a], move_features=move_features, meanfr=meanbias, init_sta=sta_init, device=device, LinNetwork=LinNetwork).to(device)
                        optimizer = optim.Adam(params=[{'params': [l1.weight], 'lr':lr_w[1], 'weight_decay':lambdas[l]},
                                                    {'params': [l1.bias], 'lr':lr_b[1]},
                                                    {'params': [l1.move_weights], 'lr':lr_m[1], 'weight_decay': lambdas_m[l]},
                                                    {'params': [l1.bias_m], 'lr':lr_b[1]}, ])
    #                     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[lr_w[0],lr_b[0],lr_m[0]], max_lr=[lr_w[1],lr_b[1],lr_m[1]], step_size_up=int(Nbatches/4), cycle_momentum=False)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(Nbatches/4))
                    model_name = 'GLM_{}_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_alph{}_lam{}_fullim.pth'.format(model_type, WC_type, int(model_dt*1000), nt_glm_lag, MovModel, Nbatches, a, l)

                    vloss_trace = np.zeros(
                        (Nbatches, output_size), dtype=np.float32)
                    tloss_trace = np.zeros(
                        (Nbatches, output_size), dtype=np.float32)
                    for batchn in np.arange(Nbatches):
                        out = l1(xtr, xtrm)
                        loss = l1.loss(out, ytr)
                        pred = l1(xte, xtem)
                        val_loss = l1.loss(pred, yte)
                        vloss_trace[batchn] = val_loss.clone(
                        ).cpu().detach().numpy()
                        tloss_trace[batchn] = loss.clone().cpu().detach().numpy()
                        bias_traces_all[a, l, batchn] = l1.bias.clone(
                        ).cpu().detach().numpy()
                        pbar.set_description('Loss: {:.03f}'.format(
                            np.mean(val_loss.clone().cpu().detach().numpy())))
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
                        w_move_cv[a, l] = l1.move_weights.clone().cpu().detach().numpy()  
                    pred = l1(xte, xtem)
                    msetest[a, l] = l1.loss(pred, yte).cpu().detach().numpy()

                    pred_cv[:, a, l] = pred.detach().cpu().numpy().squeeze()
                    out = l1(xtr, xtrm)
                    out_cv[:, a, l] = out.detach().cpu().numpy().squeeze()

                    if (a == 0) & (l == 0):
                        sta_init = l1.weight.clone().detach()
                    torch.save({'reg_alph': reg_alph,'reg_lam': reg_lam,'model_state_dict': l1.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,}, save_model / model_name)
            print('GLM: ', time.time()-start)

            if MovModel != 0:
                w_cv2 = w_cv.T.reshape((output_size, nlam, nalph, nt_glm_lag,)+nks)

            msetest[:, 0, :] = np.nan
            malph, mlam, cellnum = np.where(msetest == np.nanmin(msetest, axis=(0, 1), keepdims=True))
            cellnum, m_cinds = np.unique(cellnum, return_index=True)
            malph = malph[m_cinds]
            mlam = mlam[m_cinds]
            sortinds = cellnum.argsort()
            cellnum = cellnum[sortinds]
            malph = malph[sortinds]
            mlam = mlam[sortinds]
            sta_all = w_cv[:, malph, mlam, cellnum].T.reshape((len(cellnum), nt_glm_lag,)+nks)
            pred_all = pred_cv[:, malph, mlam, cellnum]
            bias_all = bias_cv[malph, mlam, cellnum]
            tloss_trace_all2 = tloss_trace_all[malph, mlam, :, cellnum]
            vloss_trace_all2 = vloss_trace_all[malph, mlam, :, cellnum]
            bias_traces = bias_traces_all[malph, mlam, :, cellnum]

            output_size = len(cellnum)
            if MovModel != 1:
                w_move = w_move_cv[malph, mlam, cellnum]
                w_move_traces = w_move_traces_all[malph, mlam, :, cellnum]

            bin_length = 40
            r2_all = np.zeros((output_size))
            for celln in range(output_size):
                sp_smooth = ((np.convolve(test_nsp[:, celln], np.ones(
                    bin_length), 'same')) / (bin_length * model_dt))[bin_length:-bin_length]
                pred_smooth = ((np.convolve(pred_all[:, celln], np.ones(
                    bin_length), 'same')) / (bin_length * model_dt))[bin_length:-bin_length]
                r2_all[celln] = (np.corrcoef(sp_smooth, pred_smooth)[0, 1])**2

            if MovModel == 0:
                GLM_Data = {'r2_all': r2_all,
                            'test_nsp': test_nsp,
                            'pred_all': pred_all,
                            'bias_all': bias_all,
                            'tloss_trace_all': tloss_trace_all2,
                            'vloss_trace_all': vloss_trace_all2,
                            'msetest': msetest,
                            'pred_train': pred_train,
                            'w_move': w_move}
            elif MovModel == 1:
                GLM_Data = {'r2_all': r2_all,
                            'sta_all': sta_all,
                            'test_nsp': test_nsp,
                            'pred_all': pred_all,
                            'bias_all': bias_all,
                            'tloss_trace_all': tloss_trace_all2,
                            'vloss_trace_all': vloss_trace_all2,
                            'msetest': msetest,
                            'pred_train': pred_train,
                            }
            else:
                GLM_Data = {'r2_all': r2_all,
                            'sta_all': sta_all,
                            'test_nsp': test_nsp,
                            'pred_all': pred_all,
                            'bias_all': bias_all,
                            'tloss_trace_all': tloss_trace_all2,
                            'vloss_trace_all': vloss_trace_all2,
                            'msetest': msetest,
                            'pred_train': pred_train,
                            'w_move': w_move}
            GLM_All_Data[model_type+'_MovModel{}'.format(MovModel)] = GLM_Data
            if do_shuffle:
                save_datafile = save_dir/'GLM_{}_Data_VisMov_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_shuffled.h5'.format(
                    model_type, int(model_dt*1000), nt_glm_lag, MovModel, Nbatches)
            else:
                save_datafile = save_dir/'GLM_{}_Data_VisMov_dt{:03d}_T{:02d}_MovModel{:d}_NB{}.h5'.format(
                    model_type, int(model_dt*1000), nt_glm_lag, MovModel, Nbatches)
            ioh5.save(save_datafile, GLM_Data)
            print(save_datafile)
    save_datafile = save_dir/'GLM_{}_Data_VisMov_dt{:03d}_T{:02d}_MovModel{:d}_NB{}.h5'.format('Pytorch_AllData', int(model_dt*1000), nt_glm_lag, MovModel, Nbatches)
    ioh5.save(save_datafile, GLM_All_Data)
