import argparse
import gc
import glob
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import scipy.linalg as linalg
import scipy.sparse as sparse
import xarray as xr
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import shift as imshift
from scipy.optimize import minimize_scalar
from scipy.stats import binned_statistic
from sklearn import linear_model as lm
from sklearn.metrics import mean_poisson_deviance, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from tqdm.auto import tqdm, trange

sys.path.append(str(Path('.').absolute()))
import io_dict_to_hdf5 as ioh5
from format_data import load_ephys_data_aligned
from utils import *

# pd.set_option('display.max_rows', None)
# FigPath = check_path(Path('~/Research/SensoryMotorPred_Data').expanduser(),'Figures/Decoding')

ray.init(
    ignore_reinit_error=True,
    logging_level=logging.ERROR,
)

print(f'Dashboard URL: http://{ray.get_dashboard_url()}')
print('Dashboard URL: http://localhost:{}'.format(ray.get_dashboard_url().split(':')[-1]))

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--free_move', type=bool, default=True)
    args = parser.parse_args()
    return args


def load_train_test(file_dict, save_dir, model_dt=.1, frac=.1, train_size=.7, do_shuffle=False, do_norm=False, free_move=True, has_imu=True, has_mouse=False,):
    ##### Load in preprocessed data #####
    data = load_ephys_data_aligned(file_dict, save_dir, model_dt=model_dt, free_move=free_move, has_imu=has_imu, has_mouse=has_mouse,)
    if free_move:
        ##### Find 'good' timepoints when mouse is active #####
        nan_idxs = []
        for key in data.keys():
            nan_idxs.append(np.where(np.isnan(data[key]))[0])
        good_idxs = np.ones(len(data['model_active']),dtype=bool)
        good_idxs[data['model_active']<.5] = False
        good_idxs[np.unique(np.hstack(nan_idxs))] = False
    else:
        good_idxs = np.where((np.abs(data['model_th'])<10) & (np.abs(data['model_phi'])<10))[0]
    
    data['raw_nsp'] = data['model_nsp'].copy()
    ##### return only active data #####
    for key in data.keys():
        if (key != 'model_nsp') & (key != 'model_active') & (key != 'unit_nums'):
            data[key] = data[key][good_idxs] # interp_nans(data[key]).astype(float)
        elif (key == 'model_nsp'):
            data[key] = data[key][good_idxs]
        elif (key == 'unit_nums'):
            pass
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    nT = data['model_nsp'].shape[0]
    groups = np.hstack([i*np.ones(int((frac*i)*nT) - int((frac*(i-1))*nT)) for i in range(1,int(1/frac)+1)])

    for train_idx, test_idx in gss.split(np.arange(len(data['model_nsp'])), groups=groups):
        print("TRAIN:", len(train_idx), "TEST:", len(test_idx))


    data['model_dth'] = np.diff(data['model_th'],append=0)
    data['model_dphi'] = np.diff(data['model_phi'],append=0)

    data['model_vid_sm'] = (data['model_vid_sm'] - np.mean(data['model_vid_sm'],axis=0))/np.nanstd(data['model_vid_sm'],axis=0)
    data['model_vid_sm'][np.isnan(data['model_vid_sm'])]=0
    if do_norm:
        data['model_th'] = (data['model_th'] - np.mean(data['model_th'],axis=0))/np.std(data['model_th'],axis=0) 
        data['model_phi'] = (data['model_phi'] - np.mean(data['model_phi'],axis=0))/np.std(data['model_phi'],axis=0) 
        if free_move:
            data['model_roll'] = (data['model_roll'] - np.mean(data['model_roll'],axis=0))/np.std(data['model_roll'],axis=0) 
            data['model_pitch'] = (data['model_pitch'] - np.mean(data['model_pitch'],axis=0))/np.std(data['model_pitch'],axis=0) 

    ##### Split Data by train/test #####
    data_train_test = {
        'train_vid': data['model_vid_sm'][train_idx],
        'test_vid': data['model_vid_sm'][test_idx],
        'train_nsp': shuffle(data['model_nsp'][train_idx],random_state=42) if do_shuffle else data['model_nsp'][train_idx],
        'test_nsp': shuffle(data['model_nsp'][test_idx],random_state=42) if do_shuffle else data['model_nsp'][test_idx],
        'train_th': data['model_th'][train_idx],
        'test_th': data['model_th'][test_idx],
        'train_phi': data['model_phi'][train_idx],
        'test_phi': data['model_phi'][test_idx],
        'train_roll': data['model_roll'][train_idx] if free_move else [],
        'test_roll': data['model_roll'][test_idx] if free_move else [],
        'train_pitch': data['model_pitch'][train_idx] if free_move else [],
        'test_pitch': data['model_pitch'][test_idx] if free_move else [],
        'train_t': data['model_t'][train_idx],
        'test_t': data['model_t'][test_idx],
        'train_dth': data['model_dth'][train_idx],
        'test_dth': data['model_dth'][test_idx],
        'train_dphi': data['model_dphi'][train_idx],
        'test_dphi': data['model_dphi'][test_idx],
        'train_gz': data['model_gz'][train_idx] if free_move else [],
        'test_gz': data['model_gz'][test_idx] if free_move else [],
    }

    d1 = data
    d1.update(data_train_test)
    return d1,train_idx,test_idx

# Create Tuning curve for theta
def tuning_curve(model_nsp, var, model_dt = .025, N_bins=10, Nstds=3):
    var_range = np.linspace(np.nanmean(var)-Nstds*np.nanstd(var), np.nanmean(var)+Nstds*np.nanstd(var),N_bins)
    tuning = np.zeros((model_nsp.shape[-1],len(var_range)-1))
    tuning_std = np.zeros((model_nsp.shape[-1],len(var_range)-1))
    for n in range(model_nsp.shape[-1]):
        for j in range(len(var_range)-1):
            usePts = (var>=var_range[j]) & (var<var_range[j+1])
            tuning[n,j] = np.nanmean(model_nsp[usePts,n])/model_dt
            tuning_std[n,j] = (np.nanstd(model_nsp[usePts,n])/model_dt)/ np.sqrt(np.count_nonzero(usePts))
    return tuning, tuning_std, var_range[:-1]
    
@ray.remote
def do_glm_fit_vis_skl(train_nsp, test_nsp, x_train, x_test, celln, model_type, lag_list, pbar:ActorHandle, bin_length=40, model_dt=.1):
    
    ##### Format data #####
    nt_glm_lag = len(lag_list)
    
    # Shift spikes by -lag for GLM fits
    sps_train = train_nsp[:,celln] # np.roll(train_nsp[:,celln],-lag)
    sps_test = test_nsp[:,celln] # np.roll(test_nsp[:,celln],-lag)


    if model_type == 'elasticnetcv':
        model = lm.ElasticNetCV(l1_ratio=[.05, .01, .5, .7]) # lm.RidgeCV(alphas=np.arange(100,10000,1000))) #  #MultiOutputRegressor(lm.Ridge(),n_jobs=-1)) 
        model.fit(x_train,sps_train)
        sta_all = np.reshape(model.coef_,(nt_glm_lag,)+nks)
        sp_pred = model.predict(x_test)
    elif model_type == 'ridgecv':
        lambdas = 1024 * (2**np.arange(0,16))
        model = lm.RidgeCV(alphas=lambdas)
        model.fit(x_train,sps_train)
        sta_all = np.reshape(model.coef_,(nt_glm_lag,)+nks)
        sp_pred = model.predict(x_test)
    else:
    #     lambdas = 2048 * (2**np.arange(0,16))
        lambdas = 2**np.arange(0,16)
        nlam = len(lambdas)
        # Initialze mse traces for regularization cross validation
        msetrain = np.zeros((nlam,1))
        msetest = np.zeros((nlam,1))
        pred_all =np.zeros((x_test.shape[0],nlam)) 
        w_ridge = np.zeros((x_train.shape[-1],nlam))
        w_intercept = np.zeros((nlam,1))
        # loop over regularization strength
        for l in range(len(lambdas)):
            model = lm.PoissonRegressor(alpha=lambdas[l],max_iter=300)
            # calculate MAP estimate               
            model.fit(x_train,sps_train)
            w_ridge[:,l] = model.coef_
            w_intercept[l] = model.intercept_
            pred_all[:,l] = model.predict(x_test)
            # calculate test and training rms error
            msetrain[l] = mean_poisson_deviance(sps_train,model.predict(x_train)) #np.mean((sps_train - model.predict(x_train))**2)
            msetest[l] = mean_poisson_deviance(sps_test,pred_all[:,l]) # np.mean((sps_test - model.predict(x_test))**2)
        # select best cross-validated lambda for RF
        best_lambda = np.argmin(msetest)
        w = w_ridge[:,best_lambda]
        intercept= w_intercept[best_lambda]
        ridge_rf = w_ridge[:,best_lambda]
        sta_all = np.reshape(w,(nt_glm_lag,)+nks)
        sp_pred = pred_all[:,best_lambda]
    #     model = make_pipeline(StandardScaler(), lm.PoissonRegressor(alpha=lambdas[best_lambda]))
    #     model.fit(x_train,sps_train)
    # predicted firing rate
    # bin the firing rate to get smooth rate vs time
    sp_smooth = (np.convolve(sps_test, np.ones(bin_length), 'same')) / (bin_length * model_dt)
    pred_smooth = (np.convolve(sp_pred, np.ones(bin_length), 'same')) / (bin_length * model_dt)
    # a few diagnostics
    err = np.mean((sp_smooth-pred_smooth)**2)
    cc = np.corrcoef(sp_smooth[bin_length:-bin_length], pred_smooth[bin_length:-bin_length])
    cc_all = cc[0,1]
    r2_all = r2_score(sp_smooth,pred_smooth)
    pbar.update.remote(1)
    return cc_all, sta_all, sps_test, sp_pred, r2_all

@ray.remote
def do_glm_fit_vismov_skl(train_nsp, test_nsp, x_train, x_test, move_train, move_test, perm, celln, lag_list, bin_length=40, model_dt=.05):
    ##### Format data #####
    nt_glm_lag = len(lag_list)
    w_move = np.zeros(move_train.shape[-1])
    xm_train = move_train[:,perm]
    xm_test = move_test[:,perm]
    x_train = np.concatenate((x_train,move_train),axis=-1)
    x_test = np.concatenate((x_test,move_test),axis=-1)
    # Shift spikes by -lag for GLM fits
    sps_train = train_nsp[:,celln] # np.roll(train_nsp[:,celln],-lag)
    sps_test = test_nsp[:,celln] # np.roll(test_nsp[:,celln],-lag)

    lambdas = 2**np.arange(0,16)
    nlam = len(lambdas)
    # Initialze mse traces for regularization cross validation
    error_train = np.zeros((nlam,1))
    error_test = np.zeros((nlam,1))
    pred_all =np.zeros((x_test.shape[0],nlam)) 
    w_cv = np.zeros((x_train.shape[-1],nlam))
    w_intercept = np.zeros((nlam,1))
    # loop over regularization strength
    for l in range(len(lambdas)):
        model = lm.PoissonRegressor(alpha=lambdas[l],max_iter=300)
        # calculate MAP estimate               
        model.fit(x_train,sps_train)
        w_cv[:,l] = model.coef_
        w_intercept[l] = model.intercept_
        pred_all[:,l] = model.predict(x_test)
        # calculate test and training rms error
        error_train[l] = mean_poisson_deviance(sps_train,model.predict(x_train)) #np.mean((sps_train - model.predict(x_train))**2)
        error_test[l] = mean_poisson_deviance(sps_test,pred_all[:,l]) # np.mean((sps_test - model.predict(x_test))**2)
    # select best cross-validated lambda for RF
    best_lambda = np.argmin(error_test)
    w = w_cv[:,best_lambda]
    intercept= w_intercept[best_lambda]
    sta_all = np.reshape(w[:-move_train.shape[-1]],(nt_glm_lag,)+nks)
    w_move = w[-move_train.shape[-1]:]
    sp_pred = pred_all[:,best_lambda]

    # predicted firing rate
    # bin the firing rate to get smooth rate vs time
    sp_smooth = (np.convolve(sps_test, np.ones(bin_length), 'same')) / (bin_length * model_dt)
    pred_smooth = (np.convolve(sp_pred, np.ones(bin_length), 'same')) / (bin_length * model_dt)
    # a few diagnostics
    err = np.mean((sp_smooth-pred_smooth)**2)
    cc = np.corrcoef(sp_smooth[bin_length:-bin_length], pred_smooth[bin_length:-bin_length])
    cc_all = cc[0,1]
    r2_all = r2_score(sp_smooth,pred_smooth)
    return cc_all, sta_all, sps_test, sp_pred, r2_all, w_move

if __name__ == '__main__':
    args = parse_args()

    free_move = args.free_move
    if free_move:
        stim_type = 'fm1'
    else:
        stim_type = 'hf1_wn' # 'fm1' # 

    data_dir  = Path('~/Goeppert/freely_moving_ephys/ephys_recordings/070921/J553RT').expanduser() / stim_type
    save_dir  = check_path(Path('~/Research/SensoryMotorPred_Data/data/070921/J553RT/').expanduser(), stim_type)
    # FigPath = check_path(FigPath, stim_type)


    file_dict = {'cell': 0,
                'drop_slow_frames': True,
                'ephys': list(data_dir.glob('*ephys_merge.json'))[0].as_posix(),
                'ephys_bin': list(data_dir.glob('*Ephys.bin'))[0].as_posix(),
                'eye': list(data_dir.glob('*REYE.nc'))[0].as_posix(),
                'imu': list(data_dir.glob('*imu.nc'))[0].as_posix() if stim_type=='fm1' else None,
                'mapping_json': '/home/seuss/Research/Github/FreelyMovingEphys/probes/channel_maps.json',
                'mp4': True,
                'name': '070921_J553RT_control_Rig2_'+stim_type,
                'probe_name': 'DB_P128-6',
                'save': data_dir.as_posix(),
                'speed': list(data_dir.glob('*speed.nc'))[0].as_posix() if stim_type=='hf1_wn' else None,
                'stim_type': 'light',
                'top': list(data_dir.glob('*TOP1.nc'))[0].as_posix() if stim_type=='fm1' else None,
                'world': list(data_dir.glob('*world.nc'))[0].as_posix(),}

    model_dt = .05
    do_shuffle=False
    do_norm = False

    data,train_idx,test_idx = load_train_test(file_dict, save_dir, model_dt=model_dt, do_shuffle=do_shuffle, do_norm=do_norm,free_move=free_move, has_imu=free_move, has_mouse=False)
    locals().update(data)

    lag=150 # in ms
    nt_glm_lag = 5
    # minlag = int(-lag//(1000*model_dt)); maxlag=int((lag//(1000*model_dt))+1)
    lag_list = np.array([-1,0,1,2,3]) #,np.arange(minlag,maxlag,np.floor((maxlag-minlag)/nt_glm_lag).astype(int))
    nt_glm_lag = len(lag_list)
    print(lag_list,1000*lag_list*model_dt)
    do_shuffle=False
    model_type = 'poissonregressor'

    ##### Visual Only Fitting #####
    for do_shuffle in [False,True]:
        # Load Data
        data, train_idx, test_idx = load_train_test(file_dict, save_dir, model_dt=model_dt, do_shuffle=do_shuffle, do_norm=True,free_move=free_move, has_imu=free_move, has_mouse=False)
        locals().update(data)

        ##### Start GLM Parallel Processing #####
        start = time.time()
        nks = np.shape(train_vid)[1:]; nk = nks[0]*nks[1]*nt_glm_lag
        # Reshape data (video) into (T*n)xN array
        rolled_vid = np.hstack([np.roll(model_vid_sm, nframes, axis=0) for nframes in lag_list]) # nt_glm_lag
        x_train = rolled_vid[train_idx].reshape(len(train_idx),-1)
        x_test = rolled_vid[test_idx].reshape(len(test_idx),-1)
        
        # Put data into shared memory for parallization 
        train_nsp_r = ray.put(train_nsp)
        test_nsp_r = ray.put(test_nsp)
        train_data_r = ray.put(x_train)
        test_data_r = ray.put(x_test)
        result_ids = []
        # Loop over parameters appending process ids
        for celln in range(train_nsp.shape[1]):
            result_ids.append(do_glm_fit_vis_skl.remote(train_nsp_r, test_nsp_r, train_data_r, test_data_r, celln, model_type, lag_list, model_dt=model_dt))

        print('N_proc:', len(result_ids))
        results_p = ray.get(result_ids)
        print('GLM: ', time.time()-start)

        ##### Gather Data and Find Max CC Model #####
        mcc = np.stack([results_p[i][0] for i in range(len(results_p))])
        msta = np.stack([results_p[i][1] for i in range(len(results_p))])
        msp = np.stack([results_p[i][2] for i in range(len(results_p))])
        mpred = np.stack([results_p[i][3] for i in range(len(results_p))])
        mr2 = np.stack([results_p[i][4] for i in range(len(results_p))])

        nt_glm_lag = len(lag_list)
        GLM_Data = {'mcc': mcc,
                    'msta': msta,
                    'msp': msp,
                    'mpred': mpred,
                    'mr2':mr2,}
        if do_shuffle:
            ioh5.save(save_dir/'GLM_{}_Data_VisOnly_notsmooth_dt{:03d}_T{:02d}_shuffled.h5'.format(model_type,int(model_dt*1000), nt_glm_lag), GLM_Data)
        else:
            ioh5.save(save_dir/'GLM_{}_Data_VisOnly_notsmooth_dt{:03d}_T{:02d}.h5'.format(model_type,int(model_dt*1000), nt_glm_lag), GLM_Data)
            
        del train_nsp_r, test_nsp_r, train_data_r, test_data_r, result_ids, results_p, mcc, msta, msp, mpred, mr2,
        gc.collect()


##### Visual + Movement Fitting #####
    for do_shuffle in [False,True]:
        # Load Data
        data, train_idx, test_idx = load_train_test(file_dict, save_dir, model_dt=model_dt, do_shuffle=do_shuffle, do_norm=True,free_move=free_move, has_imu=free_move, has_mouse=False)
        locals().update(data)

        # Initialize movement combinations
        titles = np.array(['Theta','Phi','Roll','Pitch']) # 'dg_p','dg_n' 'roll','pitch'
        titles_all = []
        for n in range(1,len(titles)+1):
            perms = np.array(list(itertools.combinations(np.arange(len(titles)), n)))
            for ind in range(perms.shape[0]):
                titles_all.append('_'.join([t for t in titles[perms[ind]]]))

        ##### Start GLM Parallel Processing #####
        start = time.time()
        nks = np.shape(train_vid)[1:]; nk = nks[0]*nks[1]*nt_glm_lag
        # Reshape data (video) into (T*n)xN array
        rolled_vid = np.hstack([np.roll(model_vid_sm, nframes, axis=0) for nframes in lag_list]) # nt_glm_lag
        x_train = rolled_vid[train_idx].reshape(len(train_idx),-1)
        x_test = rolled_vid[test_idx].reshape(len(test_idx),-1)
        move_train = np.hstack((train_th[:,np.newaxis],train_phi[:,np.newaxis],train_roll[:,np.newaxis],train_pitch[:,np.newaxis]))
        move_test = np.hstack((test_th[:,np.newaxis],test_phi[:,np.newaxis],test_roll[:,np.newaxis],test_pitch[:,np.newaxis]))

        # Put data into shared memory for parallization 
        train_nsp_r = ray.put(train_nsp)
        test_nsp_r = ray.put(test_nsp)
        train_data_r = ray.put(x_train)
        test_data_r = ray.put(x_test)
        train_move_r = ray.put(move_train)
        test_move_r = ray.put(move_test)
        result_ids = []
        # Loop over parameters appending process ids
        for celln in range(train_nsp.shape[1]):
            for n in range(1,len(titles)+1):
                perms = np.array(list(itertools.combinations(np.arange(len(titles)), n)))
                for ind in range(perms.shape[0]):
                    result_ids.append(do_glm_fit_vismov_skl.remote(train_nsp_r, test_nsp_r, train_data_r, test_data_r, train_move_r, test_move_r, perms[ind], celln, lag_list, model_dt=model_dt))

        print('N_proc:', len(result_ids))
        results_p = ray.get(result_ids)
        print('GLM: ', time.time()-start)
        ##### Gather Data and Find Max CC Model #####
        cc_all = np.stack([results_p[i][0] for i in range(len(results_p))])
        sta_all = np.stack([results_p[i][1] for i in range(len(results_p))])
        sp_all = np.stack([results_p[i][2] for i in range(len(results_p))])
        pred_all = np.stack([results_p[i][3] for i in range(len(results_p))])
        r2_all = np.stack([results_p[i][4] for i in range(len(results_p))])
        wmove_all = np.stack([results_p[i][5] for i in range(len(results_p))])


        cc_all = cc_all.reshape((model_nsp.shape[1],len(titles_all),) + cc_all.shape[1:])
        wmove_all = wmove_all.reshape((model_nsp.shape[1],len(titles_all),) + wmove_all.shape[1:])
        sp_all = sp_all.reshape((model_nsp.shape[1],len(titles_all),) + sp_all.shape[1:])
        pred_all = pred_all.reshape((model_nsp.shape[1],len(titles_all),) + pred_all.shape[1:])
        r2_all = r2_all.reshape((model_nsp.shape[1],len(titles_all),) + r2_all.shape[1:])
        wmove_all = wmove_all.reshape((model_nsp.shape[1],len(titles_all),) + wmove_all.shape[1:])

        m_cells, m_models,  = np.where(cc_all==np.max(cc_all,axis=(-1), keepdims=True))
        m_cells, m_cinds = np.unique(m_cells,return_index=True)
        m_models = m_models[m_cinds]
        mcc = cc_all[m_cells,m_models]
        msp = sp_raw[m_cells,m_models]
        mpred = pred_raw[m_cells,m_models]
        mw_move = w_move_all[m_cells,m_models]
        mr2 = r2_all[m_cells,m_models]

        nt_glm_lag = len(lag_list)
        GLM_Data = {'mcc': mcc,
                    'msta': msta,
                    'msp': msp,
                    'mpred': mpred,
                    'mr2':mr2,
                    'mw_move':mw_move}
        if do_shuffle:
            ioh5.save(save_dir/'GLM_{}_Data_VisMot_notsmooth_dt{:03d}_T{:02d}_shuffled.h5'.format(model_type,int(model_dt*1000), nt_glm_lag), GLM_Data)
        else:
            ioh5.save(save_dir/'GLM_{}_Data_VisMot_notsmooth_dt{:03d}_T{:02d}.h5'.format(model_type,int(model_dt*1000), nt_glm_lag), GLM_Data)

        del train_nsp_r, test_nsp_r, train_data_r, test_data_r, result_ids, results_p, mcc, msta, msp, mpred, mr2,
        gc.collect()
