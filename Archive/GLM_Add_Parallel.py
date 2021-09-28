import sys 
import ray
import logging 
import time
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# import io_dict_to_hdf5 as ioh5
import scipy.linalg as linalg

from tqdm.notebook import tqdm, trange
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import shift as imshift
from sklearn.model_selection import train_test_split

sys.path.append(str(Path('.').absolute().parent))
from utils import *
import io_dict_to_hdf5 as ioh5
from format_data import load_ephys_data_aligned

pd.set_option('display.max_rows', None)
FigPath = check_path(Path('~/Research/SensoryMotorPred_Data').expanduser(),'Figures/Decoding')

ray.init(
    ignore_reinit_error=True,
    logging_level=logging.ERROR,
)

print(f'Dashboard URL: http://{ray.get_dashboard_url()}')
print('Dashboard URL: http://localhost:{}'.format(ray.get_dashboard_url().split(':')[-1]))


@ray.remote
def do_glm_fit(train_nsp, test_nsp, train_data, test_data, move_train, move_test, celln, perms, lag, lambdas, bin_length=40, model_dt=.1):
    ##### Format data #####
    # save shape of train_data for initialization
    nks = np.shape(train_data)[1:]; nk = nks[0]*nks[1]
    
    # Shift spikes by -lag for GLM fits
    sps_train = np.roll(train_nsp[:,celln],-lag)
    sps_test = np.roll(test_nsp[:,celln],-lag)
    
    # Initialize saving movement weights 
    w_move = np.zeros(move_train.shape[1])
    
    # Take combination of movements
    move_train = move_train[:,perms]
    move_test = move_test[:,perms]
    
    # Reshape data (video) into TxN array
    x_train = train_data.reshape(train_data.shape[0],-1)
    x_train = np.append(x_train, np.ones((x_train.shape[0],1)), axis = 1) # append column of ones for fitting intercept
    x_train = np.concatenate((x_train, move_train),axis=1)
    
    x_test = test_data.reshape(test_data.shape[0],-1) 
    x_test = np.append(x_test,np.ones((x_test.shape[0],1)), axis = 1) # append column of ones
    x_test = np.concatenate((x_test, move_test),axis=1)
    
    # Prepare Design Matrix
    nlam = len(lambdas)
    XXtr = x_train.T @ x_train
    XYtr = x_train.T @ sps_train
    
    # Initialze mse traces for regularization cross validation
    msetrain = np.zeros((nlam,1))
    msetest = np.zeros((nlam,1))
    w_ridge = np.zeros((x_train.shape[-1],nlam))
    # Inverse matrix for regularization 
    Cinv = np.eye(nk)
    Cinv = linalg.block_diag(Cinv,np.zeros((1+move_test.shape[-1], 1+move_test.shape[-1])))
    # loop over regularization strength
    for l in range(len(lambdas)):  
        # calculate MAP estimate               
        w = np.linalg.solve(XXtr + lambdas[l]*Cinv, XYtr) # equivalent of \ (left divide) in matlab
        w_ridge[:,l] = w
        # calculate test and training rms error
        msetrain[l] = np.mean((sps_train - x_train@w)**2)
        msetest[l] = np.mean((sps_test - x_test@w)**2)
    
    # select best cross-validated lambda for RF
    best_lambda = np.argmin(msetest)
    w = w_ridge[:,best_lambda]
    ridge_rf = w_ridge[:,best_lambda]
    sta_all = np.reshape(w[:-(1+move_test.shape[-1])],nks)
    w_move[perms] = w[-move_test.shape[-1]:]
    
    # predicted firing rate
    sp_pred = x_test@ridge_rf
    # bin the firing rate to get smooth rate vs time
    sp_smooth = (np.convolve(sps_test, np.ones(bin_length), 'same')) / (bin_length * model_dt)
    pred_smooth = (np.convolve(sp_pred, np.ones(bin_length), 'same')) / (bin_length * model_dt)
    # a few diagnostics
    err = np.mean((sp_smooth-pred_smooth)**2)
    cc = np.corrcoef(sp_smooth, pred_smooth)
    cc_all = cc[0,1]
    
    return cc_all, sta_all, sps_test, sp_pred, w_move

start = time.time()



########## Assuming data loaded from Niell lab Freely Moving Ephys ##########
model_dt = .025
data = load_ephys_data_aligned(file_dict, save_dir, model_dt=model_dt)
nan_idxs = []
for key in data.keys():
    nan_idxs.append(np.where(np.isnan(data[key]))[0])
good_idxs = np.ones(len(data['model_active']),dtype=bool)
good_idxs[data['model_active']<.5] = False
good_idxs[np.unique(np.hstack(nan_idxs))] = False


model_dt = .025
data = load_ephys_data_aligned(file_dict, save_dir, model_dt=model_dt)
nan_idxs = []
for key in data.keys():
    nan_idxs.append(np.where(np.isnan(data[key]))[0])
good_idxs = np.ones(len(data['model_active']),dtype=bool)
good_idxs[data['model_active']<.5] = False
good_idxs[np.unique(np.hstack(nan_idxs))] = False

raw_nsp = data['model_nsp'].copy()
for key in data.keys():
    if (key != 'model_nsp') & (key != 'model_active'):
#         movement_times = (data['model_active']>.5) & (~np.isnan(data[key]))
        data[key] = data[key][good_idxs] # interp_nans(data[key]).astype(float)
    elif (key == 'model_nsp'):
        data[key] = data[key][good_idxs]
locals().update(data)

model_dth = np.diff(model_th,append=0)
model_dphi = np.diff(model_phi,append=0)

##### Group shuffle #####
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, train_size=.7, random_state=42)
nT = model_nsp.shape[0]
frac = .2
groups = np.hstack([i*np.ones(int((frac*i)*nT) - int((.2*(i-1))*nT)) for i in range(1,int(1/frac)+1)])
for train_idx, test_idx in gss.split(np.arange(len(model_nsp)), groups=groups):
    print("TRAIN:", train_idx, "TEST:", test_idx)
    
model_vid_sm = (model_vid_sm - np.mean(model_vid_sm,axis=0))/np.std(model_vid_sm,axis=0) 
model_th = (model_th - np.mean(model_th,axis=0))/np.std(model_th,axis=0) 
model_phi = (model_phi - np.mean(model_phi,axis=0))/np.std(model_phi,axis=0) 
model_roll = (model_roll - np.mean(model_roll,axis=0))/np.std(model_roll,axis=0) 
model_pitch = (model_pitch - np.mean(model_pitch,axis=0))/np.std(model_pitch,axis=0) 

train_vid = model_vid_sm[train_idx]
test_vid = model_vid_sm[test_idx]
train_nsp = model_nsp[train_idx]
test_nsp = model_nsp[test_idx]
train_th = model_th[train_idx]
test_th = model_th[test_idx]
train_phi = model_phi[train_idx]
test_phi = model_phi[test_idx]
train_roll = model_roll[train_idx]
test_roll = model_roll[test_idx]
train_pitch = model_pitch[train_idx]
test_pitch = model_pitch[test_idx]
train_t = model_t[train_idx]
test_t = model_t[test_idx]
train_dth = model_dth[train_idx]
test_dth = model_dth[test_idx]
train_dphi = model_dphi[train_idx]
test_dphi = model_dphi[test_idx]


# Initialize movement combinations
titles = np.array(['th','phi','roll','pitch'])
titles_all = []
for n in range(1,2):
    perms = np.array(list(itertools.combinations([0,1,2,3], n)))
    for ind in range(perms.shape[0]):
        titles_all.append('_'.join([t for t in titles[perms[ind]]]))
        
move_train = np.hstack((train_th[:,np.newaxis],train_phi[:,np.newaxis],train_roll[:,np.newaxis],train_pitch[:,np.newaxis])) #,train_dth[:,np.newaxis],train_dphi[:,np.newaxis]))
move_test = np.hstack((test_th[:,np.newaxis],test_phi[:,np.newaxis],test_roll[:,np.newaxis],test_pitch[:,np.newaxis])) #,test_dth[:,np.newaxis],test_dphi[:,np.newaxis]))

lag_list = [ -4, -2, 0 , 2, 4]
lambdas = 1024 * (2**np.arange(0,16))
nks = np.shape(model_vid_sm)[1:]; nk = nks[0]*nks[1]

# Put data into shared memory for parallization 
train_nsp_r = ray.put(train_nsp)
test_nsp_r = ray.put(test_nsp)
train_data_r = ray.put(train_vid)
test_data_r = ray.put(test_vid)
move_train_r = ray.put(move_train)
move_test_r = ray.put(move_test)
result_ids = []
# Loop over parameters appending process ids
for celln in range(train_nsp.shape[1]):
    for n in range(1,2):
        perms = np.array(list(itertools.combinations([0,1,2,3], n)))
        for ind in range(perms.shape[0]):
            for lag_ind, lag in enumerate(lag_list):    
                result_ids.append(do_glm_fit.remote(train_nsp_r, test_nsp_r, train_data_r, test_data_r, move_train_r, move_test_r, celln, perms[ind], lag, lambdas))
                      
results_p = ray.get(result_ids)
print('GLM: ', time.time()-start)

##### Gather Data and Find Max CC Model #####
cc_all = np.stack([results_p[i][0] for i in range(len(results_p))])
sta_all = np.stack([results_p[i][1] for i in range(len(results_p))])
sp_smooth = np.stack([results_p[i][2] for i in range(len(results_p))])
pred_smooth = np.stack([results_p[i][3] for i in range(len(results_p))])
w_move_all = np.stack([results_p[i][4] for i in range(len(results_p))])

cc_all = cc_all.reshape((model_nsp.shape[1],len(titles_all),len(lag_list),) + cc_all.shape[1:])
sta_all = sta_all.reshape((model_nsp.shape[1],len(titles_all),len(lag_list),) + sta_all.shape[1:])
sp_smooth = sp_smooth.reshape((model_nsp.shape[1],len(titles_all),len(lag_list),) + sp_smooth.shape[1:])
pred_smooth = pred_smooth.reshape((model_nsp.shape[1],len(titles_all),len(lag_list),) + pred_smooth.shape[1:])
w_move_all = w_move_all.reshape((model_nsp.shape[1],len(titles_all),len(lag_list),) + w_move_all.shape[1:])

m_cells, m_models, m_lags = np.where(cc_all==np.max(cc_all,axis=(-2,-1), keepdims=True))

mcc = cc_all[m_cells,m_models,m_lags]
msta = sta_all[m_cells,m_models,m_lags]
msp = sp_smooth[m_cells,m_models,m_lags]
mpred = pred_smooth[m_cells,m_models,m_lags]
mw_move = w_move_all[m_cells,m_models,m_lags]

GLM_add = {'cc_all': cc_all,
            'sta_all': sta_all,
            'sp_smooth': sp_smooth,
            'pred_smooth': pred_smooth,
            'w_move_all': w_move_all,}
ioh5.save(save_dir/'Add_GLM_Data_notsmooth_dt{:03d}.h5'.format(int(model_dt*1000)), GLM_add)