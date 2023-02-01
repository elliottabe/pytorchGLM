########## This .py file contains the functions related to formatting data assuming Niell Lab preprocessing pipeline ##########
import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import xarray as xr

from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from itertools import chain
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle
from scipy.signal import medfilt

import pytorchGLM.Utils.io_dict_to_hdf5 as ioh5
from pytorchGLM.Utils.utils import *
# from Utils.params import *

def format_raw_data(file_dict, params, medfiltbins=11, **kwargs):
    """ Formatting raw data for Niell Lab freely moving ephys data 

    Args:
        file_dict (dict): file dictionary containing raw data paths.
        params (dict): parameter dictionary holding key parameters for formatting.
        medfiltbins (int, optional): filter bin size for smoothing. Defaults to 11.

    Returns:
        raw_data (dict): returns formatted dictionary of raw data
        goodcells (pd.DataFrame): returns a DataFrame with ephys unit information
    """
    
    ##### Set up condition shorthand #####
    if file_dict['imu'] is not None:
        has_imu = True
        has_mouse = False
    else:
        has_imu = False
        has_mouse = True
        
    # open worldcam
    print('opening worldcam data')
    world_data = xr.open_dataset(file_dict['world'])
    world_vid_raw = np.uint8(world_data['WORLD_video'])
    # resize worldcam
    sz = world_vid_raw.shape # raw video size
    # if size is larger than the target 60x80, resize by 0.5
    if sz[1]>160:
        downsamp = 0.5
        world_vid = np.zeros((sz[0],int(sz[1]*downsamp),int(sz[2]*downsamp)), dtype = 'uint8')
        for f in range(sz[0]):
            world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(int(sz[2]*downsamp),int(sz[1]*downsamp)))
    else:
        # if the worldcam has already been resized when the nc file was written in preprocessing, don't resize
        world_vid = world_vid_raw.copy()
    del world_vid_raw
    gc.collect()

    # world timestamps
    worldT = world_data.timestamps.copy()
    if params['free_move'] == True:
        print('opening top data')
    #     # open the topdown camera nc file
        top_data = xr.open_dataset(file_dict['top'])
        top_speed = top_data.TOP1_props[:,0].data
        topT = top_data.timestamps.data.copy() # read in time timestamps
    #     top_vid = np.uint8(top_data['TOP1_video']) # read in top video
        # clear from memory
        del top_data
        gc.collect()
    else: 
        topT = []
        top_speed = []
        
    # load IMU data
    if has_imu:
        print('opening imu data')
        imu_data = xr.open_dataset(file_dict['imu'])
        try:
            accT = imu_data.IMU_data.sample # imu timestamps
            acc_chans = imu_data.IMU_data # imu dample data
        except AttributeError:
            accT = imu_data.__xarray_dataarray_variable__.sample
            acc_chans = imu_data.__xarray_dataarray_variable__
        
        # raw gyro values
        gx = np.array(acc_chans.sel(channel='gyro_x_raw'))
        gy = np.array(acc_chans.sel(channel='gyro_y_raw'))
        gz = np.array(acc_chans.sel(channel='gyro_z_raw'))
        gz = (gz-np.mean(gz))*7.5 # Rescale gz
        # gyro values in degrees
        gx_deg = np.array(acc_chans.sel(channel='gyro_x'))
        gy_deg = np.array(acc_chans.sel(channel='gyro_y'))
        gz_deg = np.array(acc_chans.sel(channel='gyro_z'))
        # pitch and roll in deg
        groll = medfilt(np.array(acc_chans.sel(channel='roll')),medfiltbins)
        gpitch = medfilt(np.array(acc_chans.sel(channel='pitch')),medfiltbins)
    else: 
        accT = []
        gz = []
        groll = []
        gpitch = []
    # load optical mouse nc file from running ball
    if file_dict['speed'] is not None:
        print('opening speed data')
        speed_data = xr.open_dataset(file_dict['speed'])
        try:
            spdVals = speed_data.BALL_data
        except AttributeError:
            spdVals = speed_data.__xarray_dataarray_variable__
        try:
            spd = spdVals.sel(move_params = 'speed_cmpersec')
            spd_tstamps = spdVals.sel(move_params = 'timestamps')
        except:
            spd = spdVals.sel(frame = 'speed_cmpersec')
            spd_tstamps = spdVals.sel(frame = 'timestamps')
    print('opening ephys data')
    # ephys data for this individual recording
    ephys_data = pd.read_json(file_dict['ephys'])
    # sort units by shank and site order
    ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)
    ephys_data = ephys_data.reset_index()
    ephys_data = ephys_data.drop('index', axis=1)
    # spike times
    ephys_data['spikeTraw'] = ephys_data['spikeT']
    print('getting good cells')
    # select good cells from phy2
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    units = goodcells.index.values
    # get number of good units
    n_units = len(goodcells)
    # plot spike raster
    plt.close()
    print('opening eyecam data')
    # load eye data
    eye_data = xr.open_dataset(file_dict['eye'])
    eye_vid = np.uint8(eye_data['REYE_video'])
    eyeT = eye_data.timestamps.copy()
    # plot eye postion across recording
    eye_params = eye_data['REYE_ellipse_params']
    # define theta, phi and zero-center
    th = np.array((eye_params.sel(ellipse_params = 'theta'))*180/np.pi)
    phi = np.array((eye_params.sel(ellipse_params = 'phi'))*180/np.pi)
    eyerad = eye_data.REYE_ellipse_params.sel(ellipse_params = 'longaxis').data

    print('adjusting camera times to match ephys')
    # adjust eye/world/top times relative to ephys
    ephysT0 = ephys_data.iloc[0,12]
    eyeT = eye_data.timestamps  - ephysT0
    if eyeT[0]<-600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = world_data.timestamps - ephysT0
    if worldT[0]<-600:
        worldT = worldT + 8*60*60
    if params['free_move'] is True and has_imu is True:
        accTraw = accT - ephysT0
    if params['free_move'] is False and has_mouse is True:
        speedT = spd_tstamps - ephysT0
    if params['free_move'] is True:
        topT = topT - ephysT0

    ##### Clear some memory #####
    del eye_data 
    gc.collect()

    if file_dict['drop_slow_frames'] is True:
        # in the case that the recording has long time lags, drop data in a window +/- 3 frames around these slow frames
        isfast = np.diff(eyeT)<=0.05
        isslow = sorted(list(set(chain.from_iterable([list(range(int(i)-3,int(i)+4)) for i in np.where(isfast==False)[0]]))))
        th[isslow] = np.nan
        phi[isslow] = np.nan


    print(world_vid.shape)
    # calculate eye veloctiy
    dEye = np.diff(th)
    accT_correction_file = params['save_dir']/'acct_correction_{}.h5'.format(params['data_name'])
    # check accelerometer / eye temporal alignment
    if (accT_correction_file.exists()):# & (reprocess==False):
        accT_correction = ioh5.load(accT_correction_file)
        offset0    = accT_correction['offset0']
        drift_rate = accT_correction['drift_rate']
        accT = accTraw - (offset0 + accTraw*drift_rate)
        found_good_offset = True
    else:
        if (has_imu):
            print('checking accelerometer / eye temporal alignment')
            lag_range = np.arange(-0.2,0.2,0.002)
            cc = np.zeros(np.shape(lag_range))
            t1 = np.arange(5,len(dEye)/60-120,20).astype(int) # was np.arange(5,1600,20), changed for shorter videos
            t2 = t1 + 60
            offset = np.zeros(np.shape(t1))
            ccmax = np.zeros(np.shape(t1))
            acc_interp = interp1d(accTraw, (gz-3)*7.5)
            for tstart in tqdm(range(len(t1))):
                for l in range(len(lag_range)):
                    try:
                        c, lag= nanxcorr(-dEye[t1[tstart]*60 : t2[tstart]*60] , acc_interp(eyeT[t1[tstart]*60:t2[tstart]*60]+lag_range[l]),1)
                        cc[l] = c[1]
                    except: # occasional problem with operands that cannot be broadcast togther because of different shapes
                        cc[l] = np.nan
                offset[tstart] = lag_range[np.argmax(cc)]    
                ccmax[tstart] = np.max(cc)
            offset[ccmax<0.1] = np.nan
            del ccmax, dEye
            gc.collect()
            if np.isnan(offset).all():
                found_good_offset = False
            else:
                found_good_offset = True

        if has_imu and found_good_offset is True:
            print('fitting regression to timing drift')
            # fit regression to timing drift
            model = LinearRegression()
            dataT = np.array(eyeT[t1*60 + 30*60])
            model.fit(dataT[~np.isnan(offset)].reshape(-1,1),offset[~np.isnan(offset)]) 
            offset0 = model.intercept_
            drift_rate = model.coef_
            del dataT
            gc.collect()
        elif file_dict['speed'] is not None or found_good_offset is False:
            offset0 = 0.1
            drift_rate = -0.000114
        if has_imu:
            accT_correction = {'offset0': offset0, 'drift_rate': drift_rate}
            ioh5.save(accT_correction_file,accT_correction)
            accT = accTraw - (offset0 + accTraw*drift_rate)
            del accTraw
            gc.collect()


    print('correcting ephys spike times for offset and timing drift')
    for i in ephys_data.index:
        ephys_data.at[i,'spikeT'] = np.array(ephys_data.at[i,'spikeTraw']) - (offset0 + np.array(ephys_data.at[i,'spikeTraw']) *drift_rate)
    goodcells = ephys_data.loc[ephys_data['group']=='good']


    ##### Calculating image norm #####
    sz = np.shape(world_vid)
    world_vid_sm = np.zeros((sz[0],int(sz[1]/params['downsamp_vid']),int(sz[2]/params['downsamp_vid'])),dtype=np.uint8)
    for f in range(sz[0]):
        world_vid_sm[f,:,:] = cv2.resize(world_vid[f,:,:],(int(sz[2]/params['downsamp_vid']),int(sz[1]/params['downsamp_vid'])))

    del world_vid
    gc.collect()

    # Build dictionary formatted for interpolating
    raw_data = {
                'eye':{
                    'th':     th,
                    'phi':    phi,
                    'eyerad': eyerad,
                    'eyeTS':  eyeT,
                },
                'acc': {
                    'gz':     gz,
                    'roll':   groll,
                    'pitch':  gpitch,    
                    'accTS':  accT,
                },
                'top':{
                    'speed': top_speed,
                    'topTS': topT,
                },
                'vid':{
                    'vidTS':    worldT,
                    'vid_sm':   world_vid_sm,
                },
                }

    return raw_data, goodcells


def interp_raw_data(raw_data, align_t, model_dt=0.05, goodcells=None):
    """Interpolates raw data based on nested dictionary. 

    Args:
        raw_data (dict): nested dictionary where first level (raw_data[key0]) represents data type
                         second level contains data and timestamps assuming the following format respectively: 
                         raw_data[key0]['datatype'] or raw_data[key0]['datatypeTS'] 
                         where datatype is the variabel name of the data.
        align_t (np.array): timestamps which to align data to.
        model_dt (float, optional): model bin size to align data. Defaults to 0.05.
        goodcells (pd.DataFrame, optional): If processing ephys data input DataFrame with size (units x features) with a column named 
                                            spikeT. spikeT containsa list spike times (row). Defaults to None.

    Returns:
        model_data (dict): dictionary containing interpolated time-aligned model data with naming convention 'model_[datatype]'
    """


    ##### Set up model interpolated time #####
    model_t = np.arange(0,np.max(align_t), model_dt)

    ##### Interpolate raw data #####
    model_data = {}
    for key0 in raw_data.keys():
            for key1 in raw_data[key0].keys():
                if ('TS' not in key1) & (np.size(raw_data[key0][key1])>0):
                    if 'vid' in key0: # Z score video then interpolate
                        std_im = np.std(raw_data[key0][key1], axis=0, dtype=float)
                        img_norm = ((raw_data[key0][key1]-np.mean(raw_data[key0][key1],axis=0,dtype=float))/std_im).astype(float)
                        std_im[std_im<20] = 0 # zero out extreme values
                        img_norm = (img_norm * (std_im>0)).astype(float)
                        interp = interp1d(raw_data[key0][key0+'TS'], img_norm,'nearest', axis=0,bounds_error = False) 
                        testimg = interp(model_t[0])
                        model_vid_sm = np.zeros((len(model_t),int(np.shape(testimg)[0]),int(np.shape(testimg)[1])),dtype=float)
                        for i in tqdm(range(len(model_t))):
                            model_vid = interp(model_t[i] + model_dt/2)
                            model_vid_sm[i,:] = model_vid
                        model_vid_sm[np.isnan(model_vid_sm)]=0
                        model_data['model_'+ key1] = model_vid_sm
                    else:
                        interp = interp1d(raw_data[key0][key0+'TS'],pd.DataFrame(raw_data[key0][key1]).interpolate(limit_direction='both').to_numpy().squeeze(),axis=0, bounds_error=False)
                        model_data['model_'+ key1] = interp(model_t+model_dt/2)
    model_data['model_t'] = model_t
    if ('acc' in raw_data.keys()) & (np.size(raw_data['acc']['gz'])>0):
        model_data['model_active'] = np.convolve(np.abs(model_data['model_gz']), np.ones(int(1/model_dt)), 'same') / len(np.ones(int(1/model_dt)))

    # get spikes / rate
    if goodcells is not None:
        n_units = len(goodcells)
        model_nsp = np.zeros((len(model_t),n_units))
        bins = np.append(model_t,model_t[-1]+model_dt)
        for i,ind in enumerate(goodcells.index):
            model_nsp[:,i],bins = np.histogram(goodcells.at[ind,'spikeT'],bins)
        model_data['model_nsp'] = model_nsp
        model_data['unit_nums'] = goodcells.index.values

    return model_data


def load_aligned_data(file_dict, params, reprocess=False):
    """ Load time aligned data from file or process raw data and return formatted data

    Args:
        file_dict (dict): file dictionary containing raw data paths.
        params (dict): parameter dictionary holding key parameters for formatting.
        reprocess (bool, optional): reprocess raw data. Defaults to False.

    Returns:
        model_data (dict): returns dictionary with time aligned model data
    """

    model_file = params['save_dir'] / 'ModelData_{}_dt{:03d}_rawWorldCam_{:d}ds.h5'.format(params['data_name'],int(params['model_dt']*1000),int(params['downsamp_vid']))
    if (model_file.exists()) & (reprocess==False):
        model_data = ioh5.load(model_file)
    else:
        raw_data, goodcells = format_raw_data(file_dict,params)
        model_data = interp_raw_data(raw_data,raw_data['vid']['vidTS'],model_dt=params['model_dt'],goodcells=goodcells)
        if params['free_move']:
            ##### Saving average and std of parameters for centering and scoring across conditions #####
            FM_move_avg = np.zeros((2,6))
            FM_move_avg[:,0] = np.array([np.nanmean(model_data['model_th']),np.nanstd(model_data['model_th'])])
            FM_move_avg[:,1] = np.array([np.nanmean(model_data['model_phi']),np.nanstd(model_data['model_phi'])])
            FM_move_avg[:,2] = np.array([np.nanmean(model_data['model_roll']),np.nanstd(model_data['model_roll'])])
            FM_move_avg[:,3] = np.array([np.nanmean(model_data['model_pitch']),np.nanstd(model_data['model_pitch'])])
            FM_move_avg[:,4] = np.array([np.nanmean(model_data['model_speed']),np.nanmax(model_data['model_speed'])])
            FM_move_avg[:,5] = np.array([np.nanmean(model_data['model_eyerad']),np.nanmax(model_data['model_eyerad'])])
            np.save(params['save_dir_fm']/'FM_MovAvg_{}_dt{:03d}.npy'.format(params['data_name'],int(params['model_dt']*1000)),FM_move_avg)
        ephys_file = params['save_dir'] / 'RawEphysData_{}.h5'.format(params['data_name'])
        goodcells.to_hdf(ephys_file,key='goodcells', mode='w')
        ioh5.save(model_file, model_data)
    return model_data


def format_data(data, params, frac=.1, shifter_train_size=.5, test_train_size=.75, do_norm=True, NKfold=1, thresh_cells=True, cut_inactive=True, move_medwin=11,**kwargs):
    """ Fully format data for model training

    Args:
        data (dict): data dictionary containing time aligned data. get as ouput from load_aligned_data
        params (dict): parameter dictionary holding key parameters for formatting.
        frac (float, optional): from of total length, size of groups for train/test split. Defaults to .1.
        shifter_train_size (float, optional): shifter train/test split fraction. Defaults to .5.
        test_train_size (float, optional): train/test size for random group shuffle. Defaults to .75.
        do_shuffle (bool, optional): shuffle spikes. Defaults to False.
        do_norm (bool, optional): normalize data. Defaults to False.
        NKfold (int, optional): How many Kfolds to make. Defaults to 1.
        thresh_cells (bool, optional): threshold out bad cells. Defaults to False.
        cut_inactive (bool, optional): cut inactive timepoints. Defaults to True.
        move_medwin (int, optional): window to smooth roll/pitch. Defaults to 11.

    Returns:
        data (dict): formatted dictionary of data
        train_idx_list: list of train indecies for CV
        test_idx_list: list of test indecies for CV
    """
    ##### Load in preprocessed data #####
    if params['free_move']:
        ##### Find 'good' timepoints when mouse is active #####
        nan_idxs = []
        for key in data.keys():
            nan_idxs.append(np.where(np.isnan(data[key]))[0])
        good_idxs = np.ones(len(data['model_active']),dtype=bool)
        good_idxs[data['model_active']<0.5] = False # .5 based on histogram, determined emperically 
        good_idxs[np.unique(np.hstack(nan_idxs))] = False
    else:
        good_idxs = np.where((np.abs(data['model_th'])<50) & (np.abs(data['model_phi'])<50))[0].astype(int)


    data['raw_nsp'] = data['model_nsp'].copy()
    if cut_inactive:
        ##### return only active data #####
        for key in data.keys():
            if (key != 'model_nsp') & (key != 'model_active') & (key != 'unit_nums') & (key != 'model_vid_sm_shift'):
                if len(data[key])>0:
                    data[key] = data[key][good_idxs] # interp_nans(data[key]).astype(float)
            elif (key == 'model_nsp'):
                data[key] = data[key][good_idxs]
            elif (key == 'unit_nums') | (key == 'model_vis_sm_shift'):
                pass
    
    
    ##### Splitting data for shifter, then split for model training #####
    if params['shifter_5050']==False:
        gss = GroupShuffleSplit(n_splits=NKfold, train_size=test_train_size, random_state=42)
        nT = data['model_nsp'].shape[0]
        groups = np.hstack([i*np.ones(int((frac*i)*nT) - int((frac*(i-1))*nT)) for i in range(1,int(1/frac)+1)])

        train_idx_list=[]
        test_idx_list = []
        for train_idx, test_idx in gss.split(np.arange(data['model_nsp'].shape[0]), groups=groups):
            train_idx_list.append(train_idx)
            test_idx_list.append(test_idx)
    else:
        ##### Need to fix 50/50 splitting ###################
        # gss = GroupShuffleSplit(n_splits=NKfold, train_size=shifter_train_size, random_state=42)
        np.random.seed(42)
        nT = data['model_nsp'].shape[0]
        
        shifter_train_size = .5
        groups = np.hstack([i*np.ones(int((frac*i)*nT) - int((frac*(i-1))*nT)) for i in range(1,int(1/frac)+1)])
        train_idx_list_shifter=[]
        test_idx_list_shifter=[]
        train_idx_list=[]
        test_idx_list = []
        for Kfold in np.arange(NKfold):
            glist = np.arange(1,1/frac+1)
            shifter_groups = np.random.choice(glist,size=int((1/frac)*shifter_train_size),replace=False)
            idx = np.arange(nT)
            sampled_inds = np.any(np.array([groups==shifter_groups[n] for n in np.arange(len(shifter_groups))]),axis=0)
            train_idx_list_shifter.append(idx[sampled_inds])
            test_idx_list_shifter.append(idx[~sampled_inds])
            glist=glist[~np.any(np.array([glist==shifter_groups[n] for n in np.arange(len(shifter_groups))]),axis=0)]
            
            if params['train_shifter']==False:
                if params['shifter_5050_run']:
                    train_idx_list = test_idx_list_shifter
                    test_idx_list  = train_idx_list_shifter
                else:
                    train_idx_list = train_idx_list_shifter
                    test_idx_list  = test_idx_list_shifter
            else:
                if params['shifter_5050_run']:
                    train_idx_list = test_idx_list_shifter
                    test_idx_list  = train_idx_list_shifter
                else:
                    train_idx_list = train_idx_list_shifter
                    test_idx_list  = test_idx_list_shifter

    if thresh_cells:
        if params['free_move']:
            if (params['save_dir_fm']/'bad_cells_{}.npy'.format(params['data_name'])).exists():
                bad_cells = np.load(params['save_dir_fm']/'bad_cells_{}.npy'.format(params['data_name']))
            else:
                mean_thresh = np.nanmean(data['model_nsp']/params['model_dt'],axis=0) < 1 # Thresholding out units under 1 Hz
                f25,l75=int((data['model_nsp'].shape[0])*.5),int((data['model_nsp'].shape[0])*.5) # Checking first 25% and last 25% firing rate for drift
                scaled_fr = (np.nanmean(data['model_nsp'][:f25], axis=0)/np.nanstd(data['model_nsp'][:f25], axis=0) - np.nanmean(data['model_nsp'][l75:], axis=0)/np.nanstd(data['model_nsp'][l75:], axis=0))/params['model_dt']
                bad_cells = np.where((mean_thresh | (np.abs(scaled_fr)>4)))[0] # Locating bad units      
                np.save(params['save_dir_fm']/'bad_cells_{}.npy'.format(params['data_name_fm']),bad_cells)
        else:
            bad_cells = np.load(params['save_dir_fm']/'bad_cells_{}.npy'.format(params['data_name_fm']))

        print('Tot_units: {}'.format(data['unit_nums'].shape))
        data['model_nsp'] = np.delete(data['model_nsp'],bad_cells,axis=1) # removing bad units
        data['unit_nums'] = np.delete(data['unit_nums'],bad_cells,axis=0) # removing bad units
        print('Good_units: {}'.format(data['unit_nums'].shape))
        
    data['model_dth'] = np.diff(data['model_th'],append=0)
    data['model_dphi'] = np.diff(data['model_phi'],append=0)
    FM_move_avg = np.load(params['save_dir_fm']/'FM_MovAvg_{}_dt{:03d}.npy'.format(params['data_name_fm'],int(params['model_dt']*1000)))
    data['model_th'] = data['model_th'] - FM_move_avg[0,0]
    data['model_phi'] = (data['model_phi'] - FM_move_avg[0,1])
    if do_norm:
        data['model_vid_sm'] = (data['model_vid_sm'] - np.mean(data['model_vid_sm'],axis=0))/np.nanstd(data['model_vid_sm'],axis=0)
        data['model_vid_sm'][np.isnan(data['model_vid_sm'])]=0
        data['model_th'] = (data['model_th'])/FM_move_avg[1,0] # np.nanstd(data['model_th'],axis=0) 
        data['model_phi'] = (data['model_phi'])/FM_move_avg[1,1] # np.nanstd(data['model_phi'],axis=0) 
        if params['free_move']:
            data['model_roll'] = (data['model_roll'] - FM_move_avg[0,2])/FM_move_avg[1,2]
            data['model_pitch'] = (data['model_pitch'] - FM_move_avg[0,3])/FM_move_avg[1,3]
            data['model_roll'] = medfilt(data['model_roll'],move_medwin)
            data['model_pitch'] = medfilt(data['model_pitch'],move_medwin)
            if params['use_spdpup']:
                data['model_speed'] = (data['model_speed']-FM_move_avg[0,4])/FM_move_avg[1,4]
                data['model_eyerad'] = (data['model_eyerad']-FM_move_avg[0,5])/FM_move_avg[1,5]
        else:
            # data['model_roll'] = (0 - FM_move_avg[0,2])/FM_move_avg[1,2])
            data['model_pitch'] = (np.zeros(data['model_phi'].shape) - FM_move_avg[0,3])/FM_move_avg[1,3]
    else:
        if params['free_move']:
            data['model_roll']   = (data['model_roll'] - FM_move_avg[0,2])
            data['model_pitch']  = (data['model_pitch'] - FM_move_avg[0,3])
            data['model_roll']   = medfilt(data['model_roll'],move_medwin)
            data['model_pitch']  = medfilt(data['model_pitch'],move_medwin)
            if params['use_spdpup']:
                data['model_speed']  = (data['model_speed'])
                data['model_eyerad'] = (data['model_eyerad'])
        else:
            data['model_pitch']  = (np.zeros(data['model_phi'].shape) - FM_move_avg[0,3])

    ##### Ensure no NaNs in data #####
    for key in data.keys():
        if (np.any(np.isnan(data[key]))) & ('vid' not in key):
            data[key][np.isnan(data[key])]=0

    return data,train_idx_list,test_idx_list


##### Load in Kfold Data #####
def load_Kfold_data(data,params,train_idx,test_idx):
    """ Create Train/Test splits 

    Args:
        data (dict): dictionary of formatted data
        params (dict): parameter dictionary holding key parameters for formatting.
        train_idx (array): training indecies
        test_idx (array): testing indecies

    Returns:
        data (dict): data dictionary with train/test splits
    """
    data['train_vid'] = data['model_vid_sm'][train_idx]
    data['test_vid'] = data['model_vid_sm'][test_idx]
    data['train_nsp'] = shuffle(data['model_nsp'][train_idx],random_state=42) if params['do_shuffle'] else data['model_nsp'][train_idx]
    data['test_nsp'] = shuffle(data['model_nsp'][test_idx],random_state=42) if params['do_shuffle'] else data['model_nsp'][test_idx]
    data['train_th'] = data['model_th'][train_idx]
    data['test_th'] = data['model_th'][test_idx]
    data['train_phi'] = data['model_phi'][train_idx]
    data['test_phi'] = data['model_phi'][test_idx]
    data['train_roll'] = data['model_roll'][train_idx] if params['free_move'] else []
    data['test_roll'] = data['model_roll'][test_idx] if params['free_move'] else []
    data['train_pitch'] = data['model_pitch'][train_idx]
    data['test_pitch'] = data['model_pitch'][test_idx]
    data['train_t'] = data['model_t'][train_idx]
    data['test_t'] = data['model_t'][test_idx]
    data['train_gz'] = data['model_gz'][train_idx] if params['free_move'] else []
    data['test_gz'] = data['model_gz'][test_idx] if params['free_move'] else []
    data['train_speed'] = data['model_speed'][train_idx] if (params['free_move']) else []
    data['test_speed'] = data['model_speed'][test_idx] if (params['free_move']) else []
    data['train_eyerad'] = data['model_eyerad'][train_idx] #if ((params['use_spdpup'])&params['free_move']) else []
    data['test_eyerad'] = data['model_eyerad'][test_idx] #if ((params['use_spdpup'])&params['free_move']) else []
    return data



if __name__ == '__main__': 
    import pytorchGLM as pglm
    # Input arguments
    args = pglm.arg_parser()
    ModelID = 1
    params, file_dict,exp = pglm.load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)

    data = load_aligned_data(file_dict, params, reprocess=True)