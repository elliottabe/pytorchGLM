import os
import argparse
import glob
import sys 
import yaml 
import glob
import h5py 
import ray
import logging 
import json
import gc
import cv2
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import xarray as xr

from tqdm.auto import tqdm, trange
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import shift as imshift
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path('.').absolute().parent))
from utils import *
import io_dict_to_hdf5 as ioh5

@ray.remote
def shift_vid_parallel(x, world_vid, warp_mode, criteria, dt):
    xshift_t = []
    yshift_t = []
    cc_t = []
    for i in range(x,x+dt):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try: 
            (cc, warp_matrix) = cv2.findTransformECC(world_vid[i,:,:], world_vid[i+1,:,:], warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
            xshift = warp_matrix[0,2]
            yshift = warp_matrix[1,2]
        except:
            cc = np.nan
            xshift=np.nan
            yshift = np.nan
        xshift_t.append(xshift)
        yshift_t.append(yshift)
        cc_t.append(cc)
    return xshift_t, yshift_t, cc_t

@ray.remote
def shift_world_pt2(f, dt, world_vid, thInterp, phiInterp, ycorrection, xcorrection):
    if (f+dt) < world_vid.shape[0]:
        world_vid2 = np.zeros((dt,world_vid.shape[1],world_vid.shape[2]))
        for n, x in enumerate(range(f,f+dt)):
            world_vid2[n,:,:] = imshift(world_vid[x,:,:],(-np.int8(thInterp[x]*ycorrection[0] + phiInterp[x]*ycorrection[1]),
                                                         -np.int8(thInterp[x]*xcorrection[0] + phiInterp[x]*xcorrection[1])))
    else: 
        world_vid2 = np.zeros((world_vid.shape[0]-f,world_vid.shape[1],world_vid.shape[2]))
        for n,x in enumerate(range(f,world_vid.shape[0])):
            world_vid2[n,:,:] = imshift(world_vid[x,:,:],(-np.int8(thInterp[x]*ycorrection[0] + phiInterp[x]*ycorrection[1]),
                                                         -np.int8(thInterp[x]*xcorrection[0] + phiInterp[x]*xcorrection[1])))
    return world_vid2


def grab_aligned_data(goodcells, worldT, accT, img_norm, gz, groll, gpitch, th_interp, phi_interp, free_move=True, model_dt=0.025):
    # get number of good units
    n_units = len(goodcells)
    # simplified setup for GLM
    # these are general parameters (spike rates, eye position)
    n_units = len(goodcells)
    print('get timing')
    model_t = np.arange(0,np.max(worldT), model_dt)
    model_nsp = np.zeros((n_units,len(model_t)))

    # get spikes / rate
    print('get spikes')
    bins = np.append(model_t,model_t[-1]+model_dt)
    for i,ind in enumerate(goodcells.index):
        model_nsp[i,:],bins = np.histogram(goodcells.at[ind,'spikeT'],bins)

    # get eye position
    print('get eye')
    model_th = th_interp(model_t+model_dt/2)
    model_phi = phi_interp(model_t+model_dt/2)
    # del thInterp, phiInterp

    # get active times
    if free_move:
        interp = interp1d(accT,(gz-np.mean(gz))*7.5,bounds_error=False)
        model_gz = interp(model_t)
        model_active = np.convolve(np.abs(model_gz), np.ones(int(1/model_dt)), 'same') / len(np.ones(int(1/model_dt)))
        use = np.where((model_active>40))[0] # (np.abs(model_th)<10) & (np.abs(model_phi)<10) & 
        roll_interp = interp1d(accT,groll,bounds_error=False)
        pitch_interp = interp1d(accT,gpitch,bounds_error=False)
        model_roll = roll_interp(model_t)
        model_pitch = pitch_interp(model_t)
    else:
        model_roll  = []
        model_pitch = []
        model_gz    = []
        model_active = []
        use = np.where((np.abs(model_th)<10) & (np.abs(model_phi)<10))[0]

    # get video ready for GLM
    print('setting up video') 
    movInterp = interp1d(worldT, img_norm,'nearest', axis = 0,bounds_error = False) 
    testimg = movInterp(model_t[0])
    # testimg = cv2.resize(testimg,(int(np.shape(testimg)[1]*downsamp), int(np.shape(testimg)[0]*downsamp)))
    if free_move:
        testimg = testimg[5:-5,5:-5]; #remove area affected by eye movement correction
    model_vid_sm = np.zeros((len(model_t),int(np.shape(testimg)[0]),int(np.shape(testimg)[1])),dtype=float)
    for i in tqdm(range(len(model_t))):
        model_vid = movInterp(model_t[i] + model_dt/2)
        # smallvid = cv2.resize(model_vid,(int(np.shape(img_norm)[2]*downsamp),int(np.shape(img_norm)[1]*downsamp)), interpolation=cv2.INTER_AREA)
        if free_move:
            model_vid_sm[i,:] = model_vid[5:-5,5:-5]
        else: 
            model_vid_sm[i,:] = model_vid
    model_vid_sm[np.isnan(model_vid_sm)]=0
    del movInterp
    gc.collect()
    return model_vid_sm, model_nsp, model_t, model_th, model_phi, model_roll, model_pitch, model_active, model_gz

def load_ephys_data_aligned(file_dict, save_dir, free_move=True, has_imu=True, has_mouse=False, max_frames=60*60, model_dt=.025):
        
    ##### Align Data #####
    if (save_dir / 'ModelData_dt{:03d}.h5'.format(int(model_dt*1000))).exists():
        data = ioh5.load((save_dir / 'ModelData_dt{:03d}.h5'.format(int(model_dt*1000))))
    else:
        ##### Loading ephys experiment data #####
        print('Starting to Load Data')
        world_data = xr.open_dataset(file_dict['world'])
        world_vid_raw = np.uint8(world_data['WORLD_video'])
        # resize worldcam
        sz = world_vid_raw.shape # raw video size
        # if size is larger than the target 60x80, resize by 0.5
        if sz[1]>160:
            downsamp = 0.5
            world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
            for f in range(sz[0]):
                world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
        else:
            # if the worldcam has already been resized when the nc file was written in preprocessing, don't resize
            world_vid = world_vid_raw.copy()
        # world timestamps
        worldT = world_data.timestamps.copy()

        # # open the topdown camera nc file
        # top_data = xr.open_dataset(file_dict['top'])
        # # get the speed of the base of the animal's tail in the topdown tracking
        # # most points don't track well enough for this to be done with other parts of animal (e.g. head points)
        # topx = top_data.TOP1_pts.sel(point_loc='tailbase_x').values; topy = top_data.TOP1_pts.sel(point_loc='tailbase_y').values
        # topdX = np.diff(topx); topdY = np.diff(topy)
        # top_speed = np.sqrt(topdX**2, topdY**2) # speed of tailbase in topdown camera
        # topT = top_data.timestamps.copy() # read in time timestamps
        # top_vid = np.uint8(top_data['TOP1_video']) # read in top video
        
        # clear from memory
        del world_vid_raw
        gc.collect()

        # load IMU data
        if file_dict['imu'] is not None:
            print('opening imu data')
            imu_data = xr.open_dataset(file_dict['imu'])
            accT = imu_data.IMU_data.sample # imu timestamps
            acc_chans = imu_data.IMU_data # imu dample data
            # raw gyro values
            gx = np.array(acc_chans.sel(channel='gyro_x_raw'))
            gy = np.array(acc_chans.sel(channel='gyro_y_raw'))
            gz = np.array(acc_chans.sel(channel='gyro_z_raw'))
            # gyro values in degrees
            gx_deg = np.array(acc_chans.sel(channel='gyro_x'))
            gy_deg = np.array(acc_chans.sel(channel='gyro_y'))
            gz_deg = np.array(acc_chans.sel(channel='gyro_z'))
            # pitch and roll in deg
            groll = np.array(acc_chans.sel(channel='roll'))
            gpitch = np.array(acc_chans.sel(channel='pitch'))
        else:
            accT = []
            gz = []
            groll = []
            gpitch = []

        print('opening ephys data')
        # ephys data for this individual recording
        ephys_data = pd.read_json(file_dict['ephys'])
        # sort units by shank and site order
        ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)
        ephys_data = ephys_data.reset_index()
        ephys_data = ephys_data.drop('index', axis=1)
        # spike times
        ephys_data['spikeTraw'] = ephys_data['spikeT']

        print('opening eyecam data')
        # load eye data
        eye_data = xr.open_dataset(file_dict['eye'])
        eye_vid = np.uint8(eye_data['REYE_video'])
        eyeT = eye_data.timestamps.copy()

        # plot eye postion across recording
        eye_params = eye_data['REYE_ellipse_params']

        # define theta, phi and zero-center
        th = np.array((eye_params.sel(ellipse_params='theta') -
                      np.nanmean(eye_params.sel(ellipse_params='theta')))*180/3.14159)
        phi = np.array((eye_params.sel(ellipse_params='phi') -
                       np.nanmean(eye_params.sel(ellipse_params='phi')))*180/3.14159)



        # adjust eye/world/top times relative to ephys
        ephysT0 = ephys_data.iloc[0,12]
        eyeT = eye_data.timestamps  - ephysT0
        if eyeT[0]<-600:
            eyeT = eyeT + 8*60*60 # 8hr offset for some data
        worldT = world_data.timestamps - ephysT0
        if worldT[0]<-600:
            worldT = worldT + 8*60*60
        if free_move is False and has_mouse is True:
            accT = imu_data.IMU_data.sample - ephysT0

        dEye = np.diff(th)
        # # load IMU data
        # if file_dict['imu'] is not None:
        #     print('opening imu data')
        #     imu_data = xr.open_dataset(file_dict['imu'])
        #     accT = imu_data.IMU_data.sample - ephysT0 # imu timestamps
        #     acc_chans = imu_data.IMU_data # imu dample data
        #     # raw gyro values
        #     gx = np.array(acc_chans.sel(channel='gyro_x_raw'))
        #     gy = np.array(acc_chans.sel(channel='gyro_y_raw'))
        #     gz = np.array(acc_chans.sel(channel='gyro_z_raw'))
        #     # gyro values in degrees
        #     gx_deg = np.array(acc_chans.sel(channel='gyro_x'))
        #     gy_deg = np.array(acc_chans.sel(channel='gyro_y'))
        #     gz_deg = np.array(acc_chans.sel(channel='gyro_z'))
        #     # pitch and roll in deg
        #     groll = np.array(acc_chans.sel(channel='roll'))
        #     gpitch = np.array(acc_chans.sel(channel='pitch'))

        if file_dict['imu'] is not None:
            lag_range = np.arange(-0.2, 0.2, 0.002)
            cc = np.zeros(np.shape(lag_range))
            t1 = np.arange(5, len(dEye)/60-120, 20).astype(int) # was np.arange(5,1600,20), changed for shorter videos
            t2 = t1 + 60
            offset = np.zeros(np.shape(t1))
            ccmax = np.zeros(np.shape(t1))
            acc_interp = interp1d(accT, (gz-3)*7.5)
            for tstart in tqdm(range(len(t1))):
                for l in range(len(lag_range)):
                    try:
                        c, _ = nanxcorr(-dEye[t1[tstart]*60: t2[tstart]*60], acc_interp(eyeT[t1[tstart]*60:t2[tstart]*60]+lag_range[l]), 1)
                        cc[l] = c[1]
                    except:  # occasional problem with operands that cannot be broadcast togther because of different shapes
                        cc[l] = np.nan
                offset[tstart] = lag_range[np.argmax(cc)]
                ccmax[tstart] = np.max(cc)
            offset[ccmax < 0.1] = np.nan
            print('fitting regression to timing drift')
            # fit regression to timing drift
            model = LinearRegression()
            dataT = np.array(eyeT[t1*60 + 30*60])
            model.fit(dataT[~np.isnan(offset)].reshape(-1, 1),
                      offset[~np.isnan(offset)])
            offset0 = model.intercept_
            drift_rate = model.coef_

            accT = accT - (offset0 + accT*drift_rate)
            fig = plt.figure()
            plt.plot(dataT, offset,'.')
            plt.plot(dataT, offset0 + dataT*drift_rate)
            plt.xlabel('secs'); plt.ylabel('offset - secs')
            plt.title('offset0 = '+str(np.round(offset0,3))+' drift_rate = '+str(np.round(drift_rate,3)))
            fig.savefig((save_dir / 'OffsetCorrection.png'),transparent=False, facecolor='w')
            plt.close()
        else:
            offset0 = 0.1
            drift_rate = -0.000114
       
        print('adjusting cvamera times to match ephys')
        print('correcting ephys spike times for offset and timing drift')

        # spike times
        for i in ephys_data.index:
            ephys_data.at[i, 'spikeT'] = np.array(ephys_data.at[i, 'spikeTraw']) - (offset0 + np.array(ephys_data.at[i, 'spikeTraw']) * drift_rate)
        print('getting good cells')
        # select good cells from phy2
        goodcells = ephys_data.loc[ephys_data['group']=='good']
        units = goodcells.index.values
        # get number of good units
        n_units = len(goodcells)

        # if free_move is False and has_mouse is True:
        #     speedT = spd_tstamps - ephysT0
        # if free_move is True:
        #     topT = topT - ephysT0

        ##### Clear some memory #####
        del world_data, eye_data, ephys_data 
        gc.collect()

        ##### Correction to world cam #####
        if (save_dir / 'corrected_worldcam_dt{:03d}.npy'.format(int(model_dt*1000))).exists():
            world_vid = np.load(save_dir / 'corrected_worldcam_dt{:03d}.npy'.format(int(model_dt*1000)), mmap_mode='r')
            # get eye displacement for each worldcam frame
            th_interp = interp1d(eyeT, th, bounds_error=False)
            phi_interp = interp1d(eyeT, phi, bounds_error=False)
        else:
            start = time.time()
            # get eye displacement for each worldcam frame
            th_interp = interp1d(eyeT, th, bounds_error=False)
            phi_interp = interp1d(eyeT, phi, bounds_error=False)
            dth = np.diff(th_interp(worldT))
            dphi = np.diff(phi_interp(worldT))
            # calculate x-y shift for each worldcam frame  
            number_of_iterations = 5000
            termination_eps = 1e-4
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
            warp_mode = cv2.MOTION_TRANSLATION

            # Parallel Testing
            world_vid_r = ray.put(world_vid)
            warp_mode_r = ray.put(warp_mode)
            criteria_r = ray.put(criteria)
            dt = 60
            result_ids = []
            [result_ids.append(shift_vid_parallel.remote(i, world_vid_r, warp_mode_r, criteria_r, dt)) for i in range(0, max_frames, dt)]
            results_p = ray.get(result_ids)
            results_p = np.array(results_p).transpose(0,2,1).reshape(-1,3)

            xshift = results_p[:,0]
            yshift = results_p[:,1]
            cc = results_p[:,2]

            xmodel = LinearRegression()
            ymodel = LinearRegression()

            # eye data as predictors
            eyeData = np.zeros((max_frames,2))
            eyeData[:,0] = dth[0:max_frames]
            eyeData[:,1] = dphi[0:max_frames]
            # shift in x and y as outputs
            xshiftdata = xshift[0:max_frames]
            yshiftdata = yshift[0:max_frames]
            # only use good data
            # not nans, good correlation between frames, small eye movements (no sacccades, only compensatory movements)
            usedata = ~np.isnan(eyeData[:,0]) & ~np.isnan(eyeData[:,1]) & (cc>0.95)  & (np.abs(eyeData[:,0])<2) & (np.abs(eyeData[:,1])<2) & (np.abs(xshiftdata)<5) & (np.abs(yshiftdata)<5)

            # fit xshift
            xmodel.fit(eyeData[usedata,:],xshiftdata[usedata])
            xmap = xmodel.coef_
            xrscore = xmodel.score(eyeData[usedata,:],xshiftdata[usedata])
            # fit yshift
            ymodel.fit(eyeData[usedata,:],yshiftdata[usedata])
            ymap = ymodel.coef_
            yrscore = ymodel.score(eyeData[usedata,:],yshiftdata[usedata])
            warp_mat_duration = time.time() - start
            print("warp mat duration =", warp_mat_duration)
            del results_p, warp_mode_r, criteria_r, result_ids
            gc.collect()

            start = time.time()
            print('estimating eye-world calibration')
            xcorrection_r = ray.put(xmap.copy())
            ycorrection_r = ray.put(ymap.copy())
            print('shifting worldcam for eyes')

            thInterp_r = ray.put(th_interp(worldT))
            phiInterp_r = ray.put(phi_interp(worldT))

            dt = 1000
            result_ids2 = []
            [result_ids2.append(shift_world_pt2.remote(f, dt, world_vid_r, thInterp_r, phiInterp_r, ycorrection_r, xcorrection_r)) for f in range(0, world_vid.shape[0], dt)] # 
            results = ray.get(result_ids2)
            world_vid = np.concatenate(results,axis=0).astype(np.uint8)
            print('saving worldcam video corrected for eye movements')
            np.save(file=(save_dir / 'corrected_worldcam_dt{:03d}.npy'.format(int(model_dt*1000))), arr=world_vid)
            shift_world_duration = time.time() - start
            print("shift world duration =", shift_world_duration)
            print('Total Duration:', warp_mat_duration + shift_world_duration)

        ##### Calculating image norm #####
        print('Calculating Image Norm')
        start = time.time()
        sz = np.shape(world_vid)
        downsamp = 0.25
        world_vid_sm = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)))
        for f in range(sz[0]):
            world_vid_sm[f,:,:] = cv2.resize(world_vid[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
        std_im = np.std(world_vid_sm, axis=0, dtype=float)
        img_norm = ((world_vid_sm-np.mean(world_vid_sm,axis=0,dtype=float))/std_im).astype(float)
        std_im[std_im<20] = 0
        img_norm = (img_norm * (std_im>0)).astype(float)
        del world_vid, world_vid_sm
        gc.collect()
        img_norm_duration = time.time() - start
        print("img_norm duration =", img_norm_duration)

        start = time.time()
        model_vid_sm, model_nsp, model_t, model_th, model_phi, model_roll, model_pitch, model_active, model_gz = grab_aligned_data(
            goodcells, worldT, accT, img_norm, gz, groll, gpitch, th_interp, phi_interp, free_move=free_move, model_dt=model_dt)

        data = {'model_vid_sm': model_vid_sm,
                'model_nsp': model_nsp.T,
                'model_t': model_t,
                'model_th': model_th,
                'model_phi': model_phi,
                'model_roll': model_roll,
                'model_pitch': model_pitch,
                'model_active': model_active,
                'model_gz': model_gz,
                'unit_nums': units}
        
        ioh5.save( (save_dir / 'ModelData_dt{:03d}.h5'.format(int(model_dt*1000))), data)
        align_data_duration = time.time() - start
        print("align_data_duration =", align_data_duration)
    print('Done Loading Aligned Data')
    return data

