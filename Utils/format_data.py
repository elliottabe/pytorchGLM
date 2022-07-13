########## This .py file contains the functions related to formatting data assuming Niell Lab preprocessing pipeline ##########

import argparse
import ray
import gc
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import xarray as xr

from tqdm.auto import tqdm
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import shift as imshift
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_pdf import PdfPages
from itertools import chain
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple
from asyncio import Event
from sklearn.utils import shuffle
from scipy.signal import medfilt
from scipy.stats import binned_statistic

from Utils.utils import *
import Utils.io_dict_to_hdf5 as ioh5


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


def grab_aligned_data(goodcells, worldT, accT, img_norm, gz, groll, gpitch, th_interp, phi_interp, top_speed, top_interp, eyerad, eyerad_interp, free_move=True, model_dt=0.05,pxcrop=5,do_worldcam_correction=True,**kwargs):

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
    model_eyerad = eyerad_interp(model_t+model_dt/2)
    # del thInterp, phiInterp

    # get active times
    if free_move:
        interp = interp1d(accT, (gz-np.mean(gz))*7.5,bounds_error=False)
        model_gz = interp(model_t) 
        model_active = np.convolve(np.abs(model_gz), np.ones(int(1/model_dt)), 'same') / len(np.ones(int(1/model_dt)))
        use = np.where((model_active>40))[0] # (np.abs(model_th)<10) & (np.abs(model_phi)<10) & 
        roll_interp = interp1d(accT,groll,bounds_error=False)
        pitch_interp = interp1d(accT,gpitch,bounds_error=False)
        model_roll = roll_interp(model_t)
        model_pitch = pitch_interp(model_t)
        model_speed = top_interp(model_t+model_dt/2)
    else:
        model_roll   = []
        model_pitch  = []
        model_gz     = []
        model_active = []
        model_speed  = []
        use = np.where((np.abs(model_th)<10) & (np.abs(model_phi)<10))[0]

    # get video ready for GLM
    print('setting up video') 
    movInterp = interp1d(worldT, img_norm,'nearest', axis = 0,bounds_error = False) 
    testimg = movInterp(model_t[0])
    # testimg = cv2.resize(testimg,(int(np.shape(testimg)[1]*downsamp), int(np.shape(testimg)[0]*downsamp)))
    if free_move & do_worldcam_correction:
        testimg = testimg[pxcrop:-pxcrop,pxcrop:-pxcrop]; #remove area affected by eye movement correction
    model_vid_sm = np.zeros((len(model_t),int(np.shape(testimg)[0]),int(np.shape(testimg)[1])),dtype=float)
    for i in tqdm(range(len(model_t))):
        model_vid = movInterp(model_t[i] + model_dt/2)
        # smallvid = cv2.resize(model_vid,(int(np.shape(img_norm)[2]*downsamp),int(np.shape(img_norm)[1]*downsamp)), interpolation=cv2.INTER_AREA)
        if free_move & do_worldcam_correction:
            model_vid_sm[i,:] = model_vid[pxcrop:-pxcrop,pxcrop:-pxcrop]
        else: 
            model_vid_sm[i,:] = model_vid
    model_vid_sm[np.isnan(model_vid_sm)]=0
    del movInterp
    gc.collect()
    return model_vid_sm, model_nsp, model_t, model_th, model_phi, model_roll, model_pitch, model_active, model_gz, model_speed, model_eyerad

def load_ephys_data_aligned(file_dict, save_dir, free_move=True, has_imu=True, has_mouse=False, max_frames=60*60, model_dt=.025, downsamp_vid = 4, do_worldcam_correction=False, reprocess=False, medfiltbins=11, **kwargs):
        
    ##### Align Data #####
    if do_worldcam_correction:
        model_file = (save_dir / 'ModelData_dt{:03d}.h5'.format(int(model_dt*1000)))
    else: 
        model_file = save_dir / 'ModelData_dt{:03d}_rawWorldCam_{:d}ds.h5'.format(int(model_dt*1000),int(downsamp_vid))
        # model_file = save_dir / 'ModelData_dt{:03d}_rawWorldCam.h5'.format(int(model_dt*1000))
        # model_file = save_dir / 'ModelData_dt050_rawWorldCam_2ds.h5'
    if (model_file.exists()) & (reprocess==False):
        data = ioh5.load(model_file)
    else:
        diagnostic_pdf = PdfPages(save_dir /(file_dict['name'] + '_diagnostic_analysis_figures.pdf'))
        # open worldcam
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
        # plot worldcam timing
        fig, axs = plt.subplots(1,2)
        axs[0].plot(np.diff(worldT)[0:-1:10]); axs[0].set_xlabel('every 10th frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('worldcam timing')
        axs[1].hist(np.diff(worldT),100);axs[1].set_xlabel('deltaT')
        plt.tight_layout()
        diagnostic_pdf.savefig()
        plt.close()
        # plot mean world image
        plt.figure()
        plt.imshow(np.mean(world_vid,axis=0)); plt.title('mean world image')
        plt.tight_layout()
        diagnostic_pdf.savefig()
        plt.close()
        if free_move == True:
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
        if file_dict['imu'] is not None:
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
            # gyro values in degrees
            gx_deg = np.array(acc_chans.sel(channel='gyro_x'))
            gy_deg = np.array(acc_chans.sel(channel='gyro_y'))
            gz_deg = np.array(acc_chans.sel(channel='gyro_z'))
            # pitch and roll in deg
            groll = medfilt(np.array(acc_chans.sel(channel='roll')),medfiltbins)
            gpitch = medfilt(np.array(acc_chans.sel(channel='pitch')),medfiltbins)
            # figure of gyro z
            plt.figure()
            plt.plot(gz_deg[0:100*60])
            plt.title('gyro z (deg)')
            plt.xlabel('frame')
            plt.tight_layout()
            diagnostic_pdf.savefig()
            plt.close()
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
        # plot eye timestamps
        fig, axs = plt.subplots(1,2)
        axs[0].plot(np.diff(eyeT)[0:-1:10]); axs[0].set_xlabel('every 10th frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('eyecam timing')
        axs[1].hist(np.diff(eyeT),100);axs[1].set_xlabel('deltaT')
        plt.tight_layout()
        diagnostic_pdf.savefig()
        plt.close()
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
        if free_move is True and has_imu is True:
            accTraw = accT - ephysT0
        if free_move is False and has_mouse is True:
            speedT = spd_tstamps - ephysT0
        if free_move is True:
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
        accT_correction_file = save_dir/'acct_correction.h5'
        # check accelerometer / eye temporal alignment
        if (accT_correction_file.exists()):# & (reprocess==False):
            accT_correction = ioh5.load(accT_correction_file)
            offset0    = accT_correction['offset0']
            drift_rate = accT_correction['drift_rate']
            accT = accTraw - (offset0 + accTraw*drift_rate)
            found_good_offset = True
        else:
            if (file_dict['imu'] is not None):
                print('checking accelerometer / eye temporal alignment')
                # plot eye velocity against head movements
                plt.figure
                plt.plot(eyeT[0:-1],-dEye,label='-dEye')
                plt.plot(accTraw,gz_deg,label='gz')
                plt.legend()
                plt.xlim(0,10); plt.xlabel('secs'); plt.ylabel('gyro (deg)')
                plt.tight_layout()
                diagnostic_pdf.savefig()
                plt.close()
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
                fig = plt.subplot(1,2,1)
                plt.plot(eyeT[t1*60],offset)
                plt.xlabel('secs'); plt.ylabel('offset (secs)')
                plt.subplot(1,2,2)
                plt.plot(eyeT[t1*60],ccmax)
                plt.xlabel('secs'); plt.ylabel('max cc')
                plt.tight_layout()
                diagnostic_pdf.savefig()
                plt.close()
                del ccmax, dEye
                gc.collect()
                if np.isnan(offset).all():
                    found_good_offset = False
                else:
                    found_good_offset = True

            if file_dict['imu'] is not None and found_good_offset is True:
                print('fitting regression to timing drift')
                # fit regression to timing drift
                model = LinearRegression()
                dataT = np.array(eyeT[t1*60 + 30*60])
                model.fit(dataT[~np.isnan(offset)].reshape(-1,1),offset[~np.isnan(offset)]) 
                offset0 = model.intercept_
                drift_rate = model.coef_
                fig = plt.figure()
                plt.plot(dataT, offset,'.')
                plt.plot(dataT, offset0 + dataT*drift_rate)
                plt.xlabel('secs'); plt.ylabel('offset - secs')
                plt.title('offset0 = '+str(np.round(offset0,3))+' drift_rate = '+str(np.round(drift_rate,3)))
                plt.tight_layout()
                diagnostic_pdf.savefig()
                plt.close()
                del dataT
                gc.collect()
            elif file_dict['speed'] is not None or found_good_offset is False:
                offset0 = 0.1
                drift_rate = -0.000114
            if file_dict['imu'] is not None:
                accT_correction = {'offset0': offset0, 'drift_rate': drift_rate}
                ioh5.save(accT_correction_file,accT_correction)
                accT = accTraw - (offset0 + accTraw*drift_rate)
                del accTraw
                gc.collect()


        print('correcting ephys spike times for offset and timing drift')
        for i in ephys_data.index:
            ephys_data.at[i,'spikeT'] = np.array(ephys_data.at[i,'spikeTraw']) - (offset0 + np.array(ephys_data.at[i,'spikeTraw']) *drift_rate)
        goodcells = ephys_data.loc[ephys_data['group']=='good']
        diagnostic_pdf.close()
        np.save(file=(save_dir / 'uncorrected_worldcam_dt{:03d}.npy'.format(int(model_dt*1000))), arr=world_vid)
        
        if do_worldcam_correction:
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
                if file_dict['imu'] is not None:
                    if (save_dir / 'FM_WorldShift_dt{:03d}.h5'.format(int(model_dt*1000))).exists():
                        world_shifts = ioh5.load((save_dir / 'FM_WorldShift_dt{:03d}.h5'.format(int(model_dt*1000))))
                        xmap = world_shifts['xmap']
                        ymap = world_shifts['ymap']
                        world_vid_r = ray.put(world_vid)
                        warp_mat_duration = 0
                    else:
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
                        world_shifts = {'xmap': xmap,
                                        'ymap': ymap,
                                        'xrscore': xrscore,
                                        'yrscore': yrscore,
                                        }
                        ioh5.save((save_dir / 'FM_WorldShift_dt{:03d}.h5'.format(int(model_dt*1000))), world_shifts)
                        warp_mat_duration = time.time() - start
                        print("warp mat duration =", warp_mat_duration)
                        del results_p, warp_mode_r, criteria_r, result_ids
                        gc.collect()
                elif file_dict['speed'] is not None:
                    world_shifts = ioh5.load((save_dir.parent / 'fm1' / 'FM_WorldShift_dt{:03d}.h5'.format(int(model_dt*1000))))
                    xmap = world_shifts['xmap']
                    ymap = world_shifts['ymap']
                    world_vid_r = ray.put(world_vid)
                    warp_mat_duration = 0
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
        else:
            th_interp = interp1d(eyeT, th, bounds_error=False)
            phi_interp = interp1d(eyeT, phi, bounds_error=False)
            eyerad_interp = interp1d(eyeT, eyerad, bounds_error=False)
            if free_move is True:
                top_interp = interp1d(topT, top_speed, bounds_error=False)
            else:
                top_interp = []

        ##### Calculating image norm #####
        print('Calculating Image Norm')
        start = time.time()
        sz = np.shape(world_vid)
        # downsamp = .5 #0.25
        world_vid_sm = np.zeros((sz[0],int(sz[1]/downsamp_vid),int(sz[2]/downsamp_vid)))
        for f in range(sz[0]):
            world_vid_sm[f,:,:] = cv2.resize(world_vid[f,:,:],(int(sz[2]/downsamp_vid),int(sz[1]/downsamp_vid)))
        std_im = np.std(world_vid_sm, axis=0, dtype=float)
        img_norm = ((world_vid_sm-np.mean(world_vid_sm,axis=0,dtype=float))/std_im).astype(float)
        std_im[std_im<20] = 0
        img_norm = (img_norm * (std_im>0)).astype(float)
        del world_vid, world_vid_sm
        gc.collect()
        img_norm_duration = time.time() - start
        print("img_norm duration =", img_norm_duration)

        start = time.time()
        model_vid_sm, model_nsp, model_t, model_th, model_phi, model_roll, model_pitch, model_active, model_gz, model_speed, model_eyerad = grab_aligned_data(
            goodcells, worldT, accT, img_norm, gz, groll, gpitch, th_interp, phi_interp, top_speed, top_interp, eyerad, eyerad_interp, 
            free_move=free_move, model_dt=model_dt,do_worldcam_correction=do_worldcam_correction,**kwargs)
        if free_move:
            FM_move_avg = np.zeros((2,6))
            FM_move_avg[:,0] = np.array([np.nanmean(model_th),np.nanstd(model_th)])
            FM_move_avg[:,1] = np.array([np.nanmean(model_phi),np.nanstd(model_phi)])
            FM_move_avg[:,2] = np.array([np.nanmean(model_roll),np.nanstd(model_roll)])
            FM_move_avg[:,3] = np.array([np.nanmean(model_pitch),np.nanstd(model_pitch)])
            FM_move_avg[:,4] = np.array([np.nanmean(model_speed),np.nanmax(model_speed)])
            FM_move_avg[:,5] = np.array([np.nanmean(model_eyerad),np.nanmax(model_eyerad)])
            np.save(save_dir/'FM_MovAvg_dt{:03d}.npy'.format(int(model_dt*1000)),FM_move_avg)
        ephys_file = save_dir / 'RawEphysData.h5'
        goodcells.to_hdf(ephys_file,key='goodcells', mode='w')
        data = {'model_vid_sm': model_vid_sm,
                'model_nsp': model_nsp.T,
                'model_t': model_t,
                'model_th': model_th,
                'model_phi': model_phi,
                'model_roll': model_roll,
                'model_pitch': model_pitch,
                'model_active': model_active,
                'model_gz': model_gz,
                'model_speed': model_speed, 
                'model_eyerad': model_eyerad,
                'unit_nums': units}
        
        ioh5.save(model_file, data)
        align_data_duration = time.time() - start
        print("align_data_duration =", align_data_duration)
    if do_worldcam_correction:
        print('Done Loading Aligned Data')
    else: 
        print('Done Loading Unaligned data')
    return data

def load_train_test(file_dict, save_dir, model_dt=.1, frac=.1, shifter_train_size=.5, test_train_size=.75, do_shuffle=False, do_norm=False, free_move=True, has_imu=True, has_mouse=False, NKfold=1,thresh_cells=False,cut_inactive=True,move_medwin=11,**kwargs):
    ##### Load in preprocessed data #####
    data = load_ephys_data_aligned(file_dict, save_dir, model_dt=model_dt, free_move=free_move, has_imu=has_imu, has_mouse=has_mouse,**kwargs)

    if free_move:
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
            if (key != 'model_nsp') & (key != 'model_active') & (key != 'unit_nums') & (key != 'model_vis_sm_shift'):
                if len(data[key])>0:
                    data[key] = data[key][good_idxs] # interp_nans(data[key]).astype(float)
            elif (key == 'model_nsp'):
                data[key] = data[key][good_idxs]
            elif (key == 'unit_nums') | (key == 'model_vis_sm_shift'):
                pass
        
    ##### Splitting data for shifter, then split for model training #####
    if kwargs['shifter_5050']==False:
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
            
            if kwargs['train_shifter']==False:
                if kwargs['shifter_5050_run']:
                    train_idx_list = test_idx_list_shifter
                    test_idx_list  = train_idx_list_shifter
                else:
                    train_idx_list = train_idx_list_shifter
                    test_idx_list  = test_idx_list_shifter
                # train_groups = np.random.choice(glist,size=int(len(glist)*test_train_size),replace=False)
                # train_idx = np.any(np.array([groups==train_groups[n] for n in np.arange(len(train_groups))]),axis=0)
                # train_idx_list.append(idx[train_idx])
                # test_idx_list.append(idx[(~train_idx)&(~sampled_inds)])
            else:
                if kwargs['shifter_5050_run']:
                    train_idx_list = test_idx_list_shifter
                    test_idx_list  = train_idx_list_shifter
                else:
                    train_idx_list = train_idx_list_shifter
                    test_idx_list  = test_idx_list_shifter

    if thresh_cells:
        print('Tot_units: {}'.format(data['unit_nums'].shape))
        if free_move:
            if (kwargs['save_dir_hf']/'bad_cells.npy').exists():
                bad_cells = np.load(kwargs['save_dir_hf']/'bad_cells.npy')
            else:
                mean_thresh = np.nanmean(data['model_nsp']/model_dt,axis=0)<1
                f25,l75=int((data['model_nsp'].shape[0])*.5),int((data['model_nsp'].shape[0])*.5)
                scaled_fr = (np.nanmean(data['model_nsp'][:f25], axis=0)/np.nanstd(data['model_nsp'][:f25], axis=0) - np.nanmean(data['model_nsp'][l75:], axis=0)/np.nanstd(data['model_nsp'][l75:], axis=0))/model_dt
                bad_cells = np.where((mean_thresh | (np.abs(scaled_fr)>4)))[0]
                np.save(kwargs['save_dir_hf']/'bad_cells.npy',bad_cells)
        else:
            bad_cells = np.load(kwargs['save_dir_hf']/'bad_cells.npy')
        data['model_nsp'] = np.delete(data['model_nsp'],bad_cells,axis=1)
        data['unit_nums'] = np.delete(data['unit_nums'],bad_cells,axis=0)
        
    data['model_dth'] = np.diff(data['model_th'],append=0)
    data['model_dphi'] = np.diff(data['model_phi'],append=0)
    FM_move_avg = np.load(kwargs['save_dir_fm']/'FM_MovAvg_dt{:03d}.npy'.format(int(model_dt*1000)))
    data['model_th'] = data['model_th'] - FM_move_avg[0,0]
    data['model_phi'] = (data['model_phi'] - FM_move_avg[0,1])
    data['model_vid_sm'] = (data['model_vid_sm'] - np.mean(data['model_vid_sm'],axis=0))/np.nanstd(data['model_vid_sm'],axis=0)
    data['model_vid_sm'][np.isnan(data['model_vid_sm'])]=0
    if do_norm:
        data['model_th'] = (data['model_th'])/FM_move_avg[1,0] # np.nanstd(data['model_th'],axis=0) 
        data['model_phi'] = (data['model_phi'])/FM_move_avg[1,1] # np.nanstd(data['model_phi'],axis=0) 
        if free_move:
            data['model_roll'] = (data['model_roll'] - FM_move_avg[0,2])/FM_move_avg[1,2]
            data['model_pitch'] = (data['model_pitch'] - FM_move_avg[0,3])/FM_move_avg[1,3]
            data['model_roll'] = medfilt(data['model_roll'],move_medwin)
            data['model_pitch'] = medfilt(data['model_pitch'],move_medwin)
            if kwargs['use_spdpup']:
                data['model_speed'] = (data['model_speed']-FM_move_avg[0,4])/FM_move_avg[1,4]
                data['model_eyerad'] = (data['model_eyerad']-FM_move_avg[0,5])/FM_move_avg[1,5]
        else:
            # data['model_roll'] = (0 - FM_move_avg[0,2])/FM_move_avg[1,2])
            data['model_pitch'] = (np.zeros(data['model_phi'].shape) - FM_move_avg[0,3])/FM_move_avg[1,3]
    else:
        if free_move:
            data['model_roll']   = (data['model_roll'] - FM_move_avg[0,2])
            data['model_pitch']  = (data['model_pitch'] - FM_move_avg[0,3])
            data['model_roll']   = medfilt(data['model_roll'],move_medwin)
            data['model_pitch']  = medfilt(data['model_pitch'],move_medwin)
            if kwargs['use_spdpup']:
                data['model_speed']  = (data['model_speed'])
                data['model_eyerad'] = (data['model_eyerad'])
        else:
            data['model_pitch']  = (np.zeros(data['model_phi'].shape) - FM_move_avg[0,3])
    return data,train_idx_list,test_idx_list

def load_Kfold_forPlots(params, file_dict={}, Kfold=0, dataset_type='test',thresh_fr = 1, tuning_thresh = .2):
    params['do_norm']=False
    data, train_idx_list, test_idx_list = load_train_test(file_dict, **params)
    train_idx = train_idx_list[Kfold]
    test_idx = test_idx_list[Kfold]
    data = load_Kfold_data(data,train_idx,test_idx,params)
    locals().update(data)
    locals().update(params)
    params['nks'] = np.shape(data['model_vid_sm'])[1:]
    if params['crop_input']!=0:
        params['nks'] = (params['nks'][0]-2*params['crop_input'],params['nks'][1]-2*params['crop_input'])
    params['nk'] = params['nks'][0]*params['nks'][1]*params['nt_glm_lag']
    params['Ncells'] = data['model_nsp'].shape[-1]

    if params['free_move']:
        if params['use_spdpup']:
            if params['only_spdpup']:
                move_train = np.hstack((data['train_speed'][:, np.newaxis],data['train_eyerad'][:, np.newaxis]))
                move_test = np.hstack((data['test_speed'][:, np.newaxis],data['test_eyerad'][:, np.newaxis]))
                model_move = np.hstack((data['model_speed'][:, np.newaxis],data['model_eyerad'][:, np.newaxis]))
                model_move = (model_move - np.nanmean(model_move,axis=0))
                move_test = model_move[test_idx]
                move_train = model_move[train_idx]
            else:
                move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis],data['train_pitch'][:, np.newaxis],data['train_roll'][:, np.newaxis],data['train_speed'][:, np.newaxis],data['train_eyerad'][:, np.newaxis]))
                move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis],data['test_pitch'][:, np.newaxis],data['test_roll'][:, np.newaxis],data['test_speed'][:, np.newaxis],data['test_eyerad'][:, np.newaxis]))
                model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis],data['model_pitch'][:, np.newaxis],data['model_roll'][:, np.newaxis],data['model_speed'][:, np.newaxis],data['model_eyerad'][:, np.newaxis]))
                model_move = (model_move - np.nanmean(model_move,axis=0))
                move_test = model_move[test_idx]
                move_train = model_move[train_idx]
        else:
            move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis],data['train_pitch'][:, np.newaxis],data['train_roll'][:, np.newaxis]))
            move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis],data['test_pitch'][:, np.newaxis],data['test_roll'][:, np.newaxis]))
            model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis],data['model_pitch'][:, np.newaxis],data['model_roll'][:, np.newaxis]))
            model_move = (model_move - np.nanmean(model_move,axis=0))
            move_test = model_move[test_idx]
            move_train = model_move[train_idx]
    else:
        move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis], np.zeros(data['train_phi'].shape)[:, np.newaxis], np.zeros(data['train_phi'].shape)[:, np.newaxis]))
        move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis], np.zeros(data['test_phi'].shape)[:, np.newaxis], np.zeros(data['test_phi'].shape)[:, np.newaxis]))
        model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis], np.zeros(data['model_phi'].shape)[:, np.newaxis], np.zeros(data['model_phi'].shape)[:, np.newaxis]))
    
    if dataset_type == 'train':
        nsp_raw = data['train_nsp']
        move_data = move_train.copy()
    else: 
        nsp_raw = data['test_nsp']
        move_data = move_test.copy()
        
    if params['free_move']:
        spk_percentile2 = np.arange(.125,1.125,.25)
        quartiles = np.arange(0,1.25,.25)
        tuning_curves = np.zeros((data['model_nsp'].shape[1],model_move.shape[-1],len(quartiles)-1))
        tuning_stds = np.zeros((data['model_nsp'].shape[1],model_move.shape[-1],1))
        tuning_curve_edges = np.zeros((data['model_nsp'].shape[1],model_move.shape[-1],len(quartiles)-1))
        ax_ylims = np.zeros((data['model_nsp'].shape[-1],model_move.shape[-1]))
        for i,modeln in enumerate(range(model_move.shape[-1])):
            for celln in np.arange(data['model_nsp'].shape[1]):
                metric = move_data[:,modeln]
                nranges = np.quantile(metric,quartiles)
                stat_range, edges, _ = binned_statistic(metric,nsp_raw[:,celln],statistic='mean',bins=nranges)
                stat_std, _, _ = binned_statistic(metric,nsp_raw[:,celln],statistic='std',bins=nranges)
                tuning_curves[celln,modeln] = stat_range/params['model_dt']
                edge_mids = np.quantile(metric,spk_percentile2)
                tuning_curve_edges[celln,modeln] = edge_mids
                tuning_stds[celln,modeln] = stat_std.max()
            ax_ylims[:,modeln] = np.nanmax(tuning_curves[:,modeln],axis=-1)
        tc_mod = (np.max(tuning_curves,axis=-1,keepdims=True)-np.min(tuning_curves,axis=-1,keepdims=True))/(np.max(tuning_curves,axis=-1,keepdims=True)+np.min(tuning_curves,axis=-1,keepdims=True))
        avg_fr = np.mean(tuning_curves,axis=(-1,-2)).squeeze()

        tuning_sig = tc_mod.copy()
        tuning_sig[avg_fr<thresh_fr,:,0] = np.nan
        tuning_sig2 = np.any(tuning_sig>tuning_thresh,axis=1).squeeze()
        tuning_idx = np.where(tuning_sig2)[0]
    else: 
        tuning_curves=tuning_stds=tuning_curve_edges=ax_ylims=tc_mod=avg_fr=tuning_sig=tuning_sig2=tuning_idx = None
    return data,move_train,move_test,model_move,nsp_raw,move_data,tuning_curves,tuning_stds,tuning_curve_edges,ax_ylims,tc_mod,avg_fr,tuning_sig,tuning_sig2,tuning_idx


##### Load in Kfold Data #####
def load_Kfold_data(data,train_idx,test_idx,params):
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
    data['train_dth'] = data['model_dth'][train_idx]
    data['test_dth'] = data['model_dth'][test_idx]
    data['train_dphi'] = data['model_dphi'][train_idx]
    data['test_dphi'] = data['model_dphi'][test_idx]
    data['train_gz'] = data['model_gz'][train_idx] if params['free_move'] else []
    data['test_gz'] = data['model_gz'][test_idx] if params['free_move'] else []
    data['train_speed'] = data['model_speed'][train_idx] if ((params['use_spdpup'])&params['free_move']) else []
    data['test_speed'] = data['model_speed'][test_idx] if ((params['use_spdpup'])&params['free_move']) else []
    data['train_eyerad'] = data['model_eyerad'][train_idx] if ((params['use_spdpup'])&params['free_move']) else []
    data['test_eyerad'] = data['model_eyerad'][test_idx] if ((params['use_spdpup'])&params['free_move']) else []
    return data


def f_add(alpha,stat_range,stat_all):
    return np.mean((stat_range - (stat_all+alpha))**2)

def f_mult(alpha,stat_range,stat_all):
    return np.mean((stat_range - (stat_all*alpha))**2)

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

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--free_move', type=str_to_bool, default=True)
    parser.add_argument('--prey_cap', type=str_to_bool, default=False)
    parser.add_argument('--fm_dark', type=str_to_bool, default=False)
    parser.add_argument('--date_ani', type=str, default='070921/J553RT') # '122021/J581RT')#
    parser.add_argument('--save_dir', type=str, default='~/Research/SensoryMotorPred_Data/data4/')
    parser.add_argument('--fig_dir', type=str, default='~/Research/SensoryMotorPred_Data/Figures2')
    parser.add_argument('--data_dir', type=str, default='~/Goeppert/nlab-nas/Dylan/freely_moving_ephys/ephys_recordings/')
    args = parser.parse_args()
    args = vars(args)
    # pd.set_option('display.max_rows', None)

    # ray.init(
    #     ignore_reinit_error=True,
    #     logging_level=logging.ERROR,
    # )

    model_dt=.05
    downsamp_vid = 4
    free_move = True
    fm_dark = False
    prey_cap=False
    if args['prey_cap']:
        fm_dir = 'fm1_prey'
    elif args['fm_dark']:
        fm_dir = 'fm1_dark'
    else:
        fm_dir = 'fm1'
    if args['free_move']:
        stim_type = fm_dir
    else:
        stim_type = 'hf1_wn'     
    dates_all = ['070921/J553RT' ,'101521/J559NC','102821/J570LT','110421/J569LT','122021/J581RT'] # '102621/J558NC' '062921/G6HCK1ALTRN',
    # dates_all = ['100821/J559TT', '101621/J559NC', '102721/J558NC', '110421/J558LT','110521/J569LT']
    date_ani = dates_all[0] #args['date_ani']
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

    file_dict = {'cell': 0,
                'drop_slow_frames': False, #True,
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

    data = load_ephys_data_aligned(file_dict, save_dir, model_dt=model_dt, free_move=free_move, has_imu=True, has_mouse=False, downsamp_vid=downsamp_vid, reprocess=True)
    # ray.shutdown()
