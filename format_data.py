import os
import argparse
import sys 
import glob
import ray
import logging 
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
from ray.actor import ActorHandle
from typing import Tuple
from asyncio import Event
from sklearn.utils import shuffle
from scipy.signal import medfilt
from scipy.stats import binned_statistic

sys.path.append(str(Path('.').absolute().parent))
sys.path.append(str(Path('.').absolute()))
from utils import *
import io_dict_to_hdf5 as ioh5

# ProgressBar
@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return

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


def grab_aligned_data(goodcells, worldT, accT, img_norm, gz, groll, gpitch, th_interp, phi_interp, free_move=True, model_dt=0.05,pxcrop=5,do_worldcam_correction=True,**kwargs):
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
    return model_vid_sm, model_nsp, model_t, model_th, model_phi, model_roll, model_pitch, model_active, model_gz

def load_ephys_data_aligned(file_dict, save_dir, free_move=True, has_imu=True, has_mouse=False, max_frames=60*60, model_dt=.025, do_worldcam_correction=False, reprocess=False, medfiltbins=11, **kwargs):
        
    ##### Align Data #####
    if do_worldcam_correction:
        model_file = (save_dir / 'ModelData_dt{:03d}.h5'.format(int(model_dt*1000)))
    else: 
        model_file = save_dir / 'ModelData_dt{:03d}_rawWorldCam.h5'.format(int(model_dt*1000))
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
        # if free_move == True:
        #     print('opening top data')
        #     # open the topdown camera nc file
        #     top_data = xr.open_dataset(file_dict['top'])
        #     # get the speed of the base of the animal's tail in the topdown tracking
        #     # most points don't track well enough for this to be done with other parts of animal (e.g. head points)
        #     topx = top_data.TOP1_pts.sel(point_loc='tailbase_x').values; topy = top_data.TOP1_pts.sel(point_loc='tailbase_y').values
        #     topdX = np.diff(topx); topdY = np.diff(topy)
        #     top_speed = np.sqrt(topdX**2, topdY**2) # speed of tailbase in topdown camera
        #     topT = top_data.timestamps.copy() # read in time timestamps
        #     top_vid = np.uint8(top_data['TOP1_video']) # read in top video
        #     # clear from memory
        #     del top_data
        #     gc.collect()
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
            spdVals = speed_data.BALL_data
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
        th = np.array((eye_params.sel(ellipse_params = 'theta'))*180/np.pi)# -np.nanmean(eye_params.sel(ellipse_params = 'theta'))
        phi = np.array((eye_params.sel(ellipse_params = 'phi'))*180/np.pi)# -np.nanmean(eye_params.sel(ellipse_params = 'phi'))
        # if free_move:
        #     FM_move_avg = np.zeros((2,4))
        #     FM_move_avg[:,0] = np.array([np.nanmean(th),np.nanstd(th)])
        #     FM_move_avg[:,1] = np.array([np.nanmean(phi),np.nanstd(phi)])
        #     FM_move_avg[:,2] = np.array([np.nanmean(groll),np.nanstd(groll)])
        #     FM_move_avg[:,3] = np.array([np.nanmean(gpitch),np.nanstd(gpitch)])
        #     np.save(save_dir/'FM_MovAvg_dt{:03d}.npy'.format(int(model_dt*1000)),FM_move_avg)
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
            accTraw = imu_data.IMU_data.sample - ephysT0
        if free_move is False and has_mouse is True:
            speedT = spd_tstamps - ephysT0
        # if free_move is True:
            # topT = topT - ephysT0
        
        ##### Clear some memory #####
        del eye_data 
        gc.collect()

        if file_dict['drop_slow_frames'] is True:
            # in the case that the recording has long time lags, drop data in a window +/- 3 frames around these slow frames
            isfast = np.diff(eyeT)<=0.05
            isslow = sorted(list(set(chain.from_iterable([list(range(int(i)-3,int(i)+4)) for i in np.where(isfast==False)[0]]))))
            th[isslow] = np.nan
            phi[isslow] = np.nan
        # check that deinterlacing worked correctly
        # plot theta and theta_switch
        # want theta_switch to be jagged, theta to be smooth
        # theta_switch_fig, th_switch = plot_param_switch_check(eye_params)
        # diagnostic_pdf.savefig()
        # plt.close()

        # calculate eye veloctiy
        dEye = np.diff(th)
        # check accelerometer / eye temporal alignment
        if file_dict['imu'] is not None:
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

        ##### Calculating image norm #####
        print('Calculating Image Norm')
        start = time.time()
        sz = np.shape(world_vid)
        downsamp = 0.25
        world_vid_sm = np.zeros((sz[0],int(sz[1]*downsamp),int(sz[2]*downsamp)))
        for f in range(sz[0]):
            world_vid_sm[f,:,:] = cv2.resize(world_vid[f,:,:],(int(sz[2]*downsamp),int(sz[1]*downsamp)))
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
            goodcells, worldT, accT, img_norm, gz, groll, gpitch, th_interp, phi_interp, free_move=free_move, model_dt=model_dt,do_worldcam_correction=do_worldcam_correction,**kwargs)
        if free_move:
            FM_move_avg = np.zeros((2,4))
            FM_move_avg[:,0] = np.array([np.nanmean(model_th),np.nanstd(model_th)])
            FM_move_avg[:,1] = np.array([np.nanmean(model_phi),np.nanstd(model_phi)])
            FM_move_avg[:,2] = np.array([np.nanmean(model_roll),np.nanstd(model_roll)])
            FM_move_avg[:,3] = np.array([np.nanmean(model_pitch),np.nanstd(model_pitch)])
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
                'unit_nums': units}
        
        ioh5.save(model_file, data)
        align_data_duration = time.time() - start
        print("align_data_duration =", align_data_duration)
    if do_worldcam_correction:
        print('Done Loading Aligned Data')
    else: 
        print('Done Loading Unaligned data')
    return data

def load_train_test(file_dict, save_dir, model_dt=.1, frac=.1, train_size=.7, do_shuffle=False, do_norm=False, free_move=True, has_imu=True, has_mouse=False, NKfold=1,thresh_cells=False,**kwargs):
    ##### Load in preprocessed data #####
    data = load_ephys_data_aligned(file_dict, save_dir, model_dt=model_dt, free_move=free_move, has_imu=has_imu, has_mouse=has_mouse,**kwargs)
    if free_move:
        ##### Find 'good' timepoints when mouse is active #####
        nan_idxs = []
        for key in data.keys():
            nan_idxs.append(np.where(np.isnan(data[key]))[0])
        good_idxs = np.ones(len(data['model_active']),dtype=bool)
        good_idxs[data['model_active']<.5] = False
        good_idxs[np.unique(np.hstack(nan_idxs))] = False
    else:
        good_idxs = np.where((np.abs(data['model_th'])<50) & (np.abs(data['model_phi'])<50))[0].astype(int)
    
    data['raw_nsp'] = data['model_nsp'].copy()
    ##### return only active data #####
    for key in data.keys():
        if (key != 'model_nsp') & (key != 'model_active') & (key != 'unit_nums'):
            data[key] = data[key][good_idxs] # interp_nans(data[key]).astype(float)
        elif (key == 'model_nsp'):
            data[key] = data[key][good_idxs]
        elif (key == 'unit_nums'):
            pass

    gss = GroupShuffleSplit(n_splits=NKfold, train_size=train_size, random_state=42)
    nT = data['model_nsp'].shape[0]
    groups = np.hstack([i*np.ones(int((frac*i)*nT) - int((frac*(i-1))*nT)) for i in range(1,int(1/frac)+1)])

    train_idx_list=[]
    test_idx_list = []
    for train_idx, test_idx in gss.split(np.arange(data['model_nsp'].shape[0]), groups=groups):
        train_idx_list.append(train_idx)
        test_idx_list.append(test_idx)

    # if thresh_cells:
        if free_move:
            if (kwargs['save_dir_hf']/'bad_cells.npy').exists():
                bad_cells = np.load(kwargs['save_dir_hf']/'bad_cells.npy')
            else:
                mean_thresh = np.nanmean(data['model_nsp']/model_dt,axis=0)<1
                f25,l75=int((data['model_nsp'].shape[0])*.5),int((data['model_nsp'].shape[0])*.5)
                scaled_fr = (np.nanmean(data['model_nsp'][:f25], axis=0)/np.nanstd(data['model_nsp'][:f25], axis=0) - np.nanmean(data['model_nsp'][l75:], axis=0)/np.nanstd(data['model_nsp'][l75:], axis=0))/model_dt
                bad_cells = np.where((mean_thresh | (np.abs(scaled_fr)>4)))[0]
                np.where(mean_thresh), np.where(np.abs(scaled_fr) > 4)
                np.save(kwargs['save_dir_hf']/'bad_cells.npy',bad_cells)
        else:
            bad_cells = np.load(kwargs['save_dir_hf']/'bad_cells.npy')
        data['model_nsp'] = np.delete(data['model_nsp'],bad_cells,axis=1)
        data['unit_nums'] = np.delete(data['unit_nums'],bad_cells,axis=0)
        
    data['model_dth'] = np.diff(data['model_th'],append=0)
    data['model_dphi'] = np.diff(data['model_phi'],append=0)
    FM_move_avg = np.load(kwargs['save_dir_fm']/'FM_MovAvg_dt{:03d}.npy'.format(int(model_dt*1000)))
    data['model_th'] = data['model_th'] - FM_move_avg[0,0]
    data['model_phi'] = data['model_phi'] - FM_move_avg[0,1]
    data['model_vid_sm'] = (data['model_vid_sm'] - np.mean(data['model_vid_sm'],axis=0))/np.nanstd(data['model_vid_sm'],axis=0)
    data['model_vid_sm'][np.isnan(data['model_vid_sm'])]=0
    if do_norm:
        data['model_th'] = (data['model_th'])/FM_move_avg[1,0] # np.nanstd(data['model_th'],axis=0) 
        data['model_phi'] = (data['model_phi'])/FM_move_avg[1,1] # np.nanstd(data['model_phi'],axis=0) 
        if free_move:
            data['model_roll'] = (data['model_roll'] - FM_move_avg[0,2])/FM_move_avg[1,2]
            data['model_pitch'] = (data['model_pitch'] - FM_move_avg[0,3])/FM_move_avg[1,3]
        else:
            # data['model_roll'] = (0 - FM_move_avg[0,2])/FM_move_avg[1,2])
            data['model_pitch'] = (np.zeros(data['model_phi'].shape) - FM_move_avg[0,3])/FM_move_avg[1,3]
    else:
        if free_move:
            data['model_roll'] = (data['model_roll'] - FM_move_avg[0,2])
            data['model_pitch'] = (data['model_pitch'] - FM_move_avg[0,3])
        else:
            data['model_pitch'] = (np.zeros(data['model_phi'].shape) - FM_move_avg[0,3])
            ### 0 - fm_pitch/sd_pitch_FM
    return data,train_idx_list,test_idx_list

def load_Kfold_forPlots(params, file_dict={}, Kfold=0, dataset_type='test',thresh_fr = 1, tuning_thresh = .2):
    params['do_norm']=False
    data, train_idx_list, test_idx_list = load_train_test(file_dict, **params)
    train_idx = train_idx_list[Kfold]
    test_idx = test_idx_list[Kfold]
    data = load_Kfold_data(data,train_idx,test_idx,params)
    locals().update(data)
    locals().update(params)

    if params['free_move']:
        move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis],data['train_roll'][:, np.newaxis], data['train_pitch'][:, np.newaxis]))
        move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis],data['test_roll'][:, np.newaxis], data['test_pitch'][:, np.newaxis]))
        model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis],data['model_roll'][:, np.newaxis], data['model_pitch'][:, np.newaxis]))
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

    sys.path.append(str(Path('.').absolute().parent))
    from utils import *
    import io_dict_to_hdf5 as ioh5
    from format_data import load_ephys_data_aligned

    pd.set_option('display.max_rows', None)
    FigPath = check_path(Path('~/Research/SensoryMotorPred_Data').expanduser(),'Figures/Encoding')

    ray.init(
        ignore_reinit_error=True,
        logging_level=logging.ERROR,
    )
    model_dt=.05
    free_move = False
    if free_move:
        stim_type = 'fm1'
    else:
        stim_type = 'hf1_wn' # 'fm1' # 
    date_ani = '070921/J553RT'  # '062921/G6HCK1ALTRN'
    data_dir  = Path('~/Goeppert/freely_moving_ephys/ephys_recordings/').expanduser() / date_ani / stim_type
    save_dir  = check_path(Path('~/Research/SensoryMotorPred_Data/data/').expanduser() / date_ani, stim_type)
    FigPath = check_path(Path('~/Research/SensoryMotorPred_Data').expanduser(),'Figures/Encoding')

    FigPath = check_path(FigPath, stim_type)
    save_dir,data_dir,FigPath
    file_dict = {'cell': 0,
                'drop_slow_frames': True,
                'ephys': list(data_dir.glob('*ephys_merge.json'))[0].as_posix(),
                'ephys_bin': list(data_dir.glob('*Ephys.bin'))[0].as_posix(),
                'eye': list(data_dir.glob('*REYE.nc'))[0].as_posix(),
                'imu': list(data_dir.glob('*imu.nc'))[0].as_posix() if stim_type=='fm1' else None,
                'mapping_json': '/home/seuss/Research/Github/FreelyMovingEphys/probes/channel_maps.json',
                'mp4': True,
                 'name': '070921_J553RT_control_Rig2_'+stim_type,  # 070921_J553RT
                'probe_name': 'DB_P128-6',
                'save': data_dir.as_posix(),
                'speed': list(data_dir.glob('*speed.nc'))[0].as_posix() if stim_type=='hf1_wn' else None,
                'stim_type': 'light',
                'top': list(data_dir.glob('*TOP1.nc'))[0].as_posix() if stim_type=='fm1' else None,
                'world': list(data_dir.glob('*world.nc'))[0].as_posix(),}

    data = load_ephys_data_aligned(file_dict, save_dir, model_dt=model_dt, free_move=free_move, has_imu=True, has_mouse=False,)
    ray.shutdown()
