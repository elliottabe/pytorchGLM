
import sys 
import glob
import ray
import logging 
import gc
import cv2
import time
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import xarray as xr
import matplotlib.gridspec as gridspec

from tqdm.notebook import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import medfilt
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import binned_statistic

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from kornia.geometry.transform import Affine
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(str(Path('.').absolute()))
from utils import *
import io_dict_to_hdf5 as ioh5
from format_data import *
from models import *
from fit_GLM import *


def plot_shifter_output(data,params):
    model_vid_sm_shift2 = {}
    pdf_name = params['fig_dir']/ 'VisMov_{}_dt{:03d}_Lags{:02d}_MovModel{:d}_CellSummary.pdf'.format('Pytorch_VisMov_AddMul_NoL1_Shifter',int(params['model_dt']*1000),params['nt_glm_lag'], MovModel)
    with PdfPages(pdf_name) as pdf:
        for l in tqdm(range(len(params['lambdas']))):
            model_name = 'GLM_{}_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_alph{}_lam{}_Kfold{:01d}.pth'.format(model_type,'UC',int(params['model_dt']*1000), params['nt_glm_lag'], MovModel, 5000,a,l,Kfold)
            # model_name = 'GLMShifter_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_pretrain_xyz.pth'.format(WC_type,int(params['model_dt']*1000), params['nt_glm_lag'] MovModel, Nbatches)
            checkpoint = torch.load(params['save_model']/model_name)
            l1.load_state_dict(checkpoint['model_state_dict'])

            ang_sweepx,ang_sweepy,ang_sweepz = np.meshgrid(np.arange(-50,50),np.arange(-50,50),np.arange(-50,50),sparse=False,indexing='ij')
            shift_mat = np.zeros((3,) + ang_sweepx.shape)
            for i in range(ang_sweepx.shape[0]):
                for j in range(ang_sweepy.shape[1]):
                    ang_sweep = torch.from_numpy(np.vstack((ang_sweepx[i,j,:],ang_sweepy[i,j,:],ang_sweepz[i,j,:])).astype(np.float32).T).to(device)
                    shift_vec = l1.shifter_nn(ang_sweep).detach().cpu().numpy()
                    shift_mat[0,i,j] = shift_vec[:,0]
                    shift_mat[1,i,j] = shift_vec[:,1]
                    shift_mat[2,i,j] = shift_vec[:,2]

                
            fig, ax = plt.subplots(1,4,figsize=(20,5))
            crange = np.max(np.abs(shift_mat[:2]))
            im1=ax[0].imshow(shift_mat[0,:,:,40].T,vmin=-crange, vmax=crange, origin='lower', cmap='RdBu_r')
            cbar1 = add_colorbar(im1)
            ax[0].set_xticks(np.arange(0,90,20))
            ax[0].set_xticklabels(np.arange(-40,50,20))
            ax[0].set_yticks(np.arange(0,90,20))
            ax[0].set_yticklabels(np.arange(-40,50,20))
            ax[0].set_xlabel('Theta')
            ax[0].set_ylabel('Phi')
            ax[0].set_title('Horizontal Shift')


            im2=ax[1].imshow(shift_mat[1,:,:,40].T,vmin=-crange, vmax=crange, origin='lower', cmap='RdBu_r')
            cbar2 = add_colorbar(im2)
            ax[1].set_xticks(np.arange(0,90,20))
            ax[1].set_xticklabels(np.arange(-40,50,20))
            ax[1].set_yticks(np.arange(0,90,20))
            ax[1].set_yticklabels(np.arange(-40,50,20))
            ax[1].set_xlabel('Theta')
            ax[1].set_ylabel('Phi')
            ax[1].set_title('Vertical Shift')

            crange = np.max(np.abs(shift_mat[2]))
            im3=ax[2].imshow(shift_mat[2,:,40,:].T,vmin=-crange, vmax=crange, origin='lower', cmap='RdBu_r')
            cbar3 = add_colorbar(im3)
            ax[2].set_xticks(np.arange(0,90,20))
            ax[2].set_xticklabels(np.arange(-40,50,20))
            ax[2].set_yticks(np.arange(0,90,20))
            ax[2].set_yticklabels(np.arange(-40,50,20))
            ax[2].set_xlabel('Theta')
            ax[2].set_ylabel('Pitch')
            ax[2].set_title('Rotational: Phi=0')

            im4=ax[3].imshow(shift_mat[2,40,:,:].T,vmin=-crange, vmax=crange, origin='lower', cmap='RdBu_r')
            cbar4 = add_colorbar(im4)
            ax[3].set_xticks(np.arange(0,90,20))
            ax[3].set_xticklabels(np.arange(-40,50,20))
            ax[3].set_yticks(np.arange(0,90,20))
            ax[3].set_yticklabels(np.arange(-40,50,20))
            ax[3].set_xlabel('Phi')
            ax[3].set_ylabel('Pitch')
            ax[3].set_title('Rotational: Theta=0')
            plt.tight_layout()

            fig.savefig(fig_dir/'ThetaPhiPitch_Shifter_lam{}.png'.format(l), facecolor='white', transparent=True, bbox_inches='tight')
            pdf.savefig()
            plt.close()
            shift_out = l1.shifter_nn(torch.from_numpy(model_move[:,(0,1,3)].astype(np.float32)).to(device))
            shift = Affine(angle=shift_out[:,-1],translation=shift_out[:,:2])
            model_vid_sm_shift = shift(torch.from_numpy(data['model_vid_sm'][:,np.newaxis].astype(np.float32)).to(device)).detach().cpu().numpy().squeeze()

            model_vid_sm_shift2['model_vid_sm_shift{}'.format(l)] = model_vid_sm_shift
    ioh5.save(save_dir/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(int(params['model_dt']*1000), MovModel),model_vid_sm_shift2)
