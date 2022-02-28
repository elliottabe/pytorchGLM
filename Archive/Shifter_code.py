import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path

import torch
import torch.nn as nn
from kornia.geometry.transform import Affine
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import io_dict_to_hdf5 as ioh5


##### Load preprocessed data #####
filepath = Path('/home/seuss/Goeppert/nlab-nas/freely_moving_ephys/ephys_recordings/070921/J553RT/fm1/ModelData_dt025_rawWorldCam_50ds.h5')
model_data_all = ioh5.load(filepath)

''' 
All data aligned to 25ms bins
model_active: zscored speed of mice to determine inactive times, threshold active > .5
model_gz: gyroz data
model_nsp: spikes per bin
model_th: horizontal angle of the eye
model_phi: vertical angle of the eye
model_pitch: vertical angle of the head
model_roll: angle of roll along head axis
model_t: time during the recording
model_vid_sm: raw downsampled video
model_vis_sm_shift: eye corrected downsampled video
unit_num: channel number for each unit
'''
model_move = np.hstack((model_data_all['model_th'][:, np.newaxis], model_data_all['model_phi'][:, np.newaxis], model_data_all['model_pitch'][:, np.newaxis],model_data_all['model_roll'][:, np.newaxis]))

##### Transforming Raw video with shifter network #####
Kfold = 0
shift_in=3
shift_hidden=50
shift_out=3
# path to trained shifter data. h5 holds all the weights across all fits and minimum in test loss is used to shift video
save_model_fm = Path('/home/seuss/Research/SensoryMotorPred_Data/data/070921/J553RT/fm1/GLM_Network/MovModel1/version_0/GLM_Pytorch_VisShifter_NoL1_dt050_T01_MovModel1_NB2000_Kfold00_all.h5')

shifter_nn = nn.Sequential(
                nn.Linear(shift_in,shift_hidden),
                nn.Softplus(),
                nn.Linear(shift_hidden, shift_out))#.to(device)
GLM_Shifter = ioh5.load(save_model_fm)
best_shifter=np.nanargmin(np.nanmean(GLM_Shifter['loss_regcv'][0],axis=-1))
reg_alph=a=0; l=best_shifter
pretrained_dict = {'.'.join(k.split('.')[1:]): torch.from_numpy(v[a,l]) for k, v in GLM_Shifter.items() if 'shift' in k}
shifter_nn.load_state_dict(pretrained_dict)

ds = 2
shift_out = shifter_nn(torch.from_numpy(model_move[:,(0,1,3)].astype(np.float32)))#.to(device))
shift = Affine(angle=torch.clamp(shift_out[:,-1],min=-45,max=45),translation=torch.clamp(shift_out[:,:2]*ds,min=-20*ds,max=20*ds))
vid_tensor=torch.from_numpy(model_data_all['model_vid_sm'][:,np.newaxis].astype(np.float32))#.to(device)
model_vid_sm_shift = shift(vid_tensor.contiguous()).detach().numpy().squeeze()