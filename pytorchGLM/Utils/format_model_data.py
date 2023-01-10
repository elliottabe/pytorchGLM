import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from filelock import FileLock


from pytorchGLM.Utils.format_raw_data import *
from pytorchGLM.params import *

def load_datasets(file_dict,params,single_trial=False):
    """Load dataset and prepare datasets for ray tune. 

    Args:
        file_dict (dict): dictionary containing paths to raw data.
        params (dict): key parameters dictionary
        single_trial (bool, optional): single_trial=True, used for debuging. Defaults to False.

    Returns:
        train_dataset (Dataset): train dataset for pytorch. Grouped shuffled data
        test_dataset  (Dataset): test dataset for pytorch. Grouped shuffled data
        network_config (dict): Dictionary containing parameters for network
    """
    data = load_aligned_data(file_dict, params, reprocess=False)
    data,train_idx_list,test_idx_list = format_data(data, params,do_norm=True,thresh_cells=True,cut_inactive=True)
    train_idx = train_idx_list[0]
    test_idx = test_idx_list[0]
    data = load_Kfold_data(data,params,train_idx,test_idx)
    xtr, xte, xtr_pos, xte_pos, ytr, yte, meanbias = format_pytorch_data(data,params,train_idx,test_idx)
    network_config = make_network_config(params,single_trial=single_trial)
    with FileLock(params['save_model']/'data.lock'):
        train_dataset = FreeMovingEphysDataset(xtr,xtr_pos,ytr)
        test_dataset  = FreeMovingEphysDataset(xte,xte_pos,yte)
    return train_dataset, test_dataset, network_config

def get_modeltype(params,load_for_training=False):
    """Creates model name based on params configuation

    Args:
        params (dict): parameter dictionary holding key parameters
        load_for_training (bool, optional): If loading for shifter. Defaults to False.

    Returns:
        params or model_type: params with model_type key or model_type string
    """
    if load_for_training==False:
        if params['ModelID'] == 0:
            model_type = 'Pytorch_Mot'
        elif params['ModelID'] == 1:
            model_type = 'Pytorch_Vis'
        elif params['ModelID'] == 2:
            model_type = 'Pytorch_Add'
        elif params['ModelID'] == 3:
            model_type = 'Pytorch_Mul'
    else:
        model_type = 'Pytorch_Vis'

    if params['train_shifter']:
        params['save_model_shift'] = params['save_model'].parent.parent / 'Shifter'
        params['save_model_shift'].mkdir(parents=True, exist_ok=True)
        params['NoL1'] = True
        params['do_norm']=True
        model_type = model_type + 'Shifter'
        if params['shifter_5050']:
            if params['shifter_5050_run']:
                model_type = model_type + 'Train_1'
            else: 
                model_type = model_type + 'Train_0'
    else:
        if params['shifter_5050']:
            if params['shifter_5050_run']:
                model_type = model_type + '1'
            else: 
                model_type = model_type + '0'

    if params['EyeHead_only']:
        if params['EyeHead_only_run']==True:
            model_type = model_type + '_EyeOnly'
        else:
            model_type = model_type + '_HeadOnly'
    if params['NoShifter']:
        model_type = model_type + 'NoShifter'

    if params['only_spdpup']:
        model_type = model_type + '_onlySpdPup'
    elif params['use_spdpup']:
        model_type = model_type + '_SpdPup'
    if params['NoL1']:
        model_type = model_type + '_NoL1'
    if params['NoL2']:
        model_type = model_type + '_NoL2'
    if params['SimRF']:
        model_type = model_type + '_SimRF'
    if load_for_training==False:
        params['model_type'] = model_type
        return params
    else: 
        return model_type


def format_pytorch_data(data,params,train_idx,test_idx, device='cuda'):
    if params['free_move']:
        if params['train_shifter']:
            pos_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis],data['train_pitch'][:, np.newaxis]))
            pos_test  = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis],data['test_pitch'][:, np.newaxis]))
            model_pos = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis],data['model_pitch'][:, np.newaxis]))
            params['shift_in'] = model_pos.shape[-1]
            params['shift_out'] = model_pos.shape[-1]
        else:
            pos_train,pos_test,model_pos = [],[],[]
            for key in params['position_vars']:
                pos_train.append(data['train_'+key][:,np.newaxis])
                pos_test.append(data['test_'+key][:,np.newaxis])
                model_pos.append(data['model_'+key][:,np.newaxis])
            pos_train = np.hstack(pos_train)
            pos_test  = np.hstack(pos_test)
            model_pos = np.hstack(model_pos)
    else: 
        pos_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis], data['train_pitch'][:, np.newaxis], np.zeros(data['train_phi'].shape)[:, np.newaxis]))
        pos_test  = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis], data['test_pitch'][:, np.newaxis], np.zeros(data['test_phi'].shape)[:, np.newaxis]))
        model_pos = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis], data['model_pitch'][:, np.newaxis], np.zeros(data['model_phi'].shape)[:, np.newaxis]))

    ##### Save dimensions #####
    params['nks'] = np.shape(data['train_vid'])[1:]
    params['nk'] = params['nks'][0]*params['nks'][1]*params['nt_glm_lag']
    if params['train_shifter']:
        rolled_vid = np.hstack([np.roll(data['model_vid_sm'], nframes, axis=0) for nframes in params['lag_list']])
        move_quantiles = np.quantile(model_pos,params['quantiles'],axis=0)
        train_range = np.all(((pos_train>move_quantiles[0]) & (pos_train<move_quantiles[1])),axis=1)
        test_range = np.all(((pos_test>move_quantiles[0]) & (pos_test<move_quantiles[1])),axis=1)
        x_train = rolled_vid[train_idx].reshape((len(train_idx), params['nt_glm_lag'])+params['nks']).astype(np.float32)[train_range]
        x_test = rolled_vid[test_idx].reshape((len(test_idx), params['nt_glm_lag'])+params['nks']).astype(np.float32)[test_range]
        pos_train = pos_train[train_range]
        pos_test = pos_test[test_range]
        ytr = torch.from_numpy(data['train_nsp'][train_range].astype(np.float32))
        yte = torch.from_numpy(data['test_nsp'][test_range].astype(np.float32))
    elif params['NoShifter']:
        if params['crop_input'] != 0:
            model_vid_sm = data['model_vid_sm'][:,params['crop_input']:-params['crop_input'],params['crop_input']:-params['crop_input']]
        rolled_vid = np.hstack([np.roll(model_vid_sm, nframes, axis=0) for nframes in params['lag_list']])
        x_train = rolled_vid[train_idx].reshape(len(train_idx), -1).astype(np.float32)
        x_test = rolled_vid[test_idx].reshape(len(test_idx), -1).astype(np.float32)
        ytr = torch.from_numpy(data['train_nsp'].astype(np.float32))
        yte = torch.from_numpy(data['test_nsp'].astype(np.float32))
        params['nks'] = np.shape(model_vid_sm)[1:]
        params['nk'] = params['nks'][0]*params['nks'][1]*params['nt_glm_lag']
    else: 
        ##### Rework for new model format ######
        model_vid_sm_shift = ioh5.load(params['save_dir']/params['exp_name']/'ModelWC_shifted_dt{:03d}_ModelID{:d}.h5'.format(int(params['model_dt']*1000), 1))['model_vid_sm_shift']  # [:,5:-5,5:-5]
        if params['crop_input'] != 0:
            model_vid_sm_shift = model_vid_sm_shift[:,params['crop_input']:-params['crop_input'],params['crop_input']:-params['crop_input']]
        params['nks'] = np.shape(model_vid_sm_shift)[1:]
        params['nk'] = params['nks'][0]*params['nks'][1]*params['nt_glm_lag']
        rolled_vid = np.hstack([np.roll(model_vid_sm_shift, nframes, axis=0) for nframes in params['lag_list']]) 
        x_train = rolled_vid[train_idx].reshape(len(train_idx), -1).astype(np.float32)
        x_test = rolled_vid[test_idx].reshape(len(test_idx), -1).astype(np.float32)

        ytr = torch.from_numpy(data['train_nsp'].astype(np.float32))
        yte = torch.from_numpy(data['test_nsp'].astype(np.float32))

    ##### Convert to Tensors #####
    if params['ModelID']==0:
        xtr = torch.from_numpy(pos_train.astype(np.float32))
        xte = torch.from_numpy(pos_test.astype(np.float32))
    else:
        xtr = torch.from_numpy(x_train.astype(np.float32))
        xte = torch.from_numpy(x_test.astype(np.float32))
    xtr_pos = torch.from_numpy(pos_train.astype(np.float32))
    xte_pos = torch.from_numpy(pos_test.astype(np.float32))
    params['pos_features'] = xtr_pos.shape[-1]
    params['Ncells'] = ytr.shape[-1]
    
    if params['SimRF']:
        SimRF_file = params['save_dir'].parent.parent.parent/'021522/SimRF/fm1/SimRF_withL1_dt050_T01_Model1_NB10000_Kfold00_best.h5'
        SimRF_data = ioh5.load(SimRF_file)
        ytr = torch.from_numpy(SimRF_data['ytr'].astype(np.float32)).to(device)
        yte = torch.from_numpy(SimRF_data['yte'].astype(np.float32)).to(device)
        params['save_model'] = params['save_model'] / 'SimRF'
        params['save_model'].mkdir(parents=True, exist_ok=True)
        meanbias = torch.from_numpy(SimRF_data['bias_sim'].astype(np.float32)).to(device)
    else:
        meanbias = torch.mean(torch.tensor(data['model_nsp'], dtype=torch.float32), axis=0)
    return xtr, xte, xtr_pos, xte_pos, ytr, yte, meanbias



class FreeMovingEphysDataset(Dataset):
    def __init__(self, input0, input1, target):
        """Pytorch dataset class for 2 inputs and 1 output

        Args:
            input0 (Tensor): tensor for model input
            input1 (Tensor): tensor for model input
            target (Tensor): Target data
        """
        self.input0 = input0
        self.input1 = input1
        self.target = target
        
    def __len__(self):
        return len(self.input0)

    def __getitem__(self, idx):
        X = self.input0[idx]
        Y = self.target[idx]
        X2 = self.input1[idx]
        return X, X2, Y



def load_model(model,params,filename,meanbias=None,device='cuda:0'):
    """ Load model parameters

    Args:
        model (nn.Module): model to load parameters onto
        params (dict): key parameters dictionary
        filename (str): path to load pytorch model
        meanbias (Tensor, optional): mean bias for network. Defaults to None.
        device (str,optional): device to load network onto
    Returns:
        model (nn.Module): model with loaded parameters
    """
    checkpoint = torch.load(filename)
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if ('posNN' not in key) & ('shifter_nn' not in key):
            if 'weight' in key:
                state_dict[key] = checkpoint['model_state_dict'][key].repeat(1,params['nt_glm_lag'])
            elif 'bias' in key:
                state_dict[key] = checkpoint['model_state_dict'][key]
    if (params['SimRF']==True):
        SimRF_file = params['save_dir'].parent.parent.parent/'121521/SimRF/fm1/SimRF_withmodel_dt050_T01_Modemodel_NB10000_Kfold00_best.h5'
        SimRF_data = ioh5.load(SimRF_file)
        model.Cell_NN[0].weight.data = torch.from_numpy(SimRF_data['sta'].astype(np.float32).T).to(params['device'])
        model.Cell_NN[0].bias.data = torch.from_numpy(SimRF_data['bias_sim'].astype(np.float32)).to(params['device'])
    if meanbias is not None:
        state_dict = model.state_dict()
        state_dict['Cell_NN.0.bias']=meanbias
    model.load_state_dict(state_dict)
    return model

def setup_model_training(model,params,network_config):
    """Set up optimizer and scheduler for training

    Args:
        model (nn.Module): Network model to train
        params (dict): key parameters 
        network_config (diot): dictionary of hyperparameters

    Returns:
        optimizer: pytorch optimizer
        scheduler: learning rate scheduler
    """
    param_list = []
    if params['train_shifter']:
        param_list.append({'params': list(model.shifter_nn.parameters()),'lr': network_config['lr_shift'],'weight_decay':.0001})
    for name,p in model.named_parameters():
        if params['ModelID']<2:
            if ('Cell_NN' in name):
                if ('weight' in name):
                    param_list.append({'params':[p],'lr':network_config['lr_w'],'weight_decay':network_config['L2_lambda']})
                elif ('bias' in name):
                    param_list.append({'params':[p],'lr':network_config['lr_b']})
        elif (params['ModelID']==2) | (params['ModelID']==3):
            if ('posNN' in name):
                if ('weight' in name):
                    param_list.append({'params':[p],'lr':network_config['lr_w'],'weight_decay':network_config['L2_lambda_m']})
                elif ('bias' in name):
                    param_list.append({'params':[p],'lr':network_config['lr_b']})

    optimizer = optim.Adam(params=param_list)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(params['Nepochs']/5))
    return optimizer, scheduler