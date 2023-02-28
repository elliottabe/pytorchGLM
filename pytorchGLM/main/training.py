import warnings 
warnings.filterwarnings('ignore')

import numpy as np

from ray import tune
from ray import air
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True


import pytorchGLM.Utils.io_dict_to_hdf5 as ioh5
from pytorchGLM.Utils.utils import *
from pytorchGLM.params import *
# from pytorchGLM.Utils.format_raw_data import *
from pytorchGLM.Utils.format_model_data import *
from pytorchGLM.main.models import *

def train_network(network_config, xtr, xte, xtr_pos, xte_pos, ytr, yte, params={}, filename=None, meanbias=None):
    """ Function to train network. For Ray Tune, need to have named inputs. 

    Args:
        network_config (dict): Dictionary containing parameters for network
        params (dict): Key parameter dictionary.
        train_dataset (Dataset): train dataset for pytorch. Grouped shuffled data
        test_dataset  (Dataset): test dataset for pytorch. Grouped shuffled data
        filename (str): path to load network not training shifter
        meanbias (Tensor): mean bias for linear network
    """
    if params['train_shifter']:
        if params['single_shifter']:
            model = model_wrapper((network_config,SingleShifterNetwork))
        else:
            model = model_wrapper((network_config,ShifterNetwork))
    elif (params['ModelID']==2) | (params['ModelID']==3):
        model = model_wrapper((network_config,MixedNetwork))
        model = load_model(model,params,filename=params['best_vis_network'],meanbias=meanbias)
    else:
        model = model_wrapper((network_config,BaseModel))
        if filename is not None:
            model = load_model(model,params,filename,meanbias=meanbias)

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
            
    model.to(device)

    optimizer, scheduler = setup_model_training(model,params,network_config)
    # train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=2, pin_memory=True,)
    # test_dataloader  = DataLoader(test_dataset,  batch_size=len(test_dataset),  num_workers=2, pin_memory=True,)

    tloss_trace = torch.zeros((params['Nepochs'], network_config['Ncells']), dtype=torch.float)
    vloss_trace = torch.zeros((params['Nepochs'], network_config['Ncells']), dtype=torch.float)
    if network_config['single_trial'] is not None:
        pbar = tqdm((range(params['Nepochs'])))
    else: 
        pbar = (range(params['Nepochs']))  
    for epoch in pbar:  # loop over the dataset multiple times
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(xtr,xtr_pos)
        loss = model.loss(outputs, ytr)
        loss.backward(torch.ones_like(loss))
        optimizer.step()

        # save loss
        tloss_trace[epoch] = loss.detach().cpu()
            
        if scheduler is not None:
            scheduler.step()

        # Validation loss
        with torch.no_grad():
            outputs = model(xte,xte_pos)
            loss = model.loss(outputs, yte)
            vloss_trace[epoch] = loss.detach().cpu()

    if network_config['single_trial'] is not None:
        model_name = '{}_ModelID{:d}_dt{:03d}_T{:02d}_NB{}_{}.pt'.format(params['model_type'], params['ModelID'],int(params['model_dt']*1000), params['nt_glm_lag'], params['Nepochs'],network_config['single_trial'])
        torch.save((model.state_dict(), optimizer.state_dict()), params['save_model']/ model_name)
        return tloss_trace,vloss_trace,model,optimizer
    else:
        # Here we save a checkpoint. It is automatically registered with Ray Tune and can be accessed through `session.get_checkpoint()`
        model_name = 'GLM_{}_ModelID{:d}_dt{:03d}_T{:02d}_NB{}_{}.pt'.format(params['model_type'], params['ModelID'],int(params['model_dt']*1000), params['nt_glm_lag'], params['Nepochs'],session.get_trial_name())
        torch.save((model.state_dict(), optimizer.state_dict()), session.get_trial_dir()+model_name)
        checkpoint = Checkpoint.from_dict({'step':epoch})
        session.report({'avg_loss':float(torch.mean(vloss_trace[-1],dim=-1).numpy())}, checkpoint=checkpoint)

    print("Finished Training")
    # return dict(avg_loss=float(torch.mean(vloss_trace[-1],dim=-1).numpy()))

def train_dataset_network(network_config, train_dataset, test_dataset, params={}, filename=None, meanbias=None):
    """ Function to train network. For Ray Tune, need to have named inputs. 

    Args:
        network_config (dict): Dictionary containing parameters for network
        params (dict): Key parameter dictionary.
        train_dataset (Dataset): train dataset for pytorch. Grouped shuffled data
        test_dataset  (Dataset): test dataset for pytorch. Grouped shuffled data
        filename (str): path to load network not training shifter
        meanbias (Tensor): mean bias for linear network
    """
    if params['train_shifter']:
        model = model_wrapper((network_config,ShifterNetwork))
    elif (params['ModelID']==2) | (params['ModelID']==3):
        model = model_wrapper((network_config,MixedNetwork))
        model = load_model(model,params,filename=params['best_vis_network'],meanbias=meanbias)
    else:
        model = model_wrapper((network_config,BaseModel))
        if filename is not None:
            model = load_model(model,params,filename,meanbias=meanbias)

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
            
    model.to(device)

    optimizer, scheduler = setup_model_training(model,params,network_config)
    train_dataloader = DataLoader(train_dataset, batch_size=network_config['batch_size'], shuffle=network_config['shuffle'], num_workers=2, pin_memory=True, drop_last=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=network_config['batch_size'], shuffle=network_config['shuffle'], num_workers=2, pin_memory=True, drop_last=True)

    tloss_trace = torch.zeros((params['Nepochs'], network_config['Ncells']), dtype=torch.float)
    vloss_trace = torch.zeros((params['Nepochs'], network_config['Ncells']), dtype=torch.float)
    if network_config['single_trial'] is not None:
        pbar = tqdm((range(params['Nepochs'])))
    else: 
        pbar = (range(params['Nepochs']))  
    for epoch in pbar:  # loop over the dataset multiple times
        for mini_batch in train_dataloader:
            xtr, xtr_pos, ytr = mini_batch
            xtr, xtr_pos, ytr = xtr.to(device), xtr_pos.to(device), ytr.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(xtr,xtr_pos)
            loss = model.loss(outputs, ytr)
            loss.backward(torch.ones_like(loss))
            optimizer.step()

            # save loss
            tloss_trace[epoch] = loss.detach().cpu()
                
            if scheduler is not None:
                scheduler.step()

        # Validation loss
        with torch.no_grad():
            for mini_batch in test_dataloader:
                xte, xte_pos, yte = mini_batch
                xte, xte_pos, yte = xte.to(device), xte_pos.to(device), yte.to(device)
                outputs = model(xte,xte_pos)
                loss = model.loss(outputs, yte)
                vloss_trace[epoch] = loss.detach().cpu()

    if network_config['single_trial'] is not None:
        model_name = '{}_ModelID{:d}_dt{:03d}_T{:02d}_NB{}_{}.pt'.format(params['model_type'], params['ModelID'],int(params['model_dt']*1000), params['nt_glm_lag'], params['Nepochs'],network_config['single_trial'])
        torch.save((model.state_dict(), optimizer.state_dict()), params['save_model']/ model_name)
        return tloss_trace,vloss_trace,model,optimizer
    else:
        # Here we save a checkpoint. It is automatically registered with Ray Tune and can be accessed through `session.get_checkpoint()`
        model_name = 'GLM_{}_ModelID{:d}_dt{:03d}_T{:02d}_NB{}_{}.pt'.format(params['model_type'], params['ModelID'],int(params['model_dt']*1000), params['nt_glm_lag'], params['Nepochs'],session.get_trial_name())
        torch.save((model.state_dict(), optimizer.state_dict()), session.get_trial_dir()+model_name)
        checkpoint = Checkpoint.from_dict({'step':epoch})
        session.report({'avg_loss':float(torch.mean(vloss_trace[-1],dim=-1).numpy())}, checkpoint=checkpoint)

    print("Finished Training")
    # return dict(avg_loss=float(torch.mean(vloss_trace[-1],dim=-1).numpy()))
def evaluate_networks(best_network,network_config,params,xte,xte_pos,yte,device='cpu'):
    """Evaluates ray tune experiment and hyperparameter search

    Args:
        best_network (str): path to best network model '.pt' file
        network_config (dict): network_config for best network
        params (dict): key parameters dictionary
        xte (Tensor): test input data 
        xte_pos (Tensor): test additional input data
        yte (Tensor): target test data
        device (str, optional): device to load data onto. Defaults to 'cpu'.
    """
    import pytorchGLM.Utils.io_dict_to_hdf5 as ioh5
    ##### Load best network from saved ray experiment ######
    state_dict, _ = torch.load(best_network,map_location='cpu')
    if params['train_shifter']:
        if params['single_shifter']:
            model = model_wrapper((network_config,SingleShifterNetwork))
        else:      
            model = model_wrapper((network_config,ShifterNetwork))
    elif (params['ModelID']==2) | (params['ModelID']==3):
        model = model_wrapper((network_config,MixedNetwork))
    else:
        model = model_wrapper((network_config,BaseModel))
    model.load_state_dict(state_dict)
    model.to(device)

    ##### Load data into device and predict test set ######
    xte, xte_pos, yte = xte.to(device), xte_pos.to(device), yte.to(device)
    with torch.no_grad():
        yhat = model(xte,xte_pos)

    ##### Smooth Firing rates and save ######
    actual_smooth = np.apply_along_axis(lambda m: np.convolve(m, np.ones(params['bin_length']), mode='same')/(params['bin_length'] * params['model_dt']), axis=0, arr=yte.detach().cpu().numpy())[params['bin_length']:-params['bin_length']]
    pred_smooth = np.apply_along_axis(lambda m: np.convolve(m, np.ones(params['bin_length']), mode='same')/(params['bin_length'] * params['model_dt']), axis=0, arr=yhat.detach().cpu().numpy())[params['bin_length']:-params['bin_length']]
    cc_test = np.array([(np.corrcoef(pred_smooth[:,celln],actual_smooth[:,celln])[0, 1]) for celln in range(pred_smooth.shape[1])])
    
    GLM_Dict = {
        'actual_smooth': actual_smooth,
        'pred_smooth': pred_smooth,
        'cc_test': cc_test,
        }

    for key in state_dict.keys():
        GLM_Dict[key]  = state_dict[key].cpu().numpy()

    model_name = '{}_ModelID{:d}_dt{:03d}_T{:02d}_NB{}_Best.h5'.format(params['model_type'], params['ModelID'],int(params['model_dt']*1000), params['nt_glm_lag'], params['Nepochs'])
    ioh5.save(params['save_model']/model_name,GLM_Dict)


def evaluate_singleshifter(args,network_config,params):
    from kornia.geometry.transform import Affine
    import pytorchGLM.Utils.io_dict_to_hdf5 as ioh5
    from matplotlib.backends.backend_pdf import PdfPages

    ##### Loading Best Shifter Network #####
    Nepochs_saved = params['Nepochs']
    if params['single_shifter']:
        single_shifter = True
    params, file_dict, _ = load_params(args,1,exp_dir_name=None,nKfold=0,debug=False,file_dict={})
    params = get_modeltype(params)
    params['single_shifter']=True
    exp_filename = list((params['save_model_shift'] / ('NetworkAnalysis/')).rglob('*experiment_data.h5'))[-1]
    df,metadata= h5load(exp_filename)
    best_network = metadata['best_network']

    state_dict,_ = torch.load(best_network,map_location='cpu')
    if params['single_shifter']:
        model = model_wrapper((network_config,SingleShifterNetwork))
    else:
        model = model_wrapper((network_config,ShifterNetwork))
    model.load_state_dict(state_dict)
    model.cpu()

    ##### Make Shifter Matricies Plots #####
    pdf_name = params['save_model_shift']/ 'VisMov_{}_dt{:03d}_Lags{:02d}_MovModel{:d}_CellSummary.pdf'.format(params['model_type'],int(params['model_dt']*1000),params['nt_glm_lag'], params['ModelID'])
    with PdfPages(pdf_name) as pdf:
        ##### Sweep -40 to 40 degrees
        FM_move_avg = np.load(params['save_dir_fm']/'FM_MovAvg_{}_dt{:03d}.npy'.format(params['data_name_fm'],int(params['model_dt']*1000)))
        th_range = 40/FM_move_avg[1,0]
        phi_range = 40/FM_move_avg[1,1]
        pitch_range = 40#/FM_move_avg[1,2]
        n_ranges = 81
        ang_sweepx,ang_sweepy,ang_sweepz = np.meshgrid(np.linspace(-th_range,th_range,n_ranges),np.linspace(-phi_range,phi_range,n_ranges),np.linspace(-pitch_range,pitch_range,n_ranges),sparse=False,indexing='ij')
        shift_mat = np.zeros((3,) + ang_sweepx.shape)
        for i in range(ang_sweepx.shape[0]):
            for j in range(ang_sweepy.shape[1]):
                # ang_sweep = torch.from_numpy(np.vstack((ang_sweepx[i,j,:],ang_sweepy[i,j,:],ang_sweepz[i,j,:])).astype(np.float32).T)
                ang_sweep = torch.from_numpy((ang_sweepx[i,j,:][None,:]).astype(np.float32).T)
                shift_vec = model.shifter_nn(ang_sweep).detach().cpu().numpy()
                shift_mat[0,i,j] = shift_vec[:,0]
                # shift_mat[1,i,j] = shift_vec[:,1]
                # shift_mat[2,i,j] = shift_vec[:,2]

        fig, ax = plt.subplots(1,4,figsize=(20,5))
        shift_titles = [r'$dx$',r'$dy$',r'$d\alpha,\phi=0$',r'$d\alpha,\theta=0$']
        ticks=np.arange(0,90,20)
        ticklabels=np.arange(-40,50,20)
        shift_matshow=np.stack((shift_mat[0,:,:,40].T,shift_mat[1,:,:,40].T,shift_mat[2,:,40,:].T,shift_mat[2,40,:,:].T))
        crange_list = np.stack((np.max(np.abs(shift_mat[:2])),np.max(np.abs(shift_mat[:2])), np.max(np.abs(shift_mat[2])), np.max(np.abs(shift_mat[2]))))
        for n in range(1):
            im1=ax[n].imshow(shift_matshow[n],vmin=-crange_list[n], vmax=crange_list[n], origin='lower', cmap='RdBu_r')
            cbar1 = pglm.add_colorbar(im1)
            ax[n].set_xticks(ticks)
            ax[n].set_xticklabels(ticklabels)
            ax[n].set_yticks(ticks)
            ax[n].set_yticklabels(ticklabels)
            ax[n].set_xlabel(r'$\theta$')
            ax[n].set_ylabel(r'$\phi$')
            ax[n].set_title(shift_titles[n])
        # plt.suptitle('Lambda={}, Best_shifter={}'.format(l,best_shifter),y=1)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    ##### Save FM Shifted World Cam #####
    data = load_aligned_data(file_dict, params, reprocess=False)
    data,train_idx_list,test_idx_list = format_data(data, params,do_norm=True,thresh_cells=True,cut_inactive=True)
    model_pos = np.hstack((data['model_th'][:, np.newaxis]))[:,None]
    ds=4/params['downsamp_vid']
    with torch.no_grad():
        shift_out = model.shifter_nn(torch.from_numpy(model_pos.astype(np.float32)))
        shift = Affine(torch.clamp(shift_out[:,-1],min=-45,max=45),translation=shift_out[:,:2]*ds)
        model_vid_sm_shift = shift(torch.from_numpy(data['model_vid_sm'][:,np.newaxis].astype(np.float32))).detach().cpu().numpy().squeeze()
    model_file = params['save_dir'] / 'ModelData_{}_dt{:03d}_rawWorldCam_{:d}ds.h5'.format(params['data_name_fm'],int(params['model_dt']*1000),int(params['downsamp_vid']))
    model_data = ioh5.load(model_file)
    model_data['model_vid_sm_shift'] = model_vid_sm_shift
    ioh5.save(model_file,model_data)
    print('saved FM shifted video')

    ##### Save HF Shifted World Cam #####
    args['free_move'] = False
    args['train_shifter']=True
    args['Nepochs'] = 5000
    params, file_dict, exp = load_params(args,1,exp_dir_name=None,nKfold=0,debug=False,file_dict={})
    params['lag_list'] = [0]
    params['nt_glm_lag']=len(params['lag_list'])
    if single_shifter:    
        params['single_shifter']=True
    data = load_aligned_data({}, params, reprocess=False)
    data,train_idx_list,test_idx_list = format_data(data, params,do_norm=True,thresh_cells=True,cut_inactive=True)
    model_pos = np.hstack((data['model_th'][:, np.newaxis]))[:,None]
    ds=4/params['downsamp_vid']
    with torch.no_grad():
        shift_out = model.shifter_nn(torch.from_numpy(model_pos.astype(np.float32)))
        shift = Affine(torch.clamp(shift_out[:,-1],min=-45,max=45),translation=shift_out[:,:2]*ds)
        model_vid_sm_shift = shift(torch.from_numpy(data['model_vid_sm'][:,np.newaxis].astype(np.float32))).detach().cpu().numpy().squeeze()
    model_file = params['save_dir'] / 'ModelData_{}_dt{:03d}_rawWorldCam_{:d}ds.h5'.format(params['data_name_hf'],int(params['model_dt']*1000),int(params['downsamp_vid']))
    model_data = ioh5.load(model_file)
    model_data['model_vid_sm_shift'] = model_vid_sm_shift
    ioh5.save(model_file,model_data)
    print('saved HF shifted video')
    args['free_move'] = True
    args['Nepochs'] = Nepochs_saved
    


def evaluate_shifter(args,network_config,params):
    from kornia.geometry.transform import Affine
    import pytorchGLM.Utils.io_dict_to_hdf5 as ioh5
    from matplotlib.backends.backend_pdf import PdfPages

    ##### Loading Best Shifter Network #####
    Nepochs_saved = params['Nepochs']
    if params['single_shifter']:
        single_shifter = True
    params, file_dict, _ = load_params(args,1,exp_dir_name=None,nKfold=0,debug=False,file_dict={})
    params = get_modeltype(params)
    params['single_shifter']=True
    exp_filename = list((params['save_model_shift'] / ('NetworkAnalysis/')).rglob('*experiment_data.h5'))[-1]
    df,metadata= h5load(exp_filename)
    best_network = metadata['best_network']

    state_dict,_ = torch.load(best_network,map_location='cpu')
    if params['single_shifter']:
        model = model_wrapper((network_config,SingleShifterNetwork))
    else:
        model = model_wrapper((network_config,ShifterNetwork))
    model.load_state_dict(state_dict)
    model.cpu()

    ##### Make Shifter Matricies Plots #####
    pdf_name = params['save_model_shift']/ 'VisMov_{}_dt{:03d}_Lags{:02d}_MovModel{:d}_CellSummary.pdf'.format(params['model_type'],int(params['model_dt']*1000),params['nt_glm_lag'], params['ModelID'])
    with PdfPages(pdf_name) as pdf:
        ##### Sweep -40 to 40 degrees
        FM_move_avg = np.load(params['save_dir_fm']/'FM_MovAvg_{}_dt{:03d}.npy'.format(params['data_name_fm'],int(params['model_dt']*1000)))
        th_range = 40/FM_move_avg[1,0]
        phi_range = 40/FM_move_avg[1,1]
        pitch_range = 40/FM_move_avg[1,2]
        n_ranges = 81
        ang_sweepx,ang_sweepy,ang_sweepz = np.meshgrid(np.linspace(-th_range,th_range,n_ranges),np.linspace(-phi_range,phi_range,n_ranges),np.linspace(-pitch_range,pitch_range,n_ranges),sparse=False,indexing='ij')
        shift_mat = np.zeros((3,) + ang_sweepx.shape)
        for i in range(ang_sweepx.shape[0]):
            for j in range(ang_sweepy.shape[1]):
                ang_sweep = torch.from_numpy(np.vstack((ang_sweepx[i,j,:],ang_sweepy[i,j,:],ang_sweepz[i,j,:])).astype(np.float32).T)
                shift_vec = model.shifter_nn(ang_sweep).detach().cpu().numpy()
                shift_mat[0,i,j] = shift_vec[:,0]
                shift_mat[1,i,j] = shift_vec[:,1]
                shift_mat[2,i,j] = shift_vec[:,2]

            
        fig, ax = plt.subplots(1,4,figsize=(20,5))
        shift_titles = [r'$dx$',r'$dy$',r'$d\alpha,\phi=0$',r'$d\alpha,\theta=0$']
        ticks=np.arange(0,90,20)
        ticklabels=np.arange(-40,50,20)
        shift_matshow=np.stack((shift_mat[0,:,:,40].T,shift_mat[1,:,:,40].T,shift_mat[2,:,40,:].T,shift_mat[2,40,:,:].T))
        crange_list = np.stack((np.max(np.abs(shift_mat[:2])),np.max(np.abs(shift_mat[:2])), np.max(np.abs(shift_mat[2])), np.max(np.abs(shift_mat[2]))))
        for n in range(4):
            im1=ax[n].imshow(shift_matshow[n],vmin=-crange_list[n], vmax=crange_list[n], origin='lower', cmap='RdBu_r')
            cbar1 = add_colorbar(im1)
            ax[n].set_xticks(ticks)
            ax[n].set_xticklabels(ticklabels)
            ax[n].set_yticks(ticks)
            ax[n].set_yticklabels(ticklabels)
            ax[n].set_xlabel(r'$\theta$')
            ax[n].set_ylabel(r'$\phi$')
            ax[n].set_title(shift_titles[n])
        # plt.suptitle('Lambda={}, Best_shifter={}'.format(l,best_shifter),y=1)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    ##### Save FM Shifted World Cam #####
    data = load_aligned_data(file_dict, params, reprocess=False)
    data,train_idx_list,test_idx_list = format_data(data, params,do_norm=True,thresh_cells=True,cut_inactive=True)
    model_pos = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis],data['model_pitch'][:, np.newaxis]))
    ds=4/params['downsamp_vid']
    with torch.no_grad():
        shift_out = model.shifter_nn(torch.from_numpy(model_pos.astype(np.float32)))
        shift = Affine(torch.clamp(shift_out[:,-1],min=-45,max=45),translation=torch.clamp(shift_out[:,:2]*ds,min=-20*ds,max=20*ds))
        model_vid_sm_shift = shift(torch.from_numpy(data['model_vid_sm'][:,np.newaxis].astype(np.float32))).detach().cpu().numpy().squeeze()
    model_file = params['save_dir'] / 'ModelData_{}_dt{:03d}_rawWorldCam_{:d}ds.h5'.format(params['data_name_fm'],int(params['model_dt']*1000),int(params['downsamp_vid']))
    model_data = ioh5.load(model_file)
    model_data['model_vid_sm_shift'] = model_vid_sm_shift
    ioh5.save(model_file,model_data)
    print('saved FM shifted video')

    ##### Save HF Shifted World Cam #####
    args['free_move'] = False
    args['train_shifter']=True
    args['Nepochs'] = 5000
    params, file_dict, exp = load_params(args,1,exp_dir_name=None,nKfold=0,debug=False,file_dict={})
    params['lag_list'] = [0]
    params['nt_glm_lag']=len(params['lag_list'])
    if single_shifter:    
        params['single_shifter']=True
    data = load_aligned_data({}, params, reprocess=False)
    data,train_idx_list,test_idx_list = format_data(data, params,do_norm=True,thresh_cells=True,cut_inactive=True)
    model_pos = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis],data['model_pitch'][:, np.newaxis]))
    ds=4/params['downsamp_vid']
    with torch.no_grad():
        shift_out = model.shifter_nn(torch.from_numpy(model_pos.astype(np.float32)))
        shift = Affine(torch.clamp(shift_out[:,-1],min=-45,max=45),translation=torch.clamp(shift_out[:,:2]*ds,min=-20*ds,max=20*ds))
        model_vid_sm_shift = shift(torch.from_numpy(data['model_vid_sm'][:,np.newaxis].astype(np.float32))).detach().cpu().numpy().squeeze()
    model_file = params['save_dir'] / 'ModelData_{}_dt{:03d}_rawWorldCam_{:d}ds.h5'.format(params['data_name_hf'],int(params['model_dt']*1000),int(params['downsamp_vid']))
    model_data = ioh5.load(model_file)
    model_data['model_vid_sm_shift'] = model_vid_sm_shift
    ioh5.save(model_file,model_data)
    print('saved HF shifted video')
    args['free_move'] = True
    args['Nepochs'] = Nepochs_saved
    