import warnings 
warnings.filterwarnings('ignore')

import numpy as np

from ray import tune
from ray import air
from ray.air import session
from ray.air.checkpoint import Checkpoint


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

def train_network(network_config, params={},train_dataset=None,test_dataset=None,filename=None,meanbias=None):
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
        model = load_model(model,params,filename,meanbias=meanbias)
    else:
        model = model_wrapper((network_config,BaseModel))
        model = load_model(model,params,filename,meanbias=meanbias)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
    model.to(device)

    optimizer, scheduler = setup_model_training(model,params,network_config)
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=2, pin_memory=True,)
    test_dataloader  = DataLoader(test_dataset,  batch_size=len(test_dataset),  num_workers=2, pin_memory=True,)

    tloss_trace = torch.zeros((params['Nepochs'], network_config['Ncells']), dtype=torch.float)
    vloss_trace = torch.zeros((params['Nepochs'], network_config['Ncells']), dtype=torch.float)

    for epoch in (range(params['Nepochs'])):  # loop over the dataset multiple times
        for i, minibatch in enumerate(train_dataloader, 0):
            # get the inputs; minibatch is a list of [vid, pos, y]
            vid,pos,y = minibatch
            vid,pos,y = vid.to(device),pos.to(device),y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(vid,pos)
            loss = model.loss(outputs, y)
            loss.backward(torch.ones_like(loss))
            optimizer.step()

        # print statistics
        tloss_trace[epoch] = loss.detach().cpu()
            
        if scheduler is not None:
            scheduler.step()

        # Validation loss
        for i, minibatch in enumerate(test_dataloader, 0):
            with torch.no_grad():
                # get the inputs; minibatch is a list of [vid, pos, y]
                vid,pos,y = minibatch
                vid,pos,y = vid.to(device),pos.to(device),y.to(device)
                outputs = model(vid,pos)
                loss = model.loss(outputs, y)
                vloss_trace[epoch] = loss.detach().cpu()

    # Here we save a checkpoint. It is automatically registered with
    # Ray Tune and can be accessed through `session.get_checkpoint()`
    # API in future iterations.
    model_name = 'GLM_{}_ModelID{:d}_dt{:03d}_T{:02d}_NB{}_{}.pt'.format(params['model_type'], params['ModelID'],int(params['model_dt']*1000), params['nt_glm_lag'], params['Nepochs'],session.get_trial_name())
    torch.save((model.state_dict(), optimizer.state_dict()), params['save_model']/ model_name)
    checkpoint = Checkpoint.from_dict({'step':epoch})
    # session.report({"avg_loss": float(torch.mean(vloss_trace[-1],dim=-1).numpy())})
    session.report({'avg_loss':float(torch.mean(vloss_trace[-1],dim=-1).numpy())}, checkpoint=checkpoint)

    print("Finished Training")
    # return dict(avg_loss=float(torch.mean(vloss_trace[-1],dim=-1).numpy()))


def evaluate_shifter(best_network,params):
    from matplotlib.backends.backend_pdf import PdfPages
    ################################################### Need to implement
    ##### Save Shifted World Camera Video if Training Shifter #####
    Best_RF={}
    for name, p in model.named_parameters():
        if 'shifter' not in name:
            Best_RF[name] = torch.from_numpy(GLM_Data[name])
    best_shift = 'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:01d}.pt'.format('Pytorch_BestShift',int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],Kfold)

    torch.save({'model_state_dict': Best_RF}, (params['save_dir_fm_exp'] / best_shift))
    torch.save({'model_state_dict': Best_RF}, (params['save_dir_hf_exp'] / best_shift))
    a = 0 

    model_vid_sm_shift2 = {}
    pdf_name = params['fig_dir']/ 'VisMov_{}_dt{:03d}_Lags{:02d}_MovModel{:d}_CellSummary.pdf'.format(params['model_type'],int(params['model_dt']*1000),params['nt_glm_lag'], params['MovModel'])
    with PdfPages(pdf_name) as pdf:
        if params['reg_lap']:
            pbar = tqdm(range(len(params['alpha_l'])))    
        else:
            pbar = tqdm(range(len(params['lambdas'])))
        for l,lam in enumerate(tqdm(pbar)):
            model_name = 'GLM_{}_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_alph{}_lam{}_Kfold{:01d}.pth'.format(params['model_type'],'UC',int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],a,l,Kfold)

            checkpoint = torch.load(params['save_model_shift']/model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.cpu()
            
            ##### Sweep -40 to 40 degrees
            FM_move_avg = np.load(params['save_dir_fm']/'FM_MovAvg_dt{:03d}.npy'.format(int(params['model_dt']*1000)))
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
            plt.suptitle('Lambda={}, Best_shifter={}'.format(l,best_shifter),y=1)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    ##### Save FM Shifted World Cam #####
    model_name = 'GLM_{}_WC{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_alph{}_lam{}_Kfold{:01d}.pth'.format(params['model_type'],'UC',int(params['model_dt']*1000), params['nt_glm_lag'], params['MovModel'], params['Nepochs'],a,best_shifter,Kfold)
    checkpoint = torch.load(params['save_model_shift']/model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cpu()
    ds=4/params['downsamp_vid']
    shift_out = model.shifter_nn(torch.from_numpy(model_move[:,(0,1,3)].astype(np.float32)))
    shift = Affine(torch.clamp(shift_out[:,-1],min=-45,max=45),translation=torch.clamp(shift_out[:,:2]*ds,min=-20*ds,max=20*ds))
    model_vid_sm_shift = shift(torch.from_numpy(data['model_vid_sm'][:,np.newaxis].astype(np.float32))).detach().cpu().numpy().squeeze()
    model_vid_sm_shift2['model_vid_sm_shift'] = model_vid_sm_shift
    ioh5.save(params['save_dir_fm_exp']/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(int(params['model_dt']*1000), params['MovModel']),model_vid_sm_shift2)


    ##### Save HF Shifted World Cam #####
    args['free_move'] = False
    args['train_shifter']=True
    args['Nepochs'] = 5000
    params, file_dict, exp = load_params(1,Kfold,args)
    params['lag_list'] = [0]
    params['nt_glm_lag']=len(params['lag_list'])
    VisNepochs = args['Nepochs']
    data, train_idx_list, test_idx_list = load_train_test(file_dict, **params)
    train_idx = train_idx_list[Kfold]
    test_idx = test_idx_list[Kfold]
    data = load_Kfold_data(data,train_idx,test_idx,params)
    params, xtr, xtrm, xte, xtem, ytr, yte, shift_in_tr, shift_in_te, input_size, output_size, meanbias, model_move = load_GLM_data(data, params, train_idx, test_idx)
    shift_out = model.shifter_nn(torch.from_numpy(model_move[:,(0,1,3)].astype(np.float32)))
    shift = Affine(angle=shift_out[:,-1],translation=shift_out[:,:2])
    model_vid_sm_shift = shift(torch.from_numpy(data['model_vid_sm'][:,np.newaxis].astype(np.float32))).detach().cpu().numpy().squeeze()
    model_vid_sm_shift2['model_vid_sm_shift'] = model_vid_sm_shift
    ioh5.save(params['save_dir_hf_exp']/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(int(params['model_dt']*1000), params['MovModel']),model_vid_sm_shift2)
    args['free_move'] = True
    args['Nepochs'] = Nepochs_saved





if __name__ == '__main__': 
    # Input arguments
    args = arg_parser()
    device = torch.device("cuda:{}".format(get_freer_gpu()) if torch.cuda.is_available() else "cpu")
    print('Device:',device)
    ModRun = [int(i) for i in args['ModRun'].split(',')] #[0,1,2,3,4] #-1,
    Kfold = args['Kfold']

    for ModelRun in ModRun:
        if ModelRun == -1: # train shifter
            Nepochs_saved = args['Nepochs']
            args['train_shifter']  = True
            args['Nepochs']        = 5000
            ModelID                = 1
            params, file_dict, exp = load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
            params['lag_list']     = [0]
            params['nt_glm_lag']   = len(params['lag_list'])
        elif ModelRun == 0: # pos only
            args['train_shifter']  = False
            ModelID                = 0
            params, file_dict, exp = load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
        elif ModelRun == 1: # vis only
            args['train_shifter']  = False
            ModelID                = 1
            params, file_dict, exp = load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
        elif ModelRun == 2: # add fit
            args['train_shifter']  = False
            # args['Nomodel']           = False
            ModelID                = 2
            params, file_dict, exp = load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
        elif ModelRun == 3: # mul. fit
            args['train_shifter']  = False
            # args['Nomodel']           = False
            ModelID                = 3
            params, file_dict, exp = load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)
        elif ModelRun == 4: # head-fixed
            args['train_shifter']  = False
            args['free_move']      = False
            ModelID                = 4
            params, file_dict, exp = load_params(args,ModelID,exp_dir_name=None,nKfold=0,debug=False)


        data = load_aligned_data(file_dict, params, reprocess=False)
        params = get_modeltype(params)
        train_dataset, test_dataset, network_config = load_datasets(file_dict,params,single_trial=False)

        sync_config = tune.SyncConfig()  # the default mode is to use use rsync
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_network,params=params,train_dataset=train_dataset,test_dataset=test_dataset),
                resources={"cpu": 2, "gpu": .5}),
            tune_config=tune.TuneConfig(metric="avg_loss",mode="min",),
            param_space=network_config,
            run_config=air.RunConfig(local_dir=params['save_model'], name="NetworkAnalysis",sync_config=sync_config,verbose=2)
        )
        results = tuner.fit()

        best_result = results.get_best_result("avg_loss", "min")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(best_result.metrics["avg_loss"]))
        df = results.get_dataframe()
        best_network = list(params['save_model'].glob('*{}.pt'.format(best_result.metrics['trial_id'])))[0]
        h5store(params['save_model'] / 'NetworkAnalysis/experiment_data.h5', df, **{'best_network':best_network})

        if ModelRun == -1:
            evaluate_shifter(best_network,params)