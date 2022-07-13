import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import xarray as xr
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import warnings 

from tqdm.notebook import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import medfilt
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import binned_statistic
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


sns.set_context("talk")


from NLMixedUtils.utils import *
import NLMixedUtils.io_dict_to_hdf5 as ioh5
from NLMixedUtils.format_data import *
from NLMixedUtils.models import *
from NLMixedScripts.fit_GLM import *



warnings.filterwarnings('ignore')

args = arg_parser(jupyter=True)
Kfold = 0
params,_,_ = load_params(1,Kfold,args,debug=True)
paper_fig_dir = (params['fig_dir'].parent.parent.parent / 'RevisionFigs')
paper_fig_dir.mkdir(parents=True, exist_ok=True)
move_clrs = ['blue','orange','green','red'] 
q_clrs = ["#f72585","#7209b7","#3f37c9","#4cc9f0"] # Magenta to Blue
# titles = [r'theta $(\theta)$',r'phi $(\phi)$',r'pitch $(\rho)$',r'roll $(\omega)$']
mod_clrs = ["#B541FF","#00A14B","#118ab2","#ef476f","#073b4c"]
fontsize=10
mod_titles = ['pos','vis','add','mul','HF']

mpl.rcParams.update({'font.size':         10,
                     'axes.linewidth':    2,
                     'xtick.major.size':  3,
                     'xtick.major.width': 2,
                     'ytick.major.size':  3,
                     'ytick.major.width': 2,
                     'axes.spines.right': False,
                     'axes.spines.top':   False,
                     'font.sans-serif':   "Arial",
                     'font.family':       "sans-serif",
                     'pdf.fonttype':      42,
                     'xtick.labelsize':   10,
                     'ytick.labelsize':   10,
                     'figure.facecolor': 'white'

                    })

def get_cellnums(All_data,pparams,exp_type='CropInputs'):
    start_ind = 0
    tot_num = 0
    CellNum_Tot = {}
    for da in range(len(pparams['dates_all'])):
        for n in range(All_data[pparams['date_ani2'][da]][exp_type]['Vis_pred_smooth'].shape[-1]):
            tot_num = tot_num + 1
        CellNum_Tot[pparams['date_ani2'][da]] = np.arange(start_ind,tot_num)
        start_ind += All_data[pparams['date_ani2'][da]][exp_type]['Vis_pred_smooth'].shape[-1]
    return CellNum_Tot

def figure2(All_data,pparams,exp_type='CropInputs',figname=None):
    vals_Vis = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_VisNoSh = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['VisNoShifter_cc_test'] for da in range(len(pparams['dates_all']))])
    r2_all = np.stack((vals_VisNoSh,vals_Vis))
    fig1 = plt.figure(constrained_layout=False, figsize=(10.8,4.25))
    gs0 = fig1.add_gridspec(nrows=3, ncols=8, wspace=.8, hspace=.9)
    lag_ls = [-100,-50,0,50,100]
    lag_list = [0,1,2,3,4]
    ########## Fig 2B ########## 
    da=0
    cells = [22,34,42,101] #np.arange(18)
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=len(cells),ncols=len(lag_list), subplot_spec=gs0[:,:4], wspace=.05, hspace=.08)
    axs1 = np.array([fig1.add_subplot(gs00[n,m]) for n in range(len(cells)) for m in range(len(lag_list))]).reshape(len(cells),len(lag_list))

    for n, cell in enumerate(cells):
        crange = np.max(np.abs(All_data[pparams['date_ani2'][da]][exp_type]['Vis_rf_up'][cell]))
        for m,lag in enumerate(lag_list):
            ax = axs1[n,m]
            im = ax.imshow(All_data[pparams['date_ani2'][da]][exp_type]['Vis_rf_up'][cell,lag],'RdBu_r',vmin=-crange,vmax=crange)
            axs1[0,m].set_title('{}ms'.format(lag_ls[m]),fontsize=fontsize)

        axs1[n,0].set_ylabel('unit {}'.format(n+1),fontsize=fontsize)
    scalebar = AnchoredSizeBar(axs1[0,0].transData,
                        20, '10 deg', 'lower left', 
                        pad=0.1,
                        color='black',
                        frameon=False,
                        size_vertical=1,
                        )
    axs1[0,0].add_artist(scalebar)  

    cbar2 = fig1.colorbar(im, ax=axs1,shrink=.8)
    cbar2.set_ticks([-crange, crange])
    cbar2.set_ticklabels(['dark', 'light'])
    cbar2.ax.tick_params(labelsize=fontsize, rotation=90,width=0,length=0)
    cbar2.outline.set_linewidth(1)

    for ax in axs1.flat:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_visible(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    ########## Fig 2C ########## 
    gs02 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0,4:],wspace=1.2,hspace=.2)
    axs2 = np.array([fig1.add_subplot(gs02[0,:2]), fig1.add_subplot(gs02[0,2])])

    celln = 42
    da = 0
    modN = 1
    ax = axs2[0]
    t = np.argmin(np.abs((np.arange(All_data[pparams['date_ani2'][da]]['actual_smooth'].shape[0])*params['model_dt'])-0)) 
    dt = int(300/params['model_dt']) 
    pred_time = (np.arange(All_data[pparams['date_ani2'][da]]['actual_smooth'].shape[0])*params['model_dt'])[0:np.abs(dt)]
    ax.plot(pred_time,All_data[pparams['date_ani2'][da]]['actual_smooth'][t:t+dt,celln],c='k',lw=.75,zorder=0,label='data')
    ax.plot(pred_time,All_data[pparams['date_ani2'][da]][exp_type][pparams['mod_titles'][modN]+'_pred_smooth'][t:t+dt,celln],c=mod_clrs[modN],lw=.75,
                label='{} cc: {:.02f}'.format(pparams['mod_titles'][modN],All_data[pparams['date_ani2'][da]][exp_type][pparams['mod_titles'][modN]+'_r2_test'][celln]),zorder=1)
    ax.legend(labelcolor='linecolor', fontsize=10, handlelength=0, handletextpad=0,loc='upper left',ncol=2)
    ax.set_xlabel('time (s)',fontsize=fontsize)
    ax.set_ylabel('sp/sec',fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize-2)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)


    ax = axs2[1]
    hbins = .02
    sta_titles = ['shifter Off','shifter On']
    NoShift_clrs = ["#050505", "#C41508"]
    lim0 = -.1
    lim1 = .7
    dlim = .2
    xlab = 'cc'
    count,edges = np.histogram(r2_all[1],bins=np.arange(lim0,lim1,hbins))
    edges_mid = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
    ax.bar(edges_mid, count/len(r2_all[1]),color='k',width=hbins, alpha=1)
    ax.set_xticks([0,.3,.6])
    ax.set_xticklabels([0,.3,.6],fontsize=fontsize-2)
    ax.set_xlabel(xlab,fontsize=fontsize)
    ax.axvline(x=np.nanmean(r2_all[1]),lw=2, c='#6D6E71',ls='--',zorder=1)
    ax.set_ylabel('fraction of units',fontsize=fontsize)
    ax.set_yticks([0,.03,.06])
    ax.set_yticklabels([0,.03,.06],fontsize=fontsize-2)


    ########## Fig 2D ########## 
    gs02 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs0[1:,4:],wspace=1,hspace=.5)
    gs02b = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs02[:2,:2],wspace=.1,hspace=.8)
    axs3 = np.array([fig1.add_subplot(gs02b[0,0]), fig1.add_subplot(gs02b[0,1]),fig1.add_subplot(gs02b[1,0]), fig1.add_subplot(gs02b[1,1])]).reshape(2,2)
    lag = 1
    da = 0
    cells = [22]
    for n, cell in enumerate(cells):
        ax = axs3[0,n]
        crange2 = np.max(np.abs(All_data[pparams['date_ani2'][da]][exp_type]['VisNoShifter_rf_up'][cell]))
        im2 = ax.imshow(All_data[pparams['date_ani2'][da]][exp_type]['VisNoShifter_rf_up'][cell,lag],'RdBu_r', vmin=-crange2, vmax=crange2)
        ax.set_title('shifter off',fontsize=fontsize)
        ax = axs3[0,n+1]
        crange2 = np.max(np.abs(All_data[pparams['date_ani2'][da]][exp_type]['Vis_rf_up'][cell,2]))
        im2 = ax.imshow(All_data[pparams['date_ani2'][da]][exp_type]['Vis_rf_up'][cell,2],'RdBu_r', vmin=-crange2, vmax=crange2)
        ax.set_title('shifter on',fontsize=fontsize)
    scalebar = AnchoredSizeBar(axs3[0,0].transData,
                        20, '10 deg', 'lower left', 
                        pad=0.1,
                        color='black',
                        frameon=False,
                        size_vertical=1,
                        )
    axs3[0,0].add_artist(scalebar)  
    for ax in axs3.flat:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_visible(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])



    ########## Fig 2E ########## 
    gs02c = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs02[:,2:],wspace=.1,hspace=1)
    axs3b = np.array([fig1.add_subplot(gs02c[0,0]), fig1.add_subplot(gs02c[1,0])])

    ax = axs3b[0]
    vis_th = r2_all[1] > .05
    ax.scatter(r2_all[0][vis_th],r2_all[1][vis_th],s=2,c='k',edgecolors='none')
    ax.scatter(r2_all[0][cells[0]],r2_all[1][cells[0]],s=5,c='r',edgecolors='none',zorder=2)
    ax.plot(np.linspace(0, .8, 100), np.linspace(0, .8, 100),lw=1, c='#6D6E71',ls='--',zorder=1)
    ax.set_xlabel(r'shifter off cc',fontsize=fontsize)
    ax.set_ylabel(r'shifter on cc',fontsize=fontsize)
    ax.set_xticks([0,.4,.8])
    ax.set_xticklabels([0,.4,.8],fontsize=fontsize-2)
    ax.set_yticks([0, .4, .8])
    ax.set_yticklabels([0, .4, .8],fontsize=fontsize-2)
    ax.axis('square')


    ########## Fig 2F ########## 
    # gs03 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[2,4:],wspace=1,hspace=.1)
    # axs4 = np.array([fig1.add_subplot(gs03[n,m]) for n in range(1) for m in range(3)])
    cc_sta = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_sta_cc_all'] for da in range(len(pparams['dates_all']))])
    vals_Vis = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_cc_test'] for da in range(len(pparams['dates_all']))])
    cc_all2 = np.stack((cc_sta,vals_Vis))
    RF_all = np.stack((np.vstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_sta_up'][:,0] for da in range(len(pparams['dates_all']))]),np.vstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_rf_up'][:,2,:,:] for da in range(len(pparams['dates_all']))])))

    lag_ls = ['STA','GLM']  
    lag_list = [0,0] #params['lag_list']
    # for da in tqdm(np.arange(len(date_ani2))):
    da = 0
    cell = 22 #np.arange(20) # np.arange(All_data[date_ani2[da]]['vis_rf_all'].shape[0]) #
    crange = np.max(np.abs(RF_all[0,cell]))
    for m,lag in enumerate(lag_list):
        ax = axs3[1,m]
        if m == 0:
            crange = np.max(np.abs(RF_all[0,cell]))
            im = ax.imshow(RF_all[0,cell],'RdBu_r',vmin=-crange,vmax=crange)
        else:
            crange = np.max(np.abs(RF_all[1,cell]))
            im = ax.imshow(RF_all[1,cell],'RdBu_r',vmin=-crange,vmax=crange)
        ax.set_title('{} \n cc={:.03f}'.format(lag_ls[m],cc_all2[m,cell]),fontsize=fontsize)

    scalebar = AnchoredSizeBar(axs3[0,1].transData, 20, '10 deg', 'lower left', pad=0.1, color='black', frameon=False, size_vertical=1)
    axs3[1,0].add_artist(scalebar)  
    for ax in axs3.flat:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_visible(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        
    lims = (-.05, .85)
    ticks = [0,.4,.8]
    ln_max = .8
    ax = axs3b[1]
    ax.scatter(cc_sta,vals_Vis,c='k',s=2,edgecolors='none')
    ax.scatter(cc_sta[cell],vals_Vis[cell],c='r',s=5,edgecolors='none')
    ax.plot(np.linspace(lims[0],ln_max),np.linspace(lims[0],ln_max),c='#6D6E71',ls='--',zorder=0)
    ax.set_xlabel(r'$cc_{STA}$',fontsize=fontsize)
    ax.set_ylabel(r'$cc_{GLM}$',fontsize=fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks,fontsize=fontsize-2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks,fontsize=fontsize-2)
    ax.axis('square')

    if figname != None:
        fig1.savefig(figname, transparent=True, bbox_inches='tight',dpi=300)



def figure3(All_data,pparams,exp_type='CropInputs',figname=None):
    shuff_FMHF = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_shuffled_FMHF_cc']  for da in range(len(pparams['dates_all']))])
    FM_RF_shuff = np.concatenate([All_data[pparams['date_ani2'][da]][exp_type]['Vis_shuffled_rf_all'][:,0]  for da in range(len(pparams['dates_all']))],axis=0)
    HF_RF_shuff = np.concatenate([All_data[pparams['date_ani2'][da]][exp_type]['HF_shuffled_rf_all'][:,0]  for da in range(len(pparams['dates_all']))],axis=0)
    FM_RF_all = np.concatenate([All_data[pparams['date_ani2'][da]][exp_type]['Vis_rf_all'][:,2]  for da in range(len(pparams['dates_all']))],axis=0)
    HF_RF_all = np.concatenate([All_data[pparams['date_ani2'][da]][exp_type]['HF_rf_all'][:,2]  for da in range(len(pparams['dates_all']))],axis=0)
    shuff_r2 = []
    for celln in np.arange(FM_RF_shuff.shape[0]):
        for celln2 in np.arange(HF_RF_shuff.shape[0]):
            shuff_r2.append(np.corrcoef(FM_RF_shuff[celln].flatten(),HF_RF_shuff[celln2].flatten())[0,1])
    shuff_r2 = np.array(shuff_r2)

    vals_FMHF  = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_FMHF_cc']  for da in range(len(pparams['dates_all']))])
    shuff_nstd = 2*np.nanstd(shuff_r2)
    shuff_nstd, np.nanmean(shuff_r2),np.nanmean(vals_FMHF)
    vals_FMHF  = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_FMHF_cc']  for da in range(len(pparams['dates_all']))])
    mean_HF_fr = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_meanfr'] for da in range(len(pparams['dates_all']))])
    sorted_r2 = np.argsort(vals_FMHF)[::-1] ##### Sorted indecies for R2
    hf_inds = np.arange(mean_HF_fr.shape[0])[mean_HF_fr<1]
    sorted_r2 = np.delete(sorted_r2,[np.argwhere(sorted_r2 == hf_inds[n])[0,0] for n in range(len(hf_inds))])
    RF_all = np.stack((np.vstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_rf_up'] for da in range(len(pparams['dates_all']))]),np.vstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_rf_up'] for da in range(len(pparams['dates_all']))])))
    
    fig1 = plt.figure(constrained_layout=False, figsize=(10.,1.8))
    gs0 = gridspec.GridSpec(nrows=2, ncols=5, figure=fig1,wspace=.15,hspace=.05)
    lag_ls = [-50,0,50]

    ########## Fig 3A ########## 
    cells = [3,167,62,101] #[58,61,34, 64, 58]#[62,61,101]# 
    num_cells = len(cells)

    gs00 = gridspec.GridSpecFromSubplotSpec(2, num_cells, subplot_spec=gs0[:,:3],wspace=.1,hspace=.05)
    axs1 = np.array([fig1.add_subplot(gs00[n,m]) for n in range(2) for m in range(num_cells)]).reshape(2,num_cells)
    xcut1=0
    xcut2=-0
    ycut1=0
    ycut2=-0
    for n, cell in enumerate(cells):
        crange2 = np.max(np.abs(RF_all[0,cell,2])) 
        im2 = axs1[0,n].imshow(RF_all[0,cell, 2],'RdBu_r', vmin=-crange2, vmax=crange2) 
        crange1 = np.max(np.abs(RF_all[1,cell,2])) 
        im1 = axs1[1,n].imshow(RF_all[1,cell, 2], 'RdBu_r', vmin=-crange1, vmax=crange1) 
        axs1[0,n].set_title('unit: {} \n cc={:.02f}'.format(n+1,vals_FMHF[cell]), fontsize=fontsize).set_multialignment('center')

    scalebar = AnchoredSizeBar(axs1[0,0].transData,
                        20, '10 deg', 'lower left', 
                        pad=0.1,
                        color='black',
                        frameon=False,
                        size_vertical=1,
                        )
    axs1[0,0].add_artist(scalebar)  
    axs1[1,0].set_ylabel('head- \n fixed', fontsize=fontsize)
    axs1[0,0].set_ylabel('freely \n moving', fontsize=fontsize)

    cbar2 = fig1.colorbar(im2, ax=axs1)
    cbar2.set_ticks([-crange2+.002, crange2])
    cbar2.set_ticklabels(['dark', 'light'])
    cbar2.ax.tick_params(labelsize=fontsize, rotation=90,width=0,length=0)
    cbar2.outline.set_linewidth(1)

    for ax in axs1.flat:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_visible(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])   

    ########## Fig 3B ########## 
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[:,3:],wspace=.3,hspace=.1)


    ### Shuffle bar
    shuff_Mot = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Mot_shuffled_cc_test'] for da in range(len(pparams['dates_all']))])
    shuff_Vis = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_shuffled_cc_test'] for da in range(len(pparams['dates_all']))])
    shuff_Add = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Add_shuffled_cc_test'] for da in range(len(pparams['dates_all']))])
    shuff_Mul = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Mul_shuffled_cc_test'] for da in range(len(pparams['dates_all']))])
    shuff_HF = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_shuffled_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Mot = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Mot_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Vis = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Add = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Add_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Mul = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Mul_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_HF = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_cc_test'] for da in range(len(pparams['dates_all']))])
    HF_meanfr = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_meanfr'] for da in range(len(pparams['dates_all']))])
    FM_meanfr = np.hstack([All_data[pparams['date_ani2'][da]]['actual_meanfr'] for da in range(len(pparams['dates_all']))])
    tot_Ncells = vals_Vis.shape[0] #128+67+56+71
    # % active HF,FM
    Nactive_HF = np.sum(HF_meanfr>1)/tot_Ncells
    Nactive_FM = np.sum(FM_meanfr>1)/tot_Ncells

    # % significant based on max of shuffle
    Nsig_HF = np.sum(vals_HF>np.nanmax(shuff_HF))/vals_Vis.shape[0]
    Nsig_FM = np.sum(vals_Vis>np.nanmax(shuff_Vis))/vals_Vis.shape[0]

    axs3 = fig1.add_subplot(gs01[0,0])
    ax = axs3
    ylim =1.25
    dylim = .25
    # patterns = ['//', '\\\\','//','\\\\']
    legend_label = [None,None,'HF','FM']
    data_f3b = [Nactive_HF,Nactive_FM,Nsig_HF,Nsig_FM]
    xticks = [0,1.25,2.5,3.75]
    clrs = 2*['k','#6D6E71']
    for n in range(4):
        ax.bar(xticks[n],data_f3b[n],color=clrs[n],edgecolor='white',label=legend_label[n]) # hatch=patterns[n],
    ax.set_yticks(np.arange(0,ylim,dylim))
    ax.set_yticklabels(np.round(np.arange(0,ylim,dylim),decimals=3),fontsize=fontsize-2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['% \nactive','% \nactive','% \nfit','% \nfit'],ha='center',fontsize=fontsize-2)
    ax.set_ylabel('fraction of units',fontsize=fontsize)
    ax.set_ylim([0,1])
    ax.legend(fontsize=fontsize, frameon=False,handletextpad=.1,loc='upper right')



    axs2 = fig1.add_subplot(gs01[0,1])

    snr_thresh=4
    noise_thresh=10#6
    FM_RF_all = np.concatenate([All_data[pparams['date_ani2'][da]][exp_type]['Vis_rf_all'][:,2]  for da in range(len(pparams['dates_all']))],axis=0)
    HF_RF_all = np.concatenate([All_data[pparams['date_ani2'][da]][exp_type]['HF_rf_all'][:,2]  for da in range(len(pparams['dates_all']))],axis=0)
    FM_RF_all_snr=FM_RF_all.copy()
    HF_RF_all_snr=HF_RF_all.copy()
    snr_FM = np.nanmax(np.abs(FM_RF_all_snr),axis=(-2,-1))/np.nanstd(FM_RF_all_snr,axis=(-2,-1))
    snr_HF = np.nanmax(np.abs(HF_RF_all_snr),axis=(-2,-1))/np.nanstd(HF_RF_all_snr,axis=(-2,-1))
    mean_HF_fr = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_meanfr'] for da in range(len(pparams['dates_all']))])
    # vals_FMHF  = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_FMHF_cc']  for da in range(len(pparams['dates_all']))])[((snr_FM>snr_thresh) & (snr_HF>snr_thresh) & (mean_HF_fr>1) & (vals_Vis>.01)&(vals_HF>.01))]
    RF_CC_HF_all = np.hstack([All_data[pparams['date_ani2'][da]]['5050split']['RF_CC_HF'] for da in range(len(pparams['dates_all']))])
    RF_CC_FM_all = np.hstack([All_data[pparams['date_ani2'][da]]['5050split']['RF_CC_FM'] for da in range(len(pparams['dates_all']))])
    thresh=.5
    vals_FMHF  = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_FMHF_cc']  for da in range(len(pparams['dates_all']))])[((RF_CC_HF_all>thresh)&(RF_CC_FM_all>thresh))]
    cc_nsig = vals_FMHF[((np.abs(vals_FMHF)<shuff_nstd))]
    cc_sig = vals_FMHF[((np.abs(vals_FMHF)>shuff_nstd))]

    ax = axs2
    hbins=.1
    lim0 = -.4
    lim1 = .65
    dlim = .2
    ylim = .3
    dylim = .1
    xticks = [-.3,0,.3,.6] 
    count,edges = np.histogram(cc_sig,bins=np.arange(lim0,lim1,hbins))
    edges_mid = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
    ax.bar(edges_mid, count/(len(cc_sig)+len(cc_nsig)),color='k',width=hbins, alpha=1,zorder=1) 
    count,edges = np.histogram(cc_nsig,bins=np.arange(lim0,lim1,hbins))
    ax.bar(edges_mid, count/(len(cc_sig)+len(cc_nsig)),color='#C1C1C3',width=hbins, alpha=1) 

    ax.set_xlabel('RF correlation',fontsize=fontsize)
    ax.set_ylabel('fraction of units',fontsize=fontsize)
    ax.set_yticks(np.arange(0,ylim,dylim))
    ax.set_yticklabels(np.round(np.arange(0,ylim,dylim),decimals=3),fontsize=fontsize-2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(xticks,decimals=1), fontsize=fontsize-2)
    ax.axvline(x=0,c='#828387',ls='--')

    vals_FMHF  = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['HF_FMHF_cc']  for da in range(len(pparams['dates_all']))])
    for cell in cells:
        ax.axvline(x=vals_FMHF[cell],ls='--',c='k')


    print('RF Significance Nsig:{:.02f}, Sig:{:.02}'.format(len(cc_nsig)/(len(cc_sig)+len(cc_nsig)),len(cc_sig)/(len(cc_sig)+len(cc_nsig))))
    plt.tight_layout()
    plt.show()
    if figname != None:
        fig1.savefig(figname, transparent=True, bbox_inches='tight')



def figure4(All_data,pparams,cell=265,exp_type='CropInputs',figname=None):
    CellNum_Tot = get_cellnums(All_data,pparams,exp_type=exp_type)
    da,celln = ([[da,np.where(CellNum_Tot[key]==cell)[0][0]] for da,key in enumerate(CellNum_Tot.keys()) if len(np.where(CellNum_Tot[key]==cell)[0])>0])[0]

    # modN = 1
    if 'OnlySpdPupil' in exp_type:
        pparams['titles'] = ['speed','pupil']
    elif 'SpdPup' in exp_type:
        pparams['titles'] = [r'theta $(\theta)$',r'phi $(\phi)$',r'pitch $(\rho)$',r'roll $(\omega)$','speed','pupil']
    else:
        pparams['titles'] = [r'theta $(\theta)$',r'phi $(\phi)$',r'pitch $(\rho)$',r'roll $(\omega)$']
    
    pparams['anglim'] = 70
    pparams['quartiles'] = np.arange(0,1.25,.25)
    pparams['spike_percentiles'] = np.arange(0,1.25,.25)
    pparams['spike_percentiles'][-1]=.99
    pparams['spk_percentile2'] = np.arange(.125,1.125,.25)

    ncells=All_data[pparams['date_ani2'][da]][exp_type]['Vis_pred_smooth'].shape[-1]
    predcell = All_data[pparams['date_ani2'][da]][exp_type]['Vis_pred_smooth'][:,celln]
    nspcell = All_data[pparams['date_ani2'][da]]['actual_smooth'][:,celln]
    pred_range = np.quantile(predcell,[.1,.9])
    test_nsp_range = np.quantile(nspcell,[.01,1])
    pred_rangelin = np.quantile(predcell,pparams['spike_percentiles'])
    xbin_pts = np.quantile(predcell,pparams['spk_percentile2'])
    stat_bins = len(pred_rangelin)
    tuning_sig_all = np.vstack([All_data[pparams['date_ani2'][da]]['tuning_sig'].squeeze() for da in range(len(pparams['dates_all']))])
    ax_ylims_all = np.vstack([All_data[pparams['date_ani2'][da]]['ax_lims'].squeeze() for da in range(len(pparams['dates_all']))])

    fig = plt.figure(constrained_layout=False, figsize=(10,6))
    gs0 = gridspec.GridSpec(ncols=4, nrows=3, figure=fig,wspace=.5,hspace=.8)

    ########## Fig 4A ########## 
    ##### Example Tuning Curve #####
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0,1:],wspace=.8,hspace=.7)
    gs01 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs0[0:,:1],wspace=.05,hspace=.8)
    axs0a = np.array([fig.add_subplot(gs00[0,0]),fig.add_subplot(gs00[0,1:])])
    axs0aa = np.array([fig.add_subplot(gs01[0,:1])])

    top_yaxs = np.max(ax_ylims_all[cell])#+.05*np.max(ax_ylims_all[cell])
    traces = np.zeros((ncells,len(pparams['titles']),len(pparams['quartiles'])-1,stat_bins-1))
    traces_mean = np.zeros((ncells,len(pparams['titles']),stat_bins-1))
    edges_all = np.zeros((ncells,len(pparams['titles']),len(pparams['quartiles'])-1,stat_bins-1))

    modeln = 1
    ax = axs0aa[0]
    # celln = 49#52#106
    # da = 3
    t = np.argmin(np.abs((np.arange(All_data[pparams['date_ani2'][da]]['actual_smooth'].shape[0])*params['model_dt'])-100)) # 200 seconds in #6000
    dt = int(100/params['model_dt']) #All_data[pparams['date_ani2'][da]]['actual_smooth'].shape[0] # dt in seconds converted to time points
    ##### Plotting Position Signal #####
    pred_time = (np.arange(All_data[pparams['date_ani2'][da]]['actual_smooth'].shape[0])*params['model_dt'])[0:np.abs(dt)]
    ax.plot(pred_time,All_data[pparams['date_ani2'][da]]['actual_smooth'][t:t+dt,celln],c='k',lw=1.5,label='firing rate',zorder=0)
    ax.set_ylabel('sp/sec',fontsize=fontsize,color='k')
    ax.set_xlabel('time (s)',fontsize=fontsize)
    ax = axs0aa[0].twinx()
    ax.plot(pred_time,All_data[pparams['date_ani2'][da]]['move_test'][t:t+dt,modeln],c='#6D6E71',lw=.5,alpha=1,label='phi')
    # ax.set_ylabel('phi (deg)',fontsize=fontsize,color='#6D6E71')
    ax.spines.right.set_visible(True)
    ax.spines['right'].set_color('#6D6E71')
    ax.tick_params(axis='y', colors='#6D6E71')
    ax.set_ylabel('phi (deg)',fontsize=fontsize, color='#6D6E71')
    # ax.yaxis.label.set_color('#6D6E71')
    # legend1 = ax.legend(['phi (deg)'],labelcolor=['#6D6E71'], fontsize=fontsize, markerscale=0, frameon=False, handlelength=0, handletextpad=0, bbox_to_anchor=(1.1, 1))

    ax = axs0a[0]
    metric = All_data[pparams['date_ani2'][da]]['move_test'][params['bin_length']:-params['bin_length'],modeln]
    nranges = np.quantile(metric,pparams['quartiles'])
    stat_range, edges, _ = binned_statistic(metric,All_data[pparams['date_ani2'][da]]['nsp_raw'][params['bin_length']:-params['bin_length'],celln],statistic='mean',bins=nranges)
    stat_range_std, _, _ = binned_statistic(metric,All_data[pparams['date_ani2'][da]]['nsp_raw'][params['bin_length']:-params['bin_length'],celln],statistic='std',bins=nranges)
    stat_range_count, _, _ = binned_statistic(metric,All_data[pparams['date_ani2'][da]]['nsp_raw'][params['bin_length']:-params['bin_length'],celln],statistic='count',bins=nranges)

    edge_mids = np.quantile(metric,pparams['spk_percentile2'])#
    # ax.plot(edge_mids,stat_range/params['model_dt'],'-', lw=3,c='k',zorder=0,markeredgewidth=0)
    ax.errorbar(edge_mids,stat_range/params['model_dt'],yerr=stat_range_std/np.sqrt(stat_range_count), elinewidth=5,c='k',zorder=0,markeredgewidth=0)
    ax.scatter(edge_mids,stat_range/params['model_dt'],s=25, c=q_clrs,zorder=1,edgecolors='none')
    ax.set_ylim(bottom=0,top=np.max(ax_ylims_all[cell])) #+.05*np.max(ax_ylims_all[cell])
    xlim_range = 50 #np.max(np.abs([nranges[0],nranges[-1]]))
    ax.set_xlim(-xlim_range,xlim_range)
    ax.set_ylim(0,5)
    ax.set_xlabel('angle (deg)',fontsize=fontsize)
    ax.set_ylabel('sp/sec',fontsize=fontsize)
    ax.set_title(pparams['titles'][modeln],fontsize=fontsize)

    ########## Fig 4B ########## 
    ##### Modulation Index Histograms #####
    # x_sc = np.stack((np.random.normal(0, 0.1, tuning_sig_all.shape[0]), np.random.normal(1, 0.1, tuning_sig_all.shape[0]),np.random.normal(2, 0.1, tuning_sig_all.shape[0]),np.random.normal(3, 0.1, tuning_sig_all.shape[0])))
    x_sc = np.stack([np.random.normal(n, 0.1, tuning_sig_all.shape[0]) for n in range(len(pparams['titles']))])
    sig_cells = np.sum(tuning_sig_all>.33,axis=0)/tuning_sig_all.shape[0]
    ax = axs0a[1]
    for modeln in np.arange(0,len(pparams['titles'])):
        ax.scatter(x_sc[modeln],tuning_sig_all[:,modeln],s=5,c='k',edgecolors='none')
        ax.text(modeln,.9,'{:.02f}'.format(sig_cells[modeln]),fontsize=fontsize)
    ax.set_xticks(np.arange(len(pparams['titles'])))
    ax.set_xticklabels(pparams['titles'],fontsize=fontsize)
    ax.set_ylim([0,1])
    ax.set_yticks([0,.5,1])
    ax.set_yticklabels([0,.5,1],fontsize=fontsize)
    ax.set_ylabel('Modulation Index (MI)',fontsize=fontsize)
    ax.axhline(y=.33,ls='--',color='#6D6E71')


    ########## Fig 4C/G ########## 
    axs1a = np.array([fig.add_subplot(gs01[1,0])])
    axs2a = np.array([fig.add_subplot(gs01[2,0])])

    cmap = mpl.colors.ListedColormap(q_clrs)

    # celln = 49#52#106
    # da = 3
    mod_titles2 = ['pos','vis','add','mul','HF']

    t = np.argmin(np.abs((np.arange(All_data[pparams['date_ani2'][da]]['actual_smooth'].shape[0])*params['model_dt'])-100)) # 200 seconds in #6000
    dt = int(100/params['model_dt']) #All_data[pparams['date_ani2'][da]]['actual_smooth'].shape[0] # dt in seconds converted to time points
    ##### Plotting Firing Rate #####
    pred_time = (np.arange(All_data[pparams['date_ani2'][da]]['actual_smooth'].shape[0])*params['model_dt'])[0:np.abs(dt)]
    ax = axs1a[0]
    zorder = (np.argsort([All_data[pparams['date_ani2'][da]][exp_type][pparams['mod_titles'][modN]+'_cc_test'][celln] for modN in range(len(pparams['mod_titles'][:-1]))])+1)[::-1]
    ax.plot(pred_time,All_data[pparams['date_ani2'][da]]['actual_smooth'][t:t+dt,celln],c='k',lw=1,label='data',zorder=0)
    for modN in range(0,2):
        ax.plot(pred_time,All_data[pparams['date_ani2'][da]][exp_type][pparams['mod_titles'][modN]+'_pred_smooth'][t:t+dt,celln],c=mod_clrs[modN],lw=1,
                label=r'$cc_{{{}}}$: {:.02f}'.format(mod_titles2[modN],All_data[pparams['date_ani2'][da]][exp_type][pparams['mod_titles'][modN]+'_cc_test'][celln]),zorder=zorder[modN])
    legend1 = ax.legend(labelcolor='linecolor', fontsize=fontsize, handlelength=0, handletextpad=0,loc='upper right',ncol=5, bbox_to_anchor=(1, 1.3))
    # legend1.texts[1].set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),path_effects.Normal()])
    ax.set_xlabel('time (s)',fontsize=fontsize)
    ax.set_ylabel('sp/sec',fontsize=fontsize)


    ##### Plotting Firing Rate #####
    pred_time = (np.arange(All_data[pparams['date_ani2'][da]]['actual_smooth'].shape[0])*params['model_dt'])[0:np.abs(dt)]
    ax = axs2a[0]
    zorder = (np.argsort([All_data[pparams['date_ani2'][da]][exp_type][pparams['mod_titles'][modN]+'_cc_test'][celln] for modN in range(len(pparams['mod_titles'][:-1]))])+1)[::-1]
    ax.plot(pred_time,All_data[pparams['date_ani2'][da]]['actual_smooth'][t:t+dt,celln],c='k',lw=1,label='data',zorder=0)
    for modN in range(2,len(pparams['mod_titles'])-2):
        ax.plot(pred_time,All_data[pparams['date_ani2'][da]][exp_type][pparams['mod_titles'][modN]+'_pred_smooth'][t:t+dt,celln],c=mod_clrs[modN],lw=1,
                label=r' $cc_{{{}}}$: {:.02f}'.format(mod_titles2[modN],All_data[pparams['date_ani2'][da]][exp_type][pparams['mod_titles'][modN]+'_cc_test'][celln]),zorder=zorder[modN])
    ax.legend(labelcolor='linecolor', fontsize=fontsize, handlelength=0, handletextpad=0,loc='upper right',ncol=5, bbox_to_anchor=(1, 1.3))
    ax.set_xlabel('time (s)',fontsize=fontsize)
    ax.set_ylabel('sp/sec',fontsize=fontsize)



    gs02 = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec=gs0[1:,1:],wspace=.5,hspace=.8)
    axs3a = np.array([fig.add_subplot(gs02[n,m]) for n in range(2) for m in range(3)]).reshape(2,3)

    ########## Fig 4D ########## 
    fr_th = 1 # hz
    vals_Mot = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Mot_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Vis = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Add = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Add_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Mul = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Mul_cc_test'] for da in range(len(pparams['dates_all']))])


    lims = (-.05, .85)
    ticks = [0,.4,.8]
    ln_max = .8
    crange=500
    ##### Vis vs Mot #####
    ax = axs3a[0,0]
    im=ax.scatter(vals_Vis,vals_Mot,c='k',s=5,edgecolors='none')
    ax.plot(np.linspace(lims[0],ln_max),np.linspace(lims[0],ln_max),c='#6D6E71',ls='--',zorder=0)
    ax.set_xlabel(r'$cc_{vis}$',fontsize=fontsize)
    ax.set_ylabel(r'$cc_{pos}$',fontsize=fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.axis('square')
    ax.set_ylim(0,.8)
    ax.set_xlim(0,.8)

    ########## Fig 4E ########## 

    ax = axs3a[0,1]
    stat_all, edges, _ = binned_statistic(predcell,nspcell,statistic='mean',bins=pred_rangelin)
    edge_mids = xbin_pts 
    traces_mean[celln,modeln]=stat_all
    max_fr = np.max(stat_all)
    for n in range(len(nranges)-1):
        ind = np.where(((metric<=nranges[n+1])&(metric>nranges[n])))[0]
        pred = predcell[ind]
        sp = nspcell[ind]
        stat_range, edges, _ = binned_statistic(pred, sp, statistic='mean',bins=pred_rangelin)
        edge_mids = xbin_pts
        traces[celln,modeln,n]=stat_range
        edges_all[celln,modeln,n]=edge_mids
        ax.plot(edge_mids, stat_range,'.-', c=q_clrs[n],label='{:.02f} : {:.02f}'.format(nranges[n],nranges[n+1]),lw=2,ms=10,alpha=.9,zorder=2,markeredgewidth=0)
        ax.set_xlabel('predicted (sp/sec)',fontsize=fontsize)
        ax.set_ylabel('actual (sp/sec)',fontsize=fontsize)

    lim_max = np.nanmax(np.hstack((edge_mids,traces[celln,modeln].flatten())))+.5*np.std(edges)
    lim_min = np.nanmin(np.hstack((edge_mids,traces[celln,modeln].flatten())))-.5*np.std(edges)
    lims = (-.05, lim_max) if (lim_min)<0 else (lim_min,lim_max) 
    ax.plot(np.linspace(lims[0],lims[1]),np.linspace(lims[0],lims[1]),c='#6D6E71',ls='--',zorder=0)
    # ax.plot(edge_mids, stat_all,'.-', c='k', lw=5, ms=20, label='All_data', alpha=.8,zorder=1)
    ax.set_xticks(np.arange(0,lims[1],int(lims[1]/3)))
    ax.set_yticks(np.arange(0,lims[1],int(lims[1]/3)))
    ax.set_xticklabels(np.arange(0,lims[1],int(lims[1]/3),dtype=int))
    ax.set_yticklabels(np.arange(0,lims[1],int(lims[1]/3),dtype=int))
    ax.axis('square')
    ax.set(xlim=lims, ylim=lims)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)


    ########## Fig 4H ########## 

    ##### Vis vs Add/Mul #####
    cmap_mod = mpl.colors.ListedColormap(mod_clrs[2:4])
    vals_AM = np.nanmax(np.stack((vals_Add,vals_Mul)),axis=0)
    AvM = vals_Add<vals_Mul
    AvM_thresh = (~np.isnan(vals_Vis)) & (~np.isnan(vals_AM))
    ax = axs3a[1,0]
    scatter=ax.scatter(vals_Vis[AvM_thresh],vals_AM[AvM_thresh],s=5,c=AvM[AvM_thresh],cmap=cmap_mod,edgecolors='none')
    ax.plot(np.linspace(lims[0],ln_max),np.linspace(lims[0],ln_max),c='#6D6E71',ls='--',zorder=0)
    ax.set_xlabel(r'$cc_{vis}$',fontsize=fontsize)
    ax.set_ylabel(r'$cc_{joint}$',fontsize=fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.axis('square')
    ax.set_ylim(-.05, .8)
    ax.set_xlim(-.05, .8)
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),fontsize=fontsize,markerscale=0, handlelength=0, handletextpad=0,loc="lower right")
    legend1.get_texts()[0].set_text('add')
    legend1.get_texts()[0].set_color(mod_clrs[2])
    legend1.get_texts()[1].set_text('mul')
    legend1.get_texts()[1].set_color(mod_clrs[3])

    move_r2_th = ((vals_Add>vals_Vis) | (vals_Mul>vals_Vis)) # (vals_Mot>.05) & (vals_Vis>.05) &
    vals_Mot = vals_Mot[move_r2_th]
    vals_Vis = vals_Vis[move_r2_th]
    vals_Add = vals_Add[move_r2_th]
    vals_Mul = vals_Mul[move_r2_th]
    # celltypes_all2 = celltypes_all[:,move_r2_th]


    ##### R2 Explained Variance V2 #####
    diff = vals_Mul-vals_Add
    increase = vals_Vis
    th1 = ((increase>0) & (vals_Vis>.22))
    th2 = ((increase>0) & (vals_Vis<.22))

    NMul = np.sum(((diff[th1]>0)&(vals_Vis[th1]>.22)))
    NAdd = np.sum(((diff[th1]<0)&(vals_Vis[th1]>.22)))
    ax=axs3a[1,1]
    im = ax.scatter(diff[th1],increase[th1],s=5,c='k',edgecolors='none')
    im = ax.scatter(diff[th2],increase[th2],s=5,c='#6D6E71',edgecolors='none')
    # im = ax.scatter(diff[th],increase[th],c=celltypes_all2[1,th],cmap='jet',s=20,vmin=-500,vmax=500)
    ax.axvline(x=0,c='#6D6E71',ls='--',zorder=0)
    ax.set_ylabel(r'$cc_{vis}$',fontsize=fontsize)
    ax.set_xlabel(r'$cc_{mul}-cc_{add}$',fontsize=fontsize)
    ax.set_title(r'$N_{{add}}$:{}, $N_{{mul}}$:{}'.format(NAdd,NMul),fontsize=fontsize+2)
    ax.set_ylim(-.05,.8)
    ax.set_xlim(-.06,.06)
    ax.set_xticks([-.04, 0, .04])
    ax.set_xticklabels(np.round([-.04,0,.04],decimals=2))
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.axhline(y=np.sqrt(.05),color='#6D6E71',ls='--',zorder=0)


    ##### Hist of Add-Mul #####
    vals_Mot = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Mot_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Vis = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Vis_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Add = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Add_cc_test'] for da in range(len(pparams['dates_all']))])
    vals_Mul = np.hstack([All_data[pparams['date_ani2'][da]][exp_type]['Mul_cc_test'] for da in range(len(pparams['dates_all']))])
    move_r2_th = ((vals_Add>vals_Vis) | (vals_Mul>vals_Vis)) & (vals_Vis>.22) # & (vals_Mul>vals_Vis)
    vals_Mot = vals_Mot[move_r2_th]
    vals_Vis = vals_Vis[move_r2_th]
    vals_Add = vals_Add[move_r2_th]
    vals_Mul = vals_Mul[move_r2_th]
    # celltypes_all2 = celltypes_all[:,move_r2_th]
    ax=axs3a[1,2]
    hbins=.005
    count,edges = np.histogram(vals_Mul-vals_Add,bins=np.arange(-.05,.05,hbins))
    edges_mid = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
    ax.bar(edges_mid, count/len(vals_Mul),color='k',width=hbins, alpha=1)
    ax.axvline(x=0,c='#6D6E71',ls='--',lw=2)
    ax.set_xlabel(r'$cc_{mul}-cc_{add}$',fontsize=fontsize)
    ax.set_ylabel('fraction of units',fontsize=fontsize)
    ax.set_xticks([-.04, 0, .04])
    ax.set_xticklabels(np.round([-.04,0,.04],decimals=2))
    ax.set_xlim([-.04, .04])
    ax.set_yticks([0,.1,.2,.3])

    plt.show()
    gs0.tight_layout(fig)
    if figname != None:
        fig.savefig(figname, transparent=True, bbox_inches='tight')


