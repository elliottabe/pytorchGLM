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
from matplotlib_scalebar.scalebar import ScaleBar

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
titles = [r'theta $(\theta)$',r'phi $(\phi)$',r'pitch $(\rho)$',r'roll $(\omega)$']
mod_clrs = ["#B541FF","#00A14B","#118ab2","#ef476f","#073b4c"]
fontsize=10
mod_titles = ['pos','vis','add','mul','HF']

mpl.rcParams.update({'font.size':         12,
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



def figure2(All_data,plotparams):
    vals_Vis = np.hstack([All_data[date_ani2[da]]['vis_cc_test'] for da in range(len(dates_all))])
    vals_VisNoSh = np.hstack([NoSh_data[date_ani2[da]]['VisNoSh_cc_test'] for da in range(len(dates_all))])
    r2_all = np.stack((vals_VisNoSh,vals_Vis))
    fontsize=10
    fig1 = plt.figure(constrained_layout=False, figsize=(10.8,4.25))
    gs0 = fig1.add_gridspec(nrows=3, ncols=8, wspace=.8, hspace=.9)
    lag_ls = [-100,-50,0,50,100]
    lag_list = [0,1,2,3,4]
    ########## Fig 2B ########## 
    da=0
    cells = [22,34,42,101] #np.arange(18)
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=len(cells),ncols=len(lag_list), subplot_spec=gs0[:,:4], wspace=.05, hspace=.08)
    axs1 = np.array([fig1.add_subplot(gs00[n,m]) for n in range(len(cells)) for m in range(len(lag_list))]).reshape(len(cells),len(lag_list))
    params['nt_glm_lag']=5
    for n, cell in enumerate(cells):
        crange = np.max(np.abs(All_data[date_ani2[da]]['vis_rf_up'][cell]))
        for m,lag in enumerate(lag_list):
            ax = axs1[n,m]
            im = ax.imshow(All_data[date_ani2[da]]['vis_rf_up'][cell,lag],'RdBu_r',vmin=-crange,vmax=crange)
            axs1[0,m].set_title('{}ms'.format(lag_ls[m]),fontsize=fontsize)

        axs1[n,0].set_ylabel('unit {}'.format(n+1),fontsize=fontsize)
        
    scale1 = ScaleBar(dx=.5, width_fraction=.05,location='lower left',scale_formatter=lambda value, unit: f'{value} deg')
    axs1[0,0].add_artist(scale1)

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
    t = np.argmin(np.abs((np.arange(All_data[date_ani2[da]]['actual_smooth'].shape[0])*params['model_dt'])-0)) # 200 seconds in #6000
    dt = int(300/params['model_dt']) #All_data[date_ani2[da]]['actual_smooth'].shape[0] # dt in seconds converted to time points
    pred_time = (np.arange(All_data[date_ani2[da]]['actual_smooth'].shape[0])*params['model_dt'])[0:np.abs(dt)]
    ax.plot(pred_time,All_data[date_ani2[da]]['actual_smooth'][t:t+dt,celln],c='k',lw=.75,zorder=0,label='data')
    ax.plot(pred_time,All_data[date_ani2[da]][mod_titles[modN]+'_pred_smooth'][t:t+dt,celln],c=mod_clrs[modN],lw=.75,
                label='{} cc: {:.02f}'.format(mod_titles[modN],All_data[date_ani2[da]][mod_titles[modN]+'_r2_test'][celln]),zorder=1)
    ax.legend(labelcolor='linecolor', fontsize=10, handlelength=0, handletextpad=0,loc='upper left',ncol=2)
    ax.set_xlabel('time (s)',fontsize=fontsize)
    ax.set_ylabel('sp/sec',fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize-2)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
    # scale1 = ScaleBar(dx=1, width_fraction=.05,location='lower left',scale_formatter=lambda value, unit: f'{value} s')
    # scale2 = ScaleBar(dx=1, width_fraction=.01,rotation="vertical",location='center left',scale_formatter=lambda value, unit: f'{value} sp/s')
    # ax.add_artist(scale1)
    # ax.add_artist(scale2)

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
    gs02b = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs02[:2,:2],wspace=.1,hspace=.4)
    axs3 = np.array([fig1.add_subplot(gs02b[0,0]), fig1.add_subplot(gs02b[0,1]),fig1.add_subplot(gs02b[1,0]), fig1.add_subplot(gs02b[1,1])]).reshape(2,2)
    lag = 1
    da = 0
    cells = [22]
    for n, cell in enumerate(cells):
        ax = axs3[0,n]
        crange2 = np.max(np.abs(NoSh_data[date_ani2[da]]['VisNoSh_rf_up'][cell]))
        im2 = ax.imshow(NoSh_data[date_ani2[da]]['VisNoSh_rf_up'][cell,lag],'RdBu_r', vmin=-crange2, vmax=crange2)
        ax.set_title('shifter off',fontsize=fontsize)
        ax = axs3[0,n+1]
        crange2 = np.max(np.abs(All_data[date_ani2[da]]['vis_rf_up'][cell,2]))
        im2 = ax.imshow(All_data[date_ani2[da]]['vis_rf_up'][cell,2],'RdBu_r', vmin=-crange2, vmax=crange2)
        ax.set_title('shifter on',fontsize=fontsize)

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

    lag = 1
    da = 0
    cells = [0]
    for n, cell in enumerate(cells):
        ax = axs3[1,n]
        crange2 = np.max(np.abs(SimRF_Data['RF_actual'][cell]))
        im2 = ax.imshow(SimRF_Data['RF_actual'][cell],'RdBu_r', vmin=-crange2, vmax=crange2)
        ax.set_title('simulated RF', fontsize=fontsize)
        ax = axs3[1,n+1]
        crange1 = np.max(np.abs(RF_SimFit[cell]))
        im2 = ax.imshow(RF_SimFit[cell,0],'RdBu_r', vmin=-crange1, vmax=crange1)
        ax.set_title('reconstructed RF', fontsize=fontsize)
        
    for ax in axs3.flat:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_visible(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    ########## Fig 2G ########## 
    # gs05 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[2,6:],wspace=.1,hspace=.1)
    # axs5 = np.array([fig1.add_subplot(gs05[0,0])])
    ax=axs3b[1]
    hbins = .05
    lim0 = .2
    lim1 = 1.1
    dlim = .5
    xlab = 'RF cc'
    simfr_mean = np.mean(SimRF_Data['yte'],axis=0)/params['model_dt']
    simfr_low = Sim_r2[2::3] #Sim_r2[((simfr_mean<5))]
    simfr_med = Sim_r2[1::3] #Sim_r2[((simfr_mean>5) & (simfr_mean<10))]
    simfr_high = Sim_r2[0::3] #Sim_r2[((simfr_mean>10))]

    np.nanmean(simfr_low),np.nanmean(simfr_med),np.nanmean(simfr_high),
    mean_fr_low = np.nanmean(simfr_mean[2::3])
    mean_fr_med = np.nanmean(simfr_mean[1::3])
    mean_fr_high = np.nanmean(simfr_mean[0::3])

    count,edges = np.histogram(simfr_med,bins=np.arange(lim0,lim1,hbins))
    edges_mid = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
    ax.bar(edges_mid, count/len(simfr_med),color='k',width=hbins, alpha=1,label='high FR')

    ax.set_xticks([0,.5,1])
    ax.set_xticklabels([0,.5,1],fontsize=fontsize-2)
    ax.set_xlabel('cc')
    # ax.legend(labelcolor='linecolor', fontsize=14, handlelength=0, handletextpad=0,loc='upper left',ncol=1)#, bbox_to_anchor=(.1, .9))

    ax.set_yticks([0,.05,.1,])
    ax.set_yticklabels([0,.05,.1,],fontsize=fontsize-2)
    ax.set_xlabel(xlab,fontsize=fontsize)
    ax.axvline(x=np.nanmean(simfr_med),lw=2,c='#6D6E71',ls='--',zorder=1)
    ax.set_ylabel('fraction of units',fontsize=fontsize)

    fig1.savefig(paper_fig_dir/('Figure2_draft_V2c_cc.pdf'), transparent=True, bbox_inches='tight',dpi=300)
