import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats, spatial
import sys, os, time, json
from pathlib import Path
import pickle as pkl

sys.path.append('../PreProcessing/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

import TreeMazeFunctions as TMF
import spike_functions as SF
import analyses_table as AT
import plot_functions as PF
import OpenFieldFunctions as OF
import spatial_tuning as ST
from importlib import reload  # Python 3.4+ only.
from pre_process_neuralynx import get_position

from scipy.interpolate import interp1d
sns.set(style="whitegrid",font_scale=1.2,rc={
    'axes.spines.bottom': False,
'axes.spines.left': False,
'axes.spines.right': False,
'axes.spines.top': False,
'axes.edgecolor':'0.5'})

################################################################################
# Constants
################################################################################
OFBehavParams = {'timeStep':0.02,'mm2bin':30, 'spThr':5, 'ang2bin':np.deg2rad(10),
 'secThr':0.05,'xLims':[-650,650],'yLims':[-100,1450],'sm_sigma':1.5,'sm_w':5}

for k,v in OFBehavParams.items():
    exec('\n{}={}'.format(k,v))

# filtering params
med_filt_window = 12 #
smooth_filt_window = 12 # in samples 12/60 = 200ms
med_ang_filt_window = 6
filtCoeff = signal.firwin(smooth_filt_window, cutoff = 0.2, window = "hanning")


def get_session_OF_pos(sePaths):
    fn = sePaths['Raw'] / 'VT1.nvt'
    se = sePaths['session']

    t,x,y,ha = get_position(fn)
    x2,y2,ha2 = ScaleRotateSmoothTrackDat(x,y,ha)

    sp,hd = ST.getVelocity(x2,y2,1/60)
    sp = sp/10 # conver to cm/s
    hd = np.rad2deg(hd) # conver to deg

    ts, xs  = ReSampleDat(t,x2,timeStep)
    _, ys   = ReSampleDat(t,y2,timeStep)
    _, hds  = ReSampleDat(t,hd,timeStep)
    _, ha2s = ReSampleDat(t,ha2,timeStep)
    _, sps  = ReSampleDat(t,sp,timeStep)
    ts = np.round(ts*1000)/1000 #round tp to ms resolution.

    Occ_Counts,_,_ =ST.getPositionMat(ys,xs,yLims,xLims,spacing=mm2bin)

    T = np.max(ts)-np.min(ts)
    nBins = Occ_Counts.size
    nOccBin = Occ_Counts.sum()
    Occ_Secs = Occ_Counts*timeStep
    Occ_SmSecs = ST.getSmoothMap(Occ_Secs,sm_w,sm_sigma)

    th = ha2s[sps>spThr]
    ha_stats = ST.angle_stats(th,step=np.rad2deg(ang2bin))

    th = hds[sps>spThr]
    hd_stats = ST.angle_stats(th,step=np.rad2deg(ang2bin))

    OFBehavDat = {'se':se,'t':ts,'x':xs,'y':ys,'sp':sps,'Occ_Counts':Occ_Counts,'Occ_Secs':Occ_Secs, 'Occ_SmSecs': Occ_SmSecs ,
              'HA':ha2s,'HD':hds,'HAo':ha,'HA_Stats': ha_stats,'HD_Stats': hd_stats}

    return OFBehavDat

def ScaleRotateSmoothTrackDat(x,y,ha):

    # pixel limits
    x_pixLims = [100, 650]
    y_pixLims = [100, 600]

    # pixel 2 mm
    y_pix2mm = 1480/444
    x_pix2mm = 1300/344

    # mask
    mask_x = np.logical_or(x<x_pixLims[0],x>x_pixLims[1])
    mask_y = np.logical_or(y<y_pixLims[0],y>y_pixLims[1])
    mask = np.logical_or(mask_x,mask_y)

    x[mask] = np.nan
    y[mask] = np.nan

    # pixel translation
    x = x - 380
    y = y - 280

    # rotation angle for the maze (for original pixel space)
    rot_ang=np.pi/2+0.08

    # y translation in pixel space to
    x_translate = 0
    y_translate = 200

    # speed thr
    spd_thr = 50 # mm/frame -> mm/frame*60frames/s*1cm/10mm = 50*6 cm/s

    ######## Operations ########
    # rotate
    x2,y2= ST.RotateXY(x,y,rot_ang)

    # re-scale
    x2 = (x2+x_translate)*x_pix2mm
    y2 = (y2+y_translate)*y_pix2mm

    # compute velocity to create speed threshold
    dx = np.append(0,np.diff(x2))
    dy = np.append(0,np.diff(y2))
    dr = np.sqrt(dx**2+dy**2)
    mask_r = np.abs(dr)>spd_thr

    #mask creating out of bound zones in mm space
    mask_y = np.logical_or(y2<yLims[0],y2>yLims[1])
    mask_x = np.logical_or(x2<xLims[0],x2>xLims[1])
    mask = np.logical_or(mask_x,mask_y)
    mask = np.logical_or(mask,mask_r)

    x2[mask]=np.nan
    y2[mask]=np.nan
    ha2 = np.array(ha)
    ha2[mask]=np.nan

    # double round of median filters to deal with NaNs
    x3 = medFiltFilt(x2,med_filt_window)
    y3 = medFiltFilt(y2,med_filt_window)
    ha3 = medFiltFilt(ha2,med_ang_filt_window)
    ha3 = medFiltFilt(ha2,med_ang_filt_window)

    # if there are still NaNs assign id to previous value
    badIds = np.where(np.logical_or(np.isnan(x3), np.isnan(y3)))[0]
    for ii in badIds:
        x3[ii] = ST.getLastNotNanVal(x3,ii)
        y3[ii] = ST.getLastNotNanVal(y3,ii)
        ha3[ii] = ST.getLastNotNanVal(ha3,ii)

    ha3 = ha3-180 # rotate angle (-180 to 180)
    # filter / spatial smoothing
    x3 = signal.filtfilt(filtCoeff,1,x3)
    y3 = signal.filtfilt(filtCoeff,1,y3)

    return x3,y3,ha3

def allOFBehavPlots(OFBehavDat):

    sp = OFBehavDat['sp']
    se = OFBehavDat['se']

    f = plt.figure(figsize=(16,18))

    gsTop = mpl.gridspec.GridSpec(2,3)
    axTop = np.full((2,3),type(gsTop[0,0]))
    for i in np.arange(2):
        for j in np.arange(3):
            axTop[i,j] = f.add_subplot(gsTop[i,j])
    gsTop.tight_layout(f,rect=[0,0.25,1,0.70])

    # xy traces
    axTop[0,0].plot(OFBehavDat['x'],OFBehavDat['y'],linewidth=1,color='k',alpha=0.5)
    axTop[0,0].set_aspect('equal', adjustable='box')
    axTop[0,0].set_axis_off()

    axTop[0,1]=sns.heatmap(OFBehavDat['Occ_Counts'],xticklabels=[],yticklabels=[],cmap='magma',square=True,robust=True,cbar=False,ax=axTop[0,1])
    axTop[0,1].invert_yaxis()

    axTop[0,2]=sns.heatmap(OFBehavDat['Occ_SmSecs'],xticklabels=[],cmap='magma',yticklabels=[],square=True,robust=True,cbar=False,ax=axTop[0,2])
    axTop[0,2].invert_yaxis()

    axTop[1,0]= sns.heatmap(OFBehavDat['Occ_SmSecs']>secThr,cmap='Greys',xticklabels=[],yticklabels=[],square=True,cbar=False,ax=axTop[1,0])
    axTop[1,0].invert_yaxis()

    axTop[1,1]= sns.distplot(sp,ax=axTop[1,1])
    axTop[1,1].set_yticklabels([])


    axTop[1,2] = sns.distplot(OFBehavDat['HAo'],ax=axTop[1,2])
    axTop[1,2].set_yticklabels([])

    titles = ['xy','mm/bin={}; maxCnt={}'.format(mm2bin,OFBehavDat['Occ_Counts'].max()),
            'max s/bin = {0:0.2f}'.format(OFBehavDat['Occ_Counts'].max()),
            'secThr = {}'.format(secThr), 'Speed [cm/s]', 'HA orig [deg]' ]

    cnt=0
    for a in axTop.flatten():
        a.set_title(titles[cnt])
        cnt+=1


    gsBot = mpl.gridspec.GridSpec(1,4)
    axBot = np.full(4,type(gsBot))
    for i in np.arange(4):
        axBot[i] = f.add_subplot(gsBot[i],projection='polar')
    gsBot.tight_layout(f,rect=[0,0,1,0.25])


    for i in [0,1]:
        if i==0:
            txt = 'HA'
        else:
            txt = 'HD'

        th = OFBehavDat[txt][sp>spThr]
        stats = OFBehavDat[txt+'_Stats']
        counts = stats['counts']
        bins = stats['bins']
        ang = stats['ang']
        r = stats['r']
        R = stats['R']
        pval = stats['pval']

        axBot[0+i*2].plot(bins,np.append(counts,counts[0]),linewidth=4)
        axBot[0+i*2].plot([0,ang],[0,r],color='k',linewidth=4)
        axBot[0+i*2].scatter(ang,r,s=50,color='red')
        #axBot[0+i*2].set_title('r={0:0.1f},th={1:0.1f},R={2:0.2f},p={3:0.2f}'.format(r,np.rad2deg(ang),R,pval) )
        axBot[0+i*2].set_xticklabels(['$0^o$','','$90^o$','','$180^o$'])
        axBot[0+i*2].set_yticks([])
        axBot[0+i*2].text(0,-0.1,'r={0:0.1f},th={1:0.1f},R={2:0.2f},p={3:0.2f}'.format(r,np.rad2deg(ang),R,pval), transform=axBot[0+i*2].transAxes)

        counts2 = counts*timeStep
        colors = plt.cm.magma(counts2/counts2.max())
        axBot[1+i*2].bar(bins[:-1],counts2,width=ang2bin,color=colors,bottom=counts2.min())
        axBot[1+i*2].set_axis_off()
        cax=getColBar(axBot[1+i*2],counts2)
        cax.yaxis.set_label('sec')

    ax = plt.axes([0,0.23,1,0.1])
    ax.text(0,0,'HD', {'fontsize':20})
    ax.plot([0,0.45],[-0.05,-0.05],color='k',linewidth=4)
    ax.text(0.5,0,'HA',{'fontsize':20})
    ax.plot([0.5,0.95],[-0.05,-0.05],color='k',linewidth=4)
    ax.set_axis_off()
    ax.set_xlim([0,1])
    ax.set_ylim([-.1,1])

    ax = plt.axes([0,0.7,1,0.05])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.text(0,0,se,{'fontsize':30},transform=ax.transAxes)

    ax.set_axis_off()
    return f

def getColBar(ax,values,cmap = 'magma'):
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width*0.85,pos.y0,0.05*pos.width,0.2*pos.height])

    cMap=mpl.colors.ListedColormap(sns.color_palette(cmap,50))
    vmax = values.max()
    vmin = values.min()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cMap)

    mapper.set_array([])
    cbar = plt.colorbar(mapper,ticks=[vmin,vmax],cax=cax)
    cax.yaxis.set_tick_params(right=False)
    return cax

def ReSampleDat(t,sig,step):
    '''Nearest neighbor interpolation for resampling.
    Inputs:
        t    ->  original time series
        sig  ->  signal to be resample
        step ->  time step for resampleing (e.g step = 0.02 -> 20ms)

    Output:
        tp   <-  new time series created with step
        out  <-  resampled signal
    e.g.
        tp,xs = ReSampleDat(t,x,0.02)
    '''

    tp = np.arange(t[0],t[-1],step)
    sig_ip = interp1d(t, sig, kind="nearest",fill_value="extrapolate")
    out = sig_ip(tp)
    return tp, out
