import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import warnings

import seaborn as sns

from pathlib import Path
import os,sys
import h5py
import sys

sys.path.append('../PreProcessing/')
sys.path.append('../TrackingAnalyses/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')
import TreeMazeFunctions as TMF

font = {'family' : 'sans-serif',
        'size'   : 20}

plt.rc('font', **font)


def getSmoothMap(fr_map,w=4,s=1):
    fr_smoothed = ndimage.filters.median_filter(fr_map,w)
    fr_smoothed = smooth_hist2d(fr_smoothed,w,s)
    return fr_smoothed

def smooth_hist2d(img,w=5,sigma=1.5):
    trunc = (((w - 1)/2)-0.5)/sigma
    return ndimage.filters.gaussian_filter(img,sigma,mode='constant',truncate=trunc)

def getPositionMat(x,y,xlimits, ylimits, spacing=30):
    xedges=np.arange(xlimits[0],xlimits[1]+1,spacing)
    yedges=np.arange(ylimits[0],ylimits[1]+1,spacing)
    position_2d,_,_ = np.histogram2d(x,y, bins=[xedges, yedges])
    return position_2d, xedges, yedges

def binSpikesToXY(bin_spikes,x,y):
    maxBinSpikes = np.max(bin_spikes)
    xPosSpikes = []
    yPosSpikes = []
    for nsp in np.arange(1,maxBinSpikes+1):
        xPosSpikes+=x[bin_spikes==nsp].tolist()*int(nsp)
        yPosSpikes+=y[bin_spikes==nsp].tolist()*int(nsp)
    assert len(xPosSpikes)==np.sum(bin_spikes), 'Spikes To Position Mismatch'
    return xPosSpikes,yPosSpikes

def binSpikesToZone(bin_spikes,Zones):
    # returns a vector of where each spike occured.
    maxBinSpikes = np.max(bin_spikes)
    zoneSpikes = []
    for nsp in np.arange(1,maxBinSpikes+1):
        zoneSpikes+=Zones[bin_spikes==nsp].tolist()*int(nsp)
    return zoneSpikes

def getZoneSpikeMaps(sp,Zones):
    ZoneSp = binSpikesToZone(sp,Zones)
    ZoneSpCounts = np.bincount(ZoneSp)
    return ZoneSpCounts

def getPosBinSpikeMaps(spikes,PosDat,spacing=25):
    xSp,ySp = binSpikesToXY(spikes,PosDat['x'],PosDat['y'])
    spikes_pos,_,_=getPositionMat(xSp,ySp,TMF.x_limit,TMF.y_limit,spacing)
    return spikes_pos

def getPosBinFRMaps(spikes_pos,occ_time):
    occ_time2 = np.array(occ_time)
    occ_time2[occ_time==0]=np.nan
    fr_pos=spikes_pos/occ_time2
    fr_pos[np.isnan(fr_pos)]=0
    return fr_pos

def getValidMovingSamples(speed, sp_thr = [5,2000]):
    # subsample to get moving segments
    return np.logical_and(speed>=sp_thr[0],speed<=sp_thr[1])

def getDirZoneSpikeMaps(spikes, PosDat, sp_thr = [5,2000]):
    SegSeq = PosDat['SegDirSeq']

    # subsample to get moving segments
    valid_moving = getValidMovingSamples(PosDat['Speed'])
    valid_samps = np.logical_and(valid_moving,SegSeq>0)

    MovSegSeq=np.array(SegSeq)
    dir_seq = MovSegSeq[valid_samps]-1
    seqInfo = getSeqInfo(dir_seq,PosDat['step'])

    dir_spikes = spikes[valid_samps]
    spikesByZoneDir = getZoneSpikeMaps(dir_spikes,dir_seq)
    return spikesByZoneDir, seqInfo

def getSeqInfo(Seq,step=0.02):
    fields = ['counts','time','prob']
    seqInfo = pd.DataFrame(np.full((3,TMF.nZones),np.nan),columns=TMF.ZonesNames,index=fields)

    counts = np.bincount(Seq.astype(int),minlength=TMF.nZones)
    seqInfo.loc['counts']  = counts
    seqInfo.loc['time'] = counts*step
    seqInfo.loc['prob'] = counts/np.sum(counts)

    return seqInfo

def getTM_OccupationInfo(PosDat,spacing=25,occ_time_thr=0.1):
    occ_counts,xed,yed = getPositionMat(PosDat['x'],PosDat['y'],TMF.x_limit,TMF.y_limit,spacing)
    occ_time = occ_counts*PosDat['step']
    occ_mask = occ_time>=occ_time_thr

    OccInfo = {}
    OccInfo['time'] = occ_time*occ_mask
    OccInfo['counts'] = occ_counts*occ_mask
    OccInfo['prob'] = occ_counts/np.nansum(occ_counts)
    return OccInfo

def atan2v(dy,dx):
    N = len(dy)
    out = np.zeros(N)
    for i in np.arange(N):
        out[i] = np.math.atan2(dy[i],dx[i])
    return out

def getVelocity(x,y,step):
    dx = np.append(0,np.diff(x))
    dy = np.append(0,np.diff(y))

    dr = np.sqrt(dx**2+dy**2)

    sp = dr/step # convert distance to speed
    an = atan2v(dy,dx)
    return sp,an

def RotateXY(x,y,angle):
    x2 = x*np.cos(angle)+y*np.sin(angle)
    y2 = -x*np.sin(angle)+y*np.cos(angle)
    return x2,y2

def getLastNotNanVal(x,i):
    if i==0:
        return 0
    elif np.isnan(x[i]):
        return getLastNotNanVal(x,i-1)
    else:
        return x[i]

def resultant_vector_length(alpha, w=None, d=None, axis=None, axial_correction=1, ci=None, bootstrap_iter=None):
    # source: https://github.com/circstat/pycircstat/blob/master/pycircstat/descriptive.py
    """
    Computes mean resultant vector length for circular data.
    This statistic is sometimes also called vector strength.
    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length
    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain resultant vector length
    r = np.abs(cmean)
    # obtain mean
    mean = np.angle(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    # obtain variance
    variance = 1 - r
    std = np.sqrt(-2 * np.log(r))
    return r,mean,variance,std

def rayleigh(alpha, w=None, d=None, axis=None):
    """
    Computes Rayleigh test for non-uniformity of circular data.
    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle
    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!
    :param alpha: sample of angles in radian
    :param w:       number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return z:    value of the z-statistic
    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    # if axis is None:
    # axis = 0
    #     alpha = alpha.ravel()

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    r,mean,variance,std = resultant_vector_length(alpha, w=w, d=d, axis=axis)
    n = np.sum(w, axis=axis)

    # compute Rayleigh's R (equ. 27.1)
    R = n * r

    # compute Rayleigh's z (equ. 27.2)
    z = R ** 2 / n

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return pval, z

def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    # REQUIRED for mean vector length calculation
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
        str(w.shape) + " do not match!"


    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            cmean = ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) / np.sum(w, axis=axis))
        except Warning as e:
            print('Could not compute complex mean for MVL calculation', e)
            cmean = np.nan
    return cmean

def getAngleHisto(th,step=10,th_units='deg'):
    if th_units=='deg':
        th = np.deg2rad(th)
        step = np.deg2rad(step)

    counts,bins=np.histogram(th,np.arange(-np.pi,np.pi+0.01,step))
    binCenters=bins[:-1]+step
    return counts, binCenters, bins

def angle_stats(th,step=10):
    counts,binCenters,bins = getAngleHisto(th,step)
    z = np.mean(counts*np.exp(1j*binCenters))
    ang = np.angle(z)
    r = np.abs(z)
    p,t=rayleigh(th)

    stats = {'r':r,'ang':ang,'R':t,'pval':p, 'counts':counts, 'bins':bins,'binCenters': binCenters}

    return stats
