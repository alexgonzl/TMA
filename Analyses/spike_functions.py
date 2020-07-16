import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats, spatial
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA

from pathlib import Path
import os,sys, json
import h5py
import pickle as pkl

sys.path.append('../PreProcessing/')
sys.path.append('../TrackingAnalyses/')
sys.path.append('../Lib/')
from filters_ag import *
from robust_stats import *

import pre_process_neuralynx as PPN
import TreeMazeFunctions as TMF


def getSessionSpikes(sessionPaths, overwrite=0,rej_thr=None):

    if (not sessionPaths['Cell_Spikes'].exists()) | overwrite:
        print('Spikes Files not Found or overwrite=1, creating them.')

        animal = sessionPaths['animal']
        task = sessionPaths['task']
        date = sessionPaths['date']

        with sessionPaths['ClusterTable'].open() as f:
            CT = json.load(f)
        sessionCellIDs = CT[animal][date][task]['cell_IDs']
        sessionMuaIDs = CT[animal][date][task]['mua_IDs']

        cell_spikes, cell_wf, cell_wfi = get_TT_spikes(sessionCellIDs,sessionPaths,rej_thr=rej_thr)
        mua_spikes, mua_wf, mua_wfi = get_TT_spikes(sessionMuaIDs,sessionPaths,rej_thr=rej_thr)

        with sessionPaths['Cell_Spikes'].open(mode='w') as f:
            json.dump(cell_spikes,f,indent=4)
        with sessionPaths['Cell_WaveForms'].open(mode='wb') as f:
            pkl.dump(cell_wf,f,pkl.HIGHEST_PROTOCOL)
        with sessionPaths['Cell_WaveFormInfo'].open(mode='wb') as f:
            pkl.dump(cell_wfi,f,pkl.HIGHEST_PROTOCOL)

        with sessionPaths['Mua_Spikes'].open(mode='w') as f:
            json.dump(mua_spikes,f,indent=4)
        with sessionPaths['Mua_WaveForms'].open(mode='wb') as f:
            pkl.dump(mua_wf,f,pkl.HIGHEST_PROTOCOL)
        with sessionPaths['Mua_WaveFormInfo'].open(mode='wb') as f:
            pkl.dump(mua_wfi,f,pkl.HIGHEST_PROTOCOL)

    else:
        with sessionPaths['Cell_Spikes'].open() as f:
            cell_spikes=json.load(f)
        with sessionPaths['Mua_Spikes'].open() as f:
            mua_spikes = json.load(f)
        print('Loaded Spike Files.')

    return cell_spikes, mua_spikes

def get_TT_spikes(IDs,sessionPaths,rej_thr=None):
    spikeBuffer = 3 # in seconds
    cluster_path = Path(sessionPaths['Clusters'])
    SR = sessionPaths['SR']
    nUnits=0
    for tt_id,cl_id in IDs.items():
        nUnits+=len(cl_id)
        pass
    spikes = {}
    spikes['nUnits'] = nUnits
    wf = {}
    wfi = {}
    cnt = 0
    for tt,cl_ids in IDs.items():

        if len(cl_ids)>0:
            ttPath = Path(cluster_path,'tt_'+tt)
            dat = np.fromfile(str(ttPath / ('tt_{}.bin'.format(tt))),dtype=np.int16)
            dat = dat.reshape(-1,4)
            nSamps = len(dat)
            sp_times = np.load(str(ttPath/'spike_times.npy'))
            clusters = np.load(str(ttPath/'spike_clusters.npy'))

            spikes[str(tt)]={}
            for cl in cl_ids:
                allspikes = sp_times[clusters==cl].flatten()
                spikes2 = np.array(allspikes).astype(np.float)

                # delete spikes in at beggining and end of recording.
                ids = (spikes2+spikeBuffer*SR)<nSamps
                spikes2 = spikes2[ids]
                ids = (spikes2-spikeBuffer*SR)>0
                spikes2 = spikes2[ids]
                spikes2 = spikes2.astype(np.int)

                wf[cnt] = getSpikeWaveforms(spikes2,dat)
                if not (rej_thr is None):
                      badSpikes = getSpikeOutliers(wf[cnt],thr = rej_thr)
                      wf[cnt] = wf[cnt][~badSpikes,:]
                      spikes2 = spikes2[~badSpikes]

                wfi[cnt] = getWaveformInfo(spikes2,wf[cnt],nSamps-2*spikeBuffer*SR,SR)
                wfi[cnt]['tt'] = tt
                wfi[cnt]['cl'] = cl
                if not (rej_thr is None):
                    wfi[cnt]['nExcSpikes'] = np.sum(badSpikes)

                spikes[str(tt)][str(cl)] = spikes2.tolist()
                cnt+=1
    return spikes , wf, wfi

def getWaveformInfo(spikes,waveforms,nSamps,SR):

    nSp = len(spikes)
    wfi = {}
    wfi['mean'] = np.nanmean(waveforms,0)
    wfi['std'] = np.nanstd(waveforms,0)
    wfi['sem'] = stats.sem(waveforms,0)
    wfi['nSp'] = nSp
    wfi['tstat'] = stats.ttest_1samp(waveforms,0,axis=0)[0]
    wfi['mFR'] = nSp/nSamps*SR

    # isi in ms
    isi = np.diff(spikes)/SR*1000
    wfi['isi_h'] = np.histogram(isi,bins=np.linspace(-1,20,25))
    wfi['cv'] = np.std(isi)/np.mean(isi)
    return wfi

def getSessionBinSpikes(sessionPaths, overwrite=0, resamp_t=None,cell_spikes=None,mua_spikes=None):
    #print(sessionPaths['Cell_Bin_Spikes'])
    if (not sessionPaths['Cell_Bin_Spikes'].exists()) or (overwrite):

        print('Binned Spikes Files not Found or overwrite=1, creating them.')
        if ((cell_spikes is None) or (mua_spikes is None)):
            cell_spikes, mua_spikes = getSessionSpikes(sessionPaths, overwrite=overwrite)
        if  (resamp_t is None):
            print('Missing resampled time input (resamp_t).')
            PosDat = TMF.getBehTrackData(sessionPaths, overwrite=0)
            resamp_t = PosDat['t']
            del PosDat

        cell_bin_spikes,cell_ids = bin_TT_spikes(cell_spikes,resamp_t,origSR=sessionPaths['SR'])
        mua_bin_spikes,mua_ids = bin_TT_spikes(mua_spikes,resamp_t,origSR=sessionPaths['SR'])

        ids = {}
        ids['cells'] = cell_ids
        ids['muas'] = mua_ids

        np.save(sessionPaths['Cell_Bin_Spikes'],cell_bin_spikes)
        np.save(sessionPaths['Mua_Bin_Spikes'],mua_bin_spikes)

        with sessionPaths['Spike_IDs'].open(mode='w') as f:
            json.dump(ids,f,indent=4)
        print('Bin Spike File Creation and Saving Completed.')

    else:
        print('Loading Binned Spikes...')
        cell_bin_spikes=np.load(sessionPaths['Cell_Bin_Spikes'])
        mua_bin_spikes=np.load(sessionPaths['Mua_Bin_Spikes'])
        with sessionPaths['Spike_IDs'].open() as f:
            ids = json.load(f)
        print('Binned Spike Files Loaded.')

    return cell_bin_spikes, mua_bin_spikes, ids

def getSessionFR(sessionPaths,overwrite=0,cell_bin_spikes=None,mua_bin_spikes=None):
    if (not sessionPaths['Cell_FR'].exists()) | overwrite:
        print('Firing Rate Files Not Found or overwrite=1, creating them.')

        if ((cell_bin_spikes is None) or (mua_bin_spikes is None)):
            cell_bin_spikes, mua_bin_spikes, ids = getSessionBinSpikes(sessionPaths)

        nCells,nTimePoints = cell_bin_spikes.shape
        cell_FR = np.zeros((nCells,nTimePoints))
        for cell in np.arange(nCells):
            cell_FR[cell] = smoothSpikesTrain(cell_bin_spikes[cell])

        nΜUAs,nTimePoints = mua_bin_spikes.shape
        mua_FR = np.zeros((nΜUAs,nTimePoints))
        for cell in np.arange(nΜUAs):
            mua_FR[cell] = smoothSpikesTrain(mua_bin_spikes[cell])

        np.save(sessionPaths['Cell_FR'],cell_FR)
        np.save(sessionPaths['Mua_FR'],mua_FR)

        print('Spike File Creation and Saving Completed.')
    else:
        print('Loading FRs ...')
        cell_FR=np.load(sessionPaths['Cell_FR'])
        mua_FR=np.load(sessionPaths['Mua_FR'])

        print('FR Loaded.')
    return cell_FR, mua_FR

def getSpikeSamps(sptimes):
    a = np.zeros((len(sptimes),64),dtype=np.int)
    cnt = 0
    for s in sptimes:
        a[cnt] = s+np.arange(64)-32
        cnt+=1
    return a

def getSpikeWaveforms(spikes,raw):
    sp_stamps = getSpikeSamps(spikes)
    return raw[sp_stamps,:]

def getSpikeOutliers(waveforms,thr=None):
    # waveforms = nSpikes x 64 x 4 np.array
    nF = 64*4
    nSp = waveforms.shape[0]
    X = np.reshape(waveforms,(nSp,nF))

    Xm = np.mean(X,0)
    Y = np.zeros(nSp)
    for s in np.arange(nSp):
        Y[s] = spatial.distance.braycurtis(Xm,X[s])
    badSpikes = Y>thr
    # pca = PCA(n_components=2)
    # pca.fit(X)
    # lls = pca.score_samples(X)
    # badSpikes = np.abs(robust_zscore(lls))>thr
    return badSpikes

def bin_TT_spikes(spikes,resamp_t,origSR=32000):
    orig_time = np.arange(resamp_t[0],resamp_t[-1],1/origSR)
    step = resamp_t[1]-resamp_t[0]
    nOrigTimePoints = len(orig_time)
    nTimePoints = len(resamp_t)
    sp_bins = np.zeros((spikes['nUnits'],nTimePoints))
    sp_ids = {}
    cnt = 0
    for tt,cl_ids in spikes.items():
        if tt!='nUnits':
            for cl in cl_ids:
                try:
                    sp = np.array(spikes[tt][cl])
                    out_of_record_spikes = sp>=nOrigTimePoints
                    if np.any(out_of_record_spikes):
                        sp = np.delete(sp,np.where(out_of_record_spikes)[0])
                    sp_ids[cnt] = (tt,cl)
                    #print(type(sp[0]))
                    sp_bins[cnt],_ = np.histogram(orig_time[sp],np.concatenate([resamp_t,[resamp_t[-1]+step]]))
                except:
                    print("Error processing Tetrode {}, Cluster {}".format(tt,cl))
                    pass
                cnt+=1

    return sp_bins,sp_ids

def smoothSpikesTrain(bin_spikes,step=0.02):
    lfwin = np.round(1.0/step).astype(int)
    return signal.filtfilt(np.ones(lfwin)/lfwin,1,bin_spikes/step)

def getSpikeList(spikes,ids,cell_num):
    t,c = ids[str(cell_num)]
    return spikes[str(t)][str(c)]
