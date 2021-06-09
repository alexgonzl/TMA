import numpy as np
import pandas as pd
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnchoredText

import seaborn as sns
font = {'family' : 'sans-serif',
        'size'   : 20}

plt.rc('font', **font)
plt.rc('text',usetex=False)

from pathlib import Path
import os,sys
import h5py
import sys
import pickle as pkl
import time

import nept
sys.path.append('../PreProcessing/')
sys.path.append('../TrackingAnalyses/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')
from filters_ag import *

from importlib import reload  # Python 3.4+ only.
import pre_process_neuralynx as PPN
import TreeMazeFunctions as TMF
import spike_functions as SF
import spatial_tuning as ST
import stats_functions as StatsF
import plot_functions as PF

oakPaths = {}
oakPaths['Root'] = Path('/mnt/o/giocomo/alexg/')
oakPaths['Clustered'] = Path('/mnt/o/giocomo/alexg/Clustered/')
oakPaths['PreProcessed'] = Path('/mnt/o/giocomo/alexg/PreProcessed/')
oakPaths['Raw'] = Path('/mnt/o/giocomo/alexg/RawData/InVivo/')
oakPaths['Analyses'] = Path('/mnt/o/giocomo/alexg/Analyses')

def createZoneAnalysesDict(ids):
    ZoneRes = {}
    ZoneRes['Zones'] = TMF.ZonesNames
    ZoneRes['Partitions'] = ['All','H1','H2','CL','CR','CO','In','Out']
    ZoneRes['InfoFields'] = ['counts','time','prob']

    partitions = ['H1','H2']
    pairs=list(combinations(partitions,2))
    partitions = ['CL','CR','CO']
    pairs+=list(combinations(partitions,2))
    partitions = ['In','Out']
    pairs+=list(combinations(partitions,2))

    pairs_str = []
    for p in pairs:
        pairs_str.append(p[0]+'-'+p[1])

    ZoneRes['ZonePairs'] = pairs
    ZoneRes['ZonePairs_Str'] = pairs_str

    eDF = pd.DataFrame(np.full((len(ZoneRes['InfoFields']),TMF.nZones),np.nan),columns=TMF.ZonesNames,index=ZoneRes['InfoFields'])
    ZoneRes['ZoneInfo'] = {}
    for p in ZoneRes['Partitions']:
        ZoneRes['ZoneInfo'][p] = eDF.copy()

    nMua = len(ids['muas'])
    nCells = len(ids['cells'])
    ZoneRes['unitIDs'] = ids
    ZoneRes['nUnits'] = {'Cells':nCells,'Mua':nMua}

    nParts = len(ZoneRes['Partitions'])
    eDF1 = pd.DataFrame(np.full((nParts,TMF.nZones),np.nan),columns=TMF.ZonesNames,index=ZoneRes['Partitions'])
    eDF2 = pd.DataFrame(np.full((nCells,nParts),np.nan),columns=ZoneRes['Partitions'])
    eDF3 = pd.DataFrame(np.full((nMua,nParts),np.nan),columns=ZoneRes['Partitions'])

    ZoneRes['FR_Zone'] = {'Cells':{},'Mua':{}}
    ZoneRes['SI_Zone'] = {'Cells':eDF2.copy(),'Mua':eDF3.copy()}
    ZoneRes['SIp_Zone'] = {'Cells':eDF2.copy(),'Mua':eDF3.copy()}
    for c in np.arange(nCells):
        ZoneRes['FR_Zone']['Cells'][c] = eDF1.copy()

    for c in np.arange(nMua):
        ZoneRes['FR_Zone']['Mua'][c] = eDF1.copy()

    nPartPairs = len(pairs_str)
    ZoneRes['FR_CorrZonePairs']={}
    ZoneRes['FR_CorrZonePairs']['Cells']= pd.DataFrame(np.full((nCells,nPartPairs),np.nan),columns=pairs_str)
    ZoneRes['FR_CorrZonePairs']['Mua']= pd.DataFrame(np.full((nMua,nPartPairs),np.nan),columns=pairs_str)

    ZoneRes['ZoneStability']={}
    ZoneRes['ZoneStability']['Cells'] = pd.DataFrame(np.full((nCells,2),np.nan),columns=['HalfCorr','HalfnRMSE'])
    ZoneRes['ZoneStability']['Mua'] = pd.DataFrame(np.full((nMua,2),np.nan),columns=['HalfCorr','HalfnRMSE'])

    ZoneRes['FR_CorrZone'] = {'Cells':{},'Mua':{}}
    for c in np.arange(nCells):
        ZoneRes['FR_CorrZone']['Cells'][c]=[]
    for c in np.arange(nMua):
        ZoneRes['FR_CorrZone']['Mua'][c]=[]

    return ZoneRes

def getDatPartitions(PosDat):
    PartitionNames = ['All','H1','H2','CL','CR','CO','In','Out']
    nT = len(PosDat['t'])
    nP = len(PartitionNames)
    nT2 = np.floor(nT/2).astype(int)

    datPart = pd.DataFrame(np.zeros((nT,nP),dtype=bool), columns=PartitionNames)
    datPart['All'] = True

    datPart['CL'] = PosDat['EventDat']['CL'].astype(bool)
    datPart['CR'] = PosDat['EventDat']['CR'].astype(bool)
    datPart['CO'] = ~np.logical_or(datPart['CL'],datPart['CR'])

    datPart['H1'][0:nT2]=True
    datPart['H2'][nT2+1:]=True

    datPart['In'] = PosDat['InSeg']
    datPart['Out'] = PosDat['OutSeg']

    return datPart

def zone_analyses(sessionPaths,overwriteAll=0,overwriteSpikes=0,overwritePos=0,doPlots=0):

    print()
    print('Starting Analyses for Sesssion {}'.format(sessionPaths['session']))

    spacing=25
    occ_time_thr = 0.1 #s
    sigma = 1
    w = 4 # number of bins

    if overwriteAll | overwritePos | overwriteSpikes:
        overwrite=1
    else:
        overwrite=0

    if (not sessionPaths['ZoneAnalyses'].exists()) | overwrite:
        print('Starting Zone Analyses.')
        # get position info
        PosDat = TMF.getBehTrackData(sessionPaths, overwrite=overwritePos)
        OccInfo = ST.getTM_OccupationInfo(PosDat,spacing=spacing,occ_time_thr=occ_time_thr)
        # get spikes and FR for cells and mua
        cell_bin_spikes, mua_bin_spikes, ids= SF.getSessionBinSpikes(sessionPaths,resamp_t=PosDat['t'],overwrite=overwriteSpikes)
        cell_FR, mua_FR = SF.getSessionFR(sessionPaths,overwrite=overwriteSpikes, cell_bin_spikes=cell_bin_spikes,mua_bin_spikes=mua_bin_spikes)
        datPart = getDatPartitions(PosDat)

        ZoneRes = createZoneAnalysesDict(ids)
        zo={}
        for p in ZoneRes['Partitions']:
            zo[p] = PosDat['PosZones'][datPart[p]]
            ZoneRes['ZoneInfo'][p] = ST.getSeqInfo(zo[p])

        unitTypes=['Cells','Mua']
        nMua = len(ids['muas'])
        nCells = len(ids['cells'])
        for ut in unitTypes:
            for c in np.arange(ZoneRes['nUnits'][ut]):
                if ut=='Cells':
                    fr = cell_FR[c]
                else:
                    fr = mua_FR[c]

                for p in ZoneRes['Partitions']:
                    ZoneRes['FR_Zone'][ut][c].loc[p] = np.bincount(zo[p],fr[datPart[p]],TMF.nZones)/ZoneRes['ZoneInfo'][p].loc['counts']
                    si,_,pval = StatsF.SIPermTest(ZoneRes['ZoneInfo'][p].loc['prob'], ZoneRes['FR_Zone'][ut][c].loc[p])
                    ZoneRes['SI_Zone'][ut][p][c] = si
                    ZoneRes['SIp_Zone'][ut][p][c] = pval

                d = ZoneRes['FR_Zone'][ut][c].T.corr(method='kendall')
                ZoneRes['FR_CorrZone'][ut][c] = d

                cnt=0
                for p in ZoneRes['ZonePairs']:
                    ps = ZoneRes['ZonePairs_Str'][cnt]
                    ZoneRes['FR_CorrZonePairs'][ut].loc[c][ps]=d.loc[p[0]][p[1]]
                    cnt+=1
                ZoneRes['ZoneStability'][ut].loc[c]['HalfCorr'] = ZoneRes['FR_CorrZonePairs'][ut].loc[c]['H1-H2']
                h1 = ZoneRes['FR_Zone'][ut][0].loc['H1']
                h2 = ZoneRes['FR_Zone'][ut][0].loc['H2']
                ZoneRes['ZoneStability'][ut].loc[c]['HalfnRMSE'] = StatsF.RMSE(h1,h2)/np.nanmean(fr)

                ZoneRes['OccInfo'] = OccInfo

        with sessionPaths['ZoneAnalyses'].open(mode='wb') as f:
            pkl.dump(ZoneRes,f,pkl.HIGHEST_PROTOCOL)
        print('Zone Analyses Completed')
    else:
        with sessionPaths['ZoneAnalyses'].open(mode='rb') as f:
            ZoneRes = pkl.load(f)
        print('Zone Results Loaded')
        if doPlots:
            # get position info
            PosDat = TMF.getBehTrackData(sessionPaths, overwrite=0)
            OccInfo = ST.getTM_OccupationInfo(PosDat,spacing=spacing,occ_time_thr=occ_time_thr)
            # get spikes and FR for cells and mua
            cell_bin_spikes, mua_bin_spikes, ids= SF.getSessionBinSpikes(sessionPaths,resamp_t=PosDat['t'],overwrite=0)
            cell_FR, mua_FR = SF.getSessionFR(sessionPaths,overwrite=0, cell_bin_spikes=cell_bin_spikes,mua_bin_spikes=mua_bin_spikes)

    if doPlots:
        print("")
        plotTMInfo(sessionPaths, PosDat, OccInfo)
        plotFRMaps(sessionPaths,ZoneRes,PosDat,OccInfo,cell_bin_spikes,cell_FR,mua_bin_spikes,mua_FR)
        #     print('Plotting Routines Completed')
        #     #plotTMWindow(sessionPaths,PosDat,cell_bin_spikes,mua_bin_spikes)
        #     plotFRMaps(sessionPaths,ZoneRes,PosDat,OccInfo,cell_bin_spikes,cell_FR,mua_bin_spikes,mua_FR)
        # try:
        #     print('Starting Plotting Routines')
        #     plotTMInfo(sessionPaths, PosDat, OccInfo)
        #     #plotTMWindow(sessionPaths,PosDat,cell_bin_spikes,mua_bin_spikes)
        #     plotFRMaps(sessionPaths,ZoneRes,PosDat,OccInfo,cell_bin_spikes,cell_FR,mua_bin_spikes,mua_FR)
        #     print('Plotting Routines Completed')
        # except:
        #     print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)


    return ZoneRes

def plotTMInfo(sessionPaths, PosDat, OccInfo):
    ##### Plot OccupancyMap ####
    f,(a1,a2,a3)=plt.subplots(1,3, figsize=(12,6),gridspec_kw= dict(width_ratios=(4,4,0.2), height_ratios=[1], wspace=0.02))
    a1 = PF.plotTM_Trace(a1,PosDat['x'],PosDat['y'],plot_raw_traces=1)
    a1.axis('equal')
    a2 = PF.plotHeatMap(a2, a3,OccInfo['time'],colbar_label='Occopancy [s]',robust=True,smooth=True)
    f.savefig(sessionPaths['Plots']/'OccupancyMap.png',dpi=500, bbox_inches='tight',pad_inches=0.2)
    plt.close(f)
    ##### Plot TMF Zone Counts ######
    f=PF.plotMaze_XY(PosDat['x'],PosDat['y'])
    f.savefig(sessionPaths['Plots']/'TreeMazeCoverage.png',dpi=500)
    plt.close(f)
    ##### Plot TMF Zone Counts ######
    f = PF.plotMazeZoneCounts(PosDat['PosMat'])
    f.savefig(sessionPaths['Plots']/'LocSampleCounts.pdf',dpi=500,bbox_inches='tight',pad_inches=0.2)
    plt.close(f)
    ### Plot Event Counts ######
    f = PF.plotEventCounts(PosDat['EventDat'])
    f.savefig(sessionPaths['Plots']/'EventSubsetCounts.pdf',dpi=500, bbox_inches='tight',pad_inches=0.2)
    plt.close(f)

def plotTMWindow(sessionPaths,PosDat,cell_bin_spikes,mua_bin_spikes):
    time_vec = PosDat['t']

    minMark = np.round(0.81*(time_vec[-1]-time_vec[0]))+time_vec[0]
    window=np.where(np.logical_and(time_vec>=minMark,time_vec<(minMark+120)))[0]
    minMark_str=str(int(np.round((minMark-time_vec[0])/60)))
    ###### Plot TMF Zone Window ######
    f,(a1)=plt.subplots(1,1, figsize=(12,8))
    a1=PF.plotBehWindow(time_vec[window],PosDat['PosMat'].iloc[window].T,TMF.ZonesNames,a1)
    a1.set_xlabel('Time[s]')
    f.savefig(sessionPaths['Plots']/('LocWindow_min{}.png').format(minMark_str),dpi=500, bbox_inches='tight',pad_inches=0.2)
    plt.close(f)
    ###### Plot TMF Segment Directions Window ######
    f,(a1)=plt.subplots(1,1, figsize=(12,8))
    a1=PF.plotBehWindow(time_vec[window],PosDat['SegDirMat'].iloc[window].T,TMF.SegDirNames,a1)
    a1.set_xlabel('Time[s]')
    f.savefig(sessionPaths['Plots']/('LocDirWindow_min{}.png').format(minMark_str),dpi=500, bbox_inches='tight',pad_inches=0.2)
    plt.close(f)
    ###### Plot TMF Events Window ######
    f,(a1)=plt.subplots(1,1, figsize=(12,8))
    ev_subset=['RH','RC','R1','R2','R3','R4','DH','DC','D1','D2','D3','D4',
                 'LH','LC','CL','CR','cTr','iTr','LDs','RDs']
    mat = PosDat['EventDat'][ev_subset].iloc[window].T
    a1 = PF.plotBehWindow(time_vec[window],mat,ev_subset,a1)
    a1.set_xlabel('Time[s]')
    f.savefig(sessionPaths['Plots']/('EventWindow_min{}.png').format(minMark_str),dpi=500, bbox_inches='tight',pad_inches=0.2)
    plt.close(f)
    ##### Behav Window with Cells #####
    f,a1,a2= PF.plotBehavZonesWindowAndSpikes(time_vec[window],PosDat['PosMat'].iloc[window].T,cell_bin_spikes[:,window])
    f.savefig(sessionPaths['Plots']/('Cells_LocWindow_min{}.png').format(minMark_str),dpi=500, bbox_inches='tight',pad_inches=0.2)
    plt.close(f)
    ##### Behav Window with MuaCells #####
    f,a1,a2=PF.plotBehavZonesWindowAndSpikes(time_vec[window],PosDat['PosMat'].iloc[window].T,mua_bin_spikes[:,window])
    f.savefig(sessionPaths['Plots']/('Mua_LocWindow_min{}.png').format(minMark_str),dpi=500, bbox_inches='tight',pad_inches=0.2)
    plt.close(f)

def plotFRMaps(sessionPaths,ZoneRes,PosDat,OccInfo,cell_bin_spikes,cell_FR,mua_bin_spikes,mua_FR):
    ##### Plot Firing Rate Maps ######
    sns.set()
    xPos = [[0.02,0.43],[0.02,0.43]]
    yPos = [[0.50,0.50],[0.05,0.05]]
    W = 0.38
    H = 0.4
    unitTypes=['Cells','Mua']

    for ut in unitTypes:
        for cell_id in np.arange(ZoneRes['nUnits'][ut]):
            if ut=='Cells':
                spikes = cell_bin_spikes[cell_id]
                sFR = cell_FR[cell_id]
            else:
                spikes = mua_bin_spikes[cell_id]
                sFR = mua_FR[cell_id]
            mFR = np.mean(sFR)
            ZoneFR = ZoneRes['FR_Zone'][ut][cell_id].loc['All']

            H1FR = ZoneRes['FR_Zone'][ut][cell_id].loc['H1']
            H2FR = ZoneRes['FR_Zone'][ut][cell_id].loc['H2']
            H1H2FR = np.log10(H1FR/H2FR)
            H1H2FR[H1H2FR==np.inf]=0.99
            H1H2FR[H1H2FR==-np.inf]=-0.99

            spikesByPos = ST.getPosBinSpikeMaps(spikes,PosDat)
            FR_ByPos = ST.getPosBinFRMaps(spikesByPos,OccInfo['time'])

            f,ax=plt.subplots(2,2, figsize=(12,10))
            ax[0][0]= PF.plotTM_Trace(ax[0][0],PosDat['x'],PosDat['y'],bin_spikes = spikes)
            ax[0][0].axis('equal')
            ax[0][0].set_position([xPos[0][0],yPos[0][0],W,H])

            FR_ByPos[np.isnan(FR_ByPos)]=0
            cax1=plt.axes([0.83,0.60,0.03,0.2])
            ax[0][1]= PF.plotHeatMap(ax[0][1],cax1,FR_ByPos,smooth=True)
            ax[0][1].set_position([xPos[0][1],yPos[0][1],W,H])

            cax2=plt.axes([0.32,0.08,0.03,0.15])
            ax[1][0],cax2=PF.plotZonesHeatMap(ax[1][0],cax2,ZoneFR-mFR,alpha=1)
            cax2.set_ylabel(' FR-mFR [sp/s]')
            ax[1][0].set_position([xPos[1][0],yPos[1][0],W,H])

            cax3=plt.axes([0.73,0.08,0.03,0.15])
            ax[1][1],cax3=PF.plotZonesHeatMap(ax[1][1],cax3,H1H2FR,alpha=1,colArray=np.linspace(-1,1,25))
            cax3.set_ylabel(' Stability log(H1/H2)')
            ax[1][1].set_position([xPos[1][1],yPos[1][1],W,H])
            r=ZoneRes['ZoneStability'][ut].loc[cell_id]['HalfCorr']
            e=ZoneRes['ZoneStability'][ut].loc[cell_id]['HalfnRMSE']

            ax[1][1].add_artist(AnchoredText('r(H1-H2)={0:.3f}\nE={1:.3f}'.format(r,e),loc='lower left',frameon=False))

            f.savefig(sessionPaths['ZoneFRPlots']/('FR_Maps_{}ID-{}.png'.format(ut,cell_id)),dpi=500, bbox_inches='tight',pad_inches=0.2)
            plt.close(f)

    ##### Plot Firing Rate Maps by CUE ######
    for ut in unitTypes:
        for cell_id in np.arange(ZoneRes['nUnits'][ut]):
            if ut=='Cells':
                sFR = cell_FR[cell_id]
            else:
                sFR = mua_FR[cell_id]
            mFR = np.mean(sFR)

            COZoneFR = ZoneRes['FR_Zone'][ut][cell_id].loc['CO']-mFR
            CLZoneFR = ZoneRes['FR_Zone'][ut][cell_id].loc['CL']-mFR
            CRZoneFR = ZoneRes['FR_Zone'][ut][cell_id].loc['CR']-mFR
            CLCRZoneFR = CLZoneFR-CRZoneFR

            f,ax=plt.subplots(2,2, figsize=(12,10))

            cax0=plt.axes([0.32,0.53,0.03,0.15])
            ax[0][0],cax0=PF.plotZonesHeatMap(ax[0][0],cax0,CLZoneFR,alpha=1)
            ax[0][0].axis('equal')
            ax[0][0].set_position([xPos[0][0],yPos[0][0],W,H])
            cax0.set_ylabel('CL FR [sp/s]')

            cax1=plt.axes([0.73,0.53,0.03,0.15])
            ax[0][1],cax1=PF.plotZonesHeatMap(ax[0][1],cax1,CRZoneFR,alpha=1)
            ax[0][1].set_position([xPos[0][1],yPos[0][1],W,H])
            cax1.set_ylabel('CR FR [sp/s]')

            cax2=plt.axes([0.32,0.08,0.03,0.15])
            ax[1][0],cax2=PF.plotZonesHeatMap(ax[1][0],cax2,COZoneFR,alpha=1)
            cax2.set_ylabel(' CO [sp/s]')
            ax[1][0].set_position([xPos[1][0],yPos[1][0],W,H])

            cax3=plt.axes([0.73,0.08,0.03,0.15])
            ax[1][1],cax3=PF.plotZonesHeatMap(ax[1][1],cax3,CLCRZoneFR,alpha=1)
            cax3.set_ylabel('CL-CR [sp/s]')
            ax[1][1].set_position([xPos[1][1],yPos[1][1],W,H])
            r=ZoneRes['FR_CorrZonePairs'][ut].loc[cell_id]['CL-CR']
            ax[1][1].add_artist(AnchoredText('r(CL-CR)={0:.3f}'.format(r),loc='lower left',frameon=False))

            f.savefig(sessionPaths['ZoneFRPlots']/('FR_Cue_Maps_{}ID-{}.png'.format(ut,cell_id)),dpi=500, bbox_inches='tight',pad_inches=0.2)
            plt.close(f)

    ##### Plot Firing Rate Maps by In/Out ######
    xPos = [0.02,0.34,0.66]
    yPos = 0.05
    W = 0.3
    H = 0.9
    Segs = ['SegA','SegB','SegC','SegD','SegE','SegF','SegG']

    for ut in unitTypes:
        for cell_id in np.arange(ZoneRes['nUnits'][ut]):
            if ut=='Cells':
                sFR = cell_FR[cell_id]
            else:
                sFR = mua_FR[cell_id]

            mFR = np.mean(sFR)
            OutZoneFR = ZoneRes['FR_Zone'][ut][cell_id].loc['Out']-mFR
            InZoneFR = ZoneRes['FR_Zone'][ut][cell_id].loc['In']-mFR
            OutInZoneFR = OutZoneFR-InZoneFR

            f,ax=plt.subplots(1,3, figsize=(18,5))

            ax[0].set_position([xPos[0],yPos,W,H])
            cax=plt.axes([xPos[0]+W*0.75,yPos+H*0.1,W*0.06,H*0.3])
            PF.plotZonesHeatMap(ax[0],cax,OutZoneFR,zones=Segs,alpha=1)
            cax.set_ylabel('Out [sp/s]')

            ax[1].set_position([xPos[1],yPos,W,H])
            cax=plt.axes([xPos[1]+W*0.75,yPos+H*0.1,W*0.06,H*0.3])
            PF.plotZonesHeatMap(ax[1],cax,InZoneFR,zones=Segs,alpha=1)
            cax.set_ylabel('In [sp/s]')

            ax[2].set_position([xPos[2],yPos,W,H])
            cax=plt.axes([xPos[2]+W*0.75,yPos+H*0.1,W*0.06,H*0.3])
            PF.plotZonesHeatMap(ax[2],cax,OutInZoneFR,zones=Segs,alpha=1)
            cax.set_ylabel('Out - In [sp/s]')

            f.savefig(sessionPaths['ZoneFRPlots']/('FR_OutIn_Maps_{}ID-{}.png'.format(ut,cell_id)),dpi=500, bbox_inches='tight',pad_inches=0.2)
            plt.close(f)

    #### Plot clustered correlation matrix ####
    for ut in unitTypes:
        for cell_id in np.arange(ZoneRes['nUnits'][ut]):
            try:
                d=ZoneRes['FR_CorrZone'][ut][cell_id]

                g=sns.clustermap(d,center=0, cmap="coolwarm",linewidths=.75, figsize=(10, 10),vmin=-1,vmax=1, annot=True,cbar_kws={'label':'tau'})
                g.savefig(sessionPaths['ZoneFRPlots']/('ZoneFRCorrMat_{}ID-{}.png'.format(ut,cell_id)),dpi=500, bbox_inches='tight',pad_inches=0.2)
                plt.close(plt.gcf())
            except:
                print( "Error plotting correlation matrix")
                print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)

def plotSI():
    #f,(a1,a2,a3)=plt.subplots(1,3, figsize=(12,3))
    # si,si_shf,p = StF.SIPermTest(ZoneInfo['prob'],FR_ByZone)
    # a1=StF.plotPermDist(a1,si_shf,si,p)
    # a1.set_title('Zones')

    # FR_ByPos[OccInfo['prob']==0]=np.nan
    # si,si_shf,p = StF.SIPermTest(OccInfo['prob'],FR_ByPos)
    # a2=StF.plotPermDist(a2,si_shf,si,p)
    # a2.set_title('Pos Bin')

    # spikesByZoneDir, seqInfo = ST.getDirZoneSpikeMaps(spikes, PosDat)
    # FR_ByZoneDir = spikesByZoneDir/seqInfo['time']
    # si,si_shf,p = StF.SIPermTest(seqInfo['prob'],FR_ByZoneDir)
    # a3=StF.plotPermDist(a3,si_shf,si,p)
    # a3.set_title('Directional')
    # f.savefig(sessionPaths['Plots']/('SI_CellID-{}.png'.format(cell_id)),dpi=500, bbox_inches='tight',pad_inches=0.2)

    # f,(a1,a2,a3)=plt.subplots(1,3, figsize=(12,3))
    # si,si_shf,p = StF.SIPermTest(CL_ZInfo['prob'],FR_CL_Zone)
    # a1=StF.plotPermDist(a1,si_shf,si,p)
    # a1.set_title('Left Cue Zones')

    # si,si_shf,p = StF.SIPermTest(CR_ZInfo['prob'],FR_CR_Zone)
    # a2=StF.plotPermDist(a2,si_shf,si,p)
    # a2.set_title('Right Cue Zones')

    # si,si_shf,p = StF.SIPermTest(CO_ZInfo['prob'],FR_CO_Zone)
    # a3=StF.plotPermDist(a3,si_shf,si,p)
    # a3.set_title('Cue Off Zones')
    # f.savefig(sessionPaths['Plots']/('Cue_SI_CellID-{}.png'.format(cell_id)),dpi=500, bbox_inches='tight',pad_inches=0.2)
    return []
