import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnchoredText
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

import seaborn as sns
font = {'family' : 'sans-serif',
        'size'   : 20}
plt.rc('font', **font)
plt.rc('text',usetex=False)

from pathlib import Path
import os,sys, time
import h5py, json
import pickle as pkl

sys.path.append('../PreProcessing/')
sys.path.append('../TrackingAnalyses/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')

import TreeMazeFunctions as TMF
import spike_functions as SF
import zone_analyses_session as ZA

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

oakPaths = {}
oakPaths['Root'] = Path('/mnt/o/giocomo/alexg/')
oakPaths['Clustered'] = oakPaths['Root']/'Clustered'
oakPaths['PreProcessed'] = oakPaths['Root']/'PreProcessed'
oakPaths['Raw'] =oakPaths['Root'] / 'RawData'/'InVivo'
oakPaths['Analyses'] =oakPaths['Root']/'Analyses'

################################################################################
ValidTraj = {'R_S1':['Home','SegA','Center','SegB','I1','SegC','G1'],
               'R_S2':['Home','SegA','Center','SegB','I1','SegD','G2'],
               'R_L1':['Home','SegA','Center','SegB','I1','SegD','G2','SegD','I1','SegC','G1'],
               'R_L2':['Home','SegA','Center','SegB','I1','SegC','G1','SegC','I1','SegD','G2'],
               'L_S3':['Home','SegA','Center','SegE','I2','SegF','G3'],
               'L_S4':['Home','SegA','Center','SegE','I2','SegG','G4'],
               'L_L3':['Home','SegA','Center','SegE','I2','SegG','G4','SegG','I2','SegF','G3'],
               'L_L4':['Home','SegA','Center','SegE','I2','SegF','G3','SegF','I2','SegG','G4'],
              }
ValidOffTraj = {}
for k,v in ValidTraj.items():
    ValidOffTraj[k] = v[::-1]

ValidTrajNames = list(ValidTraj.keys())
ValidOffTrajNames = list(ValidOffTraj.keys())

def main(sessionPaths, doPlots=False, overwrite = False, PosDat = None, cell_FR=None, mua_FR=None):
    # Load Data
    session = sessionPaths['session']
    print('\nAnalyzing Trials from session {}'.format(session))

    if ((cell_FR is None) or (cell_FR is None)):
        cell_FR, mua_FR = SF.getSessionFR(sessionPaths)
    if PosDat is None:
        PosDat = TMF.getBehTrackData(sessionPaths)

    t0 = time.time()
    # dictionary with all trial information
    fn = sessionPaths['Analyses'] / 'TrInfo.pkl'
    if (fn.exists() and (not overwrite)):
        print('Trial Info Dict exists. Loading.')
        with fn.open(mode="rb") as f:
            TrInfo = pkl.load(f)
    else:
        print('Trial Info Dict does not exists. Computing.')
        TrInfo = getTrInfo(PosDat)
        with fn.open(mode='wb') as f:
            pkl.dump(TrInfo,f,pkl.HIGHEST_PROTOCOL)
    t1 = time.time()
    print('Process Completed. Time {}s'.format(t1-t0))

    # matrix with several trial conditions of interest
    fn = sessionPaths['Analyses'] / 'TrialCondMat.csv'
    if (fn.exists() and (not overwrite)):
        print('Trial x Condition Matrix Exists. Loading.')
        TrCondMat = pd.read_csv(fn)
    else:
        print('Trial x Condition Matrix Does Not Exist. Computing.')
        TrCondMat = getTrCondMat(TrInfo,PosDat['step'])
        TrCondMat.to_csv(fn)
    t2 = time.time()
    print('Process Completed. Time {}s'.format(t2-t1))

    # long form matrix expanded by the animal positions in a trial
    fn = sessionPaths['Analyses'] / 'TrLongPosMat.csv'
    if (fn.exists() and (not overwrite)):
        print('Trial x Position Long Condition Matrix Exists. Loading.')
        TrLongMat = pd.read_csv(fn)
    else:
        print('Trial x Position Long Condition Matrix Does Not Exist. Computing.')
        TrLongMat = getTrLongMat(TrInfo,TrCondMat)
        TrLongMat.to_csv(fn)
    t3 = time.time()
    print('Process Completed. Time {}s'.format(t3-t2))

    # long form matrix of the firing rates by trial and position that matches
    # TrLongMat
    fn = sessionPaths['Analyses'] / 'TrLongPosFRDat.csv'
    if (fn.exists() and (not overwrite)):
        print('Trial x Position Long FiringRate Matrix Exists. Loading.')
        TrFRData = pd.read_csv(fn)
    else:
        print('Trial x Position Long FiringRate Matrix Does Not Exist. Computing.')
        TrFRData = getTrLongFRDat(TrInfo,cell_FR,mua_FR)
        TrFRData.to_csv(fn)
    t4 = time.time()
    print('Process Completed Time {}s'.format(t4-t3))

    # fit lm to the trial wise data to obtain what neurons are coding for.
    fn = sessionPaths['Analyses'] / 'TrModelFits2.csv'
    if (fn.exists() and (not overwrite)):
        print('Trial Model Fits Exists. Loading')
        TrModFits = pd.read_csv(fn)
    else:
        print('Trial Model Fits Does Not Exists. Computing')
        if ~np.any(TrFRData[TrLongMat['Valid']].isnull()):
            TrModFits = fitTrModels(TrLongMat,TrFRData)
            TrModFits.to_csv(fn)
        else:
            print('Long Form Matrix mismatch. Model Fits will not work')
            TrModFits = []
    t5 = time.time()
    print('Process Completed. Time {}s'.format(t5-t4))

    if doPlots:
        print('Plotting Trials for session {}s'.format(session))
        plotLinearTraj(sessionPaths['TrialPlots'],TrFRData,TrLongMat)
        t6 = time.time()
        print('Linearized FR Plots Completed. Time {}s'.format(t6-t5))

        plotTrialConds(sessionPaths['TrialPlots'],TrFRData,TrLongMat)
        t7 = time.time()
        print('Trial FR x Condition Plots Completed. Time {}s'.format(t7-t6))

    return TrInfo,TrCondMat,TrLongMat,TrFRData, TrModFits

def getTrInfo(PosDat):
    # get trial durations and samples
    TrialVec = PosDat['EventDat']['TrID']
    nTr=TrialVec.max()

    startIDs = np.zeros(nTr,dtype=int)
    endIDs = np.zeros(nTr,dtype=int)
    for tr in np.arange(nTr):
        trIDs = np.where(TrialVec==(tr+1))[0]
        startIDs[tr]=trIDs[0]
        endIDs[tr] = trIDs[-1]

    TrialDurs = endIDs-startIDs

    OffTrialDurs=np.concatenate((startIDs[1:],[len(PosDat['t'])]))-endIDs
    OffTrialVec = np.full_like(TrialVec,0)

    for tr in np.arange(nTr):
        idx = np.arange(endIDs[tr],endIDs[tr]+OffTrialDurs[tr])
        OffTrialVec[idx]=tr+1

    # Pre allocated Trial Info structure.
    TrInfo = {'All':{'Trials':[],'Co':[],'InCo':[]},'L':{'Trials':[],'Co':[],'InCo':[]},
                 'R':{'Trials':[],'Co':[],'InCo':[]},'BadTr':[],'Cues':np.full(nTr,''),'Desc':np.full(nTr,''),
                 'DurThr':45,'TrDurs':TrialDurs,
                 'TrialVec':TrialVec,'TrStSamp':startIDs,'TrEnSamp':endIDs,'TrSeq':{'Pos':{},'Samp':{},'Dur':{}, 'Sp':{}},
                 'OffTrStSamp':endIDs,'OffTrEnSamp':endIDs+OffTrialDurs,'OffTrDurs':OffTrialDurs,
                 'OffTrialVec':OffTrialVec, 'OffTrSeq':{'Pos':{},'Samp':{},'Dur':{}, 'Sp':{}},
                 'ValidSeqTrials':[],'ValidSeqOffTrials':[],'ValidSeqTrID':[],'ValidSeqOffTrID':[],
                 'ValidSeqNames':ValidTraj,'ValidSeqOffNames':ValidOffTraj}

    TrInfo['All']['Trials']=np.arange(nTr)+1
    #get separate trials and allocate by correct/incorrect
    for tr in TrInfo['All']['Trials']:
        idx= TrialVec==tr
        for s in ['L','R']:
            c = PosDat['EventDat']['C'+s][idx] #cues
            d = PosDat['EventDat'][s+'Ds'][idx] # decision
            if np.mean(d)>0.5: # descicion
                TrInfo['Desc'][tr-1]=s
            if np.mean(c)>0.5: # cue
                TrInfo[s]['Trials'].append(tr)
                TrInfo['Cues'][tr-1]=s
                if np.mean(d&c)>0.5: # correct descicion
                    TrInfo[s]['Co'].append(tr)
                else:
                    TrInfo[s]['InCo'].append(tr)
    assert set(TrInfo['R']['Trials']) & set(TrInfo['L']['Trials']) == set(), 'Trial classified as both left and right.'
    assert len(TrInfo['Cues']) ==len(TrInfo['Desc']), 'Number of trials mismatch'
    assert len(TrInfo['Cues']) ==nTr, 'Number of trials mismatch'

    for trC in ['Co', 'InCo']:
        TrInfo['All'][trC] = np.sort(TrInfo['L'][trC]+TrInfo['R'][trC])

    for i in ['All','L','R']:
        for j in ['Trials','Co','InCo']:
            TrInfo[i]['n'+j]=len(TrInfo[i][j])
            TrInfo[i][j]=np.array(TrInfo[i][j])

    # determine if the trials are too long to be included.
    TrInfo['BadTr'] = np.where(TrInfo['TrDurs']*PosDat['step']>TrInfo['DurThr'])[0]

    # get positions for each trial
    for tr in TrInfo['All']['Trials']:

        idx = TrInfo['TrialVec']==tr
        sID = TrInfo['TrStSamp'][tr-1]
        eID = TrInfo['TrEnSamp'][tr-1]

        p,s,d,sp =getPosSequence(PosDat['PosZones'][idx],PosDat['Speed'][idx],sID,eID)

        TrInfo['TrSeq']['Pos'][tr] = p
        TrInfo['TrSeq']['Samp'][tr] = s
        TrInfo['TrSeq']['Dur'][tr] = d
        TrInfo['TrSeq']['Sp'][tr] = sp

        idx = TrInfo['OffTrialVec']==tr
        sID = TrInfo['OffTrStSamp'][tr-1]
        eID = TrInfo['OffTrEnSamp'][tr-1]

        p,s,d,sp =getPosSequence(PosDat['PosZones'][idx],PosDat['Speed'][idx],sID,eID)

        TrInfo['OffTrSeq']['Pos'][tr] = p
        TrInfo['OffTrSeq']['Samp'][tr] = s
        TrInfo['OffTrSeq']['Dur'][tr] = d
        TrInfo['OffTrSeq']['Sp'][tr] = sp

    # determine if the sequence of positions are valid for each trial
    TrSeqs = {}
    vTr = []
    OffTrSeqs = {}
    vOTr = []
    for tr in TrInfo['All']['Trials']:
        seq = [TMF.Zones[a] for a in TrInfo['TrSeq']['Pos'][tr]]
        match = 0
        for vSeqN, vSeq in ValidTraj.items():
            if cmp(seq,vSeq)==0:
                match = 1
                vTr.append(tr)
                TrSeqs[tr]=vSeqN
                break
        if match==0:
            TrSeqs[tr]=[]

        seq =  [TMF.Zones[a] for a in TrInfo['OffTrSeq']['Pos'][tr]]
        match = 0
        for vSeqN, vSeq in ValidOffTraj.items():
            if cmp(seq,vSeq)==0:
                match = 1
                vOTr.append(tr)
                OffTrSeqs[tr]=vSeqN
                break
        if match==0:
            OffTrSeqs[tr]=[]

    TrInfo['ValidSeqTrials'] = vTr
    TrInfo['ValidSeqOffTrials'] = vOTr
    TrInfo['ValidSeqTrID'] = TrSeqs
    TrInfo['ValidSeqOffTrID'] = OffTrSeqs

    return TrInfo

def getTrCondMat(TrInfo,step=0.02):
    nTr=len(TrInfo['All']['Trials'])
    conds = ['Cues','Desc','Co','Traj','OTraj','Dur','Good','Length','OLength']
    TrCondMat = pd.DataFrame(np.full((nTr,len(conds)),np.nan),index=TrInfo['All']['Trials'],columns=conds)

    TrCondMat['Cues'] = TrInfo['Cues']
    TrCondMat['Desc'] = TrInfo['Desc']
    TrCondMat['Dur'] = TrInfo['TrDurs']

    TrCondMat['Co'].loc[TrInfo['All']['Co']]='Co'
    TrCondMat['Co'].loc[TrInfo['All']['InCo']]='InCo'

    vseq=TrInfo['ValidSeqTrials']
    TrCondMat['Traj'].loc[vseq]=[TrInfo['ValidSeqTrID'][s] for s in vseq]

    vseq=TrInfo['ValidSeqOffTrials']
    TrCondMat['OTraj'].loc[vseq]=[TrInfo['ValidSeqOffTrID'][s] for s in vseq]

    TrCondMat['Good'] = (~TrCondMat['Traj'].isnull()) & (TrCondMat['Dur']*step<TrInfo['DurThr'])

    x=np.full(nTr,'')
    for k,v in TrInfo['ValidSeqTrID'].items():
        if len(v)>0:
            x[k-1]=v[2]
    TrCondMat['Length']= x

    x=np.full(nTr,'')
    for k,v in TrInfo['ValidSeqOffTrID'].items():
        if len(v)>0:
            x[k-1]=v[2]
    TrCondMat['OLength']= x
    return TrCondMat

def getTrLongMat(TrInfo,TrCondMat):
    nMaxPos = 11
    nDatStack = 3 # Out, In, O-I
    nTr =len(TrInfo['All']['Trials'])

    Cols = ['trID','Pos','IO','Cue','Desc','Traj','Loc','OTraj', 'Goal','ioMatch','Co','Valid', 'Sp']
    nCols = len(Cols)
    TrLongMat = pd.DataFrame(np.full((nTr*nMaxPos*nDatStack,nCols),np.nan),columns=Cols)

    try:
        TrLongMat['trID'] = np.tile(np.tile(TrInfo['All']['Trials'],nMaxPos),nDatStack)
        TrLongMat['Pos'] = np.tile(np.repeat(np.arange(nMaxPos),nTr),nDatStack)
        TrLongMat['IO'] = np.repeat(['Out','In','O_I'],nTr*nMaxPos)

        TrLongMat['Traj'] = np.tile(np.tile(TrCondMat['Traj'],nMaxPos),nDatStack)
        TrLongMat['OTraj'] = np.tile(np.tile(TrCondMat['OTraj'],nMaxPos),nDatStack)
        TrLongMat['Co'] = np.tile(np.tile(TrCondMat['Co'],nMaxPos),nDatStack)
        TrLongMat['Cue'] = np.tile(np.tile(TrCondMat['Cues'],nMaxPos),nDatStack)
        TrLongMat['Desc'] = np.tile(np.tile(TrCondMat['Desc'],nMaxPos),nDatStack)
        TrLongMat['ioMatch'] = [traj==otraj for traj,otraj in zip(TrLongMat['Traj'],TrLongMat['OTraj'])]

        TrLongMat['Goal'] = [traj[3] if traj==traj else '' for traj in TrLongMat['Traj']]
        TrLongMat['Len'] = [traj[2] if traj==traj else '' for traj in TrLongMat['Traj']]
        TrLongMat['OLen'] = [traj[2] if traj==traj else '' for traj in TrLongMat['OTraj']]

        # get true location in each trials sequence 'Loc'
        # note that 'Pos' is a numerical indicator of the order in a sequence
        outTrSeq = pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'])
        outTrSp = pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'])
        inTrSeq = pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'])
        inTrSp = pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'])
        oiTrSeq = pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'])
        oiTrSp = pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'])

        for tr in TrInfo['All']['Trials']:
            traj = TrInfo['ValidSeqTrID'][tr]
            sp = TrInfo['TrSeq']['Sp'][tr]
            if traj in ValidTrajNames:
                seq = TrInfo['ValidSeqNames'][traj]
                if len(seq)==nMaxPos:
                    outTrSeq.loc[tr]=seq
                    outTrSp.loc[tr]=sp
                else:
                    outTrSeq.loc[tr]= seq + [np.nan]*4
                    outTrSp.loc[tr]= sp.tolist() + [np.nan]*4

            otraj = TrInfo['ValidSeqOffTrID'][tr]
            osp = TrInfo['OffTrSeq']['Sp'][tr]
            if otraj in ValidTrajNames:
                oseq =  TrInfo['ValidSeqOffNames'][otraj]
                if len(oseq)==nMaxPos:
                    inTrSeq.loc[tr]= oseq
                    inTrSp.loc[tr] = osp
                else:
                    inTrSeq.loc[tr]=[np.nan]*4+oseq
                    inTrSp.loc[tr] = [np.nan]*4+osp.tolist()

            if (traj in ValidTrajNames) and (otraj in ValidTrajNames):
                if traj==otraj:
                    if len(seq)==nMaxPos:
                        oiTrSeq.loc[tr]=seq
                    else:
                        oiTrSeq.loc[tr] = seq + [np.nan]*4
                elif traj[2]=='L' and otraj[2]=='S':
                    oiTrSeq.loc[tr] = seq[:4]+[np.nan]+seq[9:]+[np.nan]*4
                elif traj[2]=='S' and otraj[2]=='L':
                    oiTrSeq.loc[tr] = seq[:4]+[np.nan]+seq[5:]+[np.nan]*4

            oiTrSp.loc[tr] = outTrSp.loc[tr].values-inTrSp.loc[tr][::-1].values

        TrLongMat['Loc'] = pd.concat([pd.concat([outTrSeq.melt(value_name='Loc')['Loc'],
                                                 inTrSeq.melt(value_name='Loc')['Loc']]),
                                                 oiTrSeq.melt(value_name='Loc')['Loc']]).values

        TrLongMat['Sp'] = pd.concat([pd.concat([outTrSp.melt(value_name='Sp')['Sp'],
                                                 inTrSp.melt(value_name='Sp')['Sp']]),
                                                 oiTrSp.melt(value_name='Sp')['Sp']]).values

        TrLongMat['Valid'] = ~TrLongMat['Loc'].isnull()
        TrLongMat['EvenTrial'] = TrLongMat['trID']%2==0

    except:
        print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
    return TrLongMat

def getTrLongFRDat(TrInfo,cell_FR,mua_FR):
    # working version of long DF trialxPos matrix. this takes either short or long trajectories in the outbound but only short trajectories inbound
    nMaxPos = 11
    nMinPos = 7
    nTr =len(TrInfo['All']['Trials'])

    nCells = cell_FR.shape[0]
    nMua = mua_FR.shape[0]
    nTotalUnits = nCells+nMua
    nUnits = {'cell':nCells,'mua':nMua}

    cellCols = ['cell_'+str(i) for i in np.arange(nCells)]
    muaCols = ['mua_'+str(i) for i in np.arange(nMua)]
    unitCols = {'cell':cellCols,'mua':muaCols}
    allUnits = cellCols+muaCols

    mu = {'cell':np.mean(cell_FR,1),'mua':np.mean(mua_FR,1)}
    sig = {'cell':np.std(cell_FR,1),'mua':np.std(mua_FR,1)}

    try:
        TrZn={'cell':[],'mua':[]}
        OTrZn={'cell':[],'mua':[]}

        TrZn['cell'],OTrZn['cell'] = getFR_TrZone(TrInfo,cell_FR)
        TrZn['mua'],OTrZn['mua'] = getFR_TrZone(TrInfo,mua_FR)

        Out=pd.DataFrame(np.full((nTr*nMaxPos,nTotalUnits),np.nan),columns=allUnits)
        In=pd.DataFrame(np.full((nTr*nMaxPos,nTotalUnits),np.nan),columns=allUnits)
        O_I=pd.DataFrame(np.full((nTr*nMaxPos,nTotalUnits),np.nan),columns=allUnits)

        Out = Out.assign(trID = np.tile(TrInfo['All']['Trials'],nMaxPos))
        In = In.assign(trID = np.tile(TrInfo['All']['Trials'],nMaxPos))
        O_I = O_I.assign(trID = np.tile(TrInfo['All']['Trials'],nMaxPos))

        Out = Out.assign(Pos = np.repeat(np.arange(nMaxPos),nTr))
        In = In.assign(Pos = np.repeat(np.arange(nMaxPos),nTr))
        O_I = O_I.assign(Pos = np.repeat(np.arange(nMaxPos),nTr))

        Out = Out.assign(IO = ['Out']*(nTr*nMaxPos))
        In = In.assign(IO = ['In']*(nTr*nMaxPos))
        O_I = O_I.assign(IO = ['O_I']*(nTr*nMaxPos))

        for ut in ['cell','mua']:
            for cell in np.arange(nUnits[ut]):
                X=pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'],columns=np.arange(nMaxPos))
                Y=pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'],columns=np.arange(nMaxPos))
                Z=pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'],columns=np.arange(nMaxPos))

                m = mu[ut][cell]
                s = sig[ut][cell]

                for tr in TrInfo['All']['Trials']:
                    traj = TrInfo['ValidSeqTrID'][tr]
                    if traj in ValidTrajNames:
                        if traj[2]=='S':
                            X.loc[tr][0:nMinPos] = zscore(TrZn[ut][tr][cell],m,s)
                        else:
                            X.loc[tr] = zscore(TrZn[ut][tr][cell],m,s)

                    otraj = TrInfo['ValidSeqOffTrID'][tr]
                    if otraj in ValidTrajNames:
                        if otraj[2]=='S':
                            Y.loc[tr][4:] = zscore(OTrZn[ut][tr][cell],m,s)
                        else:
                            Y.loc[tr] = zscore(OTrZn[ut][tr][cell],m,s)
                    if (traj in ValidTrajNames) and (otraj in ValidTrajNames):
                        if traj==otraj:
                            Z.loc[tr] = X.loc[tr].values-Y.loc[tr][::-1].values
                        elif traj[2]=='L' and otraj[2]=='S': # ambigous interserction position, skipping that computation
                            Z.loc[tr][[0,1,2,3]] = X.loc[tr][[0,1,2,3]].values-Y.loc[tr][[10,9,8,7]].values
                            Z.loc[tr][[5,6]] = X.loc[tr][[9,10]].values-Y.loc[tr][[5,4]].values
                        elif traj[2]=='S' and otraj[2]=='L':
                            Z.loc[tr][[0,1,2,3]] = X.loc[tr][[0,1,2,3]].values-Y.loc[tr][[10,9,8,7]].values
                            Z.loc[tr][[5,6]] = X.loc[tr][[5,6]].values-Y.loc[tr][[1,0]].values

                Out[unitCols[ut][cell]]=X.melt(value_name='zFR')['zFR']
                In[unitCols[ut][cell]]=Y.melt(value_name='zFR')['zFR']
                O_I[unitCols[ut][cell]]=Z.melt(value_name='zFR')['zFR']

        Data = pd.DataFrame()
        Data =  pd.concat([Data,Out])
        Data =  pd.concat([Data,In])
        Data =  pd.concat([Data,O_I])
        Data = Data.reset_index()
    except:
        print("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
    return Data

def fitTrModels(TrLongMat,TrFRData):
    #all_params = ['Loc:IO','Loc','IO','Cue','Desc','Co']
    all_params = ['Loc:IO','Loc','IO','Cue','Sp','Co']
    param_set = getParamSet(params=['Loc','IO','Cue','Sp','Co'])
    nModels = len(param_set)
    R2thr = 0.2

    cellColIDs =  [i for i,item in enumerate(TrFRData.columns.values) if 'cell' in item]
    nCells = len(cellColIDs)
    muaColIDs =  [i for i,item in enumerate(TrFRData.columns.values) if 'mua' in item]
    nMua = len(muaColIDs)
    nTotalUnits = nCells+nMua
    nUnits = {'cell':nCells,'mua':nMua}

    cellCols = TrFRData.columns[cellColIDs]
    muaCols = TrFRData.columns[muaColIDs]
    unitCols = {'cell':cellCols,'mua':muaCols}

    perfCols = ['FullMod_tR2','modelNum','selMod','trainR2','AICc','testR2']
    Cols = ['ut']+perfCols+ all_params
    nCols = len(Cols)
    LM_Dat = pd.DataFrame(np.full((nTotalUnits,nCols),np.nan),columns=Cols)
    LM_Dat.loc[:,'ut'] = ['cell']*nCells+['mua']*nMua

    datSubset = ~(TrLongMat['IO']=='O_I') & (TrLongMat['Valid'])
    dat = []
    dat = TrLongMat[datSubset].copy()
    dat['trID'] = dat['trID'].astype('category')
    dat = dat.reset_index()
    dat['zFR'] = np.zeros(dat.shape[0])

    try:
        with Parallel(n_jobs=16) as parallel:
            cnt=0
            for ut in ['cell','mua']:
                for cell in np.arange(nUnits[ut]):
                    print('\nAnalyzing {} {}'.format(ut,cell))

                    dat.loc[:,'zFR'] = TrFRData.loc[datSubset,unitCols[ut][cell]].values

                    t1 = time.time()
                    tR2 = getModel_testR2(dat,params=all_params)
                    t2 = time.time()
                    LM_Dat.loc[cnt,'FullMod_tR2'] = tR2

                    print('Full Model Test Set Fit completed. Time = {}'.format(t2-t1))
                    if tR2>=R2thr:
                        print('Full Model passed the threshold, looking for optimal submodel.')
                        r = parallel(delayed(getModelPerf)(dat,params=params) for params in param_set.values())
                        trainR2,trainAICc,_ = zip(*r)

                        t3 = time.time()
                        print('\n\nFitting Completed for {0} {1}, total time = {2:0.3f}s'.format(ut,cell,t3-t1))
                        selMod = np.argmin(trainAICc)

                        selMod_tR2 = getModel_testR2(dat,params=param_set[selMod])
                        print('Selected Model = {}, AICc = {}, testR2 = {} '.format(selMod,trainAICc[selMod],selMod_tR2))

                        LM_Dat.loc[cnt,'modelNum']=selMod
                        LM_Dat.loc[cnt,'selMod'] = param_set[selMod]
                        LM_Dat.loc[cnt,'trainR2']=trainR2[selMod]
                        LM_Dat.loc[cnt,'AICc'] = trainAICc[selMod]
                        LM_Dat.loc[cnt,'testR2'] = selMod_tR2

                        temp = r[selMod][2].wald_test_terms()
                        LM_Dat.loc[cnt,param_set[selMod]] = np.sqrt(temp.summary_frame()['chi2'][param_set[selMod]])
                    cnt+=1
    except:
        print(sys.exc_info())
    return LM_Dat

################################################################################
######################### Auxiliary Functions ##################################
################################################################################

def Traj2Dist(traj):
    return np.array([ TMF.MazeZonesDists[p] for p in ValidTraj[traj]])

def getPosSequence(PosZones,speed,startID,endID,step=0.02):
    nSamps = len(PosZones)
    pos = []
    samp = []

    pos.append(PosZones[0])
    samp.append(0)
    for p in np.arange(nSamps-1):
        p0 = PosZones[p]
        p1 = PosZones[p+1]
        if p0!=p1:
            pos.append(p1)
            samp.append(p+1)
    pos = np.array(pos)
    samp = np.array(samp) + startID
    nPos = len(pos)
    dur = np.zeros(nPos,dtype=int)
    for p in np.arange(nPos-1):
        dur[p] = samp[p+1]-samp[p]
    if endID==samp[-1]:
        dur[-1]=1
    else:
        dur[-1] = endID-samp[-1]

    pos_speed = np.zeros(nPos)
    for p in np.arange(nPos):
        pos_speed[p] = np.nanmean(speed[np.arange(samp[p],samp[p]+dur[p],dtype=int)-startID ])/10

    return pos, samp, dur, pos_speed

def cmp(a,b):
    return (a>b)-(a<b)

def getTrials(dat,**kwargs):
    nTr = dat.shape[0]
    trials = set(np.arange(1,nTr+1))
    try:
        for k,v in kwargs.items():
            trials = trials & set(np.where(dat[k]==v)[0]+1)
    except:
        print('Invalid Selection {} {}'.format(k,v))
        pass
    return np.sort(np.array(list(trials)))

def zscore(x,mu,sig):
    return (x-mu)/sig

def getFR_TrZone(TrInfo, FRMat):
    nCells = FRMat.shape[0]
    TrZnFR = {} # FR for every zone visited in that trial
    OTrZnFR = {} # FR for every zone visited in that trial

    for tr in TrInfo['All']['Trials']:
        nPos = len(TrInfo['TrSeq']['Pos'][tr])
        trSpPos = np.zeros((nCells,nPos))
        for p in np.arange(nPos):
            s=TrInfo['TrSeq']['Samp'][tr][p]
            d=TrInfo['TrSeq']['Dur'][tr][p]
            samps = np.arange(s,s+d)
            for cell in np.arange(nCells):
                trSpPos[cell, p]=np.mean(FRMat[cell][samps])

        nPos = len(TrInfo['OffTrSeq']['Pos'][tr])
        otrSpPos = np.zeros((nCells,nPos))
        for p in np.arange(nPos):
            s=TrInfo['OffTrSeq']['Samp'][tr][p]
            d=TrInfo['OffTrSeq']['Dur'][tr][p]
            samps = np.arange(s,s+d)
            for cell in np.arange(nCells):
                otrSpPos[cell, p]=np.mean(FRMat[cell][samps])

        TrZnFR[tr] = trSpPos
        OTrZnFR[tr] = otrSpPos
    return TrZnFR, OTrZnFR

def AICc(model):
    n = model.nobs
    llf = model.llf
    k = len(model.params)
    AIC = 2*(k-llf)
    c = 2*k*(k+1)/(n-k-1)
    return AIC+c

def R2(x,y):
    return (np.corrcoef(x,y)**2)[0,1]

def aR2(model,y,fit=[]):
    if fit==[]:
        fit = model.fittedvalues
    r2 = R2(fit,y)
    n = model.nobs
    p = len(model.params)-1
    aR2 = 1-(1-r2)*(n-1)/(n-p-1)
    return aR2

def getParamSet(params = ['Loc','IO','Cue','Desc','Co']):
    '''
    Returns a dictionary of parameter sets for modeling.
    '''
    combs = []

    for i in np.arange(1, len(params)+1):
        combs+= [list(x) for x in combinations(params, i)]
    param_set = {}
    cnt=0
    for c in combs:
        param_set[cnt] = c
        cnt+=1

    for c in combs:
        if ('IO' in c) and ('Loc' in c):
            param_set[cnt] = ['Loc:IO']+c
            cnt+=1
    return param_set

def getModel_testR2(dat,formula='',params=[],mixedlm=True, verbose=False):
    '''
    Obtains the test R2 based on even/odd splits of the data
    '''

    if len(params)>0 and len(formula)==0:
        formula = getFormula(params)
    else:
        print('No Method of selecting parameters provided.')
        return np.nan,np.nan,[]
    print('\nComputing mixedlm with formula: {}'.format(formula))

    try:
        dat_even = dat[dat['EvenTrial']==True]
        dat_odd = dat[dat['EvenTrial']==False]

        try:
            if mixedlm:
                md_even = smf.mixedlm(formula, data=dat_even,groups=dat_even["trID"])
            else:
                md_even = smf.ols(formula + 'trID', data=dat_even)
        except:
            md_even = smf.ols(formula, data=dat_even)

        mdf_even = md_even.fit()
        pred_odd = mdf_even.predict(dat_odd)

        try:
            if mixedlm:
                md_odd = smf.mixedlm(formula, data=dat_odd,groups=dat_odd["trID"])
            else:
                md_odd = smf.ols(formula + 'trID', data=dat_odd)
        except:
            md_odd = smf.ols(formula, data=dat_odd)

        mdf_odd = md_odd.fit()
        pred_even = mdf_odd.predict(dat_even)

        if verbose:
            print('\nPerformance Train-Even:Test-Odd')
            print("Train_aR2 = {0:.3f}".format(aR2(mdf_even,dat_even['zFR'])))
            print("Model_AICc = {0:.3f}".format(AICc(mdf_even)))
            print("Test_R2 = {0:.3f}".format(R2(pred_odd,dat_odd['zFR'])))
            print('\nPerformance Train-Odd:Test-Even')
            print("Train_aR2 = {0:.3f}".format(aR2(mdf_odd,dat_odd['zFR'])))
            print("Model_AICc = {0:.3f}".format(AICc(mdf_odd)))
            print("Test_R2 = {0:.3f}".format(R2(pred_even,dat_even['zFR'])))

        dat['Pred']=np.zeros(dat.shape[0])
        dat.loc[dat['EvenTrial']==True,'Pred']=pred_even
        dat.loc[dat['EvenTrial']==False,'Pred']=pred_odd

        r2 = R2(dat['zFR'],dat['Pred'])
        print('\nOverall test R2: {0:.3f}'.format(r2))
        return r2
    except:
        print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
        return np.nan

def getFormula(params):
    formula = 'zFR ~ '
    nP = len(params)
    cnt=1
    for i in params:
        formula += i
        if cnt<nP:
            formula +='+'
        cnt+=1
    return formula

def getModelPerf(dat,formula='',params=[],mixedlm=True):
    '''
    Obtains the train adjusted R2, and AIC for data.
    returns aR2, AIC, and the fitted model.
    '''
    print('\nComputing mixedlm with formula: {}'.format(formula))

    if len(params)>0 and len(formula)==0:
        formula = getFormula(params)
    else:
        print('No Method of selecting parameters provided.')
        return np.nan,np.nan,[]

    if mixedlm:
        md = smf.mixedlm(formula, data=dat, groups=dat["trID"])
    else:
        md = smf.ols(formula + '+trID', data=dat)

    mdf = md.fit()
    print('\n Model Performance:')
    train_aR2 = aR2(mdf,dat['zFR'])
    print("Train_aR2 = {0:.3f}".format(train_aR2))
    aic = AICc(mdf)
    print("Model_AICc = {0:.3f}".format(aic))

    return train_aR2, aic, mdf
