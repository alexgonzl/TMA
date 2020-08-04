import numpy as np
import pandas as pd
from scipy import signal, interpolate
import sys, os, time
from pathlib import Path
import h5py

sys.path.append('../PreProcessing/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')
import spatial_tuning as ST
import stats_functions as StF
import Pre_Processing.pre_process_functions as pp
from filters_ag import *
import nept

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from shapely.geometry import Point
from shapely.geometry.polygon import LinearRing, Polygon
from collections import Counter
from descartes import PolygonPatch

font = {'family' : 'sans-serif',
        'size'   : 20}

plt.rc('font', **font)

################################################################################
# Constants
################################################################################
# y limits in mm space
y_limit = [-100,1500]
x_limit = [-1000,1000]

nWells = 6
EventNames  = ['RH','RC','R1','R2','R3','R4','RG','AR','DH','DC','D1','D2','D3','D4',
             'LH','LC','L1','L2','L3','L4','CL','CR','TrID','cTr','iTr','LDs','RDs']
nEventTypes = len(EventNames) # total number of events
# note that here I change the names from the original event names. wells 1 and 2
# are now referred to as Home and Center.

Segs = ['Seg'+s for s in ['A','B','C','D','E','F','G']]
InSeg= [s+'i' for s in Segs]
OutSeg=[s+'o' for s in Segs]
SegDirNames = InSeg+OutSeg
ZonesNames  = ['Home','SegA','Center','SegB','I1','SegC','G1',
            'SegD','G2','SegE','I2','SegF','G3', 'SegG','G4']
nZones = len(ZonesNames)

ReLength = 0.5 # fixed reward length [seconds]
DeLength = 0.1 # fixed detection length [seconds]
PostTrialDur = 1 # fixed post trial duration [seconds]
TrialExtTimeCor = 0.3 # fixed time to extend trial post a correct trial
TrialExtTimeInCor=0.1 # fixed time extend trial post an incorrect trial

# Define Zones
Zones = {}
cnt=0
for z in ZonesNames:
    Zones[cnt]=ZonesNames[cnt]
    cnt+=1

MazeZonesCoords ={'Home':[(-300, -80), (-300, 80),(300,80),(300, -80)],
                  'Center': [(-80,500),(-95,400),(-150,400),(-150,655),
                             (-75,550),(0,600),(75,550),(150,660),(150,400),
                             (95,400),(80,500)],
                  'SegA': [(-150,80),(-80,500),(80,500),(150,80)],
                  'SegB': [(0,600),(0,700),(200,1000),(330,900),(75, 550)],
                  'SegC': [(610,1180),(610,800),(330,900),(450,1180)],
                  'SegD': [(200,1000),(50,1230),(450,1230),(450,1180)],
                  'SegE': [(0,600),(0,700),(-200,1000),(-330,900),(-75, 550)],
                  'SegF': [(-200,1000),(-50,1230),(-450,1230),(-450,1180)],
                  'SegG': [(-610,1180),(-610,800),(-330,900),(-450,1180)],

                  'G1':[(610,1180),(800,1180),(800,800),(610,800)],
                  'G2': [(50,1230),(50,1450),(450,1450),(450,1230)],
                  'G3': [(-50,1230),(-50,1450),(-450,1450),(-450,1230)],
                  'G4': [(-610,1180),(-800,1180),(-800,800),(-610,800)],

                  'I1': [(200,1000),(450,1180),(330,900)],
                  'I2': [(-330,900),(-450,1180),(-200,1000)],
                 }

# expected traveling distances for each segment in cm
MazeZonesDists = {'Home': 4.0, 'Center': 4.0, 'SegA': 42.0, 'SegB': 42.0,
                        'SegC': 21.0, 'SegD': 21.0, 'SegE': 42.0, 'SegF': 21.0,
                        'SegG': 21.0, 'G1': 4.0, 'G2': 4.0, 'G3': 4.0,  'G4': 4.0,
                        'I1':6.0,'I2':6.0}

MazeZonesGeom = {}

for zo in ZonesNames:
   MazeZonesGeom[zo] = Polygon(MazeZonesCoords[zo])

# filtering params
med_filt_window = 15 # in samples  21samps/60samps/s = 350ms
smooth_filt_window = 15 # in samples 15/6 = 250ms
filtCoeff = signal.firwin(smooth_filt_window, cutoff = 0.2, window = "hanning")

################################################################################
# Main Functions: get
################################################################################
def getBehTrackData(sessionPaths, overwrite=0):
    if (not sessionPaths['BehavTrackDat'].exists()) | overwrite:
        print('Computing Position Data.')
        posPath = Path(sessionPaths['Raw'],'VT1.nvt')
        t,x,y,ha = load_nvt2(posPath)
        xs,ys,ts = processXY(x,y,t,sessionPaths['step'])

        print('Computing Event Data.')
        evPath = Path(sessionPaths['Raw'],'Events.nev')
        ev = get_events(evPath)
        EventDat = getEventMatrix(ev,ts)

        print('Comrrecting Positions with Event Info.')
        xs,ys = correctXY(EventDat,xs,ys)

        print('Creating Position Data Structure.')
        PosDat = getPositionMat(xs,ys,ts,sessionPaths['step'])
        PosDat['EventDat'] = EventDat
        PosDat['tB'] = t[0]
        PosDat['tE'] = t[-1]

        with h5py.File(sessionPaths['BehavTrackDat'], 'w') as f:
            for k,v in PosDat.items():
                f.create_dataset(k,data=v)
        print('Behavioral Tracking Variables Computed and Saved.')

    else:
        print('Loading Beh Tracking Data')
        PosDat = {}
        with h5py.File(sessionPaths['BehavTrackDat'], 'r') as f:
            for k in f.keys():
                if k=='PosMat':
                    PosDat[k]=pd.DataFrame(f.get(k)[()],columns=ZonesNames)
                elif k=='SegDirMat':
                    PosDat[k]=pd.DataFrame(f.get(k)[()],columns=SegDirNames)
                elif k=='EventDat':
                    PosDat[k]=pd.DataFrame(f.get(k)[()],columns=EventNames)
                else:
                    PosDat[k]=f.get(k)[()]
        print('Complete.')

    return PosDat

def getPositionMat(xs,ys,ts,step):
    '''
    Main Wrapper Function to obtain the animals position in the maze as defined
    by the Maze Zones. x,y,t should be as obtained from load_nvt2 in 'pre_process_neuralynx'

    Inputs:
        x       -> raw x positions from Neuralynx
        y       -> raw y positions from Neuralynx
        t       -> raw time arrays from Neuralynx
        step    -> scalar indicating the temporal step to sample
    Outputs:
        PosDat =

    Example:
        posFile = 'path/to/VT1.nvt'
        t,x,y,ha = load_nvt2(posFile)
        step = 0.02
        xs,ys,ts = processXY(x,y,t,step) # resampling of data
        PosDat = getPositionMatrix(x,y,t,step)
    '''
    # transform and smooth tracking signal @ original rate
    t1= time.time()

    PosDat = {}
    PosDat['x'] = xs
    PosDat['y'] = ys
    PosDat['t'] = ts
    PosDat['step'] = step

    # get maze positions
    PosZones = getMazeZones(xs,ys)
    t2=time.time()
    print('Converting Track x,y to TreeMaze Positions Completed: {0:.2f} s'.format(t2-t1))

    PosDat['PosZones'] = PosZones
    # get position matrix
    PosMat = PosZones2Mat(PosZones)
    PosDat['PosMat'] = pd.DataFrame(data=PosMat,columns=ZonesNames)
    t3=time.time()
    print('Creating Position Matrix Completed : {0:.2f} s'.format(t3-t2))

    # get segment directions
    SegDirMat = getSegmentDirs(PosZones,ts)
    t4=time.time()
    print('Creating Segment Direction Matrix Complete: {0:.2f} s'.format(t4-t3))
    print('Processing of Position Data Complete : {0:.2f} s'.format(t4-t1))

    PosDat['SegDirMat'] = SegDirMat
    PosDat['SegDirSeq'] = np.zeros(len(ts))
    cnt=1
    for seg in SegDirNames:
        PosDat['SegDirSeq']+=PosDat['SegDirMat'][seg]*cnt
        cnt+=1
    PosDat['InSeg'] = np.sum(SegDirMat[InSeg].values,1).astype(bool)
    PosDat['OutSeg'] = np.sum(SegDirMat[OutSeg].values,1).astype(bool)

    PosDat['Speed'], PosDat['HeadingAng'] = ST.getVelocity(xs,ys,step)
    return PosDat

def getEventMatrix(events,tp):
    '''
    Main Wrapper Function to obtain the event matrix describing the animals
    behavior during the TreeMaze Task.

    Inputs:
        events  -> dictionary of task events as outputed byb the get_events
                function in 'pre_process_neuralynx'. each event key returns the
                temporal time stamps of the event
        tp      -> rescaled time series as obtained from getPositionMat
    Outputs:
        evMat   -> a tall/skinny binary matrix of events, each column indicates
                a different event, each row indicates time such that row i occurs
                'step' seconds after row i-1.
    Example:
        evFile = 'path/to/Events.nev'
        events = get_events(evFile)
        step = 0.02
        EventMat = getEventMatrix(events,step)
    '''
    tBegin = tp[0]
    tEnd = tp[-1]
    step = tp[1]-tp[0]

    nTimePoints = len(tp)

    # change durations to samples
    NReSamps = np.int(ReLength/step) # samples to extend the reward markers
    NDeSamps = np.int(DeLength/step)
    NPostTrialSamps = np.int(PostTrialDur/step)

    # get Reward Time Stamps based on detection and reward delivery
    for well in np.arange(1,nWells+1,dtype=int):
        events['RW'+str(well)]=getRewardStamps(well, events)

    # get Durations for Cues and LEDs based on termination criteria
    # also returns time markes for correct/incorrect trials
    CueDurSamps, TrialEvents = getTrialsAndCueDurations(events,tp)
    LED_Durs = getLEDDurations(events,step)

    # Create Event Matrix
    EventMat = pd.DataFrame(data=np.zeros((nTimePoints,nEventTypes),dtype=int),columns=EventNames)
    for i in np.arange(1,nWells+1):
        if i==1:
            suf_str = 'H'
        elif i==2:
            suf_str = 'C'
        else:
            suf_str = str(i-2)

        for e in ['RW','DE','L']:
                if e=='RW':
                    EventMat['R'+suf_str] = makeEventVector(events[e+str(i)],NReSamps,tp)
                elif e=='DE':
                    EventMat['D'+suf_str] = makeEventVector(events[e+str(i)],NDeSamps,tp)
                elif e=='L':
                    EventMat['L'+suf_str] = makeEventVector(events[e+str(i)],LED_Durs[e+str(i)],tp)

    for e in CueDurSamps.keys():
        EventMat[e] = makeEventVector(events[e],CueDurSamps[e],tp)

    EventMat['cTr'] = makeEventVector(TrialEvents['cTr'],NPostTrialSamps,tp)
    EventMat['iTr'] = makeEventVector(TrialEvents['iTr'],NPostTrialSamps,tp)

    EventMat['RG'] = EventMat['R1']+EventMat['R2']+EventMat['R3']+EventMat['R4']
    EventMat['AR'] = EventMat['RH']+EventMat['RC']+EventMat['RG']

    EventMat['TrID'] = makeEventVector(TrialEvents['TrS'],TrialEvents['TrD'],tp,
    evValue=np.arange(len(TrialEvents['TrS'])))
    EventMat['LDs'] = makeEventVector(TrialEvents['LDs'],TrialEvents['LDur'],tp)
    EventMat['RDs'] = makeEventVector(TrialEvents['RDs'],TrialEvents['RDur'],tp)
    EventMat=EventMat.astype(int)
    return EventMat

################################################################################
# Auxiliary Functions for creating Position Matrix
################################################################################

def ScaleRotateSmoothTrackDat(x,y):
    #### Static parameters ####

    # rotation angle for the maze (for original pixel space)
    rot_ang=np.pi/2+0.05

    # parameters for translation and scaling
    x_translate = -255
    y_translate = 550

    y_pix2mm = 1308/305
    x_pix2mm = 1358/269

    # speed thr
    spd_thr = 50 # mm/frame -> mm/frame*60frames/s*1cm/10mm = 50*6 cm/s

    ######## Operations ########
    # rotate
    x,y= ST.RotateXY(x,y,rot_ang)

    # re-scale
    x = -(x+x_translate)*x_pix2mm
    x[x<0]=x[x<0]*1.1 # left warping
    y = (y+y_translate)*y_pix2mm

    # compute velocity to create speed threshold
    dx = np.append(0,np.diff(x))
    dy = np.append(0,np.diff(y))
    dr = np.sqrt(dx**2+dy**2)
    mask_r = np.abs(dr)>spd_thr

    # mask creating out of bound zones
    mask_y = np.logical_or(y<y_limit[0],y>y_limit[1])
    mask_x = np.logical_or(x<x_limit[0],x>x_limit[1])
    mask = np.logical_or(mask_x,mask_y)
    mask = np.logical_or(mask,mask_r)

    mask_lower = np.logical_and(np.abs(x)>400,np.abs(y)<600)
    mask =np.logical_or(mask,mask_lower)

    x[mask]=np.nan
    y[mask]=np.nan

    # double round of median filters to deal with NaNs
    x = medFiltFilt(x,med_filt_window)
    y = medFiltFilt(y,med_filt_window)

    # if there are still NaNs assign id to previous value
    badIds = np.where(np.logical_or(np.isnan(x), np.isnan(y)))[0]
    for ii in badIds:
        x[ii] = ST.getLastNotNanVal(x,ii)
        y[ii] = ST.getLastNotNanVal(y,ii)

    # filter / spatial smoothing
    x = signal.filtfilt(filtCoeff,1,x)
    y = signal.filtfilt(filtCoeff,1,y)

    return x,y

def processXY(x,y,t,step):
    # transform and smooth tracking signal @ original rate
    t1= time.time()
    xs,ys = ScaleRotateSmoothTrackDat(x,y)
    t2=time.time()
    print('Smoothing track data completed: {0:0.2f} s '.format(t2-t1))

    # resampling the data
    ts, xs   = ReSampleDat(t,xs,step)
    _, ys   = ReSampleDat(t,ys,step)
    ts = np.round(ts*1000)/1000 #round tp to ms resolution.
    t3=time.time()
    print('Resampling the Data to {0} seconds completed: {1:.2f} s '.format(step,t3-t2))

    return xs,ys,ts

def correctXY(EventDat,x,y):
    xd = [0,0,650,250,-250,-650]
    yd = [45,560,1000,1280,1280,1000]

    x2=np.array(x)
    y2=np.array(y)

    for z1 in ['D','R']:
        cnt=0
        for z2 in ['H','C','1','2','3','4']:
            z = z1+z2
            ids = EventDat[z]==1
            x2[ids] = xd[cnt]
            y2[ids] = yd[cnt]
            cnt+=1

    x2 = medFiltFilt(x2,5)
    y2 = medFiltFilt(y2,5)
    for z1 in ['D','R']:
        cnt=0
        for z2 in ['H','C','1','2','3','4']:
            ids = EventDat[z]==1
            x2[ids] = xd[cnt]
            y2[ids] = yd[cnt]
            cnt+=1

    return x2,y2

def getMazeZones(x,y):
    # Get zones that contains each x,y point
    PosZones = np.zeros(len(x),dtype=int)
    pcnt = -1
    DistThr = 100 # in mm -> 10cm
    for xp, yp in zip(x,y):
        pcnt+=1
        zcnt=0
        outZoneFlag = 0
        if not np.isnan(xp):
            pZoneDist = np.zeros(nZones)
            p = Point(xp,yp)
            for zo in ZonesNames:
                pZoneDist[zcnt]=MazeZonesGeom[zo].distance(p)
                if MazeZonesGeom[zo].contains(p):
                    PosZones[pcnt]=zcnt
                    break
                elif zo == ZonesNames[-1]:
                    if np.min(pZoneDist)<DistThr:
                        PosZones[pcnt]=np.argmin(pZoneDist)
                    else:
                        PosZones[pcnt]=PosZones[pcnt-1]
                    break
                zcnt+=1
        else:
            PosZones[pcnt]=PosZones[pcnt-1]

    return PosZones

def PosZones2Mat(PosZones):
    M = np.full((len(PosZones),nZones),0)
    for z in np.arange(nZones):
        M[PosZones==z,z]=1
    return M

def getPosSequence(PosZones,t):
    nPos = len(PosZones)
    out = []
    out.append((t[0],PosZones[0],0))
    for p in np.arange(nPos-1):
        p0 = PosZones[p]
        p1 = PosZones[p+1]
        if p0!=p1:
            out.append((t[p+1],p1,p+1))
    return out

def getSegmentDirs(PosZones,t):
    PosSeq = getPosSequence(PosZones,t)
    nPosSeqs = len(PosSeq)
    SegDirs = {}
    for io in SegDirNames:
        SegDirs[io] =[]

    for pos in np.arange(nPosSeqs-1):
        currentZone = PosSeq[pos][1]
        currentMarker = PosSeq[pos][2]
        nextZone = PosSeq[pos+1][1]
        deltaT = PosSeq[pos+1][2]  - currentMarker
        if Zones[currentZone] == 'SegA':
            if Zones[nextZone]=='Home':
                SegDirs['SegAi'].append([currentMarker,deltaT])
            else:
                SegDirs['SegAo'].append([currentMarker,deltaT])
        elif Zones[currentZone] == 'SegB':
            if Zones[nextZone]=='Center':
                SegDirs['SegBi'].append([currentMarker,deltaT])
            else:
                SegDirs['SegBo'].append([currentMarker,deltaT])
        elif Zones[currentZone] == 'SegE':
            if Zones[nextZone]=='Center':
                SegDirs['SegEi'].append([currentMarker,deltaT])
            else:
                SegDirs['SegEo'].append([currentMarker,deltaT])
        elif Zones[currentZone] == 'SegC':
            if Zones[nextZone]=='G1':
                SegDirs['SegCo'].append([currentMarker,deltaT])
            else:
                SegDirs['SegCi'].append([currentMarker,deltaT])
        elif Zones[currentZone] == 'SegD':
            if Zones[nextZone]=='G2':
                SegDirs['SegDo'].append([currentMarker,deltaT])
            else:
                SegDirs['SegDi'].append([currentMarker,deltaT])
        elif Zones[currentZone] == 'SegF':
            if Zones[nextZone]=='G3':
                SegDirs['SegFo'].append([currentMarker,deltaT])
            else:
                SegDirs['SegFi'].append([currentMarker,deltaT])
        elif Zones[currentZone] == 'SegG':
            if Zones[nextZone]=='G4':
                SegDirs['SegGo'].append([currentMarker,deltaT])
            else:
                SegDirs['SegGi'].append([currentMarker,deltaT])
    for io in SegDirNames:
        SegDirs[io] = np.array(SegDirs[io])

    nSegs = len(SegDirs)
    nSegDirNames = len(SegDirNames)
    SegDirMat = pd.DataFrame(np.zeros((len(t),nSegDirNames),int),columns=SegDirNames)
    for segID in SegDirNames:
        SegDirID = SegDirs[segID]
        nSegDirID = len(SegDirID)
        for i in np.arange(nSegDirID):
            mark = SegDirID[i][0]
            dur = SegDirID[i][1]
            SegDirMat[segID][mark:(mark+dur)]=1
    return SegDirMat


################################################################################
# Auxiliary Functions for creating Event Matrix
################################################################################

def isnear(X,Y,thr,):
    '''Find x,y points within the thr'''
    x_out= np.full_like(X, -1)
    y_out= np.full_like(Y, -1)
    match_cnt=1
    cnt1=0
    for x in X:
        cnt2=0
        for y in Y:
            if abs(x-y)<=thr:
                x_out[cnt1]=match_cnt
                y_out[cnt2]=match_cnt
                match_cnt+=1
            cnt2+=1
        cnt1+=1
    return x_out,y_out

def isClosest(t,X):
    '''Find closest sample in t that matches X'''
    t_out= np.full_like(t, -1)
    cnt1=1
    for x in X:
        idx = np.argmin(np.abs(t-x))
        if x-t[idx]>=0:
            t_out[idx]=cnt1
        else: # always assign to the earliest sample
            t_out[idx-1]=cnt1
        cnt1+=1
    return t_out

def isbefore(X,Y,thr,minTime=0):
    '''Find x,y points within the thr and such that x happens before y'''
    x_out= np.full_like(X, -1)
    y_out= np.full_like(Y, -1)
    match_cnt=0
    cnt1=0
    for x in X:
        cnt2=0
        for y in Y:
            if y-x<=thr and y-x>=minTime:
                x_out[cnt1]=match_cnt
                y_out[cnt2]=match_cnt
                match_cnt+=1
                break
            cnt2+=1

        cnt1+=1
    return x_out,y_out

def getRewardStamps(wellnum, ev):
    x1,x2=isnear(ev['DE'+str(wellnum)],ev['RD'],0.01)
    return ev['RD'][x2>0]

def makeEventVector(evTimes,evDurs,t,evValue =1):
    '''
    Creates binary vector of length len(t). It takes a list of event times,
    assigns them to the regularly sampled time vector 't', and returns a vector
    that has ones in the times of the events, and extends the events for evDurs.
    Note that evDurs can either be a vector of the same length as evTimes, or a fixed duration time.

    Inputs:
        evTimes -> list of times of the events
        evDurs  -> list of times of the duration of events in evTimes
                    or the number of Samples that the event lasts
        t       -> regularly sampled time vector that covers the length of a recording.
    Outputs:
        evVec   -> binary vector of length len(t), that is one for in the event
                    times plus the evDur
    '''
    N = len(t)
    nEvents = len(evTimes)
    tt = isClosest(t,evTimes)
    if type(evDurs)==int:
        evVec=signal.lfilter(np.ones(evDurs),1,tt>0)
        evVec[evVec>1]=evValue
        return evVec
    elif nEvents == len(evDurs):
        locs = np.where(tt>0)[0]
        evVec = np.zeros(N)
        for i in np.arange(nEvents):
            idx = np.arange(locs[i],locs[i]+evDurs[i])
            if type(evValue)==int:
                evVec[idx]=evValue
            else:
                evVec[idx]=evValue[i]
        return evVec
    else:
        print('Event and Event Duration mismatch: {} and {}'.format(nEvents,len(evDurs)))
        return []

def getTrialsAndCueDurations(ev,t):
    '''
    Function to obtain durations of the left and right cues,and trial outcome based on termination criteria.
    Criteria is based on reward events or detections at incorrect wells. Exit case also includes
    the start of another trial.

    Inputs:
        ev -> event directory. This must already include Reward events through the
        'getRewardStamps' function
        t -> regularly sampled time vector that covers the length of a recording
    Outputs:
        CueDurSamps -> dict of cue durations in samples
        TrialEvents > dict of trial start times and post correct/incorrect trial times

    '''
    tBegin = t[0]
    tEnd = t[-1]
    step = t[1]-t[0]

    nLCues = len(ev['CL'])
    nRCues = len(ev['CR'])

    AllCues = np.concatenate((np.ones(nLCues),1+np.ones(nRCues)))
    AllCuesTimes = np.concatenate((ev['CL'],ev['CR']))
    SortedCueIDs = np.argsort(AllCuesTimes)
    SortedCueTimes = AllCuesTimes[SortedCueIDs]
    SortedCues = AllCues[SortedCueIDs]
    nCues = nLCues+nRCues

    AllTrialStarts = np.array(SortedCueTimes)
    AllTrialDurs = []
    CorrectTrials  = []
    InCorrectTrials  = []
    LeftDecision = []
    LeftDurs = []
    RightDecision = []
    RightDurs = []

    nL = 0
    nR = 0
    RC_Durs = np.zeros(nRCues)
    LC_Durs = np.zeros(nLCues)
    for c in np.arange(nCues):
        t0=SortedCueTimes[c]
        if c==(nCues-1):
            t1=tEnd
        else:
            t1=SortedCueTimes[c+1]
        match = 0
        if SortedCues[c] == 1: # left
            for end_Ev in ['RW5','RW6','DE3','DE4']:
                end_Ev_id = np.logical_and(ev[end_Ev]>=t0,ev[end_Ev]<t1)
                if any(end_Ev_id):
                    if end_Ev in ['RW5','RW6']:
                        tE = ev[end_Ev][end_Ev_id][0]+TrialExtTimeCor
                        CorrectTrials.append(tE)
                        LeftDecision.append(t0)
                        LeftDurs.append(tE-t0)
                    else:
                        tE = ev[end_Ev][end_Ev_id][0]+TrialExtTimeInCor
                        InCorrectTrials.append(tE)
                        RightDecision.append(t0)
                        RightDurs.append(tE-t0)
                    LC_Durs[nL] = tE-t0
                    match = 1
                    AllTrialDurs.append(tE-t0)
                    break
            if match==0:
                LC_Durs[nL] = t1-t0
                InCorrectTrials.append(t1)
                AllTrialDurs.append(t1-t0)
            nL+=1 # note that multiple matches will override....
        elif SortedCues[c]==2: # right
            for end_Ev in ['RW3','RW4','DE5','DE6']:
                end_Ev_id = np.logical_and(ev[end_Ev]>=t0,ev[end_Ev]<t1)
                if any(end_Ev_id):
                    tE = ev[end_Ev][end_Ev_id][0]
                    RC_Durs[nR] = tE-t0
                    match = 1
                    AllTrialDurs.append(tE-t0)
                    if end_Ev in ['RW3','RW4']:
                        CorrectTrials.append(tE)
                        RightDecision.append(t0)
                        RightDurs.append(tE-t0)
                    else:
                        InCorrectTrials.append(tE)
                        LeftDecision.append(t0)
                        LeftDurs.append(tE-t0)
                    break
            if match==0:
                RC_Durs[nR] = t1-t0
                InCorrectTrials.append(t1)
                AllTrialDurs.append(t1-t0)
            nR+=1 # note that multiple matches will override....

    CueDurSamps = {}
    CueDurSamps['CL']=np.round(LC_Durs/step).astype(int)
    CueDurSamps['CR']=np.round(RC_Durs/step).astype(int)

    TrialEvents = {}
    TrialEvents['TrS'] = AllTrialStarts
    TrialEvents['TrD'] = np.round(AllTrialDurs/step).astype(int)
    TrialEvents['LDs'] = LeftDecision
    TrialEvents['LDur'] = np.round(LeftDurs/step).astype(int)
    TrialEvents['RDs'] = RightDecision
    TrialEvents['RDur'] = np.round(RightDurs/step).astype(int)
    TrialEvents['cTr'] = CorrectTrials
    TrialEvents['iTr'] = InCorrectTrials
    return CueDurSamps, TrialEvents

def getLEDDurations(ev,step):
    '''
    Function to obtain durations of the LEDs, based on termination criteria.
    Criteria is based on reward events or detections at incorrect wells. Exit case also includes
    the start of another trial. Note that this code would Only be applicable for T3 sessions.

    Inputs:
        ev -> event directory. This must already include Reward events through the
        'getRewardStamps' function
        step -> time step to convert time to samples
    Outputs:
        LED_Durs -> directory of durations in samples for the different LEDs
    '''

    defDurTime = 0.5 # default duration, in case termination criteria is not found.
    nLED_Ev = np.zeros(nWells,dtype=int)
    LED_Durs = {}
    defDur = np.round(defDurTime/step).astype(int)

    EndCriteria = {'L1':['RW1'],'L2':['RW2'],
                    'L3':['RW3','DE5','DE6'],'L4':['RW4','DE5','DE6'],
                    'L5':['RW5','DE3','DE4'],'L6':['RW6','DE3','DE4']}

    for well in np.arange(nWells):
        L_ID = 'L' + str(well+1)
        nLED_Ev[well] = len(ev[L_ID])
        LED_Durs[L_ID] = np.zeros(nLED_Ev[well],dtype=int)
        if nLED_Ev[well] > 0:

            # get event matches for all end criteria events
            all_matches = {}
            for eC in EndCriteria[L_ID]:
                all_matches[eC] = isbefore(ev[L_ID],ev[eC],200,minTime=0.01)

            #print(nLED_Ev[well],np.arange(nLED_Ev[well]))
            for eID in np.arange(nLED_Ev[well]):
                t0 = ev[L_ID][eID]

                # select minimum end criteria event
                trueMatchFlag =0
                t1 = t0+500
                for eC in EndCriteria[L_ID]:
                    if any(all_matches[eC][1]==eID):
                        t1 = min(t1,ev[eC][all_matches[eC][1]==eID][0])
                        trueMatchFlag=1

                if trueMatchFlag:
                    LED_Durs[L_ID][eID] = np.round((t1-t0)/step).astype(int)
                else:
                    LED_Durs[L_ID][eID] = defDur
    return LED_Durs
