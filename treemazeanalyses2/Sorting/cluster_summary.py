import sys,os, json, datetime, getopt, shutil, filecmp
from pathlib import Path
import numpy as np
import pandas as pd

## Fixed Parameters
overwrite = 1
nTTs = 16

def GetSessionClusters(session):
    '''
    session = Path object indicating the directory of the session
    '''

    assert session.exists(), 'Could not find session. Check Spelling/Path.'

    table = {}
    table['session'] = session.name
    table['Clustered']=0
    table['path'] = str(session)
    table['nCells'] = 0
    table['nMua'] = 0
    table['cell_IDs'] = {}
    table['mua_IDs'] = {}
    table['dateClustered']={}
    table['unClustered_TTs'] = []
    table['All_TTs_Clustered']=0

    TTs =  np.arange(1,nTTs+1)
    for tt in TTs:
        fn = session/('tt_'+str(tt))/'cluster_group.tsv'
        #assert fn.exists(), 'could not find record for tt {}; in {}'.format(tt,sessionName)
        if fn.exists():
            try:
                table['cell_IDs'][int(tt)]=[]
                table['mua_IDs'][int(tt)]=[]

                d=pd.read_csv(fn,delimiter='\t')
                # found bug here. if clusters do not follow an order [0,1,2,3]
                # this will give incorrect answers.
                #cells = np.where(d['group']=='good')[0].tolist()
                #mua = np.where(d['group']=='mua')[0].tolist()

                # correct version.
                cells = d['cluster_id'][d['group']=='good'].tolist()
                mua = d['cluster_id'][d['group']=='mua'].tolist()

                table['cell_IDs'][int(tt)]=[]
                table['mua_IDs'][int(tt)]=[]

                for cc in cells:
                    table['cell_IDs'][int(tt)].append(cc)
                for mm  in mua:
                    table['mua_IDs'][int(tt)].append(mm)

                table['nCells'] += len(cells)
                table['nMua'] += len(mua)
                table['dateClustered'][int(tt)]= datetime.datetime.fromtimestamp(int(fn.stat().st_mtime)).strftime("%B %d %Y, %I:%M%p")
            except:
                print('In Session {}, Error Processing TT {}'.format(session.name,tt))
        else:
            table['unClustered_TTs'].append(int(tt))
    if len(table['unClustered_TTs'])==0:
        table['All_TTs_Clustered']=1
        table['Clustered']=1
    elif len(table['unClustered_TTs'])==nTTs:
        table['Clustered']=0
        print("\nSession {} no TTs clustered.".format(session.name))
    else:
        table['Clustered']=1

    if table['Clustered']:
        table['dateSummary'] = datetime.datetime.today().strftime("%B %d %Y, %I:%M%p")
        print("\nResults for {}:\n nCells = {} \n nMuas = {}".format(session.name, table['nCells'], table['nMua']))
    return table

def CopyClustersToOak(localDir,oakDir):
    localDir = Path(localDir)
    oakDir = Path(oakDir)
    for SummaryFile in oakDir.glob("*_ClusteringSummary.json"):

        with SummaryFile.open(mode='r') as f:
            cluster_summary = json.load(f)

        for session in cluster_summary['Sessions']:
            # copy individual tetrode clusters
            sessionName = session+'_KSClusters'
            existsList = []
            notExistsList = []
            updatedList = []
            notUpDatedList = []
            for tt in np.arange(1,nTTs+1):
                try:
                    fn = localDir/sessionName/('tt_'+str(tt))/'cluster_group.tsv'
                    fn2 = localDir/sessionName/('tt_'+str(tt))/'spike_clusters.npy'
                    if fn.exists():
                        sp = oakDir/sessionName/('tt_'+str(tt))/'cluster_group.tsv'
                        sp2 = oakDir/sessionName/('tt_'+str(tt))/'spike_clusters.npy'
                        if sp.exists():
                            # if it exists @ destination but has been change, overwrite.
                            if not filecmp.cmp(str(fn),str(sp),shallow=True):
                                shutil.copy(str(fn),str(sp))
                                updatedList.append(tt)
                                print('{}: TT {} overwrite.'.format(session,tt))
                            else:
                            #otherwise ignore.
                                existsList.append(tt)
                                notUpDatedList.append(tt)
                        else:
                            # if it doesn't exist, copy
                            shutil.copy(str(fn),str(sp))
                            updatedList.append(tt)
                            print('{}: TT {} Copy.'.format(session,tt))

                        if not filecmp.cmp(str(fn2),str(sp2),shallow=True):
                            shutil.copy(str(fn2),str(sp2))
                            print('Updating spike_cluster IDs file')
                    else:
                        notExistsList.append(tt)
                except:
                    notUpDatedList.append(tt)
                    print("Possible Mounting Issue. Try umounting/remounting the Oak partition.")
                    print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)

            if len(existsList)==16:
                print("{}: All Files Exists in Cluster.".format(session))
            elif len(existsList)>0:
                print("{}:\n Files exists and not updated TTs {} \n Files do not exists {} ".format(session, existsList, notExistsList))
            else:
                print("{}: No Cluster Files to Copy.".format(session))

            if len(updatedList)>0:
                print("{}: Updated Cluster Files: {}".format(session, updatedList))
            # if len(notUpDatedList)==16:
            #     print("{}: No Clusters Updated. ".format(session))
            # elif len(notUpDatedList)==0:
            #     print("{}: Updated all tetrode clusters".format(session))
            # else:
            #     print("{}: Indetical cluster files, no updates for TTs {}".format(session, notUpDatedList))
            print()

def GetClusterTable(cl_summary, oakPath, localPath=''):
    '''
    Human readble summary cluster table. Tells, what needs to be done still!
    '''
    oakPath = Path(oakPath)

    colnames = ['SessionDate','Task','Animal','Clustered','nCells','nMua','BestTT']
    emptyEntry = {key: [0] for key in colnames}

    Sessions =[]
    Dates = []
    Tasks = []
    Animals =[]
    Clustered=[]
    for se,cl in cl_summary['Sessions'].items():
        tmp = se.split('_')
        Dates.append(tmp[2])
        Tasks.append(tmp[1])
        Animals.append(tmp[0])
        Sessions.append(se)
        Clustered.append(cl)

    d = pd.DataFrame(0,index = Sessions, columns=colnames)
    d['Task']=Tasks
    d['SessionDate']=Dates
    d['Animal']=Animals
    d['Clustered']=Clustered

    for sn in Sessions:
        if d.at[sn,'Clustered']:
            date = d.at[sn,'SessionDate']
            task = d.at[sn,'Task']
            animal = d.at[sn,'Animal']
            try:
                info = cl_summary[animal][date][task]
                d.at[sn,'nCells'] = info['nCells']
                d.at[sn,'nMua'] = info['nMua']
                d.at[sn,'BestTT'] = dict_argmax(info['cell_IDs'])+1
                #d.at[sn,'SummaryDate'] = info['dateSummary']
            except:
                print("Error updating session {}".format(sn))
                print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
    print(d)
    localSave=0
    fn = 'ClusterTableSummary.csv'
    try:
        if len(str(localPath))>0:
            localPath = Path(localPath)
            d.to_csv(str(localPath/fn))
            localSave=1
            print('File Saved to {}'.format(oakPath))
        else:
            localSave=0
    except:
        print('Could not save file locally.')
        print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
    try:
        d.to_csv(str(oakPath/fn))
        oakSave =1
        print('File Saved to {}'.format(oakPath))
    except:
        if localSave and (not oakSave) :
            try:
                shutil.copy2(str(localPath/fn),str(oakPath/fn))
                print('File copy from local to  {}'.format(oakPath))
            except:
                print('Could not save table to Oak.')
                print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
        else:
            print('Could not save table to Oak.')
            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)

def UpdateClusterInfo(oakPath, animal, localPath,overwrite=False):
    oakPath = Path(oakPath) # convert to Path object
    #Cluster Summary Json File. If it doesn't exist, create it.
    cnt =0
    cl_summary_fn = oakPath / ('{}_ClusteringSummary.json'.format(animal))
    if cl_summary_fn.exists() and overwrite==0:
        print('Loading Existing Summary File.')
        with cl_summary_fn.open() as f:
            cl_summary = json.load(f)
    else: # new summary
        print('Making New Summary File.')
        cl_summary = {}
        cl_summary[animal] ={}
        cl_summary['Sessions'] = {}
    for session in localPath.glob('*_KSClusters'):
        try:
            print(session.name, cl_summary[sessions][session.name]==0)
            if cl_summary[sessions][session.name]==0 or overwrite:
                updateSession=1
            else:
                updateSession=0
        except:
            updateSession=1

        if updateSession:
            try:
                tmp = session.name.split('_')
                an = tmp[0]
                #assert animal==an, 'Error, invalid session found.'
                task = tmp[1]
                date = tmp[2]
                oses = an+'_'+task+'_'+date
                print(oses)

                if an=='NE':
                    an = 'Ne'

                if not date in cl_summary[animal].keys():
                    cl_summary[an][date]={}
                if not task in cl_summary[an][date].keys() or overwrite:
                    cl_summary[an][date][task] = {}

                cl_summary[an][date][task] = GetSessionClusters(session)
                cl_summary['Sessions'][oses] = cl_summary[an][date][task]['Clustered']
            except:
                print('Unable to process session {}'.format(session))
                print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
    # Save
    try:
        localPath = Path(localPath)
        if localPath.exists():
            fn = cl_summary_fn.name
            with (localPath/fn).open(mode='w') as f:
                json.dump(cl_summary,f,indent=4)
                print('File Saved locally.')
    except:
        print('unable to save json file locally.')
        print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)

    try:
        with cl_summary_fn.open(mode='w') as f:
            json.dump(cl_summary,f,indent=4)
            print('File Saved into OAK')
    except:
        print('unable to update json cluster info file in oak. probably permission issue.')
        print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)



    GetClusterTable(cl_summary, oakPath, localPath)
    # print info
    nC=0;nM=0;nCS=0;nS=0;
    for d in cl_summary[animal].keys():
        info = cl_summary[animal][d]
        for t in info.keys():
            nS+=1
            info2 = info[t]
            if info2['Clustered']:
                nC+=info2['nCells']
                nM+=info2['nMua']
                nCS+=1
    print("\n Overall Summary for {} : \n nSessions = {} \n nClusteredSessions {} \n nCells = {} \n nMua = {}".format(animal, nS,nCS,nC,nM))
    return cl_summary
##### AUXILIARY FUNCTIONS #####
def dict_argmax(d):
    d2 = []
    for k in d.keys():
        d2.append(len(d[k]))
    return np.argmax(d2)

if __name__ == "__main__":
    animal = 'Li'
    oakPath = Path('/mnt/o/giocomo/alexg/Clustered/',animal)
    localPath = Path('/mnt/c/Users/alexg8/Documents/Data/',animal,'Clustered')

    CopyClustersToOak(localPath,oakPath)

    # Bug here, doesn't seem to just read exisiting file.
    #UpdateClusterInfo(oakPath,animal,localPath)
