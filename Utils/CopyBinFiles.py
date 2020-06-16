import os, json,sys, time, datetime, filecmp
import numpy as np
from pathlib import Path
from shutil import copy2

# PATCH! 4/25/19
# copy of tetrode binaries to clustering folders. this is required to run the current version of phy
# two solutions:
# (1) incorporate this copy in the clustering itself
# (2) make phy read file binaries at other locations other than
# where the clustering results are.

job_directory = Path("./.job")
job_directory.mkdir(parents=True, exist_ok=True)

ID = 'Ne'
date = '5_10_2019'

table = "Clustering_{}_{}.json".format(ID,date)
TasksDir = Path.cwd()/'TasksDir'
if not TasksDir.exists():
    sys.exit('Task directory not found.')

if (TasksDir/table).exists():
    with open(str(TasksDir/table), 'r') as f:
        task_table = json.load(f)

nJobs = len(task_table)
for t in [40]:#np.arange(1,nJobs+1):
    session = task_table[str(t)]['Files']
    nFiles = len(session)
    print()
    for f in np.arange(1,nFiles+1):
        try:
            fn = 'tt_{}.bin'.format(f)
            f1=session[str(f)]['filenames']
            sp=session[str(f)]['sp']
            f2=str(Path(sp,fn))
            if not filecmp.cmp(f1,f2):
                copy2(f1,sp)
                print('Copy completed for file {}'.format(f1))
            else:
                print('File {} already exists at destination. '.format(f1))
        except:
            print('Error copying {}'.format(f1))
            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
