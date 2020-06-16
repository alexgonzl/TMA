import sys
import os
import getopt
import json
import time
import warnings
import traceback
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from pre_process_functions import process_tetrode, process_events, process_video, post_process_channel_table

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":

    # Store taskID and TaskFile
    task_num = -1
    task_num_str = ''
    data_dir = ''
    subject_id = ''
    task_table_filename = ''
    overwrite_flag = False

    if len(sys.argv) < 3:
        print("Usage: %s -t task# -a 'animal id'" % sys.argv[0])
        sys.exit('Invalid input.')

    myopts, args = getopt.getopt(sys.argv[1:], "t:a:d:o")
    for opt, attr in myopts:
        if opt == '-t':
            task_num = int(attr)
            task_num_str = str(attr)
        elif opt == '-a':
            subject_id = str(attr)
            task_table_filename = 'pp_table_{}.json'.format(subject_id)
        elif opt == '-d':
            data_dir = Path(attr)
        elif opt == '-o':
            overwrite_flag = True
        else:
            print("Usage: %s -t taskID -a animal id" % sys.argv[0])
            sys.exit('Invalid input. Aborting.')

    print('PreProcess Manager: task {}, id {}, data {}'.format(task_num_str, subject_id, str(data_dir)))
    task_table_dir = data_dir / subject_id / 'TasksDir'
    if not task_table_dir.exists():
        sys.exit('Task directory not found.')

    task_table_filepath = task_table_dir / task_table_filename
    try:
        if task_table_filepath.exists():
            with task_table_filepath.open(mode='r') as f:
                task_table = json.load(f)
        else:
            sys.exit('Could not get Task Table. Aborting.')
    except:
        sys.exit('Could not get Task Table. Aborting.')

    if task_num >= 0:
        subtask_info = task_table[task_num_str]
        session_name = subtask_info['session_name']
        n_files = subtask_info['n_files']
        subtask_list = subtask_info['files']
        print("Processing Session {}".format(session_name))

        for subtask_id, subtask in subtask_list.items():
            try:
                t1 = time.time()
                task_type = subtask['task_type']
                if task_type == 'tt':
                    print("Processing Tetrode # {}, subSessionID={}".format(subtask['tt_id'], subtask['subSessionID']))
                    process_tetrode(subtask, overwrite_flag=overwrite_flag)

                elif task_type == 'ev':
                    print("Processing Events, subSessionID={}".format(subtask['subSessionID']))
                    process_events(subtask, overwrite_flag=overwrite_flag)

                elif task_type == 'vt':
                    print("Processing Video Tracking, subSessionID={}".format(subtask['subSessionID']))
                    process_video(subtask, overwrite_flag=overwrite_flag)

                # elif task_type == 'npy2bin':
                #     print("Processing Data Conversion to Binary")
                #     from dataConvert import npy2bin
                #
                #     npy2bin(task['filenames'], task['sp'], overwrite_flag=overwrite_flag)

                t2 = time.time()
                print("Task Completed. Total Task Time {0:0.2f}s".format(t2 - t1))
            except KeyboardInterrupt:
                print('Keyboard Interrept Detected. Aborting Task Processing.')
                sys.exit()
            except:
                print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                print("Unable to process subtask {} of task {}; session {}".format(subtask_id, task_num, session_name))
                traceback.print_exc(file=sys.stdout)
                # sys.exit('Error processing task {} of {}'.format(taskID,taskFile))

    elif task_num == -1:
        post_process_channel_table(subject_id, task_table)
    else:
        sys.exit('Invalid task number.')
