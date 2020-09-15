import sys
import datetime
import getopt
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def session_entry(session_name, session_path, files, save_path):
    return {'session_name': str(session_name), 'session_path': str(session_path), 'files': files, 'n_files': len(files),
            'save_path': str(save_path)}


def dict_entry(session_name, task_type, file_path, file_header_path, save_path, tt_id=-1):
    return {'session': str(session_name), 'task_type': task_type, 'file_path': str(file_path),
            'file_header_path': str(file_header_path), 'tt_id': str(tt_id), 'save_path': str(save_path)}


if __name__ == '__main__':
    # Store taskID and TaskFile
    subject_id = ''
    volume_path = ''
    save_path = ''
    sorter = 'KS2'
    # defaults for a tetrode recording
    tetrode_recording = True
    n_channels = 4
    n_tetrodes = 16
    data_format = '.npy'

    myopts, args = getopt.getopt(sys.argv[1:], "a:d:s:o:p:")
    for opt, attr in myopts:
        if opt == '-a':
            subject_id = str(attr)
            print('Subject ID: {}'.format(subject_id))
        elif opt == '-d':
            volume_path = Path(str(attr))
            print('Data Path: {}'.format(str(volume_path)))
        elif opt == '-o':
            save_path = Path(str(attr))
            print('Save Path: {}'.format(str(save_path)))
        elif opt == '-s':
            if str(attr) in ['KS2', 'SC', 'IC', 'MS4']:
                sorter = str(attr)
                print('Selected Sorter: {}'.format(sorter))
            else:
                sys.exit('Invalid Sorter {}.'.format(attr))
        elif opt == '-p':
            if str(attr) == 'NR32':
                tetrode_recording = False
                n_channels = 32
                n_tetrodes = -1
            elif str(attr) == 'TT16':
                # defaults
                pass
            else:
                sys.exit('Invalid Probe Type.')
        else:
            print("Usage: {} -a ID -d 'path/to/data' -o 'output/directory' -s 'sorter' - ".format(sys.argv[0]))
            sys.exit('Invalid input. Aborting.')

    if (subject_id == '') or (volume_path == '') or (save_path == ''):
        print("Usage: {} -a ID -s 'sorter' -d 'path/to/data' -o 'output/directory' -p 'probetye' ".format(sys.argv[0]))
        print('Example: {} -a Li -s KS2 -d /Data/Pre_Processed/ -o /Data/Sorted/'.format(sys.argv[0]))
        sys.exit('Invalid input.')

    volume_path = volume_path / subject_id
    save_path = save_path / subject_id

    if not volume_path.exists():
        sys.exit('could not find data folder')

    # get sessions with good channels
    chan_table_file = (volume_path / 'chan_table_{}.csv'.format(subject_id))
    if chan_table_file.exists():
        chan_table = pd.read_csv(str(chan_table_file), header=0, index_col=0)
    else:
        sys.exit('Channel Table for {} not found in the data directory. Aborting.'.format(subject_id))

    if tetrode_recording:
        # Sessions with at least one tetrodes that has at least one good channel:
        session_list = list(chan_table[(chan_table > 0).sum(axis=1) > 0].index)
    else:
        sys.exit('Approach has not been developed for non tetrode files.')

    date_obj = datetime.date.today()
    date_str = "%s_%s_%s" % (date_obj.month, date_obj.day, date_obj.year)

    Sessions = {}
    session_cnt = 0
    sort_cnt = 0
    if tetrode_recording:
        tetrodes = np.array(chan_table.columns)

        for session in session_list:
            session_cnt += 1
            print('Collecting Info for Session # {}, {}'.format(session_cnt, session))

            # tetrodes with good channels
            good_tetrodes = tetrodes[(chan_table.loc[session] > 0)]
            Files = {}
            task_id = 1
            session_folder = volume_path / session
            save_folder = save_path / session
            try:
                for tt in good_tetrodes:
                    tt_file = session_folder / ('tt_' + tt + data_format)
                    tt_info_file = session_folder / ('tt_' + tt + '_info.pickle')
                    sp = save_folder / ('tt_'+tt)
                    sp.mkdir(parents=True, exist_ok=True)

                    if tt_file.exists():
                        Files[task_id] = dict_entry(session, sorter, tt_file, tt_info_file, sp, tt_id=tt)
                        task_id += 1
                        sort_cnt += 1

                if len(Files) > 0:
                    Sessions[session_cnt] = session_entry(session, session_folder, Files, save_folder)
                else:
                    print('Empty Session {}, discarding.'.format(str(session)))
                    session_cnt -= 1
            except:
                print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                traceback.print_exc(file=sys.stdout)
                continue

    print('Number of Sessions to be sorted = {}'.format(session_cnt))
    print('Total Number of sorts = {}.'.format(sort_cnt))

    task_table_dir = save_path / 'TasksDir'
    task_table_dir.mkdir(parents=True, exist_ok=True)
    with open(str(task_table_dir) + '/sort_{}_{}.json'.format(subject_id, sorter), 'w') as f:
        json.dump(Sessions, f, indent=4)

    task_table_dir = volume_path / 'TasksDir'
    task_table_dir.mkdir(parents=True, exist_ok=True)
    with open(str(task_table_dir) + '/sort_{}_{}.json'.format(subject_id, sorter), 'w') as f:
        json.dump(Sessions, f, indent=4)
