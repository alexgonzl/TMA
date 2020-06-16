import sys
import datetime
import getopt
import json
import numpy as np
from pathlib import Path

MINFILESIZE = 16384


def session_entry(session_name, session_folder, files, sp, n_SubSessions, valid_SubSessions):
    return {'session_name': str(session_name), 'session_folder': str(session_folder), 'files': files,
            'n_files': len(Files), 'sp': sp, 'n_SubSessions': n_SubSessions,
            'valid_SubSessions': valid_SubSessions}


def dict_entry(session_name, task_type, fn, sp, subSessionID='0000', tt=-1):
    if task_type == 'tt':
        return {'session': session_name, 'task_type': task_type, 'filenames': fn, 'tt_id': str(tt), 'sp': sp,
                'subSessionID': subSessionID}
    else:
        return {'session': session_name, 'task_type': task_type, 'filenames': fn, 'sp': sp,
                'subSessionID': subSessionID}


if __name__ == '__main__':
    # Store taskID and TaskFile
    subject_id = ''
    volume_path = ''
    save_path = ''
    n_tetrodes = 16
    n_channels = 4
    Tetrode_Recording = True

    if len(sys.argv) < 6:
        print(
            "Usage: %s -a AnimalID -d 'Volume/path/to/folders' -o 'volume/path/to/results/' -p probetype" % sys.argv[0])
        print("Example: %s -a Li -d '/Data/Raw/' -o '/Data/Pre_Processed/' -p 'TT16' " % sys.argv[0])
        print('attributes -a, -v, -s are required.')
        sys.exit('Invalid input.')

    myopts, args = getopt.getopt(sys.argv[1:], "a:d:o:p:")

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
        elif opt == '-p':
            if str(attr) == 'NR32':
                Tetrode_Recording = False
                n_channels = 32
            elif str(attr) == 'TT16':
                # defaults
                pass
            else:
                sys.exit('Unsupported Probe Type.')
        else:
            print("Usage: %s -a AnimalID -d 'Volume/path/to/folders' -o output/folder/" % sys.argv[0])
            sys.exit('Invalid input. Aborting.')

    volume_path = volume_path / subject_id
    save_path = save_path / subject_id

    if not volume_path.exists():
        sys.exit('could not find data folder')

    date_obj = datetime.date.today()
    date_str = "%s_%s_%s" % (date_obj.month, date_obj.day, date_obj.year)

    Sessions = {}
    SessionCnt = 0
    for session in volume_path.glob('*_*[0-9]'):
        SessionCnt += 1
        print('Collecting Info for Session # {}, {}'.format(SessionCnt, session.name))
        sp0 = Path(save_path, str(session.name))
        sp_dict = {'0000': str(sp0)}
        Files = {}
        nSubSessions = len(list(session.glob('VT1*.nvt')))
        validSubSessions = []
        taskID = 1
        try:
            # valid vt
            for vt in session.glob('*.nvt'):
                try:
                    if vt.stat().st_size > MINFILESIZE:
                        if vt.match('VT1.nvt'):
                            Files[taskID] = dict_entry(session.name, 'vt', str(vt), str(sp0))
                            taskID += 1
                        else:
                            for ss in np.arange(1, nSubSessions):
                                ss_str = str(ss).zfill(4)
                                if vt.match('VT1_{}.nvt'.format(ss_str)):
                                    validSubSessions.append(ss_str)
                                    sp_dict[ss_str] = str(Path(save_path, str(session.name)+'_'+ss_str))
                                    Files[taskID] = dict_entry(session.name, 'vt', str(vt), sp_dict[ss_str], subSessionID=ss_str)
                                    taskID += 1
                except:
                    print('Could not assign task to {}'.format(vt))
                    print(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                    continue

            # valid events
            for ev in session.glob('*.nev'):
                try:
                    if ev.stat().st_size > MINFILESIZE:
                        if ev.match('Events.nev'):
                            Files[taskID] = dict_entry(session.name, 'ev', str(ev), str(sp0) )
                            taskID +=1
                        else:
                            for ss_str in validSubSessions:
                                if ev.match('Events_{}.nev'.format(ss_str)):
                                    Files[taskID] = dict_entry(session.name, 'ev', str(ev), sp_dict[ss_str], subSessionID=ss_str)
                                    taskID += 1
                except:
                    print('Could not assign task to {}'.format(ev))
                    print(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                    continue

            # Look for valid records e.g. CSC1d.ncs
            if Tetrode_Recording:
                for tt in np.arange(1, n_tetrodes + 1):
                    TT = []
                    chAbsentFlag = False
                    for ch in ['a', 'b', 'c', 'd']:
                        csc = 'CSC{}{}.ncs'.format(tt, ch)
                        try:
                            if not (session / csc).exists():
                                chAbsentFlag = True
                            elif not (session / csc).stat().st_size > MINFILESIZE:
                                chAbsentFlag = True
                            else:
                                chAbsentFlag = False
                                TT.append(str(session / csc))
                        except:
                            chAbsentFlag = True
                            print('Could not assign task for tt {} chan {}'.format(tt, ch))
                            print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                            continue

                    if not chAbsentFlag:
                        Files[taskID] = dict_entry(session.name, 'tt', TT, str(sp0), tt=tt)
                        taskID += 1

                    for ss_str in validSubSessions:
                        TT = []
                        chAbsentFlag = False
                        for ch in ['a', 'b', 'c', 'd']:
                            csc = 'CSC{}{}_{}.ncs'.format(tt, ch, ss_str)
                            try:
                                if not (session / csc).exists():
                                    chAbsentFlag = True
                                elif not (session / csc).stat().st_size > MINFILESIZE:
                                    chAbsentFlag = True
                                else:
                                    chAbsentFlag = False
                                    TT.append(str(session / csc))
                            except:
                                chAbsentFlag = True
                                print('Could not assign task for tt {} chan {}'.format(tt, ch))
                                print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                                continue
                        if not chAbsentFlag:
                            Files[taskID] = dict_entry(session.name, 'tt', TT, sp_dict[ss_str], subSessionID=ss_str, tt=tt)
                            taskID += 1

            else:
                # untested code below. ag 6/9/20
                for ss in np.arange(nSubSessions):
                    Probe = []
                    chAbsentFlag = False
                    for ch in np.arange(1, n_channels + 1):
                        try:
                            if ss == 0:
                                csc = 'CSC{}.ncs'.format(ch)
                            else:
                                csc = 'CSC{}_{}.ncs'.format(ch, str(ss).zfill(4))

                            if not (session / csc).exists():  # file does not exists
                                chAbsentFlag = True
                            elif not (session / csc).stat().st_size > MINFILESIZE:  # file exists but it is empty
                                chAbsentFlag = True
                            else:  # file exists and its valid
                                chAbsentFlag = False
                                Probe.append(str(session / csc))
                        except:
                            chAbsentFlag = True
                            # print('Could not assign task to {}'.format(csc))
                            print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                            continue
                    if not chAbsentFlag:
                        Files[taskID] = dict_entry(session.name, 'probe', Probe, sp0, subSessionID=str(ss).zfill(4))
                        taskID += 1

            if len(Files) > 0:
                Sessions[SessionCnt] = session_entry(session.name, str(session), Files, sp_dict, nSubSessions,
                                                     validSubSessions)
                for ss, sp in sp_dict.items():
                    Path(sp).mkdir(parents=True, exist_ok=True)
            else:
                print('Empty Session {}, discarding.'.format(str(session)))
                SessionCnt -= 1
        except:
            print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
            continue

    print('Number of Sessions to be proccess = {}'.format(SessionCnt))

    task_table_dir = save_path / 'TasksDir'
    task_table_dir.mkdir(parents=True, exist_ok=True)
    with (task_table_dir / ('pp_table_{}.json'.format(subject_id))).open(mode='w') as f:
        json.dump(Sessions, f, indent=4)

    task_table_dir = volume_path / 'TasksDir'
    task_table_dir.mkdir(parents=True, exist_ok=True)
    with (task_table_dir / ('pp_table_{}.json'.format(subject_id))).open(mode='w') as f:
        json.dump(Sessions, f, indent=4)
