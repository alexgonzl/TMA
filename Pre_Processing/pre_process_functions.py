import sys
import os
from datetime import datetime
import traceback
import time

from pathlib import Path
import h5py
import pickle

import nept
import scipy
import numpy as np
import pandas as pd
from fooof import FOOOF

# cwd = Path(os.getcwd())
# pkg_dir = cwd.parent
# sys.path.append(str(pkg_dir))

import Utils.robust_stats as rs

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette='muted')

########################################################################################################################
########################################################################################################################
############################################### Main Funcs #############################################################
########################################################################################################################
########################################################################################################################


def process_tetrode(task, save_format='npy', overwrite_flag=0):
    # things to be saved: processed signal, tetrode info
    chan_files = task['filenames']
    session = task['session']
    sp = Path(task['sp'])  # save path

    # Unpack the probe information.
    tt_id = task['tt_id']
    fn = 'tt_{}'.format(tt_id)
    fn_info = fn + '_info.pickle'.format(tt_id)

    if not ((sp / (fn + '.' + save_format)).exists() and (sp/fn_info).exists()) or overwrite_flag:
        tt_info = get_tt_info(task)

        # pre-allocate data
        fs = tt_info['fs']
        n_chans = tt_info['n_chans']

        sig, time_samps = get_csc(str(chan_files[0]))
        n_samps = len(sig)
        tt_info['n_samps'] = n_samps
        tt_info['tB'] = time_samps[0]
        tt_info['tE'] = time_samps[-1]
        del time_samps  # time stamps can be recreated with gathered information.

        raw_signals = np.empty((n_chans, n_samps))
        raw_signals[0] = sig
        for ch, chf in enumerate(chan_files[1:]):
            raw_signals[ch + 1], _ = get_csc(str(chf))
        del sig

        tt_info['Raw'] = get_signal_info(raw_signals, tt_info, get_clipped_segs =True)

        chan_th = ~np.isnan(tt_info['Raw']['PSD_table']['th_pk'].values.astype(float))
        chan_60 = ~np.isnan(tt_info['Raw']['PSD_table']['60_pk'].values.astype(float))
        chan_clp = tt_info['Raw']['PctChanClipped'] > tt_info['bad_chan_thr']
        chan_code = chan_th * 4 + chan_60 * 2 + chan_clp

        tt_info['chan_code'] = chan_code
        tt_info['bad_chans'] = np.logical_and(chan_code > 0, chan_code < 4)

        SOS, _ = get_sos_filter_bank(['HP', 'LP', 'Notch'], fs=fs)
        f_signals = np.zeros_like(raw_signals)
        t0 = time.time()
        for ch in range(n_chans):
            f_signals[ch] = scipy.signal.sosfiltfilt(SOS, raw_signals[ch])
            print('', end='.')
        t1 = time.time()
        print('\nTime to filter tetrode {0:0.2f}s'.format(t1 - t0))
        del raw_signals

        f_signals = f_signals.astype(np.float16)
        save_probe(f_signals, sp, fn, save_format)

        # save tetrode info
        with (sp/fn_info).open(mode='wb') as f_handle:
            pickle.dump(tt_info, f_handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_video(task, overwrite_flag=0):
    raw_vt_file = task['filenames']
    sp = Path(task['sp'])
    vt_file = 'vt.h5'

    if not (sp / vt_file).exists() or overwrite_flag:
        t, x, y, ha = get_position(raw_vt_file)
        with h5py.File(str(sp / vt_file), 'w') as hf:
            hf.create_dataset("t", data=t)
            hf.create_dataset("x", data=x)
            hf.create_dataset("y", data=y)
            hf.create_dataset("ha", data=ha)
    else:
        print('File exists and overwrite = false ')


def process_events(task, overwrite_flag=0):
    try:
        raw_ev_file = task['filenames']
        sp = Path(task['sp'])
        ss = task['subSessionID']
        evFile = 'ev.h5'

        if not (sp / evFile).exists() or overwrite_flag:
            ev = get_events(raw_ev_file)
            with h5py.File(str(sp / evFile), 'w') as hf:
                for k, v in ev.items():
                    hf.create_dataset(k, data=v)
        else:
            print('File exists and overwrite = false ')
    except:
        print("Error processing events. ", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)


def post_process_channel_table(subject_id, task_table):
    # get mapping for session and task id
    session2task = {}
    dates = {}
    for k in task_table.keys():
        session = Path(task_table[k]['session_name']).name
        session2task[session] = k
    session_list = list(session2task.keys())

    # sort sessions by date
    session_dates = pd.DataFrame()
    for session in session_list:
        try:
            session_dates.at[session, 'date'] = datetime.strptime(session.split('_')[2], '%m%d%y')
        except:
            pass

    sorted_sessions = list(session_dates.sort_values(by='date').index)

    channel_table = pd.DataFrame(columns=range(1, 17))

    # obtain results for each session
    for session_date in sorted_sessions:
        task_id = session2task[session_date]
        valid_session = task_table[task_id]['n_files'] >= 18

        if valid_session:
            # processed data paths
            for subsession_id, subsession_path in task_table[task_id]['sp'].items():
                session_name = Path(subsession_path).name
                channel_table.loc[session_name] = -1
                for tt in range(1, 17):
                    tt_info_file = Path(subsession_path, 'tt_{}_info.pickle'.format(tt))
                    if tt_info_file.exists():
                        with tt_info_file.open(mode='rb') as f:
                            tt_info = pickle.load(f)
                        channel_table.at[session_name, tt] = (~tt_info['bad_chans']).sum()
    channel_table = channel_table.apply(pd.to_numeric)

    save_path = Path(subsession_path).parent
    if save_path.exists():
        channel_table.to_csv(str(save_path / ('chan_table_{}.csv'.format(subject_id))))

########################################################################################################################
########################################################################################################################
############################################### Auxiliary Funcs ########################################################
########################################################################################################################
########################################################################################################################


def save_probe(data, save_path, fn, save_format):
    if save_format == 'h5':  # h5 format
        with h5py.File(str(save_path / (fn + '.' + save_format)), 'w') as hf:
            hf.create_dataset("tetrode", data=data)
    elif save_format == 'npy':  # numpy
        np.save(str(save_path / (fn + '.' + save_format)), data)
    elif save_format == 'csv':  # comma separeted values
        np.savetxt(str(save_path / (fn + '.' + save_format)), data, delimiter=',')
    elif save_format == 'bin':  # binary
        data.tofile(str(save_path / (fn + '.' + save_format)))
    else:
        print('Unsuported save method specified {}, saving as .npy array.'.format(save_format))
        np.save(str(save_path / (fn + '.' + save_format)), data)

    print('{}: results saved to {}'.format(fn, str(save_path)))
    print('')


def get_session_info(session):
    return session.split('_')


def get_tt_info(task):
    """

    Parameters
    ----------
    files -> Path, directory for the  data
    ttNum -> tetrode number.

    Returns
    -------
    dictionary with tetrode header information for each channel

    """
    n_chans = 4
    ads = np.zeros(n_chans)
    ref_chan = np.zeros(n_chans)
    chan_ids = np.zeros(n_chans)
    input_range = np.zeros(n_chans)

    # get headers:
    try:
        headers = []
        for chf in task['filenames']:
            headers.append(get_header(str(chf)))
    except:
        print('Header Files could not be loaded. Error reading files from')
        return []
    data_dir = Path(chf).parent

    fs = headers[0]['fs']
    for ch, header in enumerate(headers):
        ads[ch] = header['AD']
        ref_chan[ch] = header['RefChan']
        chan_ids[ch] = header['ChanID']
        input_range[ch] = header['InputRange']

    tt_geom = np.zeros([4, 2])
    tt_geom[1] = [0, 20]
    tt_geom[2] = [20, 0]
    tt_geom[3] = [20, 20]

    return {'data_dir': str(data_dir), 'session': task['session'], 'fs': fs, 'tt_num': task['tt_id'],
            'n_chans': n_chans,
            'chan_files': task['filenames'], 'a_ds': ads, 'ref_chan': ref_chan, 'chan_ids': chan_ids,
            'input_range': input_range, 'tt_geom': tt_geom, 'bad_chan_thr': 0.2}


def get_amp_hist(signals, tt_info, bin_step=50):
    """

    Parameters
    ----------
    signals -> numpy array of nchans x nsamples
    tt_info -> dictionary containing the header information for the channels
    bin_step -> amplitude binning step, 50 uV as default

    Returns
    -------
    amp_hist -> counts at each bin
    bin_centers -> center of amplitude bins

    """
    n_chans = tt_info['n_chans']

    max_amp = max(tt_info['input_range'])
    bins = np.arange(-max_amp, max_amp + 1, bin_step)
    bin_centers = bins[:-1] + bin_step / 2

    amp_hist = np.zeros((n_chans, len(bin_centers)))

    for ch in range(n_chans):
        amp_hist[ch], _ = np.histogram(signals[ch], bins)

    return amp_hist, bin_centers


def get_chans_psd(signals, fs, resamp=True):
    # find next power of 2 based on fs: e
    # for fs=32k, nperseg = 2**15 = 32768,
    # the operation belows computes this efficiently for arbitrary fs
    # assumes that sigs is nChans x nSamps

    if np.argmax(signals.shape) == 0:
        signals = signals.T
        transpose_flag = 1
    else:
        transpose_flag = 0

    fs = int(fs)
    nperseg = (1 << (fs - 1).bit_length())
    noverlap = 1 << int(fs * 0.05 - 1).bit_length()  # for at least 5% overlap.
    freqs, pxx = scipy.signal.welch(signals, fs=fs, nperseg=nperseg, noverlap=noverlap)

    if resamp:
        # resample log-linear
        samps = np.arange(100)

        maxExp = 4
        for e in np.arange(2, maxExp):
            samps = np.concatenate((samps, np.arange(10 ** e, 10 ** (e + 1), 10 ** (e - 1))))

        freqs = freqs[samps]
        if signals.ndim == 1:
            pxx = pxx[samps]
        else:
            pxx = pxx[:, samps]

        if transpose_flag:
            pxx = pxx.T

    return pxx, freqs


def get_tt_psd_peaks(freqs, pxx, theta_range=None):
    if theta_range is None:
        theta_range = [4, 12]
    n_chans = pxx.shape[0]
    sixty_range = [58, 62]

    df = pd.DataFrame(columns=['th_pk', 'th_amp', '60_pk', '60_amp', '1/f_r2', 'rmse'])
    for ch in range(n_chans):
        fm = FOOOF(max_n_peaks=2, peak_threshold=2.0, peak_width_limits=[0.5, 6.0], verbose=False)
        fm.fit(freqs, pxx[ch], [2, 100])

        pks = fm.peak_params_.flatten()[::3]
        amps = fm.peak_params_.flatten()[1::3]

        idx = (pks >= theta_range[0]) & (pks <= theta_range[1])
        theta_pk = pks[idx]
        theta_amp = amps[idx]

        idx = (pks >= sixty_range[0]) & (pks <= sixty_range[1])
        sixty_pk = pks[idx]
        sixty_amp = amps[idx]

        if len(theta_pk) == 1:
            df.at[ch, 'th_pk'] = np.around(theta_pk[0], decimals=2)
            df.at[ch, 'th_amp'] = np.around(theta_amp[0], decimals=2)
        elif len(theta_pk) > 1:
            df.at[ch, 'th_pk'] = np.around(np.mean(theta_pk), decimals=2)
            df.at[ch, 'th_amp'] = np.around(np.mean(theta_amp), decimals=2)

        if len(sixty_pk) == 1:
            df.at[ch, '60_pk'] = np.around(sixty_pk[0], decimals=2)
            df.at[ch, '60_amp'] = np.around(sixty_amp[0], decimals=2)
        elif len(sixty_pk) > 1:
            df.at[ch, '60_pk'] = np.around(np.mean(sixty_pk), decimals=2)
            df.at[ch, '60_amp'] = np.around(np.mean(sixty_amp), decimals=2)

        df.at[ch, 'rmse'] = np.around(fm.error_, decimals=3)
        df.at[ch, '1/f_r2'] = np.around(fm.r_squared_, decimals=3)
    return df


def get_clipped_segments(signals, tt_info, thr=0.99, sec_buffer=0.5):
    """
    function that takes signals and return segments of clipped signal buffered by fs*segBuffer.

    Inputs:
        signals -> nChans x nSamps np.array
        ttInfo -> dict, must contain nChans, fs, and InputRange (maxAmplitude)
        thr -> float, thr*InputRange is used threshold the signal
        segBuffer -> float, seconds to buffer the clipped signal, segments take the buffer into account

    Returns:
        Segs -> list of length nChans, each is a np.array of clipped segments for that channel (start and end indexes by # segments) in samples
        Durs -> list of length nChans, each element of the list is a np.array of length nSegs for that channel containing the durations in samples
    """

    n_chans = tt_info['n_chans']
    n_samps = tt_info['n_samps']
    fs = tt_info['fs']

    # binarize clipped segments
    samps_buffer = int(np.floor(fs * sec_buffer))

    Durs = [np.array([], dtype=int)] * n_chans
    Segs = [np.array([], dtype=int)] * n_chans
    for ch in range(n_chans):
        durs = np.array([], dtype=int)
        segs = np.array([], dtype=int)
        try:
            signal_mask = (np.abs(signals[ch]) >= tt_info['input_range'][ch] * thr).astype(int)
            diff_sig = np.concatenate(([0], np.diff(signal_mask)))
            idx_start = np.argwhere(diff_sig > 0).flatten()
            idx_end = np.argwhere(diff_sig < 0).flatten()

            # if idxStart and end match (ends>starts, #ends==#starts)
            if len(idx_start) > 0:
                if len(idx_start) == len(idx_end):
                    if np.all(idx_end > idx_start):
                        pass
                    else:  # if some reason some starts > ends..
                        print('Some start/end masking indices mismatch for ch{}'.format(ch))
                        continue  # channel loop

                # edge case of clipping near the end of the recording
                elif (len(idx_end) + 1) == len(idx_start):
                    if np.all(idx_end > idx_start[:-1]):
                        idx_end = np.concatenate((idx_end, np.array([n_samps])))
                    else:
                        print('start/end masking indices mismatch for ch{}'.format(ch))
                        continue  # channel loop
                # edge case of clipping at the beginning of recording
                elif (len(idx_start) + 1) == len(idx_end):
                    if np.all(idx_end[1:] > idx_start):
                        idx_start = np.concatenate(([0], idx_start))
                    else:
                        print('start/end masking indices mismatch for ch{}'.format(ch))
                        continue  # channel loop
                else:
                    print('unknwon error in masks for ch{}, debug independently.'.format(ch))
                    print(len(idx_start), len(idx_end))
                    continue  # channel loop

                # add seg_buffer for all segments
                idx_start = idx_start - samps_buffer
                idx_end = idx_end + samps_buffer

                # deal with start and end of recording
                ii = 0
                while True:
                    if idx_start[ii] - samps_buffer < 0:
                        idx_start[ii] = 0
                    else:
                        break  # while
                    ii += 1

                ii = 1
                while True:
                    if idx_end[-ii] + samps_buffer > n_samps:
                        idx_end[-ii] = n_samps
                    else:
                        break  # while
                    ii += 1

                # consolidate segments after the buffering
                cnt = 0
                seg_cnt = 0
                n_sub_segs = len(idx_start)
                segs = [(idx_start[0], idx_end[0])]

                # check if start of the next sub segment is inside the previous, and join if so
                while cnt < (n_sub_segs - 1):
                    while cnt < (n_sub_segs - 1):
                        if idx_start[cnt + 1] <= idx_end[cnt]:
                            segs[seg_cnt] = (segs[seg_cnt][0], idx_end[cnt + 1])
                            cnt += 1
                        else:
                            cnt += 1
                            break  # inner while

                    segs.append((idx_start[cnt], idx_end[cnt]))
                    seg_cnt += 1

                # convert to np arrays
                durs = np.array(([j - i for i, j in segs]))
                segs = np.array(segs)

        except:
            print('Channel {} Error in getting clipped segments:'.format(ch))
            print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
            traceback.print_exc(file=sys.stdout)

        # add channel to lists
        Durs[ch] = durs
        Segs[ch] = segs

    return Segs, Durs


def create_chan_masks(Segs, n_samps):
    n_chans = len(Segs)
    masks = np.ones((n_chans, n_samps), dtype=bool)
    for ch in range(n_chans):
        for seg in Segs[ch]:
            masks[ch][seg[0]:seg[1]] = False
    return masks


def get_sos_filter_bank(f_types, fs=32000.0, hp_edge_freq=None, lp_edge_freq=None, sp_edge_freq=None, notch_freq=None,
                        notch_harmonics=2, notch_q=20, gpass=0.2, gstop=60.0):
    """
    Function that creates default filters
    fTypes -> list of filters that have to be ['HP', 'LP', 'Notch', 'Sp'].
    fs -> integer, sampling frequency in samps/s defualt to 32000
    *_EdgeFreq -> edge frequencies for each filter type.
        Defaults are: HP-2Hz, LP-5000Hz, Sp->300Hz [High Pass], Notch->60Hz (3 Harmonics)
    Notch_Harmonics -> int, # of harmonics from Notch_Freq to filter [as default]
    gpass -> allowed oscillation gain in the pass bands [def. 0.2dB ~ to up to 1.03 multiplication of the pass band  ]
    gstop -> required suppresion in the stopband [def. 60dB ~ at least to 0.001 multiplication of the stop band  - ]
    returns SOS a N sections x 6 second order sections filter matrix.
    """

    SOS = np.zeros((0, 6))
    for f in f_types:
        if f not in ['HP', 'LP', 'Notch', 'Sp']:
            print('filter type {} not supported.'.format(f))
            print('skipping filter.')

        # settings for low pass and bandpass
        if f in ['LP', 'HP']:
            if f is 'LP':
                if lp_edge_freq is None:
                    cut_freq = 5000.0
                    cut_buffer = 5500.0
                else:
                    cut_freq = lp_edge_freq
                    cut_buffer = lp_edge_freq + lp_edge_freq * 0.1
            elif f is 'HP':
                if hp_edge_freq is None:
                    cut_freq = 2.0
                    cut_buffer = 0.2
                else:
                    cut_freq = hp_edge_freq
                    cut_buffer = hp_edge_freq * 0.1

            sos = scipy.signal.iirdesign(cut_freq / (fs / 2), cut_buffer / (fs / 2), gpass, gstop, output='sos')
            SOS = np.vstack((SOS, sos))

        if f is 'Notch':

            n_notches = notch_harmonics + 1

            if notch_freq is None:
                cut_freq = np.arange(1, n_notches + 1) * 60.0
            else:
                cut_freq = np.arange(1, n_notches + 1) * notch_freq

            if notch_q is None:
                q = np.array(cut_freq)  # changing Quality factor to keep notch bandwidth constant.

            elif type(notch_q) is np.ndarray:
                if len(notch_q) >= n_notches:
                    q = np.array(notch_q)
                # if length of quality factor array don't match the number of harmonics default to the first one
                elif len(notch_q) < n_notches:
                    q = np.ones(n_notches) * notch_q[0]
            else:
                # Q = np.ones(nNotches)*Notch_Q
                q = np.arange(1, n_notches + 1) * notch_q

            for i, notch in enumerate(cut_freq):
                b, a = scipy.signal.iirnotch(notch, q[i], fs=fs)
                sos = scipy.signal.tf2sos(b, a)
                SOS = np.vstack((SOS, sos))

        if f is 'Sp':
            if sp_edge_freq is None:
                cut_freq = 350.0
                cut_buffer = 300.0
            else:
                cut_freq = sp_edge_freq
                cut_buffer = sp_edge_freq - sp_edge_freq * 0.1
            sos = scipy.signal.iirdesign(cut_freq / (fs / 2), cut_buffer / (fs / 2), gpass, gstop, output='sos')
            SOS = np.vstack((SOS, sos))

    zi = scipy.signal.sosfilt_zi(SOS)
    return SOS, zi


def get_signal_info(signals, tt_info, get_hist=True, get_psd=True, get_clipped_segs=False):
    out = {}

    t0 = time.time()
    if get_hist:
        out['AmpHist'], out['HistBins'] = get_amp_hist(signals, tt_info)
        print('Time to get amplitude histograms = {}s'.format(time.time() - t0))

    t0 = time.time()
    if get_psd:
        out['PSD'], out['PSD_freqs'] = get_chans_psd(signals, tt_info['fs'], resamp=True)
        df = get_tt_psd_peaks(out['PSD_freqs'], out['PSD'])
        out['PSD_table'] = df
        print('Time to get compute power spectral densities = {}s'.format(time.time() - t0))

    t0 = time.time()
    if get_clipped_segs:
        Segs, Durs = get_clipped_segments(signals, tt_info)
        out['ClippedSegs'] = Segs
        out['ClippedDurs'] = Durs
        out['PctChanClipped'] = np.array([np.sum(d) / tt_info['n_samps'] for d in Durs])
        print('Time to get signal segment clips {}s'.format(time.time() - t0))

    return out


def get_signals_mad(signals, mask=None):
    if signals.ndim == 1:
        if mask is None:
            return rs.mad(signals)
        else:
            return rs.mad(signals[mask])
    elif signals.ndim > 1:
        n_chans, n_samps = signals.shape

    sig_mad = np.zeros(n_chans)
    if mask is None:
        for ch in range(n_chans):
            sig_mad[ch] = rs.mad(signals[ch])
    elif mask.shape[0] == n_chans:
        for ch in range(n_chans):
            sig_mad[ch] = rs.mad(signals[ch, mask[ch]])
    elif mask.ndim == 1 and len(mask) == n_samps:
        for ch in range(n_chans):
            sig_mad[ch] = rs.mad(signals[ch], mask)
    else:
        print('Mask does not match the data. Ignoring,')
        for ch in range(n_chans):
            sig_mad[ch] = rs.mad(signals[ch])

    return sig_mad


########################################################################################################################
########################################################################################################################
############################################### Neuralynx Read Funcs ###################################################
########################################################################################################################
########################################################################################################################

def get_csc(fn):
    ''' returns signal in uV and time stamps'''
    temp = nept.load_lfp(fn)
    return np.float32(temp.data.flatten() * 1e6), temp.time


def get_header(fn):
    h = nept.load_neuralynx_header(fn)
    for line in h.split(b'\n'):
        if line.strip().startswith(b'-ADBitVolts'):
            try:
                AD = np.array(float(line.split(b' ')[1].decode()))
            except ValueError:
                AD = 1
        if line.strip().startswith(b'-ReferenceChannel'):
            try:
                RefChan = line.split(b' ')[3].decode()
                RefChan = int(RefChan[:-2])
            except ValueError:
                RefChan = -1
        if line.strip().startswith(b'-SamplingFrequency'):
            try:
                fs = int(line.split(b' ')[1].decode())
            except ValueError:
                fs = 32000
        if line.strip().startswith(b'-ADChannel'):
            try:
                ChanID = int(line.split(b' ')[1].decode())
            except ValueError:
                ChanID = -1
        if line.strip().startswith(b'-InputRange'):
            try:
                InputRange = int(line.split(b' ')[1].decode())
            except ValueError:
                InputRange = -1

    header = {'AD': AD, 'RefChan': RefChan, 'fs': fs, 'ChanID': ChanID, 'InputRange': InputRange}

    return header


def get_position(fn):
    # Neuralynx files have a 16kbyte header
    # copy and modified from Nept Pckg.
    #

    f = open(fn, 'rb')
    header = f.read(2 ** 14).strip(b'\x00')

    # The format for .nvt files according the the neuralynx docs is
    # uint16 - beginning of the record
    # uint16 - ID for the system
    # uint16 - size of videorec in bytes
    # uint64 - timestamp in microseconds
    # uint32 x 400 - points with the color bitfield values
    # int16 - unused
    # int32 - extracted X location of target
    # int32 - extracted Y location of target
    # int32 - calculated head angle in degrees clockwise from the positive Y axis
    # int32 x 50 - colored targets using the same bitfield format used to extract colors earlier
    dt = np.dtype([('filler1', '<h', 3), ('time', '<Q'), ('points', '<i', 400),
                   ('filler2', '<h'), ('x', '<i'), ('y', '<i'), ('head_angle', '<i'),
                   ('targets', '<i', 50)])
    data = np.fromfile(f, dt)

    t = data['time'] * 1e-6
    x = np.array(data['x'], dtype=float)
    y = np.array(data['y'], dtype=float)
    ha = np.array(data['head_angle'], dtype=float)
    return t, x, y, ha


def get_events(fn):
    events = {'DE1': 'DE1', 'DE2': 'DE2', 'DE3': 'DE3', 'DE4': 'DE4', 'DE5': 'DE5', 'DE6': 'DE6',
              'L1': 'L1', 'L2': 'L2', 'L3': 'L3', 'L4': 'L4', 'L5': 'L5', 'L6': 'L6',
              'RD': 'RD', 'CL': 'CL', 'CR': 'CR', 'Start': 'Starting Recording', 'Stop': 'Stopping Recording'}

    ev = nept.load_events(fn, events)
    return ev


########################################################################################################################
########################################################################################################################
#########################################    Preprocessing Plottting Functions   #######################################
########################################################################################################################
########################################################################################################################

def plot_test_sos_filter(SOS, fs, cos_f=None, cos_a=None, noise=None):
    if noise is None:
        noise = [0.1]
    if cos_a is None:
        cos_a = [0.2, 0.25]
    if cos_f is None:
        cos_f = [60, 8]

    n_samps = int(fs * 5)
    t = np.arange(n_samps) / fs

    x = (np.arange(n_samps) < fs / 3) + ((np.arange(n_samps) > 1.5 * fs) & (np.arange(n_samps) < 2 * fs))
    for ii in range(len(cos_f)):
        x = x + cos_a[ii] * np.cos(2 * np.pi * cos_f[ii] * t)

    for jj in range(len(noise)):
        x = x + np.random.randn(n_samps) * noise[jj]

    x[x >= 1] = 1
    x[x <= -1] = -1

    xf = scipy.signal.sosfiltfilt(SOS, x)

    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    fig.tight_layout(pad=1.5)

    w, h = scipy.signal.sosfreqz(SOS,
                                 worN=np.concatenate((np.arange(0, 200, 1 / 2 ** 8), np.arange(200, 6000, 1 / 2 ** 8))),
                                 fs=fs)
    _ = plot_freq_response(w / fs * (2 * np.pi), h, fs, ax=ax2)

    ax1.plot(t, x, 'k-', label='x', alpha=0.5)
    ax1.plot(t, xf, alpha=0.75, linewidth=2, label='filtfilt')
    ax1.legend(loc='best', frameon=False)
    ax1.set_title('Test signal')
    ax1.set_xlabel(' time [s] ')

    pxx, freqs = get_chans_psd(x, fs, resamp=False)
    ax3.semilogx(freqs, 20 * np.log10(pxx))
    pxx, freqs = get_chans_psd(xf, fs, resamp=False)
    ax3.semilogx(freqs, 20 * np.log10(pxx))
    ax3.set_xlim([0.1, fs / 2])
    ax3.set_ylim([-200, 20])
    ax3.set_xlabel('Frequency Hz')


def plot_freq_response(w, h, fs, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.set_title('Digital filter frequency response')
    ax.semilogx(w / np.pi * (fs / 2), 20 * np.log10(abs(h)), 'b')
    ax.set_ylabel('Amplitude [dB]', color='b')
    ax.set_xlabel('Frequency Hz')
    ax.set_ylim([-120, 20])
    ax.set_xlim([0.1, fs / 2])

    return fig, ax


def plot_tt_psd(freqs, pxx, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    ls = ['-', '--', ':', '-.']
    n_chans = pxx.shape[0]
    for ch in range(n_chans):
        ax.semilogx(freqs, 20 * np.log10(pxx[ch]), alpha=0.75, linestyle=ls[ch], linewidth=3)
    ax.legend(['Ch{}'.format(ch) for ch in range(n_chans)], frameon=False)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels([1, 10, 100, 1000, 10000])

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r"PSD $\frac{V^2}{Hz}$")
    ax.set_ylim(-100, ax.get_ylim()[1])
    ax.set_title('Power Spectrum Density')
    return ax


def plot_tt_psd_table(freqs, pxx, df, ax=None):
    if ax is None:
        f, ax = plt.subplots(2, 1, figsize=(5, 5))

    ax[0] = plot_tt_psd(freqs, pxx, ax=ax[0])

    ax[1].axis('off')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    cl = prop_cycle.by_key()['color']
    t = pd.plotting.table(ax[1], df.round(2), loc='center', rowColours=cl)
    t.auto_set_font_size(False)
    t.set_fontsize(14)

    return ax


def plot_amp_hist(hist, bins, ax=None, **kwags):
    if ax is None:
        fig, ax = plt.subplots()
        ax = ax.flatten()
    else:
        ax.bar(bins, np.log10(hist), width=(bins[1] - bins[0]), **kwags)

    ax.set_xlabel(r'Amplitude $\mu V$')
    ax.set_ylabel(r'$log_{10}(nSamps)$')

    return ax


def plot_tt_amps_hists(hists, bin_centers, tt_info, ax=None):
    n_chans = tt_info['n_chans']
    xlim = np.max(tt_info['input_range']) * 1.1
    fs = tt_info['fs']
    ax_flag = 0
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax = ax.flatten()
    elif len(ax) != n_chans:
        print('Warning. # of signals mismatch # of given axes to plot historgrams, creating new figure.')
        fig, ax = plt.subplots((2, 2), figsize=(10, 8))
        ax = ax.flatten()
    else:
        ax_flag = 1

    prop_cycle = plt.rcParams['axes.prop_cycle']
    cl = prop_cycle.by_key()['color']
    for ch in range(n_chans):
        ax[ch] = plot_amp_hist(hists[ch], bin_centers, ax=ax[ch], color=cl[ch])
        ax[ch].set_title('Channel {}'.format(ch))
        if ax_flag == 0:
            if ch < 2:
                ax[ch].set_xlabel('')
        ax[ch].set_xlim(-xlim, xlim)
        ax[ch].axhline(np.log10(fs), color='k', linestyle='--', alpha=0.75)
    return ax


def plot_tt_summary(tt_info, sig_type='Raw'):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    ax = ax.flatten()

    _ = plot_tt_amps_hists(tt_info[sig_type]['AmpHist'], tt_info[sig_type]['HistBins'], tt_info, ax[[0, 1, 3, 4]])
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    for ch, ax_ii in enumerate([0, 1, 3, 4]):
        clp = tt_info['Raw']['PctChanClipped'][ch] * 100
        if clp > 25:
            ax[ax_ii].text(x=0.65, y=0.75, s='Clipped %{0:0.1f}'.format(clp), color='r', transform=ax[ax_ii].transAxes)
        else:
            ax[ax_ii].text(x=0.65, y=0.75, s='Clipped %{0:0.1f}'.format(clp), transform=ax[ax_ii].transAxes)
    _ = plot_tt_psd_table(tt_info[sig_type]['PSD_freqs'], tt_info[sig_type]['PSD'], tt_info[sig_type]['PSD_table'],
                          ax[[2, 5]])
    fig.tight_layout(pad=1.0)
    ax[5].set_position([0.66, 0.05, 0.33, 0.5])
    return fig, ax
