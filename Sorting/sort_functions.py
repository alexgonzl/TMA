import sys
import os
import time
import warnings

import traceback
import signal as sg

from pathlib import Path
import pickle

import scipy
import numpy as np
import pandas as pd

import spikesorters as ss
import spikeextractors as se
import spiketoolkit as st

from spikemetrics.metrics import isi_violations

import Pre_Processing.pre_process_functions as pp

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette='muted')

icPath = '/home/alexgonzalez/Documents/MATLAB/ironclust/matlab/'
ks2Path = '/home/alexgonzalez/Documents/MATLAB/Kilosort2/'

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        sg.signal(sg.SIGALRM, self.handle_timeout)
        sg.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        sg.alarm(0)


def sort_main(task, overwrite_flag=0):
    try:
        save_path = Path(task['save_path'], task['task_type'])

        if (not (save_path / 'recording.dat').exists()) or overwrite_flag:
            # load task data
            data = np.load(task['file_path'])
            with open(task['file_header_path'], 'rb') as f:
                data_info = pickle.load(f)

            # prepare filter
            sos, _ = pp.get_sos_filter_bank(['Sp'], fs=data_info['fs'])
            spk_data = np.zeros_like(data)
            assert data_info['n_chans'] == spk_data.shape[0], "Inconsistent formating in the data files. Aborting."

            # spk filter (high pass)
            t0 = time.time()
            for ch in range(data_info['n_chans']):
                spk_data[ch] = scipy.signal.sosfiltfilt(sos, data[ch])
                print('', end='.')
            t1 = time.time()
            print('\nTime to spk filter data {0:0.2f}s'.format(t1 - t0))

            chan_masks = pp.create_chan_masks(data_info['Raw']['ClippedSegs'], data_info['n_samps'])
            chan_mad = pp.get_signals_mad(spk_data, chan_masks)
            data_info['Spk'] = {'mad': chan_mad}

            # convert data to spikeinterface format
            spk_data_masked = se.NumpyRecordingExtractor(timeseries=spk_data * chan_masks, geom=data_info['tt_geom'],
                                                         sampling_frequency=data_info['fs'])

            # sort data
            sort = sort_data(spk_data_masked, save_path, sorter=task['task_type'])
            if sort is not None:
                # export data to phy
                st.postprocessing.export_to_phy(recording=spk_data_masked, sorting=sort,
                                                output_folder=str(save_path),
                                                compute_pc_features=False, compute_amplitudes=False,
                                                max_channels_per_template=4)

                # get cluster stats
                spk_times_list = sort.get_units_spike_train()
                cluster_stats = get_cluster_stats(spk_times_list, spk_data_masked.get_traces(), data_info)
                cluster_stats_file_path = Path(save_path, 'cluster_stats.csv')
                cluster_stats.to_csv(cluster_stats_file_path)

                print('downSuccessful sort.')
            else:
                print('Uncesseful sort.')

            # save header
            updated_file_header_path = Path(task['save_path'], Path(task['file_header_path']).name)
            with updated_file_header_path.open(mode='wb') as file_handle:
                pickle.dump(data_info, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            print('Sorting Done and overwrite flag is False, skipping this sort.')
    except KeyboardInterrupt:
        print('Keyboard Interrupt Detected. Aborting Task Processing.')
        sys.exit()

    except:
        print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
        traceback.print_exc(file=sys.stdout)


def sort_data(recording, store_dir, sorter='KS2', dthr=5):
    valid_sorters = ['MS4', 'KS2', 'SC', 'IC', 'HS', 'TDC', 'Klusta']

    # the timeout is the greater between 3 mins or 10 percent of the recording in seconds
    timeout_dur = int(np.max([0.1 * recording.get_num_frames() / recording.get_sampling_frequency(), 180]))

    out = None
    if sorter in valid_sorters:
        t0 = time.time()
        if sorter == 'MS4':
            params = ss.Mountainsort4Sorter.default_params()
            params['detect_threshold'] = dthr
            params['curation'] = True
            params['whiten'] = False
            algorithm = 'mountainsort4'
            algo_class = ss.Mountainsort4Sorter
        elif sorter == 'IC':
            ss.IronClustSorter.set_ironclust_path(icPath)
            params = ss.IronClustSorter.default_params()
            params['detect_threshold'] = dthr
            params['filter'] = False
            algorithm = 'ironclust'
            algo_class = ss.IronClustSorter
        elif sorter == 'KS2':
            ss.Kilosort2Sorter.set_kilosort2_path(ks2Path)
            params = ss.Kilosort2Sorter.default_params()
            params['detect_threshold'] = dthr
            params['car'] = False
            algorithm = 'kilosort2'
            algo_class = ss.Kilosort2Sorter
        elif sorter == 'SC':
            params = ss.SpykingcircusSorter.default_params()
            params['filter'] = False
            params['detect_threshold'] = dthr
            params['num_workers'] = 16
            algorithm = 'spykingcircus'
            algo_class = ss.SpykingcircusSorter
        elif sorter == 'HS':
            params = ss.HerdingspikesSorter.default_params()
            params['pca_whiten'] = False
            params['filter'] = False
            algorithm = 'herdingspikes'
            algo_class = ss.HerdingspikesSorter
        elif sorter == 'TDC':
            params = ss.TridesclousSorter.default_params()
            params['relative_threshold'] = dthr
            algorithm = 'tridesclous'
            algo_class = ss.TridesclousSorter
        elif sorter == 'Klusta':
            params = ss.KlustaSorter.default_params()
            params['num_starting_clusters'] = 20
            algorithm = 'klusta'
            algo_class = ss.KlustaSorter
        else:
            print('Sorter Error.')
            return None

        sort_folder = str(store_dir) + '/tmp_' + sorter
        try:
            with timeout(seconds=timeout_dur):
                print()
                print('Sorting with {}'.format(sorter))
                out = ss.run_sorter(sorter_name_or_class=algo_class, recording=recording,
                                    output_folder=sort_folder,
                                    delete_output_folder=True, raise_error=True, verbose=False, **params)
                t1 = time.time()
                print('Time to sort using {0} = {1:0.2f}s'.format(sorter, t1 - t0))
                print('Found {} clusters'.format(len(out.get_unit_ids())))

        except TimeoutError:
            print()
            print('Sorter {} Timeout.'.format(sorter))
            out = None

        except PermissionError:
            print()
            print('Permission Error. Trying to run algorithm locally.')
            try:
                exfile = 'sh {}/run_{}.sh'.format(sort_folder, algorithm)
                os.system(exfile)
                tmp = algo_class(recording=recording, output_folder=sort_folder)
                out = tmp.get_result_from_folder(sort_folder)
                t1 = time.time()
                print('Time to sort using {0} = {1:0.2f}s'.format(sorter, t1 - t0))
                print('Found {} clusters'.format(len(out.get_unit_ids())))
            except:
                traceback.print_exc(file=sys.stdout)
                out = None

        except KeyboardInterrupt:
            print()
            print('Keyboard Interrupt. Sorting is aborted.')
            out = None

        except:
            print()
            print('Error running sorter {}'.format(sorter))
            print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
            traceback.print_exc(file=sys.stdout)
            out = None

    else:
        print()
        print('Invalid Sorter'.format(sorter))
        out = None

    return out


def get_cluster_snr(spike_train, signals, sig_mad=None):
    """
    Inputs:
    spike_train -> array of samples indicating the peak of the detected spike
    hp_signals -> high passed signal for spike detection
    sig_mad -> the median absolute deviation for each channel, if not passed, it will be computed

    Returns:
    snr -> float indicating the SNR (higher number the better.)
    amp_mad -> measure of the stability of the amplitudes (lower the number the better)
    """
    amps = signals[:, spike_train]  # amplitudes of the spikes
    median_amps = np.abs(np.median(amps, axis=1))  # median amplitude of spikes for all channels

    max_amp_ch = np.argmax(median_amps)  # chan with max median amp
    max_amp = np.max(median_amps)  # value of the max median amp

    amp_mad = pp.get_signals_mad(amps[max_amp_ch])  # deviation of the median amp

    # get the mad of channel with max amplitude
    if sig_mad is None:
        sig_mad_max_ch = pp.rs.mad(signals[max_amp_ch])
    else:
        sig_mad_max_ch = sig_mad[max_amp_ch]

    snr = max_amp / sig_mad_max_ch  # signal to noise ratio

    return snr, max_amp_ch, max_amp, amp_mad


def get_sorter_cluster_snr(sorter, hp_signals, sig_mad=None, mask=None):
    cluster_ids = sorter.get_unit_ids()
    n_clusters = len(cluster_ids)

    cl_snrs = {}
    cl_amp_mad = {}
    for cl in cluster_ids:
        spike_train = sorter.get_unit_spike_train(cl)
        cl_snrs[cl], cl_amp_mad[cl], _, _ = get_cluster_snr(spike_train, hp_signals, sig_mad=sig_mad, mask=mask)
    return cl_snrs, cl_amp_mad


def get_cluster_stats(spike_times_dict, hp_data, data_info, sorter_id='KS2',
                      save_path=None, snr_thr=5, fr_low_thr=0.25, fr_high_thr=90, isi_thr=0.001):
    cluster_stats = pd.DataFrame()
    cluster_num = 0

    #units = sort_results.get_unit_ids()
    units = list(spike_times_dict.keys())

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        if len(units) > 0:
            for unit in units:
                key = sorter_id + '_' + str(unit)
                cluster_stats.loc[key, 'cl_num'] = int(unit)
                cluster_stats.loc[key, 'sorter'] = sorter_id

                # sort_results.get_unit_spike_train(unit)
                unit_spk_train = spike_times_dict[unit]
                n_spikes = len(unit_spk_train)

                cluster_stats.loc[key, 'fr'] = n_spikes / data_info['n_samps'] * data_info['fs']

                dur = data_info['n_samps'] / data_info['fs']
                isi_viol_rate, n_isi_viol = isi_violations(unit_spk_train / data_info['fs'], dur, isi_thr)

                cluster_stats.loc[key, 'isi_viol_rate'] = isi_viol_rate
                cluster_stats.loc[key, 'n_spikes'] = int(n_spikes)
                cluster_stats.loc[key, 'n_isi_viol'] = int(n_isi_viol)

                cluster_stats.loc[key, 'snr'], cluster_stats.at[key, 'amp_ch'], cluster_stats.at[key, 'amp_med_ch'], \
                cluster_stats.loc[key, 'amp_mad_ch'] = \
                    get_cluster_snr(unit_spk_train, hp_data, sig_mad=data_info['Spk']['mad'])

                cluster_num += 1

        valid_cluster_flag = cluster_stats['fr'] >= fr_low_thr
        valid_cluster_flag &= cluster_stats['fr'] <= fr_high_thr
        valid_cluster_flag &= cluster_stats['snr'] >= snr_thr
        cluster_stats['valid'] = valid_cluster_flag
        cluster_stats['tt_num'] = data_info['tt_num']

        if save_path is not None:
            cluster_stats_file_path = Path(save_path, 'cluster_stats.csv')
            cluster_stats.to_csv(cluster_stats_file_path)

    return cluster_stats


def save_sorted(sort, high_passed_data, output_folder):
    st.postprocessing.export_to_phy(recording=high_passed_data, sorting=sort,
                                    output_folder=str(output_folder),
                                    compute_pc_features=False, compute_amplitudes=False,
                                    max_channels_per_template=4)


def get_waveforms(spk_train, hp_data, wf_lims=None, n_wf=1000):
    if wf_lims is None:
        wf_lims = [-12, 20]
    n_spikes = len(spk_train)
    n_chans = hp_data.shape[0]

    sampled_spikes = spk_train[np.random.randint(n_spikes, size=n_wf)]
    wf_samps = np.arange(wf_lims[0], wf_lims[1])

    mwf = np.zeros((n_chans, len(wf_samps)))
    for samp_spk in sampled_spikes:
        mwf += hp_data[:, wf_samps + samp_spk]
    mwf /= n_wf

    return mwf


def load_hp_binary_data(file_name, n_channels=4, data_type=np.float16):
    """
    function to load high pass binary data (usually from KS temporary output recording.dat)
    :param file_name: string or posix path
    :param n_channels: numbe r
    :param data_type: dtype for loading data, default np.float16
    :return: n
    """
    ### bug found on 7/27/2022. recording.dat is not in the right format, hence loading in this manner produces garbage.
    ### workaround is to use SubjectInfo Class and the get_tt_data() function to load the data,
    #  and then use _spk_filter_data() to get the high pass data.
    raise NotImplementedError

    data = np.fromfile(file_name, dtype=data_type)
    data = data.reshape(-1, n_channels).T
    return data


def plot_isi_dist(spk_train, fs=32000, dt=2, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    xlims = [0, 50]

    isi = np.diff(spk_train) / fs * 1000
    bins = np.arange(0, xlims[1], dt)
    ax = sns.distplot(isi, bins=bins, norm_hist=False, kde=True, kde_kws={'bw': dt, 'lw': 3, 'clip': xlims})
    ax.set_xlim(xlims)
    ax.axvline(0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('ISI [ms]')

    return ax
