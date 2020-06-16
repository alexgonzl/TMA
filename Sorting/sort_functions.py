import sys
import os
import time

import traceback
import signal as sg

from pathlib import Path
import h5py
import json
import pickle

import scipy
import numpy as np
import pandas as pd

import spikesorters as ss

import Pre_Processing.pre_process_functions as pp

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette='muted')


# cwd = Path(os.getcwd())
# pkg_dir = cwd.parent
# sys.path.append(str(pkg_dir))

icPath = '/home/alexgonzalez/Documents/MATLAB/ironclust/'
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


def get_cluster_snr(spike_train, hp_signals, sig_mad=None, mask=None):
    """
    Inputs:
    spike_train -> array of samples indicating the peak of the detected spike
    hp_signals -> high passed signal for spike detection
    sig_mad -> the median absolute deviation for each channel, if not passed, it will be computed
    mask -> mask of valid values for the hp_signal

    Returns:
    snr -> float indicating the SNR (higher number the better.)
    amp_mad -> measure of the stability of the amplitudes (lower the number the better)
    """
    amps = hp_signals[:, spike_train]  # amplitdes of the spikes
    median_amps = np.abs(np.median(amps, axis=1))  # median amplitude of spikes for all channels

    max_amp_ch = np.argmax(median_amps)  # chan with max median amp
    max_amp = np.max(median_amps)  # value of the max median amp

    amp_mad = pp.get_signals_mad(amps[max_amp_ch])  # deviation of the median amp

    if sig_mad is None:
        sig_mad = pp.rs.mad(hp_signals[max_amp_ch], mask[max_amp_ch])
    else:
        sig_mad = sig_mad[max_amp_ch]

    snr = max_amp / sig_mad  # signal to noise ratio

    return snr, max_amp_ch, max_amp, amp_mad


def get_sorter_cluster_snr(sorter, hp_signals, sig_mad=None, mask=None):
    cluster_ids = sorter.get_unit_ids()
    n_clusters = len(cluster_ids)

    cl_snrs = {}
    cl_amp_mad = {}
    for cl in cluster_ids:
        spike_train = sorter.get_unit_spike_train(cl)
        cl_snrs[cl], cl_amp_mad[cl] = get_cluster_snr(spike_train, hp_signals, sig_mad=sig_mad, mask=mask)
    return cl_snrs, cl_amp_mad


def coeff_variation_ln(isi):
    sln = np.std(np.log(isi))
    return np.sqrt(np.exp(sln ** 2) - 1)


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


def sort_data(recording, store_dir, sorter='KS2', dthr=5):
    valid_sorters = ['MS4', 'KS2', 'SC', 'IC', 'HS', 'TDC', 'Klusta']

    # the timeout is the greater between 3 mins or 10 percent of the recording in seconds
    timeout_dur = int(np.max([0.1 * recording.get_num_frames() / recording.get_sampling_frequency(), 180]))
    out = {}

    if sorter in valid_sorters:
        t0 = time.time()
        if sorter is 'MS4':
            params = ss.Mountainsort4Sorter.default_params()
            params['detect_threshold'] = dthr
            params['curation'] = True
            params['whiten'] = False
            algorithm = 'mountainsort4'
            algo_class = ss.Mountainsort4Sorter
        elif sorter is 'IC':
            ss.IronClustSorter.set_ironclust_path(icPath)
            params = ss.IronClustSorter.default_params()
            params['detect_threshold'] = dthr
            params['filter'] = False
            algorithm = 'ironclust'
            algo_class = ss.IronClustSorter
        elif sorter is 'KS2':
            ss.Kilosort2Sorter.set_kilosort2_path(ks2Path)
            params = ss.Kilosort2Sorter.default_params()
            params['detect_threshold'] = dthr
            params['car'] = False
            algorithm = 'kilosort2'
            algo_class = ss.Kilosort2Sorter
        elif sorter is 'SC':
            params = ss.SpykingcircusSorter.default_params()
            params['filter'] = False
            params['detect_threshold'] = dthr
            params['num_workers'] = 16
            algorithm = 'spykingcircus'
            algo_class = ss.SpykingcircusSorter
        elif sorter is 'HS':
            params = ss.HerdingspikesSorter.default_params()
            params['pca_whiten'] = False
            params['filter'] = False
            algorithm = 'herdingspikes'
            algo_class = ss.HerdingspikesSorter
        elif sorter is 'TDC':
            params = ss.TridesclousSorter.default_params()
            params['relative_threshold'] = dthr
            algorithm = 'tridesclous'
            algo_class = ss.TridesclousSorter
        elif sorter is 'Klusta':
            params = ss.KlustaSorter.default_params()
            params['num_starting_clusters'] = 20
            algorithm = 'klusta'
            algo_class = ss.KlustaSorter
        else:
            print('Unknow Sorter. Aborting')
            return None

        sort_folder = str(store_dir) + '/tmp_' + sorter
        try:
            with timeout(seconds=timeout_dur):
                print()
                print('Sorting with {}'.format(sorter))
                out[sorter] = ss.run_sorter(sorter_name_or_class=algo_class, recording=recording,
                                            output_folder=sort_folder,
                                            delete_output_folder=True, raise_error=True, verbose=False, **params)
                t1 = time.time()
                print('Time to sort using {0} = {1:0.2f}s'.format(sorter, t1 - t0))
                print('Found {} clusters'.format(len(out[sorter].get_unit_ids())))

        except TimeoutError:
            print()
            print('Sorter {} Timeout.'.format(sorter))
            out[sorter] = None

        except PermissionError:
            print()
            print('Permission Error. Trying to run algorithm locally.')
            try:
                exfile = ('sh {}/run_{}.sh').format(sort_folder, algorithm)
                os.system(exfile)
                tmp = algo_class(recording=recording, output_folder=sort_folder)
                out[sorter] = tmp.get_result_from_folder(sort_folder)
                t1 = time.time()
                print('Time to sort using {0} = {1:0.2f}s'.format(sorter, t1 - t0))
                print('Found {} clusters'.format(len(out[sorter].get_unit_ids())))
            except:
                traceback.print_exc(file=sys.stdout)
                out[sorter] = None

        except KeyboardInterrupt:
            print()
            print('Keyboard Interrupt. Sorting is aborted.')
            out[sorter] = None

        except:
            print()
            print('Error running sorter {}'.format(sorter))
            print("Error", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
            traceback.print_exc(file=sys.stdout)
            out[sorter] = None

    else:
        print()
        print('Invalid Sorter'.format(sorter))
        out[sorter] = None

    return out


def get_cluster_stats(sort_results, spk_signals, data_info, snr_thr=5, fr_low_thr=0.5, fr_high_thr=90):
    cluster_stats = pd.DataFrame()
    cluster_num = 0
    for sorter_id, sort in sort_results.items():
        if sort is not None:
            units = sort.get_unit_ids()
            if len(units) > 0:
                for unit in units:
                    key = sorter_id + '_' + str(unit)
                    cluster_stats.at[key, 'cl_num'] = cluster_num
                    cluster_stats.at[key, 'sorter'] = sorter_id
                    cluster_stats.at[key, 'sorter_cl_num'] = unit

                    unit_spk_train = sort.get_unit_spike_train(unit)
                    n_spikes = len(unit_spk_train)

                    cluster_stats.at[key, 'fr'] = n_spikes / data_info['n_samps'] * data_info['fs']

                    isi = np.diff(unit_spk_train) / data_info['fs'] * 1000
                    c, b = np.histogram(isi, np.arange(50))
                    isi_viol = np.sum(c[b[:-1] <= 2]) / n_spikes
                    cv = coeff_variation_ln(isi)

                    cluster_stats.at[key, 'isi_viol'] = isi_viol
                    cluster_stats.at[key, 'cv'] = cv
                    cluster_stats.at[key, 'n_spikes'] = n_spikes

                    cluster_stats.at[key, 'snr'], cluster_stats.at[key, 'amp_ch'], cluster_stats.at[key, 'amp_med_ch'],
                    cluster_stats.at[key, 'amp_mad_ch'] \
                        = get_cluster_snr(unit_spk_train, spk_signals, sig_mad=data_info['Spk']['mad'], mask=tt_masks)

                    cluster_num += 1

    valid_cluster_flag = cluster_stats['fr'] >= fr_low_thr
    valid_cluster_flag &= cluster_stats['fr'] <= fr_high_thr
    valid_cluster_flag &= cluster_stats['snr'] >= snr_thr
    cluster_stats['valid'] = valid_cluster_flag

    return cluster_stats


def get_waveforms(spk_train, hp_data, wf_lims=[-12, 20], n_wf=1000):
    n_spikes = len(spk_train)
    n_chans = hp_data.shape[0]

    sampled_spikes = spk_train[np.random.randint(n_spikes, size=n_wf)]
    wf_samps = np.arange(wf_lims[0], wf_lims[1])

    mwf = np.zeros((n_chans, len(wf_samps)))
    for samp_spk in sampled_spikes:
        mwf += hp_data[:, wf_samps + samp_spk]
    mwf /= n_wf

    return mwf
