import numpy as np
from scipy import stats, spatial

from pathlib import Path
import json
import pickle as pkl
from .subject_info import SubjectInfo

# last edit: 8.4.20 -ag


def get_session_spikes(subject_info, session, overwrite=0, return_numpy=True, rej_thr=None):
    """
    Wrapper function to obtain all the spikes from all the curated clusters for the specified session.
    Function will also save the outputs in the predefined paths found in subject_info.
    :param SubjectInfo subject_info: instance of class SubjectInfo for a particular subject
    :param str session:  session id
    :param bool overwrite: if true overwrites. ow. returns disk data
    :param bool return_numpy: if true, returns clusters as a numpy array of spike trains, and dict of ids.
    :param float rej_thr: currently not in use.
    :return: np.ndarray spikes: object array containing spike trains per cluster
    :return: dict tt_cl: [for return_numpy] dictinonary with cluster keys and identification for each cluster
    """
    session_paths = subject_info.session_paths[session]
    params = subject_info.params
    params['spk_wf_dist_rej_thr'] = rej_thr  # not used

    # spike_buffer in seconds
    # ignore spikes at the edges of recording
    if 'spk_recording_buffer' not in params:
        params['spk_recording_buffer'] = 3

    if (not session_paths['Cell_Spikes'].exists()) | overwrite:
        print('Spikes Files not Found or overwrite=1, creating them.')

        clusters = subject_info.clusters[session]
        spikes = {'Cell': {'n_units': 0}, 'Mua': {'n_units': 0}}
        wfi = {'Cell': {}, 'Mua': {}}
        for tt in clusters['curated_TTs']:
            unit_ids = {'Cell': {}, 'Mua': {}}
            n_tt_units = 0
            for ut in ['Cell', 'Mua']:
                try:  # when loading subject_info keys are strings
                    unit_ids[ut] = clusters[ut.lower() + '_IDs'][str(tt)]
                except KeyError:  # if creating new dict, keys can be integers not strings
                    unit_ids[ut] = clusters[ut.lower() + '_IDs'][tt]
                n_tt_units += len(unit_ids[ut])

            # if there are clusters, load data and get the spikes
            if n_tt_units > 0:
                tt_dat = np.load(session_paths['PreProcessed'] / 'tt_{}.npy'.format(tt))
                sort_dir = subject_info.get_session_sorted_tt_dir(session, tt)
                spk_times = np.load(sort_dir / 'spike_times.npy')
                cluster_spks = np.load(sort_dir / 'spike_clusters.npy')
                for ut in ['Cell', 'Mua']:
                    spikes[ut], wfi[ut] = get_tt_spikes(tt, unit_ids[ut], tt_dat, spk_times,
                                                        cluster_spks, params, spikes[ut], wfi[ut])

        # save Cell/Mua spike dictionaries and waveform info
        for ut in ['Cell', 'Mua']:
            with session_paths[ut + '_Spikes'].open(mode='w') as f:
                json.dump(spikes[ut], f, indent=4)
            with session_paths[ut + '_WaveFormInfo'].open(mode='wb') as f:
                pkl.dump(wfi[ut], f, pkl.HIGHEST_PROTOCOL)

        # convert spike dictionaries to numpy and a json dict with info
        cell_spikes, cell_tt_cl = get_spikes_numpy(spikes['Cell'])
        mua_spikes, mua_tt_cl = get_spikes_numpy(spikes['Mua'])
        spikes_numpy, tt_cl = aggregate_spikes_numpy(cell_spikes, cell_tt_cl, mua_spikes, mua_tt_cl)

        # save numpy spikes
        np.save(session_paths['cluster_spikes'], spikes_numpy)
        with session_paths['cluster_spikes_ids'].open(mode='w') as f:
            json.dump(tt_cl, f, indent=4)

    else:  # Load data.
        if return_numpy:
            spikes_numpy = np.load(session_paths['cluster_spikes'])
            with session_paths['cluster_ids'].open() as f:
                tt_cl = json.load(f)
        else:
            with session_paths['Cell_Spikes'].open() as f:
                cell_spikes = json.load(f)
            with session_paths['Mua_Spikes'].open() as f:
                mua_spikes = json.load(f)
            spikes = {'Cell': cell_spikes, 'Mua': mua_spikes}

    if return_numpy:
        return spikes_numpy, tt_cl
    else:
        return spikes


def get_tt_spikes(tt, unit_ids, tt_dat, spk_times, cluster_spks, params, spikes, wfi):
    """
    Returns spikes and waveform information for the clusters in the tetrode.
    :param int tt:
    :param list unit_ids:
    :param np.array float16 tt_dat: n_chans x n_samps
    :param np.array ints spk_times: ordered spikes for all clusters
    :param np.array ints cluster_spks: cluster id of each spk in spk_times
    :param dict params:
    :param dict spikes: dictionary of spikes by tetrode and cluster
    :param dict wfi: waveform info by tetrode
    :return: dict spikes
    :return: dict wfi
    """
    spike_buffer = params['spk_recording_buffer']
    samp_rate = params['samp_rate']

    spk_buffer_int = int(spike_buffer * samp_rate)
    cnt = spikes['n_units']
    spikes[str(tt)] = {}
    for cl_id in unit_ids:
        n_samps = tt_dat.shape[1]

        allspikes = spk_times[cluster_spks == cl_id].flatten()
        spikes2 = np.array(allspikes)

        # delete spikes in at beginning and end of recording.
        unit_ids = (spikes2 + spk_buffer_int) < n_samps
        spikes2 = spikes2[unit_ids]
        unit_ids = (spikes2 - spk_buffer_int) > 0
        spikes2 = spikes2[unit_ids]
        spikes2 = spikes2.astype(np.int)

        wf = get_waveforms(spikes2, tt_dat)

        wfi[cnt] = get_waveform_info(spikes2, wf, int(n_samps - 2 * spk_buffer_int), samp_rate)
        wfi[cnt]['tt'] = tt
        wfi[cnt]['cl'] = cl_id

        spikes[str(tt)][str(cl_id)] = spikes2.tolist()
        cnt += 1

    spikes['n_units'] = cnt
    return spikes, wfi


def aggregate_spikes_numpy(cell_spikes, cell_tt_cl, mua_spikes, mua_tt_cl):
    """
    Wrapper function for get_spikes_numpy
    Deals with Cell and Mua subcategories
    :param cell_spikes: numpy output of get_spikes_numpy
    :param cell_tt_cl: dict output of get_spikes_numpy
    :param mua_spikes: numpy output of get_spikes_numpy
    :param mua_tt_cl: dict output of get_spikes_numpy
    :returns: spikes. single object array containing both cell and mua spikes
    :returns: tt_cl. single dict containing the indices and identification info for each cluster
    """
    spikes = np.concatenate((cell_spikes, mua_spikes))
    n_cell_units = len(cell_spikes)
    n_mua_units = len(mua_spikes)
    n_units = n_cell_units + n_mua_units
    tt_cl = {}
    for unit in range(n_cell_units):
        tt_cl[unit] = ('Cell',) + cell_tt_cl[unit]
    for unit in range(n_mua_units):
        tt_cl[unit + n_cell_units] = ('Mua',) + mua_tt_cl[unit]

    return spikes, tt_cl


def get_spikes_numpy(spikes_dict):
    """
    :param spikes_dict: dictionary of spikes
        contains n_units and tetrodes, each tetrode is a dict of clusters
    :return:
        spikes: a numpy object array of length n_units, each element is a spike train
        tt_cl: a dict with cluster number in the spikes array as keys and tt,cl as values
    """
    n_units = spikes_dict['n_units']
    spikes2 = np.empty(n_units, dtype=object)
    tt_cl = {}
    cnt = 0
    for tt in spikes_dict.keys():
        if tt == 'n_units':
            continue
        for cl, spks in spikes_dict[tt].items():
            spikes2[cnt] = np.array(spks).astype(np.int32)
            tt_cl[cnt] = tt, cl
            cnt += 1

    return spikes2, tt_cl


def get_waveform_info(spikes, waveforms, n_samps, samp_rate):
    """
    :param np.array int spikes:
    :param np.ndarray float16 waveforms:
    :param int n_samps:
    :param int samp_rate:
    :return dict wfi: waveform information dictionary
    """
    waveforms = waveforms.astype(np.float32)
    n_spk = len(spikes)
    wfi = {'mean': np.nanmean(waveforms, axis=0),
           'std': np.nanstd(waveforms, axis=0),
           'sem': stats.sem(waveforms, axis=0),
           'nSp': n_spk,
           't_stat': stats.ttest_1samp(waveforms, 0, axis=0)[0],
           'm_fr': n_spk / n_samps * samp_rate}

    # isi in ms
    isi = np.diff(spikes) / samp_rate * 1000
    wfi['isi_h'] = np.histogram(isi, bins=np.linspace(-1, 20, 25))
    wfi['cv'] = np.std(isi) / np.mean(isi)
    return wfi


def get_wf_samps(spikes):
    """
    Returns a matrix of samples centered on each spike for creating waveform plots
    :param spikes: spikes 1d np.array of integers indicating indices of spikes
    :return: np.ndarray n_spikes x 64 of integers
    """

    a = np.zeros((len(spikes), 64), dtype=np.int)
    cnt = 0
    for s in spikes:
        a[cnt] = s + np.arange(64) - 32
        cnt += 1
    return a


def get_waveforms(spikes, data):
    """
    Returns waveforms for each spikes and for each channel [can be a big array]
    :param np.array int spikes:
    :param np.ndarray float16 data: size n_channels x n_samps
    :returns: np.ndarray float16 n_spikes x 64 x n_chans
    """
    wf_samps = get_wf_samps(spikes)
    return np.moveaxis(data[:, wf_samps], 0, -1)


def get_wf_outliers(waveforms, thr=None):
    ###### NOT WORKING #####
    # waveforms = nSpikes x 64 x 4 np.array
    nF = 64 * 4
    nSp = waveforms.shape[0]
    X = np.reshape(waveforms, (nSp, nF))

    Xm = np.mean(X, 0)
    Y = np.zeros(nSp)
    for s in np.arange(nSp):
        Y[s] = spatial.distance.braycurtis(Xm, X[s])
    badSpikes = Y > thr
    # pca = PCA(n_components=2)
    # pca.fit(X)
    # lls = pca.score_samples(X)
    # badSpikes = np.abs(robust_zscore(lls))>thr
    return None

# def getSessionBinSpikes(sessionPaths, overwrite=0, resamp_t=None, cell_spikes=None, mua_spikes=None):
#     #print(sessionPaths['Cell_Bin_Spikes'])
#     if (not sessionPaths['Cell_Bin_Spikes'].exists()) or (overwrite):
#
#         print('Binned Spikes Files not Found or overwrite=1, creating them.')
#         if (cell_spikes is None) or (mua_spikes is None):
#             cell_spikes, mua_spikes = getSessionSpikes(sessionPaths, overwrite=overwrite)
#         if resamp_t is None:
#             print('Missing resampled time input (resamp_t).')
#             PosDat = TMF.getBehTrackData(sessionPaths, overwrite=0)
#             resamp_t = PosDat['t']
#             del PosDat
#
#         cell_bin_spikes,cell_ids = bin_TT_spikes(cell_spikes,resamp_t,origSR=sessionPaths['SR'])
#         mua_bin_spikes,mua_ids = bin_TT_spikes(mua_spikes,resamp_t,origSR=sessionPaths['SR'])
#
#         ids = {}
#         ids['cells'] = cell_ids
#         ids['muas'] = mua_ids
#
#         np.save(sessionPaths['Cell_Bin_Spikes'],cell_bin_spikes)
#         np.save(sessionPaths['Mua_Bin_Spikes'],mua_bin_spikes)
#
#         with sessionPaths['Spike_IDs'].open(mode='w') as f:
#             json.dump(ids,f,indent=4)
#         print('Bin Spike File Creation and Saving Completed.')
#
#     else:
#         print('Loading Binned Spikes...')
#         cell_bin_spikes=np.load(sessionPaths['Cell_Bin_Spikes'])
#         mua_bin_spikes=np.load(sessionPaths['Mua_Bin_Spikes'])
#         with sessionPaths['Spike_IDs'].open() as f:
#             ids = json.load(f)
#         print('Binned Spike Files Loaded.')
#
#     return cell_bin_spikes, mua_bin_spikes, ids
#
# def getSessionFR(sessionPaths,overwrite=0,cell_bin_spikes=None,mua_bin_spikes=None):
#     if (not sessionPaths['Cell_FR'].exists()) | overwrite:
#         print('Firing Rate Files Not Found or overwrite=1, creating them.')
#
#         if ((cell_bin_spikes is None) or (mua_bin_spikes is None)):
#             cell_bin_spikes, mua_bin_spikes, ids = getSessionBinSpikes(sessionPaths)
#
#         nCells,nTimePoints = cell_bin_spikes.shape
#         cell_FR = np.zeros((nCells,nTimePoints))
#         for cell in np.arange(nCells):
#             cell_FR[cell] = smoothSpikesTrain(cell_bin_spikes[cell])
#
#         nΜUAs,nTimePoints = mua_bin_spikes.shape
#         mua_FR = np.zeros((nΜUAs,nTimePoints))
#         for cell in np.arange(nΜUAs):
#             mua_FR[cell] = smoothSpikesTrain(mua_bin_spikes[cell])
#
#         np.save(sessionPaths['Cell_FR'],cell_FR)
#         np.save(sessionPaths['Mua_FR'],mua_FR)
#
#         print('Spike File Creation and Saving Completed.')
#     else:
#         print('Loading FRs ...')
#         cell_FR=np.load(sessionPaths['Cell_FR'])
#         mua_FR=np.load(sessionPaths['Mua_FR'])
#
#         print('FR Loaded.')
#     return cell_FR, mua_FR
#
#
# def bin_TT_spikes(spikes,resamp_t,origSR=32000):
#     orig_time = np.arange(resamp_t[0],resamp_t[-1],1/origSR)
#     step = resamp_t[1]-resamp_t[0]
#     nOrigTimePoints = len(orig_time)
#     nTimePoints = len(resamp_t)
#     sp_bins = np.zeros((spikes['nUnits'],nTimePoints))
#     sp_ids = {}
#     cnt = 0
#     for tt,cl_ids in spikes.items():
#         if tt!='nUnits':
#             for cl in cl_ids:
#                 try:
#                     sp = np.array(spikes[tt][cl])
#                     out_of_record_spikes = sp>=nOrigTimePoints
#                     if np.any(out_of_record_spikes):
#                         sp = np.delete(sp,np.where(out_of_record_spikes)[0])
#                     sp_ids[cnt] = (tt,cl)
#                     #print(type(sp[0]))
#                     sp_bins[cnt],_ = np.histogram(orig_time[sp],np.concatenate([resamp_t,[resamp_t[-1]+step]]))
#                 except:
#                     print("Error processing Tetrode {}, Cluster {}".format(tt,cl))
#                     pass
#                 cnt+=1
#
#     return sp_bins,sp_ids
#
# def smoothSpikesTrain(bin_spikes,step=0.02):
#     lfwin = np.round(1.0/step).astype(int)
#     return signal.filtfilt(np.ones(lfwin)/lfwin,1,bin_spikes/step)
#
# def getSpikeList(spikes,ids,cell_num):
#     t,c = ids[str(cell_num)]
#     return spikes[str(t)][str(c)]
