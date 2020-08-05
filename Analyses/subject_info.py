# creates a table with progress of analyses for each session
# update analyses table

from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
import sys
import traceback
import Analyses.spike_functions as sf

n_tetrodes = 16


##### TO DO!!! GET cluster stats for merged clusters #############


class SubjectInfo(object):
    def __init__(self, subject, sorter='KS2', data_root='BigPC', load=0, overwrite=0, time_step=0.02, samp_rate=32000):
        self.subject = subject
        self.sorter = sorter
        self.params = {'time_step': time_step, 'samp_rate': samp_rate, 'n_tetrodes': n_tetrodes,
                       'fr_temporal_smoothing': 0.125, 'spk_outlier_thr': None}

        if data_root == 'BigPC':
            if subject in ['Li', 'Ne']:
                self.root_path = Path('/Data_SSD2T/Data')
            elif subject in ['Cl']:
                self.root_path = Path('/Data2_SSD2T/Data')
            self.raw_path = Path('/RawData/Data', subject)
            self.sorted_path = self.root_path / 'Sorted' / subject
            self.results_path = self.root_path / 'Results' / subject
        elif data_root == 'oak':
            self.root_path = Path('/mnt/o/giocomo/alexg/')
            self.raw_path = self.root_path / 'RawData/InVivo' / subject
            self.sorted_path = self.root_path / 'Clustered' / subject
            self.results_path = self.root_path / 'Analyses' / subject
        self.preprocessed_path = self.root_path / 'PreProcessed' / subject

        self.data_paths_file = self.results_path / ('data_paths_{}_{}.pickle'.format(sorter, subject))
        # check if instance of DataPaths for subject and sorter exists already
        if load and not overwrite:
            self.load_subject_info()
        else:
            # get channel table
            self._channel_table_file = self.preprocessed_path / ('chan_table_{}.csv'.format(subject))
            if self._channel_table_file.exists():
                self.channel_table = pd.read_csv(self._channel_table_file, index_col=0)
                self.sessions = list(self.channel_table.index)
                self.n_sessions = len(self.sessions)

                self.session_paths = {}
                for session in self.sessions:
                    self.session_paths[session] = self.get_session_paths(session, time_step=time_step,
                                                                         samp_rate=samp_rate)

            # get cluster information
            try:
                self._clusters_file = self.sorted_path / ('clusters_{}_{}.json'.format(sorter, subject))
                if self._clusters_file.exists() and not overwrite:  # load
                    with self._clusters_file.open(mode='r') as f:
                        self.clusters = json.load(f)
                else:  # create
                    self.clusters = {}
                    for session in self.sessions:
                        self.clusters[session] = self.get_session_clusters(session)
                    with self._clusters_file.open(mode='w') as f:
                        json.dump(self.clusters, f, indent=4)
            except:
                print("Error with Clusters.")
                print(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                traceback.print_exc(file=sys.stdout)

            # get sorting tables
            try:
                self._sort_table_ids = ['tt', 'valid', 'curated', 'summary']
                self._sort_table_files = {ii: Path(self.sorted_path, 'sort_{}_{}_{}'.format(ii, sorter, subject)) for ii
                                          in
                                          self._sort_table_ids}
                if self._sort_table_files['summary'].exists() and not overwrite:
                    self.sort_tables = {ii: [] for ii in self._sort_table_ids}
                    for ii in self._sort_table_ids:
                        self.sort_tables[ii] = pd.read_csv(self._sort_table_files[ii], index_col=0)
                else:
                    self.sort_tables = self.get_sort_tables()
                    for ii in self._sort_table_ids:
                        self.sort_tables[ii].to_csv(self._sort_table_files[ii])
            except:
                print('Error with Tables')
                print(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                traceback.print_exc(file=sys.stdout)

            self.save_subject_info()

    def load_subject_info(self):
        with self.data_paths_file.open(mode='rb') as f:
            return pickle.load(f)

    def save_subject_info(self):
        with self.data_paths_file.open(mode='wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_session_paths(self, session):
        time_step = self.params['time_step']
        samp_rate = self.params['samp_rate']

        tmp = session.split('_')
        subject = tmp[0]
        assert subject == self.subject, 'Session does not match with expected animal.'

        task = tmp[1]
        date = tmp[2]

        paths = {'session': session, 'subject': subject, 'task': task, 'date': date, 'step': time_step, 'SR': samp_rate,
                 'Sorted': self.sorted_path / session, 'Raw': self.raw_path / session,
                 'PreProcessed': self.preprocessed_path / session, 'Results': self.results_path / session}

        paths['Results'].mkdir(parents=True, exist_ok=True)

        paths['BehavTrackDat'] = paths['Results'] / ('BehTrackVariables_{}ms.h5'.format(int(time_step * 1000)))

        # these paths are mostly legacy
        paths['Spike_IDs'] = paths['Results'] / 'Spike_IDs.json'
        for ut in ['Cell', 'Mua']:
            paths[ut + '_wf_info'] = paths['Results'] / (ut + '_wf_info.pkl')
            paths[ut + '_Spikes'] = paths['Results'] / (ut + '_Spikes.json')
            paths[ut + '_WaveForms'] = paths['Results'] / (ut + '_WaveForms.pkl')
            paths[ut + '_Bin_Spikes'] = paths['Results'] / ('{}_Bin_Spikes_{}ms.npy'.format(ut, int(time_step * 1000)))
            paths[ut + '_FR'] = paths['Results'] / ('{}_FR_{}ms.npy'.format(ut, int(time_step * 1000)))

        paths['cluster_spikes'] = paths['Results'] / 'spikes.npy'
        paths['cluster_spikes_ids'] = paths['Results'] / 'spikes_ids.json'
        paths['cluster_wf_info'] = paths['Results'] / 'wf_info.pkl'
        paths['cluster_binned_spikes'] = paths['Results'] / f'binned_spikes_{int(time_step*1000)}ms.npy'
        paths['cluster_fr'] = paths['Results'] / 'fr.npy'

        paths['ZoneAnalyses'] = paths['Results'] / 'ZoneAnalyses.pkl'

        paths['TrialInfo'] = paths['Results'] / 'TrInfo.pkl'
        paths['TrialCondMat'] = paths['Results'] / 'TrialCondMat.csv'
        paths['TrLongPosMat'] = paths['Results'] / 'TrLongPosMat.csv'
        paths['TrLongPosFRDat'] = paths['Results'] / 'TrLongPosFRDat.csv'
        paths['TrModelFits'] = paths['Results'] / 'TrModelFits.csv'
        paths['TrModelFits2'] = paths['Results'] / 'TrModelFits2.csv'

        paths['CueDesc_SegUniRes'] = paths['Results'] / 'CueDesc_SegUniRes.csv'
        paths['CueDesc_SegDecRes'] = paths['Results'] / 'CueDesc_SegDecRes.csv'
        paths['CueDesc_SegDecSumRes'] = paths['Results'] / 'CueDesc_SegDecSumRes.csv'
        paths['PopCueDesc_SegDecSumRes'] = paths['Results'] / 'PopCueDesc_SegDecSumRes.csv'

        # plots directories
        paths['Plots'] = paths['Results'] / 'Plots'
        paths['Plots'].mkdir(parents=True, exist_ok=True)
        paths['SampCountsPlots'] = paths['Plots'] / 'SampCountsPlots'
        paths['SampCountsPlots'].mkdir(parents=True, exist_ok=True)

        paths['ZoneFRPlots'] = paths['Plots'] / 'ZoneFRPlots'
        paths['ZoneFRPlots'].mkdir(parents=True, exist_ok=True)

        paths['ZoneCorrPlots'] = paths['Plots'] / 'ZoneCorrPlots'
        paths['ZoneCorrPlots'].mkdir(parents=True, exist_ok=True)
        paths['SIPlots'] = paths['Plots'] / 'SIPlots'
        paths['SIPlots'].mkdir(parents=True, exist_ok=True)

        paths['TrialPlots'] = paths['Plots'] / 'TrialPlots'
        paths['TrialPlots'].mkdir(parents=True, exist_ok=True)

        paths['CueDescPlots'] = paths['Plots'] / 'CueDescPlots'
        paths['CueDescPlots'].mkdir(parents=True, exist_ok=True)

        return paths

    def get_session_time_vectors(self, session, overwrite=False):
        _file_path = self.session_paths[session]['PreProcessed'] / 'time_vectors.npz'
        if _file_path.exists() and not overwrite:
            temp = np.load(_file_path)
            return temp['time_rsamp'], temp['time_orig']

        else:
            samp_rate = self.params['samp_rate']
            time_step = self.params['time_step']

            tt_info = self.get_session_tt_info(session, 1)

            n_samps = tt_info['n_samps']

            # get time vector with original sampling rate
            tB = tt_info['tB']
            tE = n_samps / samp_rate + tB
            time_orig = np.arange(tB, tE, 1 / samp_rate).astype(np.float32)

            # compute resampled time
            rsamp_rate = int(1 / time_step)
            n_rsamps = int(n_samps * rsamp_rate / samp_rate)
            trE = n_rsamps / rsamp_rate + tB
            time_rsamp = np.arange(tB, trE, 1 / rsamp_rate).astype(np.float32)

            # save & return
            np.savez(_file_path, time_rsamp=time_rsamp, time_orig=time_orig)
            return time_orig, time_rsamp

    def get_session_clusters(self, session):
        table = {'session': session, 'path': str(self.session_paths[session]['Sorted']),
                 'n_cell': 0, 'n_mua': 0, 'n_noise': 0, 'n_unsorted': 0, 'sorted_TTs': [], 'curated_TTs': [],
                 'cell_IDs': {}, 'mua_IDs': {}, 'noise_IDs': {}, 'unsorted_IDs': {}, 'clusters_snr': {},
                 'clusters_fr': {}, 'clusters_valid': {}, 'clusters_isi_viol_rate': {}}

        sort_paths = table['path']

        _cluster_stats = ['fr', 'snr', 'isi_viol_rate', 'valid']
        tetrodes = np.arange(1, n_tetrodes + 1)
        for tt in tetrodes:
            _cluster_groups_file = Path(sort_paths, ('tt_' + str(tt)), self.sorter, 'cluster_group.tsv')
            _cl_stat_file = Path(sort_paths, ('tt_' + str(tt)), self.sorter, 'cluster_stats.csv')
            if _cl_stat_file.exists():
                table['sorted_TTs'].append(int(tt))
                d = pd.read_csv(_cl_stat_file, index_col=0)
                n_clusters = d.shape[0]
                for st in _cluster_stats:
                    table['clusters_' + st][int(tt)] = {}
                    for cl in range(n_clusters):
                        if st == 'valid':
                            table['clusters_' + st][int(tt)][cl] = int(d.iloc[cl][st])
                        else:
                            table['clusters_' + st][int(tt)][cl] = np.around(d.iloc[cl][st], 2)

            if _cluster_groups_file.exists():
                d = pd.read_csv(_cluster_groups_file, delimiter='\t')
                try:
                    _group_types = ['good', 'mua', 'noise', 'unsorted']
                    _unit_types = ['cell', 'mua', 'noise', 'unsorted']
                    if any(d['group'] != 'unsorted'):
                        table['curated_TTs'].append(int(tt))

                    for ii, gt in enumerate(_group_types):
                        ut = _unit_types[ii]
                        table[ut + '_IDs'][int(tt)] = d['cluster_id'][d['group'] == gt].tolist()
                        table['n_' + ut] += len(table[ut + '_IDs'][int(tt)])
                except:
                    print('In Session {}, Error Processing TT {}'.format(session, tt))

        return table

    def get_sort_tables(self):
        sort_tables = {ii: pd.DataFrame(index=self.sessions,
                                        columns=range(1, n_tetrodes + 1)) for ii in ['tt', 'curated', 'valid']}

        sort_tables['summary'] = pd.DataFrame(index=self.sessions,
                                              columns=['n_tt', 'n_tt_sorted', 'n_tt_curated',
                                                       'n_valid_clusters', 'n_cell', 'n_mua',
                                                       'n_noise', 'n_unsorted'])
        #
        # for tbl in self._sort_table_ids:
        #     sort_tables[tbl] = sort_tables[tbl].fillna(-1)

        for session in self.sessions:
            _clusters_info = self.clusters[session]
            for tbl in self._sort_table_ids:
                if tbl == 'tt':
                    sort_tables[tbl].at[session, _clusters_info['sorted_TTs']] = 1
                if tbl == 'curated':
                    sort_tables[tbl].at[session, _clusters_info['curated_TTs']] = 1
                if tbl == 'valid':
                    _valid_cls = _clusters_info['clusters_valid']
                    for tt, cls in _valid_cls.items():
                        sort_tables[tbl].at[session, int(tt)] = len(cls)

            sort_tables['summary'].at[session, 'n_tt'] = n_tetrodes
            sort_tables['summary'].at[session, 'n_tt_sorted'] = len(_clusters_info['sorted_TTs'])
            sort_tables['summary'].at[session, 'n_tt_curated'] = len(_clusters_info['curated_TTs'])
            _unit_types = ['cell', 'mua', 'noise', 'unsorted']
            for ut in _unit_types:
                sort_tables['summary'].at[session, 'n_' + ut] = _clusters_info['n_' + ut]

        return sort_tables

    def get_session_sorted_tt_dir(self, session, tt):
        return self.session_paths[session]['Sorted'] / f'tt_{tt}' / self.sorter

    def get_session_tt_info(self, session, tt):
        with (self.session_paths[session]['PreProcessed'] / f'tt_{tt}_info.pickle').open(mode='rb') as f:
            return pickle.load(f)

    def get_session_tt_data(self, session, tt):
        return np.load(self.session_paths[session]['PreProcessed'] / f'tt_{tt}.npy')

    # methods from spike functions
    def get_session_spikes(self, session, return_numpy=True, save_spikes_dict=False, overwrite=False):
        return sf.get_session_spikes(self, session, return_numpy=return_numpy, save_spikes_dict=save_spikes_dict,
                                     rej_thr=self.params['spk_outlier_thr'], overwrite=overwrite)

    def get_session_binned_spikes(self, session, spike_trains=None, overwrite=False):
        return sf.get_session_binned_spikes(self, session, spike_trains=spike_trains, overwrite=overwrite)

    def get_session_fr(self, session, bin_spikes=None, overwrite=False):
        return sf.get_session_fr(self, session, bin_spikes=bin_spikes,
                                 temporal_smoothing=self.params['fr_temporal_smoothing'], overwrite=overwrite)


