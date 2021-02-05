from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
import h5py
import sys
import traceback
import Analyses.spike_functions as spike_funcs
import Analyses.open_field_functions as of_funcs
import Pre_Processing.pre_process_functions as pp_funcs
import Sorting.sort_functions as sort_funcs

import scipy.signal as signal

"""
Classes in this file will have several retrieval processes to acquire the required information for each
subject and session. 

:class SubjectInfo
    ->  class that takes a subject as an input. contains general information about what processes have been performed,
        clusters, and importantly all the session paths. The contents of this class are saved as a pickle in the results 
        folder.
    
:class SubjectSessionInfo
    ->  children class of SubjectInfo, takes session as an input. This class contains session specific retrieval methods
        Low level things, like reading position (eg. 'get_track_dat') are self contained in the class. Higher level 
        functions like 'get_spikes', are outsourced to the appropriate submodules in the Analyses folder. 
        If it is the first time calling a retrieval method, the call will save the contents according the paths variable 
        Otherwise the contents will be loaded from existing data, as opposed to recalculation. Exception is the get_time 
        method, as this is easily regenerated on each call.

- last edit: 8.6.20 -ag
"""


class SubjectInfo:
    def __init__(self, subject, sorter='KS2', data_root='BigPC', overwrite=False, time_step=0.02,
                 samp_rate=32000, n_tetrodes=16, fr_temporal_smoothing=0.125, spk_outlier_thr=None,
                 overwrite_cluster_stats=False):

        self.subject = subject
        self.sorter = sorter
        self.params = {'time_step': time_step, 'samp_rate': samp_rate, 'n_tetrodes': n_tetrodes,
                       'fr_temporal_smoothing': fr_temporal_smoothing, 'spk_outlier_thr': spk_outlier_thr,
                       'spk_recording_buffer': 3}
        self.tetrodes = np.arange(n_tetrodes)+1

        if data_root == 'BigPC':
            if subject in ['Li', 'Ne']:
                self.root_path = Path('/mnt/Data1_SSD2T/Data')
            elif subject in ['Cl']:
                self.root_path = Path('/mnt/Data2_SSD2T/Data')
            elif subject in ['Ca', 'Mi', 'Al']:
                self.root_path = Path('/mnt/Data3_SSD2T/Data')

            self.raw_path = Path('/mnt/RawData/Data', subject)

        elif data_root == 'oak':
            self.root_path = Path('/mnt/o/giocomo/alexg/')
            self.raw_path = self.root_path / 'RawData/InVivo' / subject
            # self.sorted_path = self.root_path / 'Clustered' / subject
            # self.results_path = self.root_path / 'Analyses' / subject
        else:
            self.root_path = Path(data_root)
            self.raw_path = self.root_path / 'RawData' / subject

        self.preprocessed_path = self.root_path / 'PreProcessed' / subject
        self.sorted_path = self.root_path / 'Sorted' / subject
        self.results_path = self.root_path / 'Results' / subject

        self.subject_info_file = self.results_path / ('subject_info_{}_{}.pkl'.format(sorter, subject))
        # check if instance of DataPaths for subject and sorter exists already
        if self.subject_info_file.exists() and not overwrite:
            self.load_subject_info()
        else:
            # get channel table
            self._channel_table_file = self.preprocessed_path / ('chan_table_{}.csv'.format(subject))
            if not self._channel_table_file.exists():
                _task_fn = self.preprocessed_path / 'TasksDir' / f"pp_table_{self.subject}.json"
                if _task_fn.exists():
                    with _task_fn.open(mode='r') as f:
                        _task_table = json.load(f)
                    pp_funcs.post_process_channel_table(self.subject, _task_table)
                else:
                    sys.exit(f"Error. Task table for pre-processing does not exists: {_task_fn}")

            self.channel_table = pd.read_csv(self._channel_table_file, index_col=0)

            # get sessions from channel table information
            self.sessions = list(self.channel_table.index)
            self.n_sessions = len(self.sessions)

            self.session_paths = {}
            for session in self.sessions:
                self.session_paths[session] = self._session_paths(session)

            # get cluster information
            try:
                if overwrite_cluster_stats:
                    # overwrite cluster stats & clusters tables
                    self.update_clusters()
                else:
                    # load tables
                    self.session_clusters = self.get_session_clusters(overwrite=overwrite)
                    self.sort_tables = self.get_sort_tables(overwrite=overwrite)
            except:
                print("Error obtaining clusters.")
                print(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                traceback.print_exc(file=sys.stdout)

            # get tetrode depths & match sessions
            self.sessions_tt_positions = self.get_sessions_tt_position()
            self.tt_depth_match = self.get_tetrode_depth_match()

            self.save_subject_info()

    def update_clusters(self, verbose=False):
        for session in self. sessions:
            self._cluster_stats(session)
            if verbose:
                print('.', end='')
        self.get_session_clusters(overwrite=True)
        self.get_sort_tables(overwrite=True)
        self.save_subject_info()

    def get_sessions_tt_position(self):
        p = Path(self.results_path.parent / f"{self.subject}_tetrodes.csv")

        if p.exists():
            tt_pos = pd.read_csv(p)
            tt_pos['Date'] = pd.to_datetime(tt_pos['Date']).dt.strftime('%m%d%y')
            tt_pos = tt_pos.set_index('Date')
            tt_pos = tt_pos[['TT' + str(tt) + '_overall' for tt in self.tetrodes]]

            session_dates = {session: session.split('_')[2] for session in self.sessions}
            sessions_tt_pos = pd.DataFrame(index=self.sessions, columns=['tt_' + str(tt) for tt in self.tetrodes])

            for session in self.sessions:
                date = session_dates[session]
                # below if is to correct for incorrect session dates for Cl
                if (date in ['010218', '010318', '010418']) & (self.subject == 'Cl'):
                    date = date[:5] + '9'
                sessions_tt_pos.loc[session] = tt_pos.loc[date].values
            return sessions_tt_pos
        else:
            print(f"Tetrode depth table not found at '{str(p)}'")
            return None

    def get_depth_wf(self):
        raise NotImplementedError

    def get_tetrode_depth_match(self):

        tt_pos = self.sessions_tt_positions

        try:
            tt_depth_matchs = {tt: {} for tt in self.tetrodes}
            for tt in self.tetrodes:
                tt_str = 'tt_' + str(tt)
                tt_depths = tt_pos[tt_str].unique()
                for depth in tt_depths:
                    tt_depth_matchs[tt][depth] = list(tt_pos[tt_pos[tt_str] == depth].index)
            return tt_depth_matchs

        except:
            print("Error Matching Sessions based on tetrode depth")
            print(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
            traceback.print_exc(file=sys.stdout)
        return None

    def load_subject_info(self):
        with self.subject_info_file.open(mode='rb') as f:
            loaded_self = pickle.load(f)
            self.__dict__.update(loaded_self.__dict__)
            return self

    def save_subject_info(self):
        with self.subject_info_file.open(mode='wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_session_clusters(self, overwrite=False):
        _clusters_file = self.sorted_path / ('clusters_{}_{}.json'.format(self.sorter, self.subject))
        if _clusters_file.exists() and not overwrite:  # load
            with _clusters_file.open(mode='r') as f:
                session_clusters = json.load(f)
        else:  # create
            session_clusters = {}
            for session in self.sessions:
                self._cluster_stats(session)
                session_clusters[session] = self._session_clusters(session)
            with _clusters_file.open(mode='w') as f:
                json.dump(session_clusters, f, indent=4)
        return session_clusters

    def get_sort_tables(self, overwrite=False):
        _sort_table_ids = ['tt', 'valid', 'curated', 'summary']
        _sort_table_files = {ii: Path(self.sorted_path, 'sort_{}_{}_{}'.format(ii, self.sorter, self.subject))
                                  for ii in _sort_table_ids}
        if _sort_table_files['summary'].exists() and not overwrite:
            sort_tables = {ii: [] for ii in _sort_table_ids}
            for ii in _sort_table_ids:
                sort_tables[ii] = pd.read_csv(_sort_table_files[ii], index_col=0)
        else:
            sort_tables = self._sort_tables()
            for ii in _sort_table_ids:
                sort_tables[ii].to_csv(_sort_table_files[ii])
        return sort_tables

    # private methods
    def _session_paths(self, session):
        time_step = self.params['time_step']
        samp_rate = self.params['samp_rate']

        tmp = session.split('_')
        subject = tmp[0]
        task = tmp[1]
        date = tmp[2]

        paths = {'session': session, 'subject': subject, 'task': task, 'date': date, 'step': time_step, 'SR': samp_rate,
                 'Sorted': self.sorted_path / session, 'Raw': self.raw_path / session,
                 'PreProcessed': self.preprocessed_path / session, 'Results': self.results_path / session}

        paths['Results'].mkdir(parents=True, exist_ok=True)

        paths['behav_track_data'] = paths['Results'] / ('behav_track_data{}ms.pkl'.format(int(time_step * 1000)))

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
        paths['cluster_binned_spikes'] = paths['Results'] / f'binned_spikes_{int(time_step * 1000)}ms.npy'
        paths['cluster_fr'] = paths['Results'] / 'fr.npy'

        paths['cluster_spike_maps'] = paths['Results'] / 'spike_maps.npy'
        paths['cluster_fr_maps'] = paths['Results'] / 'maps.npy'
        paths['cluster_OF_metrics'] = paths['Results'] / 'OF_metrics.csv'
        paths['cluster_OF_encoding_models'] = paths['Results'] / 'OF_encoding.csv'

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

    def _cluster_stats(self, session):
        sort_path = self.session_paths[session]['Sorted']

        for tt in self.tetrodes:
            tt_str = 'tt_' + str(tt)
            _cluster_spike_time_fn = Path(sort_path, tt_str, self.sorter, 'spike_times.npy')
            _cluster_spike_ids_fn = Path(sort_path, tt_str, self.sorter, 'spike_clusters.npy')
            _cluster_groups_fn = Path(sort_path, ('tt_' + str(tt)), self.sorter, 'cluster_group.tsv')
            _hp_data_fn = Path(sort_path, tt_str, self.sorter, 'recording.dat')
            _hp_data_info_fn = Path(sort_path, tt_str, tt_str+'_info.pickle')
            _cluster_stats_file_path = Path(sort_path, tt_str, self.sorter, 'cluster_stats_curated.csv')

            try:
                spike_times = np.load(_cluster_spike_time_fn)
                spike_ids = np.load(_cluster_spike_ids_fn)
                cluster_groups = pd.read_csv(_cluster_groups_fn, sep='\t')
                units = np.unique(spike_ids)

                valid_units_idx = cluster_groups.group.isin(['good', 'mua'])
                valid_units = cluster_groups.cluster_id[valid_units_idx].values

                spike_times_dict = {unit: spike_times[spike_ids == unit].flatten() for unit in valid_units}
                #print(spike_times_dict[0])
                hp_data = sort_funcs.load_hp_binary_data(_hp_data_fn)

                with _hp_data_info_fn.open(mode='rb') as f:
                    hp_data_info = pickle.load(f)

                cluster_stats = sort_funcs.get_cluster_stats(spike_times_dict, hp_data, hp_data_info)
                cluster_stats.to_csv(_cluster_stats_file_path)
            except:
                pass

    def get_session_tt_wf(self, session, tt, cluster_ids=None, wf_lims=None, n_wf=200):

        if wf_lims is None:
            wf_lims = [-12, 20]
        tt_str = 'tt_' + str(tt)
        _sort_path = Path(self.session_paths[session]['Sorted'],  tt_str, self.sorter)

        _cluster_spike_time_fn = _sort_path / 'spike_times.npy'
        _cluster_spike_ids_fn = _sort_path / 'spike_clusters.npy'
        _hp_data_fn = _sort_path / 'recording.dat'

        spike_times = np.load(_cluster_spike_time_fn)
        spike_ids = np.load(_cluster_spike_ids_fn)
        hp_data = sort_funcs.load_hp_binary_data(_hp_data_fn)

        wf_samps = np.arange(wf_lims[0], wf_lims[1])
        if cluster_ids is None:
            cluster_ids = np.unique(spike_ids)

        n_clusters = len(cluster_ids)
        out = np.zeros( (n_clusters, n_wf, len(wf_samps)*4), dtype=np.float16)

        for cl_idx, cluster in enumerate(cluster_ids):
            cl_spk_times = spike_times[spike_ids == cluster]
            n_cl_spks = len(cl_spk_times)
            if n_wf == 'all':
                sampled_spikes = cl_spk_times
            elif n_wf > n_cl_spks:
                # Note that if number of spikes < n_wf, spikes will be repeated such that sampled_spikes has n_wf
                sampled_spikes = cl_spk_times[np.random.randint(n_cl_spks, size=n_wf)]
            else:  # sample from spikes
                sampled_spikes = cl_spk_times[np.random.choice(np.arange(n_cl_spks), size=n_wf, replace=False)]

            for wf_idx, samp_spk in enumerate(sampled_spikes):
                out[cl_idx, wf_idx] = hp_data[:, wf_samps + samp_spk].flatten()

        return out

    def _session_clusters(self, session):
        table = {'session': session, 'path': str(self.session_paths[session]['Sorted']),
                 'n_cell': 0, 'n_mua': 0, 'n_noise': 0, 'n_unsorted': 0, 'sorted_TTs': [], 'curated_TTs': [],
                 'cell_IDs': {}, 'mua_IDs': {}, 'noise_IDs': {}, 'unsorted_IDs': {}, 'clusters_snr': {},
                 'clusters_fr': {}, 'clusters_valid': {}, 'clusters_isi_viol_rate': {}}

        sort_paths = table['path']
        _cluster_stats_names = ['fr', 'snr', 'isi_viol_rate', 'valid']
        for tt in self.tetrodes:
            _cluster_groups_file = Path(sort_paths, ('tt_' + str(tt)), self.sorter, 'cluster_group.tsv')

            # check what stats file to load
            if Path(sort_paths, ('tt_' + str(tt)), self.sorter, 'cluster_stats_curated.csv').exists():
                _cl_stat_file = Path(sort_paths, ('tt_' + str(tt)), self.sorter, 'cluster_stats_curated.csv')
            else:
                _cl_stat_file = Path(sort_paths, ('tt_' + str(tt)), self.sorter, 'cluster_stats.csv')

            if _cl_stat_file.exists():
                table['sorted_TTs'].append(int(tt))
                d = pd.read_csv(_cl_stat_file, index_col=0)
                n_clusters = d.shape[0]
                for st in _cluster_stats_names:
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

    def _sort_tables(self):
        n_tetrodes = self.params['n_tetrodes']

        sort_tables = {ii: pd.DataFrame(index=self.sessions,
                                        columns=self.tetrodes) for ii in ['tt', 'curated', 'valid']}

        sort_tables['summary'] = pd.DataFrame(index=self.sessions,
                                              columns=['n_tt', 'n_tt_sorted', 'n_tt_curated',
                                                       'n_valid_clusters', 'n_cell', 'n_mua',
                                                       'n_noise', 'n_unsorted'])
        _unit_types = ['cell', 'mua', 'noise', 'unsorted']
        _sort_table_ids = ['tt', 'valid', 'curated', 'summary']

        #
        # for tbl in self._sort_table_ids:
        #     sort_tables[tbl] = sort_tables[tbl].fillna(-1)

        for session in self.sessions:
            _clusters_info = self.session_clusters[session]
            for tbl in _sort_table_ids:
                if tbl == 'tt':
                    sort_tables[tbl].at[session, _clusters_info['sorted_TTs']] = 1
                if tbl == 'curated':
                    n_cells = np.zeros(n_tetrodes)
                    n_mua = np.zeros(n_tetrodes)

                    tt_cell_clusters = _clusters_info['cell_IDs']
                    for tt, clusters in tt_cell_clusters.items():
                        n_cells[tt-1] = len(clusters)

                    tt_mua_clusters = _clusters_info['mua_IDs']
                    for tt, clusters in tt_mua_clusters.items():
                        n_mua[tt - 1] = len(clusters)

                    sort_tables[tbl].loc[session] = n_cells+n_mua
                if tbl == 'valid':
                    _valid_cls = _clusters_info['clusters_valid']
                    for tt, cls in _valid_cls.items():
                        sort_tables[tbl].at[session, int(tt)] = len(cls)

            sort_tables['summary'].at[session, 'n_tt'] = n_tetrodes
            sort_tables['summary'].at[session, 'n_tt_sorted'] = len(_clusters_info['sorted_TTs'])
            sort_tables['summary'].at[session, 'n_tt_curated'] = len(_clusters_info['curated_TTs'])

            for ut in _unit_types:
                sort_tables['summary'].at[session, 'n_' + ut] = _clusters_info['n_' + ut]

        return sort_tables


class SubjectSessionInfo(SubjectInfo):
    def __init__(self, subject, session, sorter='KS2', data_root='BigPC', time_step=0.02,
                 samp_rate=32000, n_tetrodes=16, fr_temporal_smoothing=0.125, spk_outlier_thr=None):
        super().__init__(subject, sorter=sorter, data_root=data_root, time_step=time_step,
                         samp_rate=samp_rate, n_tetrodes=n_tetrodes, fr_temporal_smoothing=fr_temporal_smoothing,
                         spk_outlier_thr=spk_outlier_thr)

        self.session = session
        self.paths = self.session_paths[session]
        self.clusters = self.session_clusters[session]
        if len(session.split('_')) == 3:
            self.subject, self.task, self.date = session.split('_')
            self.subject = self.subject.capitalize()
            self.sub_session_id = '0000'
        elif len(session.split('_')) == 4:
            self.subject, self.task, self.date, self.sub_session_id = session.split('_')
            self.subject = self.subject.capitalize()

        self.task_params = get_task_params(self)
        self.n_units = self.clusters['n_cell'] + self.clusters['n_mua']
        #print('number of units in session {}'.format(self.clusters['n_cell'] + self.clusters['n_mua']))
        if self.paths['cluster_spikes_ids'].exists():
            with self.paths['cluster_spikes_ids'].open('r') as f:
                self.cluster_ids = json.load(f)
                self.n_units = len(self.cluster_ids)
            self.cell_ids = np.array([v[0] == 'cell' for k, v in self.cluster_ids.items()])
            self.mua_ids = ~self.cell_ids

        self._analyses = self._check_analyses()

    def __str__(self):
        print()
        print(f'Session Information for subject {self.subject}, session {self.session}')
        print(f'Number of curated units: {self.n_units}')
        print('Methods listed below can be executed with get_{method}(), eg. get_spikes():')
        for a, m in self._analyses.items():
            print(f'  -> {a}. Executed = {m[1]}')
        print()
        print('To run all analyses use run_analyses().')
        return ''

    def print_task_params(self):
        print()
        print('Task/track and analysis parameters. ')
        print()
        for param, val in self.task_params.items():
            if param[-1] != '_':  # not a vector
                print(f'  -> {param}: {val}')

    def _check_analyses(self):
        if self.task == 'OF':
            # analyses is a dictionary with keys as the desired output, and values that
            # include the method and if it has been run (file exists)
            analyses = {
                'track_data': (self.get_track_data, self.paths['behav_track_data'].exists()),
                'time': (self.get_time, True),
                'spikes': (self.get_spikes, self.paths['cluster_spikes'].exists()),
                'binned_spikes': (self.get_binned_spikes, self.paths['cluster_binned_spikes'].exists()),
                'fr': (self.get_fr, self.paths['cluster_fr'].exists()),
                'spike_maps': (self.get_spike_maps, self.paths['cluster_spike_maps'].exists()),
                'maps': (self.get_fr_maps, self.paths['cluster_fr_maps'].exists()),
                'scores': (self.get_scores, self.paths['cluster_OF_metrics'].exists()),
                'encoding_models': (self.get_encoding_models, self.paths['cluster_OF_encoding_models'].exists())
            }
        else:
            raise NotImplementedError
        return analyses

    def run_analyses(self, overwrite=False):
        """
        Method to execute all analyses in the analyses list.
        :param bool overwrite: overwrite flag.
        :return: None
        """
        if self.n_units>0:
            for a, m in self._analyses.items():
                if a == 'time':
                    continue
                if not m[1] or overwrite:
                    try:
                        # calls methods in _analyses
                        _ = m[0](overwrite=True)
                        print(f'Analysis {a} completed.')
                    except NotImplementedError:
                        print(f'Analysis {a} not implemented.')
                    except FileNotFoundError:
                        print(f'Analysis {a} did not find the dependent files.')
                    except KeyboardInterrupt:
                        print('Keyboard Interrupt')
                        break
                    except:
                        print(f'Analysis {a} failed.')
            # update analyses
            self._analyses = self._check_analyses()
        else:
            print('This session does not have units. No analyses were ran. ')

    #  default methods
    def get_time(self, which='resamp'):
        """
        :param str which:   if 'resamp' [default], returns resampled time at the time_step found in params
                            if 'orig', returns a time vector sampled at the original sampling rate
        :return np.array time: array of time, float32.
        """
        samp_rate = self.params['samp_rate']
        time_step = self.params['time_step']

        tt_info = self.get_tt_info(1)
        n_samps = tt_info['n_samps']
        tB = tt_info['tB']

        # get time vector with original sampling rate
        if which == 'orig':
            tE = n_samps / samp_rate + tB
            time = np.arange(tB, tE, 1.0 / samp_rate).astype(np.float32)
        elif which == 'resamp':
            # compute resampled time
            rsamp_rate = int(1 / time_step)
            n_rsamps = int(n_samps * rsamp_rate / samp_rate)
            trE = n_rsamps / rsamp_rate + tB
            time = np.arange(tB, trE, 1.0 / rsamp_rate).astype(np.float32)
            time = np.round(time * 1000.0) / 1000.0  # round time to ms resolution
        else:
            print('Invalid which argument.')
            time = None

        return time

    def get_raw_track_data(self):
        """
        Loads video tracking data:
        :return: np.arrays of tracking data.
            t -> time sampled at the camera's sampling rate, usually 60Hz
            x -> x position of the animal
            y -> y position of the animal
            ha -> estimate of the head angle of the animal
        """
        _track_data_file = self.paths['PreProcessed'] / 'vt.h5'

        with h5py.File(_track_data_file, 'r') as f:
            t = np.array(f['t'])
            x = np.array(f['x'])
            y = np.array(f['y'])
            ha = np.array(f['ha'])

        return t, x, y, ha

    def get_sorted_tt_dir(self, tt):
        return self.paths['Sorted'] / f'tt_{tt}' / self.sorter

    def get_tt_info(self, tt):
        try:
            with (self.paths['PreProcessed'] / f'tt_{tt}_info.pickle').open(mode='rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f'Tetrode {tt} information not found.')
            return None

    def get_tt_data(self, tt):
        try:
            return np.load(self.paths['PreProcessed'] / f'tt_{tt}.npy')
        except FileNotFoundError:
            print(f'Tetrode {tt} data not found.')
            return None

    # behavioral methods
    def get_track_data(self, overwrite=False):
        if self.task == 'OF':
            if not self.paths['behav_track_data'].exists() or overwrite:
                print('Open Field Track Data not Found or Overwrite= True, creating them.')
                of_track_dat = of_funcs.get_session_track_data(self)
                with self.paths['behav_track_data'].open(mode='wb') as f:
                    pickle.dump(of_track_dat, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with self.paths['behav_track_data'].open(mode='rb') as f:
                    of_track_dat = pickle.load(f)
            return of_track_dat
        else:
            return None

    # methods from spike functions
    def get_spikes(self, return_numpy=True, save_spikes_dict=False, overwrite=False):
        """
        :param bool return_numpy: if true, returns clusters as a numpy array of spike trains, and dict of ids.
        :param bool save_spikes_dict: saves dictionary for the spikes for cells and mua separetely
        :param bool overwrite: if true overwrites. ow. returns disk data
        :return: np.ndarray spikes: object array containing spike trains per cluster
        :return: dict tt_cl: [for return_numpy] dictinonary with cluster keys and identification for each cluster
        """
        if self.n_units == 0:
            print('No units in the session.')
            return None

        session_paths = self.paths

        spikes = None
        spikes_numpy = None
        tt_cl = None
        wfi = None
        wfi2 = None

        if (not session_paths['cluster_spikes'].exists()) or overwrite:
            print('Spikes Files not Found or overwrite=1, creating them.')

            spikes, wfi = spike_funcs.get_session_spikes(self)

            # convert spike dictionaries to numpy and a json dict with info
            cell_spikes, cell_tt_cl = spike_funcs.get_spikes_numpy(spikes['Cell'])
            mua_spikes, mua_tt_cl = spike_funcs.get_spikes_numpy(spikes['Mua'])
            spikes_numpy, tt_cl, wfi2 = spike_funcs.aggregate_spikes_numpy(cell_spikes, cell_tt_cl, mua_spikes,
                                                                           mua_tt_cl, wfi)

            self.cluster_ids = tt_cl
            self.n_units = len(self.cluster_ids)
            self.cell_ids = np.array([v[0] == 'cell' for k, v in self.cluster_ids.items()])
            self.mua_ids = ~self.cell_ids

            # save numpy spikes
            np.save(session_paths['cluster_spikes'], spikes_numpy)
            with session_paths['cluster_spikes_ids'].open(mode='w') as f:
                json.dump(tt_cl, f, indent=4)

            # save waveform info
            with session_paths['cluster_wf_info'].open(mode='wb') as f:
                pickle.dump(wfi2, f, pickle.HIGHEST_PROTOCOL)

            # save Cell/Mua spike dictionaries and waveform info
            if save_spikes_dict:
                for ut in ['Cell', 'Mua']:
                    with session_paths[ut + '_Spikes'].open(mode='w') as f:
                        json.dump(spikes[ut], f, indent=4)
                    with session_paths[ut + '_WaveForms'].open(mode='w') as f:
                        pickle.dump(wfi[ut], f, pickle.HIGHEST_PROTOCOL)

        else:  # Load data.
            if return_numpy:
                spikes_numpy = np.load(session_paths['cluster_spikes'], allow_pickle=True)
                with session_paths['cluster_spikes_ids'].open() as f:
                    tt_cl = json.load(f)
                with session_paths['cluster_wf_info'].open(mode='rb') as f:
                    wfi2 = pickle.load(f)
            else:
                with session_paths['Cell_Spikes'].open() as f:
                    cell_spikes = json.load(f)
                with session_paths['Mua_Spikes'].open() as f:
                    mua_spikes = json.load(f)
                spikes = {'Cell': cell_spikes, 'Mua': mua_spikes}

        if return_numpy:
            return spikes_numpy, tt_cl, wfi2
        else:
            return spikes, wfi

    def get_binned_spikes(self, spike_trains=None, overwrite=False):
        """
        :param np.ndarray spike_trains: if not provided, these are computed or loaded
        :param bool overwrite: overwrite flag. if false, loads data from the subject_info paths
        :returns: np.ndarray bin_spikes: shape n_clusters x n_bin_samps. simply the spike counts for each cluster/bin
        """

        if self.n_units == 0:
            print('No units.')
            return None

        session_paths = self.paths
        if (not session_paths['cluster_binned_spikes'].exists()) or overwrite:
            print('Binned Spikes Files not Found or overwrite=1, creating them.')
            bin_spikes = spike_funcs.get_session_binned_spikes(self, spike_trains=spike_trains)
            # save
            np.save(session_paths['cluster_binned_spikes'], bin_spikes)
        else:  # load
            bin_spikes = np.load(session_paths['cluster_binned_spikes'])
        return bin_spikes

    def get_fr(self, bin_spikes=None, overwrite=False):
        """
        :param np.ndarray bin_spikes: shape [n_clusters x n_timebins]
        :param bool overwrite: flag, if false loads data
        :return: np.ndarray fr: firing rate shape [n_clusters x n_timebins]
        """

        if self.n_units == 0:
            print('No units.')
            return None

        session_paths = self.paths
        if (not session_paths['cluster_fr'].exists()) | overwrite:
            print('Firing Rate Files Not Found or overwrite=1, creating them.')
            fr = spike_funcs.get_session_fr(self, bin_spikes=bin_spikes)
            # save data
            np.save(session_paths['cluster_fr'], fr)

        else:  # load firing rate
            fr = np.load(session_paths['cluster_fr'])
        return fr

    def get_spike_maps(self, overwrite=False):
        if self.n_units == 0:
            print('No units.')
            return None
        if self.task == 'OF':
            if not self.paths['cluster_spike_maps'].exists() or overwrite:
                print('Open Field Spike Maps not Found or Overwrite= True, creating them.')
                of_track_dat = of_funcs.get_session_spike_maps(self)
                with self.paths['cluster_spike_maps'].open(mode='wb') as f:
                    pickle.dump(of_track_dat, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with self.paths['cluster_spike_maps'].open(mode='rb') as f:
                    of_track_dat = pickle.load(f)
            return of_track_dat
        else:
            return None

    def get_fr_maps(self, overwrite=False, traditional=False):
        """
        obtains 2 dimensional representation of the firing rate across the spatial positions.
        :param bool traditional: if true: computes firing rate map through spike counts at each spatial position
                                 if False [default]: uses continous firing rate to estimate maps, then gets normalized
                                 by occupation.
        :param bool overwrite:
        :return:
        """
        if self.n_units == 0:
            print('No units.')
            return None

        if self.task == 'OF':
            if not self.paths['cluster_fr_maps'].exists() or overwrite:
                print('Open Field Firing Rate Maps not Found or Overwrite= True, creating them.')
                if traditional:
                    fr_maps = of_funcs.get_session_fr_maps(self)
                else:
                    fr_maps = of_funcs.get_session_fr_maps_cont(self)

                np.save(self.paths['cluster_fr_maps'], fr_maps)
            else:
                fr_maps = np.load(self.paths['cluster_fr_maps'])
            return fr_maps
        else:
            return None

    def get_scores(self, overwrite=False):
        """
        obtains a series of pandas data frames quantifying the extent of coding to environmental variables
        :param overwrite:
        :returns: dictionary of pandas data frames.
        """
        if self.n_units == 0:
            print('No units.')
            return None

        if self.task == 'OF':
            if not self.paths['cluster_OF_metrics'].exists() or overwrite:
                print('Open Field Score Metrics do not exits or overwrite=True, creating them.')
                scores = of_funcs.get_session_scores(self)
                scores.to_csv(self.paths['cluster_OF_metrics'])
                # with self.paths['cluster_OF_metrics'].open(mode='wb') as f:
                #     pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                scores = pd.read_csv(self.paths['cluster_OF_metrics'], index_col=0)
                # with self.paths['cluster_OF_metrics'].open(mode='rb') as f:
                #     scores = pickle.load(f)
        else:
            print('Method not develop for other tasks.')
            raise NotImplementedError

        return scores

    def get_encoding_models(self, overwrite=False):
        """
        obtains a series of pandas data frames quantifying the extent of coding to environmental variables
        :param overwrite:
        :returns: dictionary of pandas data frames.
        """
        if self.n_units == 0:
            print('No units.')
            return None

        if self.task == 'OF':
            if not self.paths['cluster_OF_encoding_models'].exists() or overwrite:
                print('Encoding Models do not exist or overwrite=True, creating them.')
                scores = of_funcs.get_session_encoding_models(self)
                scores.to_csv(self.paths['cluster_OF_encoding_models'])
                # with self.paths['cluster_OF_metrics'].open(mode='wb') as f:
                #     pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                scores = pd.read_csv(self.paths['cluster_OF_encoding_models'], index_col=0)
                # with self.paths['cluster_OF_metrics'].open(mode='rb') as f:
                #     scores = pickle.load(f)
        else:
            print('Method not develop for other tasks.')
            raise NotImplementedError

        return scores


def get_task_params(session_info):
    """
    This utility function returns a dictionary of parameters specific to the task. Given that some animals have
    different maze shapes due to camera movement, this will be different for different animals or even sessions.
    :param SubjectSessionInfo session_info:
    :return: dictionary of parameters
    """
    task = session_info.task
    subject = session_info.subject
    time_step = session_info.params['time_step']

    task_params = {'time_step': time_step}
    if task == 'OF':
        if subject in ['Li', 'Ne', 'Cl', 'Ca', 'Al']:
            conv_params = {
                # pixel params
                'x_pix_lims': [100, 650],  # camera field of view x limits [pixels]
                'y_pix_lims': [100, 500],  # camera field of view y limits [pixels]
                'x_pix_bias': -380,  # factor for centering the x pixel position
                'y_pix_bias': -280,  # factor for centering the y pixel position
                'vt_rate': 1.0 / 60.0,  # video acquisition frame rate
                'xy_pix_rot_rad': np.pi / 2 + 0.08,  # rotation of original xy pix camera to experimenter xy

                # conversion params
                'x_pix_mm': 1300.0 / 344.0,  # pixels to mm for the x axis [pix/mm]
                'y_pix_mm': 1450.0 / 444.0,  # pixels to mm for the y axis [pix/mm]
                'x_mm_bias': 20,  # factor for centering the x mm position
                'y_mm_bias': 650,  # factor for centering the y mm position
                'x_mm_lims': [-630, 630],  # limits on the x axis of the maze [mm]
                'y_mm_lims': [-60, 1350],  # limits on the y axis of the maze [mm]
                'x_cm_lims': [-63, 63],  # limits on the x axis of the maze [cm]
                'y_cm_lims': [-6, 135],  # limits on the y axis of the maze [cm]
            }
        elif subject in ['Mi']:
            conv_params = {

                # pixel params
                'x_pix_lims': [200, 600],  # camera field of view x limits [pixels]
                'y_pix_lims': [100, 450],  # camera field of view y limits [pixels]
                'x_pix_bias': -390,  # factor for centering the x pixel position
                'y_pix_bias': -265,  # factor for centering the y pixel position
                'vt_rate': 1.0 / 60.0,  # video acquisition frame rate
                'xy_pix_rot_rad': np.pi / 2,  # rotation of original xy pix camera to experimenter xy

                # conversion params
                'x_pix_mm': 1300.0 / 290.0,  # pixels to mm for the x axis [pix/mm]
                'y_pix_mm': 1450.0 / 370.0,  # pixels to mm for the y axis [pix/mm]
                'x_mm_bias': 0,  # factor for centering the x mm position
                'y_mm_bias': 750,  # factor for centering the y mm position
                'x_mm_lims': [-630, 630],  # limits on the x axis of the maze [mm]
                'y_mm_lims': [0, 1450],  # limits on the y axis of the maze [mm]
                'x_cm_lims': [-63, 63],  # limits on the x axis of the maze [cm]
                'y_cm_lims': [0, 145],  # limits on the y axis of the maze [cm]
            }
            pass
        else:
            conv_params = {}
        default_task_params = {
            # binning parameters
            'mm_bin': 30,  # millimeters per bin [mm]
            'cm_bin': 3,  # cm per bin [cm]
            'max_speed_thr': 80,  # max speed threshold for allowing valid movement [cm/s]
            'min_speed_thr': 2,  # min speed threshold for allowing valid movement [cm/s]
            'rad_bin': np.deg2rad(10),  # angle radians per bin [rad]
            'occ_num_thr': 3,           # number of occupation times threshold [bins
            'occ_time_thr': time_step * 3,  # time occupation threshold [sec]
            'speed_bin': 2,                # speed bin size [cm/s]

            # filtering parameters
            'spatial_sigma': 2,  # spatial smoothing sigma factor [au]
            'spatial_window_size': 5,  # number of spatial position bins to smooth [bins]
            'temporal_window_size': 11,  # smoothing temporal window for filtering [bins]
            'temporal_angle_window_size': 11,  # smoothing temporal window for angles [bins]
            'temporal_window_type': 'hann',  # window type for temporal window smoothing

            # statistical tests parameters:
            'sig_alpha': 0.02,  # double sided alpha level for significance testing
            'n_perm': 200,  # number of permutations

            # type of encoding model. see spatial_funcs.get_border_encoding_features
            'reg_type': 'poisson',
            # these are ignoed if border_enc_model_type is linear.
            'border_enc_model_feature_params__': {'center_gaussian_spread': 0.2,  # as % of environment
                                                  'sigmoid_slope_thr': 0.15,  # value of sigmoid at border width
                                                 },

            'border_score_params__': {'fr_thr': 0.25,  # firing rate threshold
                                      'width_bins': 3,  # distance from border to consider it a border cell [bins]
                                      'min_field_size_bins': 10},  # minimum area for fields in # of bins

            'grid_score_params__': {'ac_thr': 0.1,  # autocorrelation threshold for finding fields
                                      'radix_range': [0.5, 2.0],  # range of radii for grid score in the autocorr
                                      'apply_sigmoid': True,  # apply sigmoid to rate maps
                                      'sigmoid_center': 0.5,  # center for sigmoid
                                      'sigmoid_slope': 10,   # slope for sigmoid
                                      'find_fields': True},  # mask fields before autocorrelation

            # grid encoding model
            'grid_fit_type': 'auto_corr',  # ['auto_corr', 'moire'], how to find parameters for grid
            'pos_feat_type': 'pca',  # feature type for position encoding model
            'pos_feat_n_comp': 0.95,  # variance explaiend for pca in position feautrues

        }

        task_params.update(conv_params)
        task_params.update(default_task_params)

        # derived parameters

        # -- filter coefficients --
        task_params['filter_coef_'] = signal.get_window(task_params['temporal_window_type'],
                                                        task_params['temporal_window_size'],
                                                        fftbins=False)
        task_params['filter_coef_'] /= task_params['filter_coef_'].sum()

        task_params['filter_coef_angle_'] = signal.get_window(task_params['temporal_window_type'],
                                                              task_params['temporal_angle_window_size'],
                                                              fftbins=False)
        task_params['filter_coef_angle_'] /= task_params['filter_coef_angle_'].sum()

        # -- bins --
        task_params['ang_bin_edges_'] = np.arange(0, 2*np.pi+task_params['rad_bin'], task_params['rad_bin'])
        task_params['ang_bin_centers_'] = task_params['ang_bin_edges_'][:-1] + task_params['rad_bin']/2
        task_params['n_ang_bins'] = len(task_params['ang_bin_centers_'])

        task_params['sp_bin_edges_'] = np.arange(task_params['min_speed_thr'],
                                                 task_params['max_speed_thr'] + task_params['speed_bin'],
                                                 task_params['speed_bin'])
        task_params['sp_bin_centers_'] = task_params['sp_bin_edges_'][:-1]+task_params['speed_bin']/2
        task_params['n_sp_bins'] = len(task_params['sp_bin_centers_'])

        task_params['x_bin_edges_'] = np.arange(task_params['x_cm_lims'][0],
                                                task_params['x_cm_lims'][1]+task_params['cm_bin'],
                                                task_params['cm_bin'])
        task_params['x_bin_centers_'] = task_params['x_bin_edges_'][:-1] + task_params['cm_bin']/2
        task_params['n_x_bins'] = len(task_params['x_bin_centers_'])
        task_params['n_width_bins'] = task_params['n_x_bins']
        task_params['width'] = task_params['n_x_bins']

        task_params['y_bin_edges_'] = np.arange(task_params['y_cm_lims'][0],
                                                task_params['y_cm_lims'][1] + task_params['cm_bin'],
                                                task_params['cm_bin'])
        task_params['y_bin_centers_'] = task_params['y_bin_edges_'][:-1] + task_params['cm_bin']/2
        task_params['n_y_bins'] = len(task_params['y_bin_centers_'])
        task_params['n_height_bins'] = task_params['n_y_bins']
        task_params['height'] = task_params['n_y_bins']

    elif task[:2] == 'T3':
        pass

    return task_params


# def save_dict_to_hdf5(dic, filename):
#     """
#     ....
#     """
#     with h5py.File(filename, 'w') as h5file:
#         recursively_save_dict_contents_to_group(h5file, '/', dic)
#
#
# def recursively_save_dict_contents_to_group(h5file, path, dic):
#     """
#     ....
#     """
#     for key, item in dic.items():
#         if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
#             h5file[path + key] = item
#         elif isinstance(item, dict):
#             recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
#         else:
#             raise ValueError('Cannot save %s type' % type(item))
#
#
# def load_dict_from_hdf5(filename):
#     """
#     ....
#     """
#     with h5py.File(filename, 'r') as h5file:
#         return recursively_load_dict_contents_from_group(h5file, '/')
#
#
# def recursively_load_dict_contents_from_group(h5file, path):
#     """
#     ....
#     """
#     ans = {}
#     for key, item in h5file[path].items():
#         if isinstance(item, h5py._hl.dataset.Dataset):
#             ans[key] = item.value
#         elif isinstance(item, h5py._hl.group.Group):
#             ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
#     return ans
