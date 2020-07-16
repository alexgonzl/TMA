# creates a table with progress of analyses for each session
# update analyses table

from pathlib import Path
import numpy as np
import pandas as pd
import pickle
n_tetrodes = 16


class DataPaths(object):
    def __init__(self, subject, sorter='KS2', data_root='BigPC', load=0, overwrite=0, time_step=0.02, samp_rate=32000):
        self.subject = subject
        self.sorter = sorter

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

        self.data_paths_file = self.results_path / ('data_paths_{}_{}.pickle'.format(sorter,subject))
        # check if instance of DataPaths for subject and sorter exists already
        if load and not overwrite:
            self.load_data_paths()
        else:
            self._cluster_table_file = self.sorted_path / ('clusters_{}.json'.format(subject))
            self._channel_table_file = self.preprocessed_path / ('chan_table_{}.csv'.format(subject))
            self._sort_summary_file = self.sorted_path / ('sort_summary_{}_{}.csv'.format(sorter, subject))

            _sort_file_ids = ['tt', 'valid', 'curated']
            self.sorter_tables = {}
            for ii in _sort_file_ids:
                _file_name = self.sorted_path / ('sort_{}_{}_{}.csv'.format(ii, sorter, subject))
                if _file_name.exists():
                    self.sorter_tables[ii] = pd.read_csv(_file_name, index_col=0)

            if self._sort_summary_file.exists():
                self.sort_summary = pd.read_csv(self._sort_summary_file, index_col=0)

            if self._channel_table_file.exists():
                self.channel_table = pd.read_csv(self._channel_table_file, index_col=0)
                self.sessions = list(self.channel_table.index)

                self.session_paths = {}
                for session in self.sessions:
                    self.session_paths[session] = self.get_session_paths(session, time_step=time_step, samp_rate=samp_rate)

                self.session_clusters = {}
                for session in self.sessions:
                    self.session_clusters[session] = self.get_session_clusters(session)

            self.save_data_paths()

    def load_data_paths(self):
        with self.data_paths_file.open(mode='rb') as f:
            return pickle.load(f)

    def save_data_paths(self):
        with self.data_paths_file.open(mode='wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_session_paths(self, session, sorter='KS2', time_step=0.02, samp_rate=32000):
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

        for ut in ['Cell', 'Mua']:
            paths[ut + '_Spikes'] = paths['Results'] / (ut + '_Spikes.json')
            paths[ut + '_WaveForms'] = paths['Results'] / (ut + '_WaveForms.pkl')
            paths[ut + '_WaveFormInfo'] = paths['Results'] / (ut + '_WaveFormInfo.pkl')
            paths[ut + '_Bin_Spikes'] = paths['Results'] / ('{}_Bin_Spikes_{}ms.npy'.format(ut, int(time_step * 1000)))
            paths[ut + '_FR'] = paths['Results'] / ('{}_FR_{}ms.npy'.format(ut, int(time_step * 1000)))

        paths['Spike_IDs'] = paths['Results'] / 'Spike_IDs.json'
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

    def get_session_clusters(self, session):
        table = {'session': session, 'Clustered': 0, 'path': str(self.session_paths[session]['Sorted']),
                 'n_cells': 0, 'n_mua': 0, 'n_noise':0, 'n_unsorted':0, 'curated_TTs': [],
                 'cell_IDs': {}, 'mua_IDs': {}, 'noise_IDs': {}, 'unsorted_IDs': {}}

        sort_paths = table['path']

        tetrodes = np.arange(1, n_tetrodes + 1)
        for tt in tetrodes:
            fn = Path(sort_paths, ('tt_' + str(tt)), self.sorter, 'cluster_group.tsv')
            tt_curated = 0
            if fn.exists():
                try:
                    table['cell_IDs'][int(tt)] = []
                    table['mua_IDs'][int(tt)] = []
                    table['noise_IDs'][int(tt)] = []
                    table['unsorted_IDs'][int(tt)] = []

                    d = pd.read_csv(fn, delimiter='\t')
                    group_types = d['group'].unique()
                    if any([gt in group_types for gt in ['mua', 'good', 'noise']]):
                        tt_curated = 1.0
                    else:
                        tt_curated = 0.0

                    if 'good' in group_types:
                        cells = d['cluster_id'][d['group'] == 'good'].tolist()
                        for cc in cells:
                            table['cell_IDs'][int(tt)].append(cc)
                        table['n_cells'] += len(cells)
                    if 'mua' in group_types:
                        mua = d['cluster_id'][d['group'] == 'mua'].tolist()
                        for mm in mua:
                            table['mua_IDs'][int(tt)].append(mm)
                        table['n_mua'] += len(mua)
                    if 'noise' in group_types:
                        noise = d['cluster_id'][d['group'] == 'noise'].tolist()
                        for nn in noise:
                            table['noise_IDs'][int(tt)].append(nn)
                        table['n_noise'] += len(noise)
                    if 'unsorted' in group_types:
                        unsorted = d['cluster_id'][d['group'] == 'unsorted'].tolist()
                        for uu in unsorted:
                            table['unsorted_IDs'][int(tt)].append(uu)
                        table['n_unsorted'] += len(unsorted)
                except:
                    print('In Session {}, Error Processing TT {}'.format(session, tt))
            if tt_curated:
                table['curated_TTs'].append(int(tt))
        return table

# def checkRaw(sePaths, aTable):
#     for se in aTable.index:
#         rawFlag = 1
#         for ch in ['a', 'b', 'c', 'd']:
#             for tt in np.arange(1, n_tetrodes + 1):
#                 if not (sePaths[se]['Raw'] / ('CSC{}{}.ncs'.format(tt, ch))).exists():
#                     rawFlag = 0
#                     break
#         aTable.loc[se, 'Raw'] = rawFlag
#     return aTable
#
#
# def checkPrePro(sePaths, aTable):
#     for se in aTable.index:
#         allTTFlag = 1
#         partialFlag = 0
#         for tt in np.arange(1, n_tetrodes + 1):
#             if not (sePaths[se]['PreProcessed'] / ('tt_{}.bin'.format(tt))).exists():
#                 allTTFlag = 0
#             else:
#                 partialFlag = 1
#
#         aTable.loc[se, 'PP'] = partialFlag
#         aTable.loc[se, 'PP_A'] = allTTFlag
#     return aTable
#
#
# def checkSort(sePaths, aTable):
#     for se in aTable.index:
#         allTTFlag = 1
#         partialFlag = 0
#         for tt in np.arange(1, n_tetrodes + 1):
#             if not (sePaths[se]['Clusters'] / ('tt_{}'.format(tt)) / 'rez.mat').exists():
#                 allTTFlag = 0
#             else:
#                 partialFlag = 1
#
#         aTable.loc[se, 'Sort'] = partialFlag
#         aTable.loc[se, 'Sort_A'] = allTTFlag
#     return aTable
#
#
# def checkClust(sePaths, aTable):
#     for se in aTable.index:
#         allTTFlag = 1
#         partialFlag = 0
#         for tt in np.arange(1, n_tetrodes + 1):
#             if not (sePaths[se]['Clusters'] / ('tt_{}'.format(tt)) / 'cluster_group.tsv').exists():
#                 allTTFlag = 0
#             else:
#                 partialFlag = 1
#
#         aTable.loc[se, 'Clust'] = partialFlag
#         aTable.loc[se, 'Clust_A'] = allTTFlag
#     return aTable
#
#
# def checkFR(sePaths, aTable):
#     for se in aTable.index:
#         allFR = 1
#         partialFR = 0
#         if not (sePaths[se]['Cell_Bin_Spikes']).exists():
#             allFR = 0
#         else:
#             dat = np.load(sePaths[se]['Cell_Bin_Spikes'])
#             if np.all(dat.sum(axis=1) > 0):
#                 partialFR = 1
#             else:
#                 allFR = 0
#
#         aTable.loc[se, 'FR'] = partialFR
#         aTable.loc[se, 'FR_A'] = allFR
#     return aTable
#
#
# def checkZoneAnalyses(sePaths, aTable):
#     for se in aTable.index:
#         if not (aTable.loc[se, 'Task'] == 'OF'):
#             if sePaths[se]['ZoneAnalyses'].exists():
#                 aTable.loc[se, 'Zone'] = 1
#             else:
#                 aTable.loc[se, 'Zone'] = 0
#     return aTable
#
#
# def checkTrialAnalyses(sePaths, aTable):
#     for se in aTable.index:
#         if not (aTable.loc[se, 'Task'] == 'OF'):
#             if sePaths[se]['TrialCondMat'].exists():
#                 aTable.loc[se, 'Trial'] = 1
#             else:
#                 aTable.loc[se, 'Trial'] = 0
#
#             if sePaths[se]['TrModelFits'].exists():
#                 aTable.loc[se, 'TrModels'] = 1
#             else:
#                 aTable.loc[se, 'TrModels'] = 0
#
#     return aTable
#
#
# def loadSessionData(sessionPaths, vars=['all']):
#     if 'all' in vars:
#         vars = ['wfi', 'bin_spikes', 'fr', 'ids', 'za', 'PosDat', 'TrialLongMat',
#                 'TrialFRLongMat', 'fitTable', 'TrialConds']
#
#     dat = {}
#
#     mods = {}
#     params = TA.getParamSet()
#     for k, pp in params.items():
#         s = ''
#         for p in pp:
#             s += '-' + p
#         mods[k] = s[1:]
#
#     for a in ['wfi', 'bin_spikes', 'fr']:
#         if a in vars:
#             dat[a] = {}
#
#     for ut in ['Cell', 'Mua']:
#         if 'wfi' in vars:
#             with sessionPaths[ut + '_WaveFormInfo'].open(mode='rb') as f:
#                 dat['wfi'][ut] = pkl.load(f)
#         if 'bin_spikes' in vars:
#             dat['bin_spikes'][ut] = np.load(sessionPaths[ut + '_Bin_Spikes'])
#         if 'fr' in vars:
#             dat['fr'][ut] = np.load(sessionPaths[ut + '_FR'])
#
#     if 'ids' in vars:
#         with sessionPaths['Spike_IDs'].open() as f:
#             dat['ids'] = json.load(f)
#     if 'za' in vars:
#         with sessionPaths['ZoneAnalyses'].open(mode='rb') as f:
#             dat['za'] = pkl.load(f)
#
#     if 'PosDat' in vars:
#         dat['PosDat'] = TMF.getBehTrackData(sessionPaths)
#
#     if 'TrialLongMat' in vars:
#         dat['TrialLongMat'] = pd.read_csv(sessionPaths['TrLongPosMat'], index_col=0)
#
#     if 'TrialFRLongMat' in vars:
#         dat['TrialFRLongMat'] = pd.read_csv(sessionPaths['TrLongPosFRDat'], index_col=0)
#
#     if 'TrialConds' in vars:
#         dat['TrialConds'] = pd.read_csv(sessionPaths['TrialCondMat'], index_col=0)
#
#     if 'fitTable' in vars:
#
#         def addModelName(fitTable, fitNum):
#             mods = {}
#             if fitNum == 1:
#                 params = TA.getParamSet()
#             else:
#                 params = TA.getParamSet(params=['Loc', 'IO', 'Cue', 'Sp', 'Co'])
#
#             for k, pp in params.items():
#                 s = ''
#                 for p in pp:
#                     s += '-' + p
#                 mods[k] = s[1:]
#             selModels = []
#
#             for u in fitTable['modelNum']:
#                 if u > -1:
#                     selModels.append(mods[int(u)])
#                 else:
#                     selModels.append('UnCla')
#             fitTable['selMod'] = selModels
#             return fitTable
#
#         if sessionPaths['TrModelFits'].exists():
#             dat['fitTable'] = pd.read_csv(sessionPaths['TrModelFits'], index_col=0)
#             if not ('selMod' in dat['fitTable'].columns):
#                 dat['fitTable'] = addModelName(dat['fitTable'], 1)
#         if sessionPaths['TrModelFits2'].exists():
#             dat['fitTable2'] = pd.read_csv(sessionPaths['TrModelFits2'], index_col=0)
#             if not ('selMod' in dat['fitTable2'].columns):
#                 dat['fitTable2'] = addModelName(dat['fitTable2'], 2)
#
#         # if isinstance(dat['fitTable'] ,pd.core.frame.DataFrame):
#         #    nUnits = dat['fitTable'] .shape[0]
#         #    x=[]
#         #    for i in np.arange(nUnits):
#         #        if np.isnan(dat['fitTable'] ['modelNum'][i]):
#         #            x.append('UnCla')
#         #        else:
#         #            x.append(mods[dat['fitTable'] ['modelNum'][i]])
#         #    dat['fitTable']['selMod'] = x
#
#     return dat
