from pathlib import Path
import numpy as np
import pandas as pd
import time
import pickle
import json
import h5py
import sys
import traceback
import datetime
from importlib import reload
import warnings
from joblib import delayed, Parallel

import Analyses.spike_functions as spike_funcs
import Analyses.spatial_functions as spatial_funcs
import Analyses.open_field_functions as of_funcs

import Pre_Processing.pre_process_functions as pp_funcs
import Utils.robust_stats as rs
import Analyses.tree_maze_functions as tmf
import Analyses.plot_functions as pf

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

"""


class SummaryInfo:
    subjects = ['Li', 'Ne', 'Cl', 'Al', 'Ca', 'Mi']
    min_n_units = 1
    min_n_trials = 50  # task criteria
    min_pct_coverage = 0.75  # open field criteria
    invalid_sessions = ['Li_OF_080718']
    figure_names = [f"f{ii}" for ii in range(5)]

    _root_paths = dict(GD=Path("/home/alexgonzalez/google-drive/TreeMazeProject/"),
                       BigPC=Path("/mnt/Data_HD2T/TreeMazeProject/"))

    def __init__(self, data_root='BigPC'):
        self.main_path = self._root_paths[data_root]
        self.paths = self._get_paths()
        self.unit_table = self.get_unit_table()
        self.analyses_table = self.get_analyses_table()
        self.valid_track_table = self.get_track_validity_table()

        self.sessions_by_subject = {}
        self.tasks_by_subject = {}
        for s in self.subjects:
            self.sessions_by_subject[s] = self.unit_table[self.unit_table.subject == s].session.unique()
            self.tasks_by_subject[s] = self.unit_table[self.unit_table.subject == s].task.unique()

    def run_analyses(self, task='all', which='all', verbose=False, overwrite=False,
                     overwrite_old=False, overwrite_old_days=1, parallel=False, n_jobs=5, **params):

        if parallel:
            parallel_obj = Parallel(n_jobs=n_jobs)
            params.update({'parallel': parallel_obj})
        interrupt_flag = False
        for subject in self.subjects:
            if not interrupt_flag:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    try:
                        if task == 'all':
                            pass
                        elif task not in session:
                            continue
                        else:
                            pass

                        if verbose:
                            t0 = time.time()
                            print(f'Processing Session {session}')

                        session_info = SubjectSessionInfo(subject, session)
                        session_info.run_analyses(which=which, overwrite=overwrite, verbose=verbose,
                                                  overwrite_old=overwrite_old, overwrite_old_days=overwrite_old_days,
                                                  **params)
                        if verbose:
                            t1 = time.time()
                            print(f"Session Processing Completed: {t1 - t0:0.2f}s")
                            print()
                        else:
                            print(".", end='')
                    except KeyboardInterrupt:
                        interrupt_flag = True
                        break
                    except ValueError:
                        pass
                    except FileNotFoundError:
                        pass
                    except:
                        print(f"{session} analysis {which} failed.")
                        if verbose:
                            traceback.print_exc(file=sys.stdout)
                        pass
            else:
                break
            if verbose:
                print(f"Subject {subject} Analyses Completed.")

    def get_analyses_table(self, overwrite=False):

        if not self.paths['analyses_table'].exists() or overwrite:
            analyses_table = pd.DataFrame()
            for subject in self.subjects:
                analyses_table = analyses_table.append(SubjectInfo(subject).get_sessions_analyses())

            analyses_table.to_csv(self.paths['analyses_table'])
        else:
            analyses_table = pd.read_csv(self.paths['analyses_table'], index_col=0)

        self.analyses_table = analyses_table
        return analyses_table

    def get_track_validity_table(self, overwrite=False):
        if not self.paths['valid_track_table'].exists() or overwrite:
            valid_track_table = pd.DataFrame()
            for subject in self.subjects:
                valid_track_table = valid_track_table.append(SubjectInfo(subject).valid_track_table)

            valid_track_table.to_csv(self.paths['valid_track_table'])
        else:
            valid_track_table = pd.read_csv(self.paths['valid_track_table'], index_col=0)

        return valid_track_table

    def get_behav_perf(self, overwrite=False):
        if not self.paths['behavior'].exists() or overwrite:
            perf = pd.DataFrame()
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    if 'T3' in session:
                        try:
                            session_info = SubjectSessionInfo(subject, session)
                            b = session_info.get_event_behavior()
                            sp = b.get_session_perf()

                            sp['session'] = session
                            sp['task'] = session_info.task
                            sp['subject'] = subject
                            sp['n_units'] = session_info.n_units
                            sp['n_cells'] = session_info.n_cells
                            sp['n_mua'] = session_info.n_mua
                            perf = pd.concat((perf, sp), ignore_index=True)
                        except:
                            pass

            perf.to_csv(self.paths['behavior'])
        else:
            perf = pd.read_csv(self.paths['behavior'], index_col=0)

        return perf

    def _get_paths(self, root_path=None):
        if root_path is None:
            results_path = self.main_path / 'Results_Summary'
            figures_path = self.main_path / 'Figures'
        else:
            results_path = root_path / 'Results_Summary'
            figures_path = root_path / 'Figures'

        paths = dict(
            analyses_table=results_path / 'analyses_table.csv',
            valid_track_table=results_path / 'valid_track_table.csv',
            behavior=results_path / 'behavior_session_perf.csv',
            units=results_path / 'all_units_table.csv',
            match_table=results_path / "match_table.csv",
            combined_scores_table=results_path / "combined_scores_table.csv",
            of_metric_scores=results_path / 'of_metric_scores_summary_table.csv',
            of_model_scores=results_path / 'of_model_scores_summary_table_agg.csv',
            zone_rates_comps=results_path / 'zone_rates_comps_summary_table.csv',
            zone_rates_remap=results_path / 'zone_rates_remap_summary_table.csv',
            pop_zone_rates_remap=results_path / 'pop_zone_rates_remap_summary_table.csv',
            bal_conds_seg_rates=results_path / 'bal_conds_seg_rates_summary_table.csv',
            zone_encoder=results_path / 'zone_encoder.csv',
            zone_encoder_comps=results_path / 'zone_encoder_comps.csv',
            segment_rate_comps=results_path / 'segment_rate_comps.csv',
            zone_encoder_lag=results_path / 'zone_encoder_lag.csv',
            zone_encoder_cue=results_path / 'zone_encoder_cue.csv',
            zone_decoder=results_path / 'zone_decoder.csv',
            zone_decoder_dec=results_path / 'zone_decoder_dec.csv',
        )
        paths['results'] = results_path
        paths['figures'] = figures_path

        return paths

    def update_paths(self):
        for subject in self.subjects:
            si = SubjectInfo(subject)
            si.update_paths()

    def get_zone_rates_comps(self, overwrite=False, **params):
        """
        Aggregates tables across sessions and adds unit information.
        Note, that overwrite only overwrites the aggregate table and does not perform the analysis on each session.
        :param overwrite:
        :return:
        pandas data frame with n_units as index

        """

        if not self.paths['zone_rates_comps'].exists() or overwrite:
            sessions_validity = self.get_track_validity_table()
            zone_rates = pd.DataFrame()
            unit_count = 0

            valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_rates_comps == True].index)
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    if session in valid_sessions:
                        session_info = SubjectSessionInfo(subject, session)
                        n_session_units = session_info.n_units
                        if n_session_units > 0:
                            try:
                                session_zone_rate_comp_table = session_info.get_zone_rates_remapping()
                                comp_table_columns = session_zone_rate_comp_table.columns

                                session_table = pd.DataFrame(index=np.arange(n_session_units),
                                                             columns=['unit_id', 'subject', 'session',
                                                                      'session_pct_cov', 'session_valid',
                                                                      'session_unit_id', 'unit_type', 'tt', 'tt_cl',
                                                                      'cl_name'])

                                session_table['session'] = session
                                session_table['subject'] = session_info.subject
                                session_table['session_unit_id'] = np.arange(n_session_units)

                                session_table['unit_id'] = np.arange(n_session_units) + unit_count
                                session_table['unit_type'] = [v[0] for k, v in session_info.cluster_ids.items()]
                                session_table['tt'] = [v[1] for k, v in session_info.cluster_ids.items()]
                                session_table['tt_cl'] = [v[2] for k, v in session_info.cluster_ids.items()]

                                if session in sessions_validity.columns:
                                    session_table['session_pct_cov'] = sessions_validity[session]
                                    session_table['session_valid'] = 1
                                else:
                                    session_table['session_pct_cov'] = 0
                                    session_table['session_valid'] = 0

                                cl_names = []
                                for k, v in session_info.cluster_ids.items():
                                    tt = v[1]
                                    cl = v[2]
                                    depth = subject_info.sessions_tt_positions.loc[session, f"tt_{tt}"]
                                    cl_name = f"{session}-tt{tt}_d{depth}_cl{cl}"
                                    cl_names.append(cl_name)

                                session_table['cl_name'] = cl_names

                                unit_count += n_session_units

                                session_table = session_table.join(session_zone_rate_comp_table)
                            except:
                                print(f'Error Processing Session {session}')
                                traceback.print_exc(file=sys.stdout)
                                continue
                            zone_rates = zone_rates.append(session_table)

            zone_rates = zone_rates.reset_index(drop=True)
            zone_rates.to_csv(self.paths['zone_rates_comps'])
        else:
            zone_rates = pd.read_csv(self.paths['zone_rates_comps'], index_col=0)

        return zone_rates

    def get_bal_conds_seg_rates(self, segment_type='bigseg', overwrite=False):

        fn = self.paths['bal_conds_seg_rates']
        if segment_type != 'bigseg':
            name = fn.name.split('.')
            name2 = name[0] + segment_type + name[1]
            fn = fn.parent / name2

        if not fn.exists() or overwrite:
            sessions_validity = self.get_track_validity_table()
            seg_rates = pd.DataFrame()
            unit_count = 0

            valid_sessions = list(self.analyses_table.loc[self.analyses_table.bal_conds_seg_rates == True].index)
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    if session in valid_sessions:
                        session_info = SubjectSessionInfo(subject, session)
                        n_session_units = session_info.n_units
                        if n_session_units > 0:
                            try:
                                session_zone_rate_comp_table = session_info.get_bal_conds_seg_rates(
                                    segment_type=segment_type)
                                comp_table_columns = session_zone_rate_comp_table.columns

                                session_table = pd.DataFrame(index=np.arange(n_session_units),
                                                             columns=['unit_id', 'subject', 'session',
                                                                      'session_pct_cov', 'session_valid',
                                                                      'session_unit_id', 'unit_type', 'tt', 'tt_cl',
                                                                      'cl_name'])

                                session_table['session'] = session
                                session_table['subject'] = session_info.subject
                                session_table['session_unit_id'] = np.arange(n_session_units)

                                session_table['unit_id'] = np.arange(n_session_units) + unit_count
                                session_table['unit_type'] = [v[0] for k, v in session_info.cluster_ids.items()]
                                session_table['tt'] = [v[1] for k, v in session_info.cluster_ids.items()]
                                session_table['tt_cl'] = [v[2] for k, v in session_info.cluster_ids.items()]

                                if session in sessions_validity.columns:
                                    session_table['session_pct_cov'] = sessions_validity[session]
                                    session_table['session_valid'] = 1
                                else:
                                    session_table['session_pct_cov'] = 0
                                    session_table['session_valid'] = 0

                                cl_names = []
                                for k, v in session_info.cluster_ids.items():
                                    tt = v[1]
                                    cl = v[2]
                                    depth = subject_info.sessions_tt_positions.loc[session, f"tt_{tt}"]
                                    cl_name = f"{session}-tt{tt}_d{depth}_cl{cl}"
                                    cl_names.append(cl_name)

                                session_table['cl_name'] = cl_names

                                unit_count += n_session_units

                                session_table = session_table.join(session_zone_rate_comp_table)
                            except:
                                print(f'Error Processing Session {session}')
                                traceback.print_exc(file=sys.stdout)
                                continue
                            seg_rates = seg_rates.append(session_table)

            seg_rates = seg_rates.reset_index(drop=True)
            seg_rates.to_csv(fn)
        else:
            seg_rates = pd.read_csv(fn, index_col=0)

        return seg_rates

    def get_zone_rates_remap(self, overwrite=False, **remap_params):

        params = dict(zr_method='trial',
                      corr_method='kendall',
                      reward_blank=False,
                      not_inzone_blank=True,
                      valid_transitions_blank=True,
                      sp_blank=False,
                      trial_end='tE_2')

        params.update(remap_params)

        fn = self.paths['zone_rates_remap']
        if len(remap_params) > 0:
            fn = append_analysis_mods_2_filename(fn, params)

        if not fn.exists() or overwrite:
            sessions_validity = self.get_track_validity_table()
            zone_rates = pd.DataFrame()
            unit_count = 0

            valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_rates_comps == True].index)
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    if session in valid_sessions:
                        session_info = SubjectSessionInfo(subject, session)
                        n_session_units = session_info.n_units
                        if n_session_units > 0:
                            try:
                                session_zone_rate_comp_table = session_info.get_zone_rates_remap(overwrite=False,
                                                                                                 **params)
                                comp_table_columns = session_zone_rate_comp_table.columns

                                session_table = pd.DataFrame(index=np.arange(n_session_units),
                                                             columns=['unit_id', 'subject', 'session',
                                                                      'session_pct_cov', 'session_valid',
                                                                      'session_unit_id', 'unit_type', 'tt', 'tt_cl',
                                                                      'cl_name'])

                                session_table['session'] = session
                                session_table['subject'] = session_info.subject
                                session_table['session_unit_id'] = np.arange(n_session_units)

                                session_table['unit_id'] = np.arange(n_session_units) + unit_count
                                session_table['unit_type'] = [v[0] for k, v in session_info.cluster_ids.items()]
                                session_table['tt'] = [v[1] for k, v in session_info.cluster_ids.items()]
                                session_table['tt_cl'] = [v[2] for k, v in session_info.cluster_ids.items()]

                                if session in sessions_validity.columns:
                                    session_table['session_pct_cov'] = sessions_validity[session]
                                    session_table['session_valid'] = 1
                                else:
                                    session_table['session_pct_cov'] = 0
                                    session_table['session_valid'] = 0

                                cl_names = []
                                for k, v in session_info.cluster_ids.items():
                                    tt = v[1]
                                    cl = v[2]
                                    depth = subject_info.sessions_tt_positions.loc[session, f"tt_{tt}"]
                                    cl_name = f"{session}-tt{tt}_d{depth}_cl{cl}"
                                    cl_names.append(cl_name)

                                session_table['cl_name'] = cl_names

                                unit_count += n_session_units

                                session_table = session_table.join(session_zone_rate_comp_table)
                            except:
                                print(f'Error Processing Session {session}')
                                traceback.print_exc(file=sys.stdout)
                                continue
                            zone_rates = zone_rates.append(session_table)

            zone_rates = zone_rates.reset_index(drop=True)
            zone_rates.to_csv(fn)
        else:
            zone_rates = pd.read_csv(fn, index_col=0)

        return zone_rates

    def get_pop_zone_rates_remap(self, overwrite=False, **remap_params):

        params = dict(corr_method='kendall',
                      reward_blank=False,
                      not_inzone_blank=True,
                      valid_transitions_blank=True,
                      sp_blank=False,
                      trial_end='tE_2')

        params.update(remap_params)

        fn = self.paths['pop_zone_rates_remap']
        if len(remap_params) > 0:
            fn = append_analysis_mods_2_filename(fn, params)

        if not fn.exists() or overwrite:
            sessions_validity = self.get_track_validity_table()
            zone_rates = pd.DataFrame()

            cnt = 0
            valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_rates_comps == True].index)
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    if session in valid_sessions:
                        session_info = SubjectSessionInfo(subject, session)
                        n_session_units = session_info.n_units
                        n_session_cells = session_info.n_cells
                        n_session_muas = session_info.n_mua

                        if n_session_units > 0:
                            try:
                                session_zone_rate_comp_table = session_info.get_pop_zone_rates_remap(**params)
                                comp_table_columns = session_zone_rate_comp_table.columns

                                session_table = pd.DataFrame(index=np.arange(3),
                                                             columns=['subject', 'session',
                                                                      'session_pct_cov', 'session_valid'])

                                session_table['session'] = session
                                session_table['subject'] = session_info.subject

                                if session in sessions_validity.columns:
                                    session_table['session_pct_cov'] = sessions_validity[session]
                                    session_table['session_valid'] = 1
                                else:
                                    session_table['session_pct_cov'] = 0
                                    session_table['session_valid'] = 0

                                session_table['n_session_all'] = n_session_units
                                session_table['n_session_cells'] = n_session_cells
                                session_table['n_session_muas'] = n_session_muas

                                temp = session_zone_rate_comp_table.reset_index()
                                temp = temp.rename(columns={'index': 'pop_type'})
                                temp.loc[temp.pop_type == 'units', 'pop_type'] = 'all'
                                session_table = session_table.join(temp)
                            except:
                                print(f'Error Processing Session {session}')
                                traceback.print_exc(file=sys.stdout)
                                continue
                            zone_rates = zone_rates.append(session_table)

            zone_rates = zone_rates.reset_index(drop=True)
            zone_rates.to_csv(fn)
        else:
            zone_rates = pd.read_csv(fn, index_col=0)

        return zone_rates

    def get_zone_encoder(self, overwrite=False):
        fn = self.paths['zone_encoder']
        exp_vars = ['max_lag', 'cue_type', 'rw_type', 'sp_type', 'dir_type', 'trial_seg']
        valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_encoder == 1].index)

        if not fn.exists() or overwrite:
            encoder_res = pd.DataFrame()

            unit_count = 0
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    session_info = SubjectSessionInfo(subject, session)

                    try:
                        if session not in valid_sessions:
                            continue

                        if not (session_info.session_analyses_table.zone_encoder == 1).values[0]:
                            # skip if it hasn't been run
                            continue
                        if session_info.n_units == 0:
                            # skip if no units
                            continue

                        n_session_units = session_info.n_units

                        session_unit_table = session_info.session_unit_table
                        # session_encoder_res = session_info.get_zone_encoder_cue()
                        res = session_info.get_zone_encoder()
                        res = res[res.split == 'test']

                        res = res.groupby(['unit', 'unit_type'] + exp_vars,
                                          observed=True).mean().reset_index()
                        res = res.drop(columns=['fold'])
                        res.rename(columns={'unit': 'session_unit_id'}, inplace=True)

                        s_enc_table = pd.concat((res,
                                                 session_unit_table.loc[res.session_unit_id].reset_index(
                                                     drop=True)), axis=1)
                        s_enc_table = s_enc_table.loc[:, ~s_enc_table.columns.duplicated()]
                        s_enc_table['unit_id'] = s_enc_table['session_unit_id'] + unit_count
                        unit_count += n_session_units
                    except:
                        print(f'Error Processing Session {session}')
                        traceback.print_exc(file=sys.stdout)
                        continue
                    encoder_res = encoder_res.append(s_enc_table)

            encoder_res = encoder_res.reset_index(drop=True)
            encoder_res.to_csv(fn)

        else:
            encoder_res = pd.read_csv(fn, index_col=0)

        encoder_res['cue_type'] = encoder_res['cue_type'].astype(
            pd.api.types.CategoricalDtype(['none', 'fixed', 'inter']))
        encoder_res['rw_type'] = encoder_res['rw_type'].astype(
            pd.api.types.CategoricalDtype(['none', 'fixed', 'inter']))

        return encoder_res

    def get_zone_encoder_comps(self, overwrite=False):
        fn = self.paths['zone_encoder_comps']
        exp_vars = ['max_lag', 'cue_type', 'rw_type', 'sp_type', 'dir_type', 'trial_seg']
        valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_encoder == 1].index)

        encoder_comps_dict = tmf.zone_encoder_comps_dict()

        def _select_data_rows(data, selections):
            bool_idx = _get_condition_row_idx(data, **selections)
            return data.loc[bool_idx].copy()

        def _get_condition_row_idx(data, **selections):
            bool_idx = np.ones(len(data), dtype=bool)
            for col, val in selections.items():
                if isinstance(val, list):
                    bool_idx &= (data[col].isin(val))
                else:
                    bool_idx &= (data[col] == val)
            return bool_idx

        def _get_comp_subtables(data, expt, comp):
            exp_data_selections = encoder_comps_dict[expt]['selections']
            exp_comp = encoder_comps_dict[expt]['comps'][comp]

            subdata = _select_data_rows(data, selections=exp_data_selections)
            test = subdata[subdata[exp_comp['col']] == exp_comp['test']].pivot(index='unit',
                                                                               columns='fold',
                                                                               values='score').T

            null = subdata[subdata[exp_comp['col']] == exp_comp['null']].pivot(index='unit',
                                                                               columns='fold',
                                                                               values='score').T
            return test, null

        if not fn.exists() or overwrite:
            encoder_res = pd.DataFrame()

            unit_count = 0
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    session_info = SubjectSessionInfo(subject, session)

                    try:
                        if session not in valid_sessions:
                            continue

                        if not (session_info.session_analyses_table.zone_encoder == 1).values[0]:
                            # skip if it hasn't been run
                            continue
                        if session_info.n_units == 0:
                            # skip if no units
                            continue

                        n_session_units = session_info.n_units

                        session_unit_table = session_info.session_unit_table
                        # session_encoder_res = session_info.get_zone_encoder_cue()
                        res = session_info.get_zone_encoder()
                        res = res[res.split == 'test']

                        res_comp = pd.DataFrame(index=range(n_session_units * len(encoder_comps_dict) * 3),
                                                columns=['unit', 'expt', 'comp', 'mean_test', 'mean_null', 'test_cond',
                                                         'null_cond', 'uz', 'md'])

                        units = np.arange(n_session_units)
                        block_idx = np.arange(n_session_units)
                        for expt in encoder_comps_dict.keys():
                            for comp in encoder_comps_dict[expt]['comps'].keys():
                                test, null = _get_comp_subtables(res, expt, comp)
                                uz = rs.mannwhitney_z(test, null)
                                md = (test - null).median().values

                                test_name = encoder_comps_dict[expt]['comps'][comp]['test']
                                null_name = encoder_comps_dict[expt]['comps'][comp]['null']

                                res_comp.loc[block_idx, 'unit'] = units
                                res_comp.loc[block_idx, 'expt'] = expt
                                res_comp.loc[block_idx, 'comp'] = comp
                                res_comp.loc[block_idx, 'test_cond'] = test_name
                                res_comp.loc[block_idx, 'null_cond'] = null_name
                                res_comp.loc[block_idx, 'mean_test'] = test.mean().values
                                res_comp.loc[block_idx, 'mean_null'] = null.mean().values
                                res_comp.loc[block_idx, 'uz'] = uz
                                res_comp.loc[block_idx, 'md'] = md

                                block_idx = block_idx + n_session_units

                        res_comp.rename(columns={'unit': 'session_unit_id'}, inplace=True)
                        # res = res.groupby(['unit', 'unit_type'] + exp_vars,
                        #                                        observed=True).mean().reset_index()
                        # res = res.drop(columns=['fold'])
                        # res.rename(columns={'unit': 'session_unit_id'}, inplace=True)

                        s_enc_table = pd.concat((res_comp,
                                                 session_unit_table.loc[res_comp.session_unit_id].reset_index(
                                                     drop=True)), axis=1)
                        s_enc_table = s_enc_table.loc[:, ~s_enc_table.columns.duplicated()]
                        s_enc_table['unit_id'] = s_enc_table['session_unit_id'] + unit_count
                        unit_count += n_session_units
                    except:
                        print(f'Error Processing Session {session}')
                        traceback.print_exc(file=sys.stdout)
                        continue
                    encoder_res = encoder_res.append(s_enc_table)

            encoder_res = encoder_res.reset_index(drop=True)
            encoder_res.to_csv(fn)

        else:
            encoder_res = pd.read_csv(fn, index_col=0)

        return encoder_res

    def get_segment_rate_comps(self, verbose=False, overwrite=False):
        fn = self.paths['segment_rate_comps']
        valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_rates_comps == True].index)

        comps = ['cue', 'rw', 'dir']
        if not fn.exists() or overwrite:
            out_df = pd.DataFrame()
            unit_count = 0
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    session_info = SubjectSessionInfo(subject, session)

                    try:
                        if session not in valid_sessions:
                            continue

                        if session_info.n_units == 0:
                            # skip if no units
                            continue

                        n_session_units = session_info.n_units

                        ta = tmf.TrialAnalyses(session_info)

                        res_dict = {}
                        for comp in comps:
                            res_dict[comp] = tmf.rate_segment_comp_analysis(session_info, comp=comp, ta=ta)
                            res_dict[comp]['comp'] = comp

                        res = pd.DataFrame()
                        for comp in comps:
                            res = pd.concat((res, res_dict[comp]))

                        res = res.reset_index(drop=True)
                        res.rename(columns={'unit': 'session_unit_id'}, inplace=True)

                        session_table = session_info.session_unit_table
                        res_table = pd.concat((res,
                                               session_table.loc[res.session_unit_id].reset_index(
                                                   drop=True)), axis=1)
                        res_table = res_table.loc[:, ~res_table.columns.duplicated()]
                        res_table['unit_id'] = res_table['session_unit_id'] + unit_count
                        unit_count += n_session_units

                        out_df = out_df.append(res_table)
                        print('.', end='')
                    except:
                        if verbose:
                            print(f'Error Processing Session {session}')
                            traceback.print_exc(file=sys.stdout)
                        continue

            out_df = out_df.reset_index(drop=True)
            out_df.to_csv(fn)
        else:
            out_df = pd.read_csv(fn, index_col=0)

        out_df['segment'] = out_df['segment'].astype(
            pd.api.types.CategoricalDtype(['left', 'stem', 'right']))

        return out_df

    # def get_zone_encoder_lag(self, overwrite=False):
    #
    #     fn = self.paths['zone_encoder_lag']
    #
    #     valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_encoder_lag == True].index)
    #     if not fn.exists() or overwrite:
    #         encoder_res = pd.DataFrame()
    #
    #         unit_count = 0
    #         for subject in self.subjects:
    #             subject_info = SubjectInfo(subject)
    #             for session in subject_info.sessions:
    #                 session_info = SubjectSessionInfo(subject, session)
    #
    #                 try:
    #                     if session not in valid_sessions:
    #                         continue
    #
    #                     if not (session_info.session_analyses_table.zone_encoder_lag == 1).values[0]:
    #                         # skip if it hasn't been run
    #                         continue
    #                     if session_info.n_units == 0:
    #                         # skip if no units
    #                         continue
    #
    #                     n_session_units = session_info.n_units
    #
    #                     # session_encoder_res = session_info.get_zone_encoder_lag()
    #                     session_encoder_res = session_info.get_zone_encoder()
    #                     session_encoder_res = session_encoder_res[(session_encoder_res.cue_type == 'none') &
    #                                                               (session_encoder_res.rw_type == 'none') &
    #                                                               (session_encoder_res.trial_seg == 'out')
    #                                                               ]
    #                     session_encoder_res.rename(columns={'unit': 'session_unit_id'}, inplace=True)
    #                     session_table = session_info.session_unit_table
    #
    #                     s_enc_table = pd.concat((session_encoder_res,
    #                                              session_table.loc[session_encoder_res.session_unit_id].reset_index(
    #                                                  drop=True)), axis=1)
    #                     s_enc_table = s_enc_table.loc[:, ~s_enc_table.columns.duplicated()]
    #                     s_enc_table['unit_id'] = s_enc_table['session_unit_id'] + unit_count
    #                     unit_count += n_session_units
    #                 except:
    #                     print(f'Error Processing Session {session}')
    #                     traceback.print_exc(file=sys.stdout)
    #                     continue
    #                 encoder_res = encoder_res.append(s_enc_table)
    #
    #         encoder_res = encoder_res.reset_index(drop=True)
    #
    #         encoder_res.to_csv(fn)
    #     else:
    #         encoder_res = pd.read_csv(fn, index_col=0)
    #
    #     # encoder_res = encoder_res.astype(dtype=dict(trial='int16', zones='category', session_unit_id='int16',
    #     #                                             sp='float16', fr='float16', fr_hat='float16', resid='float16',
    #     #                                             r2='float16', nrmse='float16', fold='float16', lag='category',
    #     #                                             decay='category', cue_type='category', unit_type='category',
    #     #                                             subject='category', session='category', session_pct_cov='float16',
    #     #                                             task='category', unit_id='int16'), errors='ignore')
    #     return encoder_res
    #
    # def get_zone_encoder_cue(self, overwrite=False):
    #     fn = self.paths['zone_encoder_cue']
    #
    #     valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_encoder_cue == True].index)
    #     if not fn.exists() or overwrite:
    #         encoder_res = pd.DataFrame()
    #
    #         unit_count = 0
    #         for subject in self.subjects:
    #             subject_info = SubjectInfo(subject)
    #             for session in subject_info.sessions:
    #                 session_info = SubjectSessionInfo(subject, session)
    #
    #                 try:
    #                     if session not in valid_sessions:
    #                         continue
    #
    #                     if not (session_info.session_analyses_table.zone_encoder_cue == 1).values[0]:
    #                         # skip if it hasn't been run
    #                         continue
    #                     if session_info.n_units == 0:
    #                         # skip if no units
    #                         continue
    #
    #                     n_session_units = session_info.n_units
    #
    #                     # session_encoder_res = session_info.get_zone_encoder_cue()
    #                     session_encoder_res = session_info.get_zone_encoder()
    #                     session_encoder_res = session_encoder_res[(session_encoder_res.max_lag == 50) &
    #                                                               (session_encoder_res.rw_type == 'none') &
    #                                                               (session_encoder_res.trial_seg == 'out')
    #                                                               ]
    #                     session_encoder_res.rename(columns={'unit': 'session_unit_id'}, inplace=True)
    #                     session_table = session_info.session_unit_table
    #
    #                     s_enc_table = pd.concat((session_encoder_res,
    #                                              session_table.loc[session_encoder_res.session_unit_id].reset_index(
    #                                                  drop=True)), axis=1)
    #                     s_enc_table = s_enc_table.loc[:, ~s_enc_table.columns.duplicated()]
    #                     s_enc_table['unit_id'] = s_enc_table['session_unit_id'] + unit_count
    #                     unit_count += n_session_units
    #                 except:
    #                     print(f'Error Processing Session {session}')
    #                     traceback.print_exc(file=sys.stdout)
    #                     continue
    #                 encoder_res = encoder_res.append(s_enc_table)
    #
    #         encoder_res = encoder_res.reset_index(drop=True)
    #         # encoder_res = encoder_res.astype(dtype=dict(trial='int16', zones='category', session_unit_id='int16',
    #         #                                             sp='float16', fr='float16', fr_hat='float16', resid='float16',
    #         #                                             r2='float16', nrmse='float16', fold='float16', lag='category',
    #         #                                             decay='category', cue_type='category', unit_type='category',
    #         #                                             subject='category', session='category', task='category',
    #         #                                             session_pct_cov='float16', tt='category', tt_cl='category',
    #         #                                             cl_name='category', unit_id='int16'), errors='ignore')
    #
    #         encoder_res.to_csv(fn)
    #         # encoder_res.to_pickle(fn)
    #     else:
    #         encoder_res = pd.read_csv(fn, index_col=0)
    #         # encoder_res = pd.read_pickle(fn, index_col=0)
    #
    #     # tmz = tmf.TreeMazeZones()
    #     # encoder_res['zones'] = encoder_res['zones'].astype(pd.api.types.CategoricalDtype(tmz.zones2))
    #     encoder_res['cue_type'] = encoder_res['cue_type'].astype(
    #         pd.api.types.CategoricalDtype(['none', 'fixed', 'inter']))
    #
    #     return encoder_res

    def get_zone_decoder(self, overwrite=False):
        fn = self.paths['zone_decoder']

        valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_decoder == True].index)
        if not fn.exists() or overwrite:
            decoder_res = pd.DataFrame()

            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    session_info = SubjectSessionInfo(subject, session)

                    try:
                        if session not in valid_sessions:
                            continue

                        if not (session_info.session_analyses_table.zone_decoder == 1).values[0]:
                            # skip if it hasn't been run
                            continue
                        if session_info.n_units == 0:
                            # skip if no units
                            continue

                        n_session_units = session_info.n_units

                        session_decoder_res = session_info.get_zone_decoder()
                        session_decoder_res = session_decoder_res.groupby(['encoder_type', 'feature_type',
                                                                           'target_type', 'fold', 'zones', 'cue'])[
                            'acc', 'bac', 'dist', 'logit_dist'].mean().reset_index()
                        session_decoder_res['subject'] = subject
                        session_decoder_res['session'] = session
                        session_decoder_res['task'] = session_info.task
                        session_decoder_res['n_units'] = n_session_units

                    except:
                        print(f'Error Processing Session {session}')
                        traceback.print_exc(file=sys.stdout)
                        continue
                    decoder_res = decoder_res.append(session_decoder_res)

            decoder_res = decoder_res.reset_index(drop=True)
            decoder_res.to_csv(fn)
        else:
            decoder_res = pd.read_csv(fn, index_col=0)

        tmz = tmf.TreeMazeZones()
        decoder_res['zones'] = decoder_res['zones'].astype(pd.api.types.CategoricalDtype(tmz.zones2))
        return decoder_res

    def get_zone_decoder_2_subj_behav(self, overwrite=False):
        fn = self.paths['zone_decoder_dec']

        zones2 = tmf.TreeMazeZones().zones2
        stem_zones = zones2[:9]
        branch_zones = zones2[9:]
        valid_sessions = list(self.analyses_table.loc[self.analyses_table.zone_decoder == True].index)
        if not fn.exists() or overwrite:
            decoder_res = pd.DataFrame()

            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    session_info = SubjectSessionInfo(subject, session)

                    try:
                        if session not in valid_sessions:
                            continue

                        if not (session_info.session_analyses_table.zone_decoder == 1).values[0]:
                            # skip if it hasn't been run
                            continue
                        if session_info.n_units == 0:
                            # skip if no units
                            continue

                        n_session_units = session_info.n_units

                        session_decoder_res = session_info.get_zone_decoder()
                        session_decoder_res = session_decoder_res[session_decoder_res.target_type.isin(['dec', 'cue'])]
                        session_decoder_res['cue_bias'] = session_decoder_res['pred'] == session_decoder_res['cue']
                        session_decoder_res['subj_perf'] = session_decoder_res['correct'] == 1
                        session_decoder_res['subj_perf_match'] = session_decoder_res['subj_perf'] & \
                                                                 (session_decoder_res['acc'] == 1)

                        session_decoder_res.loc[session_decoder_res.zones.isin(stem_zones), 'seg'] = 'stem'
                        session_decoder_res.loc[session_decoder_res.zones.isin(branch_zones), 'seg'] = 'branch'

                        session_decoder_res = \
                            session_decoder_res.groupby(['encoder_type', 'target_type', 'zones', 'seg'])[[
                                'acc', 'cue_bias', 'subj_perf', 'subj_perf_match']].mean().reset_index()
                        session_decoder_res = session_decoder_res.dropna().reset_index(drop=True)

                        session_decoder_res['subject'] = subject
                        session_decoder_res['session'] = session
                        session_decoder_res['task'] = session_info.task
                        session_decoder_res['n_units'] = n_session_units

                    except:
                        print(f'Error Processing Session {session}')
                        traceback.print_exc(file=sys.stdout)
                        continue
                    decoder_res = decoder_res.append(session_decoder_res)

            decoder_res = decoder_res.reset_index(drop=True)
            decoder_res.to_csv(fn)
        else:
            decoder_res = pd.read_csv(fn, index_col=0)

        tmz = tmf.TreeMazeZones()
        decoder_res['zones'] = decoder_res['zones'].astype(pd.api.types.CategoricalDtype(tmz.zones2))
        return decoder_res

    def get_of_results(self, overwrite=False):

        curate_flag = False
        # get metrics
        if not self.paths['of_metric_scores'].exists() or overwrite:
            metric_scores = self._get_of_metric_scores()
            curate_flag = True
        else:
            metric_scores = pd.read_csv(self.paths['of_metric_scores'], index_col=0)

        # get models
        if not self.paths['of_model_scores'].exists() or overwrite:
            model_scores = self._get_of_models_scores()
            curate_flag = True
        else:
            model_scores = pd.read_csv(self.paths['of_model_scores'], index_col=0)

        if curate_flag:
            metric_scores, model_scores = self._match_unit_ids(metric_scores, model_scores)

            for session in self.invalid_sessions:
                unit_idx = self.unit_table[self.unit_table.session == session].unique_cl_name
                metric_scores.loc[metric_scores.cl_name.isin(unit_idx), 'session_valid'] = False
                model_scores.loc[model_scores.cl_name.isin(unit_idx), 'session_valid'] = False

            metric_scores.to_csv(self.paths['of_metric_scores'])
            model_scores.to_csv(self.paths['of_model_scores'])

        return metric_scores, model_scores

    def _get_of_metric_scores(self, overwrite=False):
        if not self.paths['of_metric_scores'].exists() or overwrite:
            analyses = ['speed', 'hd', 'border', 'grid', 'stability']
            output_scores_names = ['score', 'sig']
            n_analyses = len(analyses)

            unit_count = 0
            metric_scores = pd.DataFrame()
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)
                for session in subject_info.sessions:
                    if 'OF' in session:
                        session_info = SubjectSessionInfo(subject, session)
                        if session_info.n_units > 0:
                            temp = session_info.get_scores()

                            session_scores = pd.DataFrame(index=np.arange(session_info.n_units * n_analyses),
                                                          columns=['unit_id', 'subject', 'session',
                                                                   'session_pct_cov', 'session_valid',
                                                                   'session_unit_id', 'unit_type', 'tt', 'tt_cl',
                                                                   'cl_name', 'analysis_type',
                                                                   'score', 'sig', ])

                            session_scores['analysis_type'] = np.repeat(np.array(analyses), session_info.n_units)
                            session_scores['session'] = session_info.session
                            session_scores['subject'] = session_info.subject
                            session_scores['session_unit_id'] = np.tile(np.arange(session_info.n_units), n_analyses)
                            session_scores['unit_id'] = np.tile(np.arange(session_info.n_units),
                                                                n_analyses) + unit_count
                            session_scores['unit_type'] = [v[0] for k, v in
                                                           session_info.cluster_ids.items()] * n_analyses
                            session_scores['tt'] = [v[1] for k, v in session_info.cluster_ids.items()] * n_analyses
                            session_scores['tt_cl'] = [v[2] for k, v in
                                                       session_info.cluster_ids.items()] * n_analyses

                            behav = session_info.get_track_data()
                            # noinspection PyTypeChecker
                            coverage = np.around(behav['pos_valid_mask'].mean(), 2)
                            session_scores['session_pct_cov'] = coverage
                            session_scores['session_valid'] = coverage >= self.min_pct_coverage

                            cl_names = []
                            for k, v in session_info.cluster_ids.items():
                                tt = v[1]
                                cl = v[2]
                                depth = subject_info.sessions_tt_positions.loc[session, f"tt_{tt}"]
                                cl_name = f"{session}-tt{tt}_d{depth}_cl{cl}"
                                cl_names.append(cl_name)

                            session_scores['cl_name'] = cl_names * n_analyses

                            unit_count += session_info.n_units
                            try:
                                for ii, analysis in enumerate(analyses):
                                    indices = np.arange(session_info.n_units) + ii * session_info.n_units
                                    session_scores.at[indices, 'sig'] = temp[analysis + '_sig'].values

                                    if analysis == 'stability':
                                        session_scores.at[indices, 'score'] = temp[analysis + '_corr'].values
                                    else:
                                        session_scores.at[indices, 'score'] = temp[analysis + '_score'].values

                            except:
                                print(f'Error Processing Session {session}')
                                traceback.print_exc(file=sys.stdout)
                                pass

                            session_scores[output_scores_names] = session_scores[output_scores_names].astype(float)
                            metric_scores = metric_scores.append(session_scores)

            metric_scores = metric_scores.reset_index(drop=True)
            metric_scores.to_csv(self.paths['of_metric_scores'])
        else:
            metric_scores = pd.read_csv(self.paths['of_metric_scores'], index_col=0)
        return metric_scores

    def _get_of_models_scores(self):

        models = ['speed', 'hd', 'border', 'grid', 'pos', 'agg_all', 'agg_sdp', 'agg_sdbg']
        metrics = ['r2', 'map_r', 'n_err', 'coef', 'agg_all_coef', 'agg_sdbg_coef', 'agg_sdp_coef']
        splits = ['train', 'test']

        unit_count = 0
        model_scores = pd.DataFrame()
        for subject in self.subjects:
            subject_info = SubjectInfo(subject)
            for session in subject_info.sessions:
                if 'OF' in session:
                    session_info = SubjectSessionInfo(subject, session)
                    n_session_units = session_info.n_units
                    if n_session_units > 0:
                        try:

                            temp = session_info.get_encoding_models_scores()
                            if temp.empty:
                                continue
                            # noinspection PyTypeChecker
                            mask = (temp['metric'].isin(metrics)) & (temp['model'].isin(models))

                            session_models_scores = pd.DataFrame(index=range(mask.sum()),
                                                                 columns=['unit_id', 'subject', 'session',
                                                                          'session_unit_id',
                                                                          'unit_type', 'session_pct_cov',
                                                                          'session_valid',
                                                                          'tt', 'tt_cl', 'model', 'split', 'metric',
                                                                          'value'])

                            session_models_scores.loc[:, ['model', 'split', 'metric', 'value']] = \
                                temp.loc[mask, ['model', 'split', 'metric', 'value']].values

                            session_models_scores['session'] = session_info.session
                            session_models_scores['subject'] = session_info.subject
                            session_models_scores['session_unit_id'] = temp.loc[mask, 'unit_id'].values

                            session_models_scores['unit_id'] = session_models_scores['session_unit_id'] + unit_count

                            for session_unit_id, cluster_info in session_info.cluster_ids.items():
                                mask = session_models_scores.session_unit_id == int(session_unit_id)

                                tt = cluster_info[1]
                                cl = cluster_info[2]
                                depth = subject_info.sessions_tt_positions.loc[session, f"tt_{tt}"]
                                cl_name = f"{session}-tt{tt}_d{depth}_cl{cl}"

                                session_models_scores.loc[mask, 'unit_type'] = cluster_info[0]
                                session_models_scores.loc[mask, 'tt'] = tt
                                session_models_scores.loc[mask, 'tt_cl'] = cl
                                session_models_scores.loc[mask, 'cl_name'] = cl_name

                            behav = session_info.get_track_data()

                            # noinspection PyTypeChecker
                            coverage = np.around(behav['pos_valid_mask'].mean(), 2)
                            session_models_scores['session_pct_cov'] = coverage
                            session_models_scores['session_valid'] = coverage >= self.min_pct_coverage

                            #
                            model_scores = model_scores.append(session_models_scores)
                            unit_count += n_session_units

                        except ValueError:
                            traceback.print_exc(file=sys.stdout)
                            pass
        #
        model_scores = model_scores.reset_index(drop=True)
        model_scores = model_scores.astype({"value": float})
        model_scores = model_scores.to_csv(self.paths['of_model_scores'])

        return model_scores

    def _match_unit_ids(self, metric_scores, model_scores):

        session_unit_id_array = metric_scores[['session', 'session_unit_id']].values
        session_unit_id_tuple = [tuple(ii) for ii in session_unit_id_array]
        sid_2_uid = {}
        uid_2_sid = {}

        used_ids = []
        unique_id_cnt = 0
        for suid in session_unit_id_tuple:
            if not suid in used_ids:
                sid_2_uid[suid] = unique_id_cnt
                uid_2_sid[unique_id_cnt] = suid

                unique_id_cnt += 1
                used_ids += [suid]

        session_unit_id_array = model_scores[['session', 'session_unit_id']].values
        session_unit_id_tuple = [tuple(ii) for ii in session_unit_id_array]
        model_scores['unit_id'] = [sid_2_uid[suid] for suid in session_unit_id_tuple]

        metric_scores.to_csv(self.paths['of_metric_scores'])
        model_scores = model_scores.to_csv(self.paths['of_model_scores'])

        return metric_scores, model_scores

    def get_unit_table(self, overwrite=False):
        if not self.paths['units'].exists() or overwrite:

            # initialize counters
            unit_cnt = 0
            unit_con_match_cnt = 0
            unit_lib_match_cnt = 0

            # iterate through subjects
            unit_table = pd.DataFrame()
            for subject in self.subjects:
                subject_info = SubjectInfo(subject)

                tbl = subject_info.get_units_table(overwrite=True)
                tbl['cl_id'] = tbl.subject_cl_id + unit_cnt
                tbl['cl_match_con_id'] = tbl.subject_cl_match_con_id + unit_con_match_cnt
                tbl['cl_match_lib_id'] = tbl.subject_cl_match_lib_id + unit_lib_match_cnt

                unit_cnt = np.nanmax(tbl.cl_id) + 1
                unit_con_match_cnt = np.nanmax(tbl.cl_match_con_id) + 1
                unit_lib_match_cnt = np.nanmax(tbl.cl_match_lib_id) + 1

                unit_table = pd.concat((unit_table, tbl))

            unit_table = unit_table.reset_index(drop=True)

            # find multi task unit matches
            def _isT3(a):
                return 'T3' in a

            # fix error on session naming for session Li_OF_080718, this is a T3g session- bug discovered on 3/25/18
            unit_table['task2'] = unit_table.task
            unit_table.loc[unit_table.session == "Li_OF_080718", 'task'] = 'T3g'
            unit_table.loc[unit_table.session == "Li_OF_080718", 'task2'] = 'T3'
            unit_table.loc[list(map(_isT3, unit_table.task)), 'task2'] = 'T3'

            unit_table['match_lib_multi_task_id'] = -1
            unit_matches = np.arange(np.nanmax(unit_table.cl_match_lib_id) + 1)
            cnt = 0
            for cm in unit_matches:
                tasks_in_cm = unit_table[unit_table.cl_match_lib_id == cm].task2.values
                if ('OF' in tasks_in_cm) & ('T3' in tasks_in_cm):
                    unit_table.loc[unit_table.cl_match_lib_id == cm, 'match_lib_multi_task_id'] = cnt
                    cnt += 1

            unit_table['match_con_multi_task_id'] = -1
            unit_matches = np.arange(np.nanmax(unit_table.cl_match_con_id) + 1)
            cnt = 0
            for cm in unit_matches:
                tasks_in_cm = unit_table[unit_table.cl_match_con_id == cm].task2.values
                if ('OF' in tasks_in_cm) & ('T3' in tasks_in_cm):
                    unit_table.loc[unit_table.cl_match_con_id == cm, 'match_con_multi_task_id'] = cnt
                    cnt += 1

            unit_table.to_csv(self.paths['units'])
        else:
            unit_table = pd.read_csv(self.paths['units'], index_col=0)

        return unit_table

    def get_unit_match_table(self, match_type='lib', overwrite=False):
        """
        returns a dataframe table index by match #, includes sessions across the tasks
        :param match_type: str, ['lib', 'con']
        :param overwrite: bool, overwrite the table
        :return: dataframe of matches
        """

        fn = self.paths['match_table']
        if fn.exists() and not overwrite:
            match_table = pd.read_csv(fn, index_col=0)
            return match_table

        unit_table = self.get_unit_table()
        match_table = pd.DataFrame(columns=['match_cl_id', 'subject',
                                            'session_T3', 'session_OF',
                                            'cl_id_T3', 'cl_id_OF',
                                            'cl_name_T3', 'cl_name_OF',
                                            'session_cl_id_T3', 'session_cl_id_OF',
                                            'match_pair_number'])

        if match_type in ['lib', 'con']:
            match_column = f'match_{match_type}_multi_task_id'
        else:
            raise ValueError

        matched_units_bool = unit_table[match_column] >= 0
        task_matches = unit_table[match_column][matched_units_bool].unique()
        task_matches.sort()

        cnt = 0
        for tm in task_matches:
            matched_T3_bool = (unit_table[match_column] == tm) & (unit_table.task2 == 'T3')
            matched_OF_bool = (unit_table[match_column] == tm) & (unit_table.task2 == 'OF')

            T3_idx = unit_table[matched_T3_bool].index
            OF_idx = unit_table[matched_OF_bool].index

            match_pair_cnt = 0
            for ii in T3_idx:
                for jj in OF_idx:
                    match_table.loc[cnt, 'match_cl_id'] = tm
                    match_table.loc[cnt, 'match_pair_number'] = match_pair_cnt
                    match_table.loc[cnt, 'subject'] = unit_table.loc[ii].subject

                    match_table.loc[cnt, 'session_T3'] = unit_table.loc[ii].session
                    match_table.loc[cnt, 'cl_id_T3'] = unit_table.loc[ii].cl_id
                    match_table.loc[cnt, 'session_cl_id_T3'] = unit_table.loc[ii].session_cl_id
                    match_table.loc[cnt, 'cl_name_T3'] = unit_table.loc[ii].unique_cl_name

                    match_table.loc[cnt, 'session_OF'] = unit_table.loc[jj].session
                    match_table.loc[cnt, 'cl_id_OF'] = unit_table.loc[jj].cl_id
                    match_table.loc[cnt, 'session_cl_id_OF'] = unit_table.loc[jj].session_cl_id
                    match_table.loc[cnt, 'cl_name_OF'] = unit_table.loc[jj].unique_cl_name

                    match_pair_cnt += 1
                    cnt += 1

        match_table.to_csv(fn)

        return match_table

    def get_combined_scores_matched_units(self, overwrite=False, trial_remap_params=None):
        """

        :param overwrite:
        :param trial_remap_params:
        :return:
        """

        fn = self.paths['combined_scores_table']
        if fn.exists() and not overwrite:
            match_table = pd.read_csv(fn, index_col=0)
            return match_table

        match_table = self.get_unit_match_table()
        tm_tables = {}
        if trial_remap_params is None:
            trial_remap_params = dict()

        ########## ----- remap ----- ##########
        tm_remap_score = self.get_zone_rates_remap(**trial_remap_params)
        z_vars = ['CR_bo-CL_bo-Even_bo-Odd_bo-corr_zm', 'Co_bi-Inco_bi-Even_bi-Odd_bi-corr_zm', 'Even_bo-Odd_bo-corr_z']
        tm_remap_table = tm_remap_score.set_index('cl_name')[z_vars]
        tm_remap_table = tm_remap_table.rename(
            columns={z_vars[0]: 'remap_cue', z_vars[1]: 'remap_rw', z_vars[2]: 'stability_tm'})

        tm_tables['remap'] = tm_remap_table

        ########## ----- encoder ----- ##########
        tm_enc_scores = self.get_zone_encoder_comps()
        tm_enc_selection = dict(expt=['cue', 'rw'],
                                comp='inter_v_fixed',
                                unit_type='cell')
        tm_enc_comp_table = self._select_table_rows(tm_enc_scores, **tm_enc_selection)
        tm_enc_comp_table = tm_enc_comp_table.pivot_table(index='cl_name', columns=['expt'],
                                                          values=['uz', 'mean_test', 'mean_null'])
        tm_enc_comp_table.columns = ["_".join(a) for a in tm_enc_comp_table.columns.to_flat_index()]
        c = tm_enc_comp_table.columns
        c = [s.replace('mean_null', 'rate') for s in c]
        c = [s.replace('mean_test', 'global') for s in c]
        c = [s.replace('uz', 'enc_uz') for s in c]
        tm_enc_comp_table.columns = c

        r2_score_names = ['rate_cue', 'rate_rw', 'global_cue', 'global_rw']
        for col in r2_score_names:
            invalid_val_bool_idx = np.abs(tm_enc_comp_table[col]) > 1
            tm_enc_comp_table.loc[invalid_val_bool_idx, col] = np.nan

        tm_tables['encoder'] = tm_enc_comp_table

        ########## ----- FR comp ----- ##########
        tm_seg_rates = self.get_segment_rate_comps(**trial_remap_params)
        tm_fr_comp_table = self._select_table_rows(tm_seg_rates, **dict(unit_type='cell', comp=['cue', 'rw']))
        tm_fr_comp_table['uz_val'] = np.abs(tm_fr_comp_table['uz_val'])
        tm_fr_comp_table = tm_fr_comp_table.pivot_table(index='cl_name', columns=['comp'], values=['uz_val'])
        tm_fr_comp_table.columns = ['fr_uz_cue', 'fr_uz_rw']

        tm_tables['fr_comp'] = tm_fr_comp_table

        ########## ----- OF results ----- ##########
        of_tables = {}
        of_metric_scores, of_model_scores = self.get_of_results()
        of_selection = dict(split='train',
                            metric=['r2', 'map_r', 'agg_sdp_coef', 'coef'],
                            unit_type='cell',
                            model=['speed', 'hd', 'pos', 'agg_sdp'])

        sub_table = self._select_table_rows(of_model_scores, **of_selection)
        of_model_table = sub_table.pivot_table(index=['cl_name'], columns=['model', 'metric'], values='value')
        of_model_table.columns = ["-".join(a) for a in of_model_table.columns.to_flat_index()]

        of_tables['of_models'] = of_model_table

        of_selection = dict(
            unit_type='cell',
            analysis_type=['hd', 'speed', 'border', 'grid', 'stability'])
        sub_table = self._select_table_rows(of_metric_scores, **of_selection)
        of_metric_score_table = sub_table.pivot_table(index=['cl_name'], columns='analysis_type', values='score')
        of_metric_score_table.columns = ['metric_score_' + s for s in of_metric_score_table.columns]
        of_metric_score_table.rename(columns={'metric_score_stability': 'metric_score_pos'}, inplace=True)

        of_tables['of_metrics'] = of_metric_score_table

        ########## ----- Combine ----- ##########

        tm_columns = []
        for k, v in tm_tables.items():
            tm_columns += list(v.columns)

        of_columns = []
        for k, v in of_tables.items():
            of_columns += list(v.columns)

        combined_columns = tm_columns + of_columns
        combined_table = pd.DataFrame(index=match_table.index.values,
                                      columns=['match_cl_id', 'subject'] + combined_columns)
        combined_table['match_cl_id'] = match_table.match_cl_id
        combined_table['subject'] = match_table.subject

        for kk, table in tm_tables.items():
            bool_table_match_idx = match_table.cl_name_T3.isin(table.index)
            idx = match_table.cl_name_T3[bool_table_match_idx]
            combined_table.loc[bool_table_match_idx, table.columns] = table.loc[idx].values.astype(float)

        for kk, table in of_tables.items():
            bool_table_match_idx = match_table.cl_name_OF.isin(table.index)
            idx = match_table.cl_name_OF[bool_table_match_idx]
            combined_table.loc[bool_table_match_idx, table.columns] = table.loc[idx].values.astype(float)

        combined_table[combined_columns] = combined_table[combined_columns].astype(float)
        combined_table.rename(columns={c: f'TM-{c}' for c in tm_columns}, inplace=True)
        combined_table.rename(columns={c: f'OF-{c}' for c in of_columns}, inplace=True)
        combined_table.to_csv(fn)

        return combined_table

    def plot(self, fig_id=1, save=False, dpi=1000, root_dir=None, fig_format='jpg'):

        if fig_id == 1:
            f = pf.Fig1()
            f.plot_all()
        elif fig_id == 2:
            f = pf.Fig2()
            f.plot_all()
        else:
            return
        if save:
            fn = f"{fig_id}.{fig_format}"
            if root_dir is None:
                f.savefig(self.paths['figures'] / fn, dpi=dpi, bbox_inches='tight')
            else:
                if root_dir in self._root_paths.keys():
                    paths = self._get_paths(self._root_paths[root_dir])
                    f.savefig(paths['figures'] / fn, dpi=dpi, bbox_inches='tight')
        return f

    @staticmethod
    def _select_table_rows(table, **kwargs):
        idx = np.ones(len(table), dtype=bool)
        for k, v in kwargs.items():
            if isinstance(v, list):
                idx &= table[k].isin(v)
            else:
                idx &= table[k] == v
        return table[idx].reset_index(drop=True)


class SubjectInfo:

    def __init__(self, subject, sorter='KS2', data_root='BigPC', overwrite=False, time_step=0.02,
                 samp_rate=32000, n_tetrodes=16, fr_temporal_smoothing=0.125, spk_outlier_thr=None,
                 overwrite_cluster_stats=False, overwrite_session_clusters=False):

        subject = str(subject.title())
        self.subject = subject
        self.sorter = sorter
        self.params = {'time_step': time_step, 'samp_rate': samp_rate, 'n_tetrodes': n_tetrodes,
                       'fr_temporal_smoothing': fr_temporal_smoothing, 'spk_outlier_thr': spk_outlier_thr,
                       'spk_recording_buffer': 3}
        self.tetrodes = np.arange(n_tetrodes, dtype=int) + 1

        if data_root == 'BigPC':
            if subject in ['Li', 'Ne']:
                self.root_path = Path('/mnt/Data1_SSD2T/Data')
            elif subject in ['Cl']:
                self.root_path = Path('/mnt/Data2_SSD2T/Data')
            elif subject in ['Ca', 'Mi', 'Al']:
                self.root_path = Path('/mnt/Data3_SSD2T/Data')

            self.raw_path = Path('/mnt/Raw_Data/Data', subject)

        elif data_root == 'oak':
            self.root_path = Path('/mnt/o/giocomo/alexg/')
            self.raw_path = self.root_path / 'RawData/InVivo' / subject
            # self.sorted_path = self.root_path / 'Clustered' / subject
            # self.results_path = self.root_path / 'Analyses' / subject
        else:
            self.root_path = Path(data_root)
            self.raw_path = self.root_path / 'Raw_Data' / subject

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
                    self.session_clusters = self.get_session_clusters(overwrite=overwrite_session_clusters)
                    self.sort_tables = self.get_sort_tables(overwrite=overwrite_session_clusters)
            except:
                print("Error obtaining clusters.")
                print(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2].tb_lineno)
                traceback.print_exc(file=sys.stdout)

            # get tetrode depths & match sessions
            self.sessions_tt_positions = self.get_sessions_tt_position()
            self.tt_depth_match = self.get_tetrode_depth_match()

            # hack because get sessions analyses calls subjects info before it saves,
            # so need to save it first. alternative is to feed subject session info the subject_info object instead.
            # TO DO #
            self.save_subject_info()

            # check analyses table
            self.analyses_table = self.get_sessions_analyses()
            self.valid_track_table = self.check_track_data_validty()

            self.save_subject_info()

    def update_paths(self):
        for session in self.sessions:
            self.session_paths[session].update(self._session_paths(session))
        self.save_subject_info()

    def load_subject_info(self):
        with self.subject_info_file.open(mode='rb') as f:
            loaded_self = pickle.load(f)
            self.__dict__.update(loaded_self.__dict__)
            return self

    def save_subject_info(self):
        with self.subject_info_file.open(mode='wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_sessions_analyses(self):

        analyses_table = pd.DataFrame()
        for session in self.sessions:
            session_info = SubjectSessionInfo(self.subject, session)
            analyses_table = analyses_table.append(session_info.session_analyses_table)

        analyses_table.fillna(-1, inplace=True)
        return analyses_table

    def check_track_data_validty(self):
        df = pd.DataFrame(index=self.sessions, columns=['task', 'validity'])
        for session in self.sessions:
            if self.analyses_table.loc[session, 'track_data'] == 1:
                session_info = SubjectSessionInfo(self.subject, session)
                df.loc[session, 'task'] = session_info.task
                df.loc[session, 'validity'] = session_info.check_track_data_validity()
        return df

    # tetrode methods
    def update_clusters(self):
        self.session_clusters = self.get_session_clusters(overwrite=True)
        self.sort_tables = self.get_sort_tables(overwrite=True)
        self.save_subject_info()

    def get_sessions_tt_position(self):
        p = Path(self.results_path / f"{self.subject}_tetrodes.csv")

        if p.exists():
            tt_pos = pd.read_csv(p)
            tt_pos['Date'] = pd.to_datetime(tt_pos['Date']).dt.strftime('%m%d%y')
            tt_pos = tt_pos.set_index('Date')
            tt_pos = tt_pos[['TT' + str(tt) + '_overall' for tt in self.tetrodes]]

            session_dates = {session: session.split('_')[2] for session in self.sessions}
            sessions_tt_pos = pd.DataFrame(index=self.sessions, columns=['tt_' + str(tt) for tt in self.tetrodes])
            tt_pos_dates = tt_pos.index
            prev_date = tt_pos_dates[0]
            for session in self.sessions:
                date = session_dates[session]
                # below if is to correct for incorrect session dates for Cl
                if (date in ['010218', '010318', '010418']) & (self.subject == 'Cl'):
                    date = date[:5] + '9'

                # this part accounts for missing dates by assigning it to the previous update
                if date in tt_pos_dates:
                    sessions_tt_pos.loc[session] = tt_pos.loc[date].values
                    prev_date = str(date)
                else:
                    sessions_tt_pos.loc[session] = tt_pos.loc[prev_date].values

            return sessions_tt_pos
        else:
            print(f"Tetrode depth table not found at '{str(p)}'")
            return None

    def get_depth_wf(self):
        raise NotImplementedError

    def get_session_tt_wf(self, session, tt, cluster_ids=None, wf_lims=None, n_wf=200):
        import Sorting.sort_functions as sort_funcs

        if wf_lims is None:
            wf_lims = [-12, 20]
        tt_str = 'tt_' + str(tt)
        _sort_path = Path(self.session_paths[session]['Sorted'], tt_str, self.sorter)

        _cluster_spike_time_fn = _sort_path / 'spike_times.npy'
        _cluster_spike_ids_fn = _sort_path / 'spike_clusters.npy'
        _hp_data_fn = _sort_path / 'recording.dat'

        if _hp_data_fn.exists():
            hp_data = sort_funcs.load_hp_binary_data(_hp_data_fn)
        else:  # filter data
            hp_data = self._spk_filter_data(session, tt)

        spike_times = np.load(_cluster_spike_time_fn)
        spike_ids = np.load(_cluster_spike_ids_fn)

        wf_samps = np.arange(wf_lims[0], wf_lims[1])
        if cluster_ids is None:
            cluster_ids = np.unique(spike_ids)

        n_clusters = len(cluster_ids)
        out = np.zeros((n_clusters, n_wf, len(wf_samps) * 4), dtype=np.float16)

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

            try:
                with _clusters_file.open(mode='w') as f:
                    json.dump(session_clusters, f, indent=4)
            except TypeError:
                print(session)
        return session_clusters

    # cluster matching methods
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

    def get_session_match_analysis(self):
        # # determine sessions/tt to match based on depth
        # matching_analyses = []
        # for tt in np.arange(1, 17):
        #     tt_depths = list(self.tt_depth_match[tt].keys())
        #
        #     for tt_d in tt_depths:
        #         tt_d_sessions = self.tt_depth_match[tt][tt_d]
        #         # check if there are more 2 or more sessions with units
        #         n_cells_session = np.zeros(len(tt_d_sessions), dtype=int)
        #
        #         for ii, session in enumerate(tt_d_sessions):
        #             session_cell_ids = self.session_clusters[session]['cell_IDs']
        #             if tt in session_cell_ids.keys():
        #                 n_cells_session[ii] = len(session_cell_ids[tt])
        #         sessions_with_cells = np.where(n_cells_session > 0)[0]
        #
        #         if len(sessions_with_cells) >= 2:
        #             n_units = n_cells_session[sessions_with_cells].sum()
        #             matching_analyses.append((tt, tt_d, np.array(tt_d_sessions)[sessions_with_cells].tolist(),
        #                                       n_units, n_cells_session[sessions_with_cells].tolist()))

        ## version as a dict ##
        matching_analyses = {}
        cnt = 0
        for tt in np.arange(1, 17):
            tt_depths = list(self.tt_depth_match[tt].keys())

            for tt_d in tt_depths:
                tt_d_sessions = self.tt_depth_match[tt][tt_d]
                # check if there are more 2 or more sessions with units
                n_cells_session = np.zeros(len(tt_d_sessions), dtype=int)

                for ii, session in enumerate(tt_d_sessions):
                    session_cell_ids = self.session_clusters[session]['cell_IDs']
                    if tt in session_cell_ids.keys():
                        n_cells_session[ii] = len(session_cell_ids[tt])

                sessions_with_cells = np.where(n_cells_session > 0)[0]
                n_units = n_cells_session[sessions_with_cells].sum()
                if len(sessions_with_cells) >= 1:
                    matching_analyses[cnt] = {'tt': tt, 'd': tt_d, 'n_units': n_units,
                                              'sessions': np.array(tt_d_sessions)[sessions_with_cells].tolist(),
                                              'n_session_units': n_cells_session[sessions_with_cells].tolist()}

                    cnt += 1

        return matching_analyses

    def get_cluster_dists(self, overwrite=False, **kwargs):
        import Analyses.cluster_match_functions as cmf
        params = {'dim_reduc_method': 'umap', 'n_wf': 1000, 'zscore_wf': True}
        params.update(kwargs)

        cl_dists_fn = self.results_path / f"cluster_dists.pickle"
        if not cl_dists_fn.exists() or overwrite:

            matching_analyses = self.get_session_match_analysis()
            n_wf = params['n_wf']
            dim_reduc_method = params['dim_reduc_method']
            n_samps = 32 * 4
            cluster_dists = {k: {} for k in np.arange(len(matching_analyses))}

            for analysis_id, analysis in matching_analyses.items():
                tt, d, sessions = analysis['tt'], analysis['d'], analysis['sessions']
                n_units, n_session_units = analysis['n_units'], analysis['n_session_units']

                # Obtain cluster labels & mapping between labels [this part can be improved]
                cl_names = []
                for session_num, session in enumerate(sessions):
                    cluster_ids = self.session_clusters[session]['cell_IDs'][tt]
                    for cl_num, cl_id in enumerate(cluster_ids):
                        cl_name = f"{session}-tt{tt}_d{d}_cl{cl_id}"
                        cl_names.append(cl_name)

                # load waveforms
                X = np.empty((0, n_wf, n_samps), dtype=np.float16)
                for session in sessions:
                    cluster_ids = self.session_clusters[session]['cell_IDs'][tt]
                    session_cell_wf = self.get_session_tt_wf(session, tt, cluster_ids=cluster_ids, n_wf=n_wf)
                    X = np.concatenate((X, session_cell_wf), axis=0)

                if params['zscore_wf']:
                    X = rs.robust_zscore(X, axis=2)
                X[np.isnan(X)] = 0
                X[np.isinf(X)] = 0

                # Obtain cluster label namess
                clusters_label_num = np.arange(n_units).repeat(n_wf)

                # Reduce dims
                X_2d = cmf.dim_reduction(X.reshape(-1, X.shape[-1]), method=dim_reduc_method)

                # compute covariance and location
                clusters_loc, clusters_cov = cmf.get_clusters_moments(data=X_2d, labels=clusters_label_num)

                # compute distance metrics
                dist_mats = cmf.get_clusters_all_dists(clusters_loc, clusters_cov, data=X_2d, labels=clusters_label_num)

                # create data frames with labeled cluster names
                dists_mats_df = {}
                for metric, dist_mat in dist_mats.items():
                    dists_mats_df[metric] = pd.DataFrame(dist_mat, index=cl_names, columns=cl_names)

                # store
                clusters_loc = {k: v for k, v in zip(cl_names, clusters_loc)}
                clusters_cov = {k: v for k, v in zip(cl_names, clusters_cov)}

                cluster_dists[analysis_id] = {'analysis': analysis, 'cl_names': cl_names,
                                              'clusters_loc': clusters_loc, 'clusters_cov': clusters_cov,
                                              'dists_mats': dists_mats_df}

                print(".", end="")

            with cl_dists_fn.open(mode='wb') as f:
                pickle.dump(cluster_dists, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with cl_dists_fn.open(mode='rb') as f:
                cluster_dists = pickle.load(f)

        return cluster_dists

    def match_clusters(self, overwrite=False, require_subsets=True, **kwargs):
        import Analyses.cluster_match_functions as cmf
        params = {'dist_metric': 'pe', 'dist_metric_thr': 0.5, 'select_lower': True}
        params.update(kwargs)

        dist_metric = params['dist_metric']
        dist_metric_thr = params['dist_metric_thr']
        select_lower = params['select_lower']

        if require_subsets:  # rs -> require subsets, conservative in grouping clusters
            cl_match_results_fn = self.results_path / f"cluster_matches_rs_{params['dist_metric']}.pickle"
        else:  # nrs -> doesn't require subsets, results in more sessions being grouped
            cl_match_results_fn = self.results_path / f"cluster_matches_nrs_{params['dist_metric']}.pickle"

        if not cl_match_results_fn.exists() or overwrite:
            cluster_dists = self.get_cluster_dists()

            matching_analyses = self.get_session_match_analysis()
            # [cluster_dists[k]['analysis'] for k in cluster_dists.keys()]
            cluster_match_results = {k: {} for k in np.arange(len(matching_analyses))}
            for analysis_id, analysis in matching_analyses.items():
                dist_mat = cluster_dists[analysis_id]['dists_mats'][dist_metric]

                matches_dict = cmf.find_session_cl_matches(dist_mat, thr=dist_metric_thr,
                                                           session_cl_sep="-", select_lower=select_lower)
                unique_matches_sets, unique_matches_dict = \
                    cmf.matches_dict_to_unique_sets(matches_dict, dist_mat, select_lower=select_lower,
                                                    require_subsets=require_subsets)

                cluster_match_results[analysis_id] = {'analysis': analysis,
                                                      'matches_dict': unique_matches_dict,
                                                      'matches_sets': unique_matches_sets
                                                      }

            with cl_match_results_fn.open(mode='wb') as f:
                pickle.dump(cluster_match_results, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:

            with cl_match_results_fn.open(mode='rb') as f:
                cluster_match_results = pickle.load(f)

        return cluster_match_results

    # sort/unit tables methods
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

    def get_units_table(self, overwrite=False):

        units_table_fn = self.results_path / f"units_table.csv"
        if not units_table_fn.exists() or overwrite:

            # matches
            subject_cl_matches_con = self.match_clusters()

            matches_con_sets = {}
            matches_con_set_num = {}
            matches_con_dict = {}
            cnt = 0
            for k, cma in subject_cl_matches_con.items():
                matches_con_sets.update({cnt + ii: clx_set for ii, clx_set in enumerate(cma['matches_sets'])})
                cnt = len(matches_con_sets)
                matches_con_dict.update(cma['matches_dict'])

            # creates a dict indexing each session to a set number
            for set_num, clm_set in matches_con_sets.items():
                for cl in clm_set:
                    matches_con_set_num[cl] = set_num

            subject_cl_matches_lib = self.match_clusters(require_subsets=False)
            matches_lib_sets = {}
            matches_lib_set_num = {}
            matches_lib_dict = {}
            cnt = 0
            for k, cma in subject_cl_matches_lib.items():
                matches_lib_sets.update({cnt + ii: clx_set for ii, clx_set in enumerate(cma['matches_sets'])})
                cnt = len(matches_lib_sets)
                matches_lib_dict.update(cma['matches_dict'])

            for set_num, clm_set in matches_lib_sets.items():
                for cl in clm_set:
                    matches_lib_set_num[cl] = set_num

            #### fill table
            n_total_units = 0
            for session in self.sessions:
                n_total_units += self.session_clusters[session]['n_cell']
                n_total_units += self.session_clusters[session]['n_mua']

            subject_units_table = pd.DataFrame(index=np.arange(n_total_units),
                                               columns=["subject_cl_id", "subject", "session", "task", "date",
                                                        "subsession", "tt", "depth", "unique_cl_name",
                                                        "session_cl_id", "unit_type", "n_matches_con",
                                                        "subject_cl_match_con_id", "n_matches_lib",
                                                        "subject_cl_match_lib_id",
                                                        "snr", "fr", "isi_viol_rate"])

            subject_units_table["subject"] = self.subject

            unit_cnt = 0
            for session in self.sessions:
                n_cells = self.session_clusters[session]['n_cell']
                n_muas = self.session_clusters[session]['n_mua']
                n_session_units = n_cells + n_muas
                session_unit_idx = np.arange(n_session_units) + unit_cnt

                subject_units_table.loc[session_unit_idx, "unit_type"] = ["cell"] * n_cells + ["mua"] * n_muas
                subject_units_table.loc[session_unit_idx, "session_cl_id"] = np.arange(n_session_units)
                subject_units_table.loc[session_unit_idx, "subject_cl_id"] = session_unit_idx

                session_details = session.split("_")
                if len(session_details) > 3:
                    subsession = session_details[3]
                else:
                    subsession = "0000"
                subject_units_table.loc[session_unit_idx, "session"] = session
                subject_units_table.loc[session_unit_idx, "task"] = session_details[1]
                subject_units_table.loc[session_unit_idx, "date"] = session_details[2]
                subject_units_table.loc[session_unit_idx, "subsession"] = subsession

                unit_cnt += n_session_units

            unit_cnt2 = 0
            for session in self.sessions:
                try:
                    session_clusters = self.session_clusters[session]
                    for unit_type in ['cell', 'mua']:
                        for tt, tt_clusters in session_clusters[f'{unit_type}_IDs'].items():
                            if len(tt_clusters) > 0:
                                depth = self.sessions_tt_positions.loc[session, f"tt_{tt}"]
                                for cl in tt_clusters:
                                    cl_name = f"{session}-tt{tt}_d{depth}_cl{cl}"
                                    subject_units_table.loc[unit_cnt2, "unique_cl_name"] = cl_name
                                    subject_units_table.loc[unit_cnt2, "tt"] = tt
                                    subject_units_table.loc[unit_cnt2, "depth"] = depth

                                    if cl not in session_clusters["clusters_snr"][tt]:
                                        cl2 = str(cl)
                                    else:
                                        cl2 = cl

                                    subject_units_table.loc[unit_cnt2, "snr"] = session_clusters["clusters_snr"][tt][
                                        cl2]
                                    subject_units_table.loc[unit_cnt2, "fr"] = session_clusters["clusters_fr"][tt][cl2]
                                    subject_units_table.loc[unit_cnt2, "isi_viol_rate"] = \
                                        session_clusters["clusters_isi_viol_rate"][tt][cl2]

                                    if unit_type == 'cell':
                                        # add fields of conservative cluster matching (requires subset)
                                        if cl_name in matches_con_dict.keys():
                                            cl_matches = matches_con_dict[cl_name][0]
                                            subject_units_table.loc[unit_cnt2, "n_matches_con"] = len(cl_matches)
                                            subject_units_table.loc[unit_cnt2, "subject_cl_match_con_id"] = \
                                                matches_con_set_num[cl_name]

                                        # add fields of liberal cluster matching ( does not require subset matching)
                                        if cl_name in matches_lib_dict.keys():
                                            cl_matches = matches_lib_dict[cl_name][0]
                                            subject_units_table.loc[unit_cnt2, "n_matches_lib"] = len(cl_matches)
                                            subject_units_table.loc[unit_cnt2, "subject_cl_match_lib_id"] = \
                                                matches_lib_set_num[cl_name]
                                    unit_cnt2 += 1

                except:
                    print(session)
                    traceback.print_exc(file=sys.stdout)
                    pass

            subject_units_table.to_csv(units_table_fn)
        else:
            subject_units_table = pd.read_csv(units_table_fn, index_col=0)

        return subject_units_table

    # private methods
    def _spk_filter_data(self, session, tt):
        tt_str = 'tt_' + str(tt)
        sos, _ = pp_funcs.get_sos_filter_bank(['Sp'], fs=self.params['samp_rate'])
        sig = np.load(self.session_paths[session]['PreProcessed'] / (tt_str + '.npy'))

        hp_data = np.zeros_like(sig)
        for ch in range(4):
            hp_data[ch] = signal.sosfiltfilt(sos, sig[ch])

        return hp_data

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

        if task == 'OF':
            paths['cluster_OF_metrics'] = paths['Results'] / 'OF_metrics.csv'
            paths['cluster_OF_encoding_models'] = paths['Results'] / 'OF_encoding.csv'
            paths['cluster_OF_encoding_agg_coefs'] = paths['Results'] / 'OF_encoding_agg_coefs.csv'
        else:

            paths['trial_table'] = paths['Results'] / 'trial_table.csv'
            paths['event_table'] = paths['Results'] / 'event_table.csv'
            paths['track_table'] = paths['Results'] / 'track_table.csv'
            paths['event_time_series'] = paths['Results'] / 'event_time_series.csv'
            paths['not_valid_pos_samps'] = paths['Results'] / 'not_valid_pos_samps.npy'
            paths['pos_zones'] = paths['Results'] / 'pos_zones.npy'
            paths['pos_zones_invalid_samps'] = paths['Results'] / 'pos_zones_invalid_samps.npy'
            paths['trial_zone_rates'] = paths['Results'] / 'trial_zone_rates.npy'
            paths['zone_rates_comps'] = paths['Results'] / 'zone_rates_comps.csv'
            paths['zone_rates_remap'] = paths['Results'] / 'zone_rates_remap.csv'
            paths['pop_zone_rates_remap'] = paths['Results'] / 'pop_zone_rates_remap.csv'
            paths['bal_conds_seg_rates'] = paths['Results'] / 'bal_conds_seg_rates.csv'
            paths['bal_conds_seg_boot_rates'] = paths['Results'] / 'bal_conds_seg_boot_rates.csv'

            paths['zone_encoder'] = paths['Results'] / 'zone_encoder.csv'
            paths['zone_encoder_lag'] = paths['Results'] / 'zone_encoder_lag.csv'
            paths['zone_encoder_cue'] = paths['Results'] / 'zone_encoder_cue.csv'

            paths['zone_decoder'] = paths['Results'] / 'zone_decoder.csv'

            paths['zone_analyses'] = paths['Results'] / 'ZoneAnalyses.pkl'
            paths['TrialInfo'] = paths['Results'] / 'TrInfo.pkl'
            paths['TrialCondMat'] = paths['Results'] / 'TrialCondMat.csv'
            paths['TrLongPosMat'] = paths['Results'] / 'TrLongPosMat.csv'
            paths['TrLongPosFRDat'] = paths['Results'] / 'TrLongPosFRDat.csv'
            paths['TrModelFits2'] = paths['Results'] / 'TrModelFits2.csv'

            paths['CueDesc_SegUniRes'] = paths['Results'] / 'CueDesc_SegUniRes.csv'
            paths['CueDesc_SegDecRes'] = paths['Results'] / 'CueDesc_SegDecRes.csv'
            paths['CueDesc_SegDecSumRes'] = paths['Results'] / 'CueDesc_SegDecSumRes.csv'
            paths['PopCueDesc_SegDecSumRes'] = paths['Results'] / 'PopCueDesc_SegDecSumRes.csv'

        # plots directories
        # paths['Plots'] = paths['Results'] / 'Plots'
        # # paths['Plots'].mkdir(parents=True, exist_ok=True)
        # paths['SampCountsPlots'] = paths['Plots'] / 'SampCountsPlots'
        # # paths['SampCountsPlots'].mkdir(parents=True, exist_ok=True)
        #
        # paths['ZoneFRPlots'] = paths['Plots'] / 'ZoneFRPlots'
        # # paths['ZoneFRPlots'].mkdir(parents=True, exist_ok=True)
        #
        # paths['ZoneCorrPlots'] = paths['Plots'] / 'ZoneCorrPlots'
        # # paths['ZoneCorrPlots'].mkdir(parents=True, exist_ok=True)
        # paths['SIPlots'] = paths['Plots'] / 'SIPlots'
        # # paths['SIPlots'].mkdir(parents=True, exist_ok=True)
        #
        # paths['TrialPlots'] = paths['Plots'] / 'TrialPlots'
        # # paths['TrialPlots'].mkdir(parents=True, exist_ok=True)
        #
        # paths['CueDescPlots'] = paths['Plots'] / 'CueDescPlots'
        # # paths['CueDescPlots'].mkdir(parents=True, exist_ok=True)

        return paths

    def _cluster_stats(self, session):
        import Sorting.sort_functions as sort_funcs

        sort_path = self.session_paths[session]['Sorted']

        for tt in self.tetrodes:
            tt_str = 'tt_' + str(tt)
            _cluster_spike_time_fn = Path(sort_path, tt_str, self.sorter, 'spike_times.npy')
            _cluster_spike_ids_fn = Path(sort_path, tt_str, self.sorter, 'spike_clusters.npy')
            _cluster_groups_fn = Path(sort_path, ('tt_' + str(tt)), self.sorter, 'cluster_group.tsv')
            _cluster_stats_fn = Path(sort_path, ('tt_' + str(tt)), self.sorter, 'cluster_stats.csv')

            _hp_data_fn = Path(sort_path, tt_str, self.sorter, 'recording.dat')
            _hp_data_info_fn = Path(sort_path, tt_str, tt_str + '_info.pickle')
            _cluster_stats_fn2 = Path(sort_path, tt_str, self.sorter, 'cluster_stats_curated.csv')

            try:
                # load
                cluster_groups = pd.read_csv(_cluster_groups_fn, sep='\t')
                try:
                    cluster_stats = pd.read_csv(_cluster_stats_fn2, index_col=0)
                except:
                    cluster_stats = pd.DataFrame(columns=['cl_num'])

                # get units and units already with computed stats
                valid_units = cluster_groups.cluster_id.values
                unit_keys_with_stats = cluster_stats.index.values
                units_with_stats = cluster_stats.cl_num.values
                unit_overlap = np.intersect1d(valid_units, units_with_stats)

                missing_units = np.setdiff1d(valid_units, units_with_stats)

                # get stats for overlapping units
                cluster_stats2 = cluster_stats.loc[cluster_stats.cl_num.isin(valid_units)].copy()

                if len(missing_units) > 0:
                    spike_times = np.load(_cluster_spike_time_fn)
                    spike_ids = np.load(_cluster_spike_ids_fn)
                    spike_times_dict = {unit: spike_times[spike_ids == unit].flatten() for unit in missing_units}
                    # print(spike_times_dict[0])

                    if _hp_data_fn.exists():
                        hp_data = sort_funcs.load_hp_binary_data(_hp_data_fn)
                    else:  # filter data
                        hp_data = self._spk_filter_data(session, tt)

                    with _hp_data_info_fn.open(mode='rb') as f:
                        hp_data_info = pickle.load(f)

                    cluster_stats_missing = sort_funcs.get_cluster_stats(spike_times_dict, hp_data, hp_data_info)
                    # cluster_stats2 = cluster_stats_missing
                    cluster_stats2 = cluster_stats2.append(cluster_stats_missing)
                    cluster_stats2 = cluster_stats2.sort_values('cl_num')
                    cluster_stats2 = cluster_stats2.drop_duplicates()

                # attached curated group labels to table
                cluster_stats2.loc[cluster_stats2.cl_num.isin(valid_units), 'group'] \
                    = cluster_groups.loc[cluster_groups.cluster_id.isin(valid_units), 'group'].values
                cluster_stats2.to_csv(_cluster_stats_fn2)
            except FileNotFoundError:
                pass
            except:
                print(f"Error Computing Cluster Stats for {session}")
                pass

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
                d = d.astype({'cl_num': int, 'valid': bool})
                keys = d.index.values

                for st in _cluster_stats_names:
                    if st == 'valid':
                        table['clusters_valid'][int(tt)] = {int(d.loc[k, 'cl_num']):
                                                                int(d.loc[k, 'valid']) for k in keys}
                    else:
                        try:
                            table['clusters_' + st][int(tt)] = {int(d.loc[k, 'cl_num']):
                                                                    np.around(d.loc[k, st], 2) for k in keys}
                        except TypeError:
                            print(st, keys, tt, session)
                            sys.exit()

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
                        n_cells[tt - 1] = len(clusters)

                    tt_mua_clusters = _clusters_info['mua_IDs']
                    for tt, clusters in tt_mua_clusters.items():
                        n_mua[tt - 1] = len(clusters)

                    sort_tables[tbl].loc[session] = n_cells + n_mua
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
            self.sub_session_id = '0000'
        elif len(session.split('_')) == 4:
            self.subject, self.task, self.date, self.sub_session_id = session.split('_')
        self.subject = self.subject.capitalize()

        self.valid_tetrodes = np.where(self.channel_table.loc[session])[0] + 1
        self.n_orig_samps = 0
        self.tB = 0
        if len(self.valid_tetrodes) > 0:
            tt_info = self.get_tt_info(self.valid_tetrodes[0])
            if tt_info is not None:
                self.n_orig_samps = tt_info['n_samps']
                self.tB = tt_info['tB']

        self.session_analyses_table = pd.DataFrame(index=[self.session])
        self._analyses = self._check_analyses()
        self._analyses_names = list(self._analyses.keys())

        self.task_params = get_task_params(self)
        self.n_cells = self.clusters['n_cell']
        self.n_mua = self.clusters['n_mua']
        self.n_units = self.n_cells + self.n_mua

        self.n_samps = len(self.get_time())

        if self.paths['cluster_spikes_ids'].exists():
            with self.paths['cluster_spikes_ids'].open('r') as f:
                self.cluster_ids = json.load(f)
                self.n_units = len(self.cluster_ids)
            self.cell_ids = np.array([v[0] == 'cell' for k, v in self.cluster_ids.items()])
            self.mua_ids = ~self.cell_ids

            unit_type = np.zeros(self.n_units, dtype='object')
            unit_type[self.cell_ids] = 'cell'
            unit_type[self.mua_ids] = 'mua'
            self.unit_type_map = dict(zip(np.arange(self.n_units), unit_type))

        self.enc_models = None

        self.session_unit_table = self._session_unit_table()

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

    def _session_unit_table(self):

        session = self.session
        session_table = pd.DataFrame(index=np.arange(self.n_units),
                                     columns=['session_unit_id', 'subject', 'session', 'session_pct_cov',
                                              'unit_type', 'tt', 'tt_cl',
                                              'cl_name'])

        session_table['session'] = session
        session_table['subject'] = self.subject
        session_table['task'] = self.task
        session_table['session_unit_id'] = np.arange(self.n_units)

        session_table['session_pct_cov'] = self.check_track_data_validity()

        try:
            session_table['unit_type'] = [v[0] for k, v in self.cluster_ids.items()]
            session_table['tt'] = [v[1] for k, v in self.cluster_ids.items()]
            session_table['tt_cl'] = [v[2] for k, v in self.cluster_ids.items()]

            cl_names = []
            for k, v in self.cluster_ids.items():
                tt = v[1]
                cl = v[2]
                depth = self.sessions_tt_positions.loc[session, f"tt_{tt}"]
                cl_name = f"{session}-tt{tt}_d{depth}_cl{cl}"
                cl_names.append(cl_name)

            session_table['cl_name'] = cl_names
        except:
            pass

        return session_table

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
                'spikes': (self.get_spikes, self.paths['cluster_spikes'].exists()),
                'binned_spikes': (self.get_binned_spikes, self.paths['cluster_binned_spikes'].exists()),
                'fr': (self.get_fr, self.paths['cluster_fr'].exists()),
                'scores': (self.get_scores, self.paths['cluster_OF_metrics'].exists()),
                'encoding_models': (self.get_encoding_models, self.paths['cluster_OF_encoding_models'].exists())
            }
        elif self.task[:2] == 'T3':
            analyses = {
                'track_data': (self.get_track_data, self.paths['track_table'].exists()),
                'spikes': (self.get_spikes, self.paths['cluster_spikes'].exists()),
                'binned_spikes': (self.get_binned_spikes, self.paths['cluster_binned_spikes'].exists()),
                'fr': (self.get_fr, self.paths['cluster_fr'].exists()),
                'pos_zones': (self.get_pos_zones, self.paths['pos_zones'].exists()),
                'event_table': (self.get_event_behavior, self.paths['event_table'].exists()),
                'trial_zone_rates': (self.get_trial_zone_rates, self.paths['trial_zone_rates'].exists()),
                'zone_rates_comps': (self.get_zone_rates_comps, self.paths['zone_rates_comps'].exists()),
                'zone_rates_remap': (self.get_zone_rates_remap, self.paths['zone_rates_remap'].exists()),
                'pop_zone_rates_remap': (self.get_pop_zone_rates_remap, self.paths['pop_zone_rates_remap'].exists()),
                'bal_conds_seg_rates': (self.get_bal_conds_seg_rates, self.paths['bal_conds_seg_rates'].exists()),
                'bal_conds_seg_boot_rates': (self.get_bal_conds_seg_boot_rates, np.nan),
                'zone_encoder': (self.get_zone_encoder, self.paths['zone_encoder'].exists()),
                # 'zone_encoder_lag': (self.get_zone_encoder_lag, self.paths['zone_encoder_lag'].exists()),
                # 'zone_encoder_cue': (self.get_zone_encoder_cue, self.paths['zone_encoder_cue'].exists()),
                'zone_decoder': (self.get_zone_decoder, self.paths['zone_decoder'].exists()),

            }
        else:
            return NotImplementedError

        for k in analyses:
            if analyses[k][1] == 1:
                self.session_analyses_table[k] = 1
            else:
                self.session_analyses_table[k] = 0

        return analyses

    def run_analyses(self, which='all', overwrite=False, overwrite_old=False, overwrite_old_days=1, verbose=False,
                     **params):
        """
        Method to execute all analyses in the analyses list.
        :param which: str or list. which analysis to perform, if 'all', runs all analyses for that task
        :param bool overwrite: overwrite flag.
        :param bool overwrite_old: if file is older than 24hts
        :return: None
        """

        if self.n_units > 0:
            if which == 'all':
                analyses_to_perfom = self._analyses_names
            elif which in self._analyses_names:
                analyses_to_perfom = list([which])
            else:
                print("Invalid analysis.")
                return

            if overwrite:
                overwrite_flag = True
            else:
                overwrite_flag = False

            for a in analyses_to_perfom:
                method = self._analyses[a][0]
                file_exists_flag = self._analyses[a][1]

                if overwrite_old:
                    file_path = self.paths[a]
                    file_mod_time = file_path.stat().st_mtime
                    if (datetime.datetime.now() - datetime.datetime.fromtimestamp(file_mod_time)) >= datetime.timedelta(
                            days=overwrite_old_days):
                        overwrite_flag = True
                    else:
                        overwrite_flag = False

                if a == 'time':
                    continue
                if (not file_exists_flag) or overwrite_flag:
                    try:
                        # calls methods in _analyses
                        _ = method(overwrite=True, **params)
                        if verbose:
                            print(f'Analysis {a} completed.')
                    except NotImplementedError:
                        print(f'Analysis {a} not implemented.')
                    except FileNotFoundError:
                        print(f'Analysis {a} did not find the dependent files.')
                    except KeyboardInterrupt:
                        print('Keyboard Interrupt')
                        break
                    except:
                        print(f'{self.session} analysis {a} failed.')
                        if verbose:
                            traceback.print_exc(file=sys.stdout)
            # update analyses
            self._analyses = self._check_analyses()
        else:
            if verbose:
                print(f"{self.session} This session does not have units. No analyses were ran.")

    #  default methods
    def get_time(self, which='resamp'):
        """
        :param str which:   if 'resamp' [default], returns resampled time at the time_step found in params
                            if 'orig', returns a time vector sampled at the original sampling rate
        :return np.array time: array of time, float32.
        """
        samp_rate = self.params['samp_rate']
        time_step = self.params['time_step']

        n_samps = self.n_orig_samps
        tB = self.tB

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

    def get_tt_info(self, tt, verbose=False):
        try:
            with (self.paths['PreProcessed'] / f'tt_{tt}_info.pickle').open(mode='rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            if verbose:
                print(f'Tetrode {tt} information not found.')
            return None

    def get_tt_data(self, tt):
        try:
            return np.load(self.paths['PreProcessed'] / f'tt_{tt}.npy')
        except FileNotFoundError:
            print(f'Tetrode {tt} data not found.')
            return None

    # behavioral methods
    def get_track_data(self, return_nan_idx=False, overwrite=False):
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
            if not self.paths['track_table'].exists() or overwrite:
                print("Tree Maze Track Data Not Found or Overwrite = True, processing...")
                t1 = time.time()
                t_vt, x_vt, y_vt, ha_vt = self.get_raw_track_data()
                t_rs = self.get_time()
                x, y, ha, nan_idx = tmf.pre_process_track_data(x_vt, y_vt, ha_vt, t_vt, t_rs, self.task_params)

                df = pd.DataFrame(columns=['t', 'x', 'y', 'ha'])
                df['x'] = x
                df['y'] = y
                df['ha'] = ha
                df['t'] = t_rs

                df.to_csv(self.paths['track_table'])
                np.save(self.paths['not_valid_pos_samps'], nan_idx)

                print(f"Processing Completed: {time.time() - t1:0.2f} seconds")
            else:
                df = pd.read_csv(self.paths['track_table'], index_col=0)
                nan_idx = np.load(self.paths['not_valid_pos_samps'])

            if return_nan_idx:
                return df, nan_idx
            else:
                return df

    def get_pos_zones(self, overwrite=False, return_invalid_pz=False):

        if self.task[:2] == 'T3':
            if not self.paths['pos_zones'].exists() or overwrite:
                track_data = self.get_track_data()
                tree_maze = tmf.TreeMazeZones()
                pos_zones, invalid_pz_samps = tree_maze.get_pos_zone_ts(track_data.x, track_data.y)
                np.save(self.paths['pos_zones'], pos_zones)
                np.save(self.paths['pos_zones_invalid_samps'], invalid_pz_samps)
            else:
                pos_zones = np.load(self.paths['pos_zones'], allow_pickle=True)

                if return_invalid_pz:
                    invalid_pz_samps = np.load(self.paths['pos_zones_invalid_samps'])

            if return_invalid_pz:
                return pos_zones, invalid_pz_samps
            else:
                return pos_zones
        else:
            raise NotImplementedError

    def get_pos_zones_mat(self):
        if self.task[:2] == 'T3':
            tree_maze = tmf.TreeMazeZones()
            pos_zones = self.get_pos_zones()
            return tree_maze.get_pos_zone_mat(pos_zones)
        else:
            raise NotImplementedError

    def get_raw_events(self):
        if self.task[:2] == 'T3':
            return pp_funcs.get_events(self.paths['Raw'] / 'Events.nev')
        else:
            print("Method not available for this task")
            return None

    def get_event_behavior(self, overwrite=False):
        if self.task[:2] == 'T3':
            return tmf.BehaviorData(self, overwrite)
        else:
            print("Method not available for this task")
            return None

    def check_track_data_validity(self):
        try:
            if self.session_analyses_table.track_data == 1:
                if self.task == 'OF':
                    behav = self.get_track_data()
                    # noinspection PyTypeChecker
                    coverage = np.around(behav['pos_valid_mask'].mean(), 2)
                    return coverage
                else:
                    df, nan_idx = self.get_track_data(return_nan_idx=True)
                    pct_nan = len(nan_idx) / len(df)
                    return 1 - pct_nan
            else:
                return np.nan
        except:
            return np.nan

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

    def get_fr_maps(self, traditional=False):
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
            print('Open Field Firing Rate Maps not Found or Overwrite= True, creating them.')
            if traditional:
                fr_maps = of_funcs.get_session_fr_maps(self)
            else:
                fr_maps = of_funcs.get_session_fr_maps_cont(self)

            np.save(self.paths['cluster_fr_maps'], fr_maps)
        else:

            # get firing rate and track data
            fr = self.get_fr()
            track_data = self.get_track_data()

            # get environment bins
            x_edges = self.task_params['x_bin_edges_']
            y_edges = self.task_params['y_bin_edges_']
            n_x_bins = len(x_edges) - 1
            n_y_bins = len(y_edges) - 1

            # get occupancy map
            pos_count_map = spatial_funcs.histogram_2d(track_data.x, track_data.y, x_edges, y_edges)

            # pre-allocate and set up the map function to be looped
            fr_maps = np.zeros((self.n_units, n_y_bins, n_x_bins))
            fr_map_function = spatial_funcs.firing_rate_2_rate_map
            args = dict(x=track_data.x, y=track_data.y,
                        x_bin_edges=x_edges, y_bin_edges=y_edges, pos_count_map=pos_count_map,
                        occ_num_thr=self.task_params['occ_num_thr'],
                        spatial_window_size=self.task_params['spatial_window_size'],
                        spatial_sigma=self.task_params['spatial_sigma'])
            for unit in range(self.n_units):
                fr_maps[unit] = fr_map_function(fr[unit], **args)

        return fr_maps

    # analyses methods
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

    def get_encoding_model(self, model):
        """
        :param model: one or more of ['speed', 'hd', 'ha', 'border', 'position', 'grid']
        :return:dictionary containing model coefficients, training and test performance
        """

        if self.enc_models is None:
            return of_funcs.get_session_encoding_models(self, models=model)
        else:
            if isinstance(model, str):
                model_method = f"{model}_model"
                if hasattr(self.enc_models, model_method):
                    return getattr(self.enc_models, model_method)
                else:
                    print(f"{model} model")
            else:
                print()

    def get_encoding_models(self, overwrite=False):
        """
        obtains a object with all the models
        :returns: enc_models object
        """
        if (self.enc_models is None) or overwrite:
            if self.n_units == 0:
                print('No units.')
                return None

            if self.task == 'OF':
                print("Getting Encoding Models")
                self.enc_models = of_funcs.get_session_encoding_models(self)
            else:
                print('Method not develop for other tasks.')
                raise NotImplementedError

        return self.enc_models

    def get_encoding_models_scores(self, overwrite=False):
        """
        obtains a series of pandas data frames quantifying the extent of coding to environmental variables
        :param overwrite:
        :returns: dictionary of pandas data frames.
        """
        if self.n_units == 0:
            print('No units.')
            return None, None

        if self.task == 'OF':
            if not self.paths['cluster_OF_encoding_models'].exists() or overwrite:
                sem = self.get_encoding_models()
                # get scores and save
                scores = sem.scores
                scores.to_csv(self.paths['cluster_OF_encoding_models'])
            else:
                scores = pd.read_csv(self.paths['cluster_OF_encoding_models'], index_col=0)
        else:
            print('Method not develop for other tasks.')
            raise NotImplementedError

        return scores

    def get_trial_zone_rates(self, overwrite=False, **trial_params):
        params = dict(reward_blank=False,
                      not_inzone_blank=True,
                      valid_transitions_blank=True,
                      sp_blank=False,
                      trial_end='tE_2')

        params.update(trial_params)

        fn = self.paths['trial_zone_rates']
        if len(trial_params) > 0:
            fn = append_analysis_mods_2_filename(fn, params)

        if self.task[:2] == 'T3':

            if not fn.exists() or overwrite:
                ta = tmf.TrialAnalyses(self, **params)
                fr_tz = ta.get_all_trial_zone_rates(occupation_thr=self.task_params['occ_trial_zone_thr'])
                np.save(fn, fr_tz['out'])
            else:
                fr_tz = np.load(fn, allow_pickle=True)
            return fr_tz
        else:
            raise NotImplementedError

    def get_zone_rates_comps(self, overwrite=False):
        if self.task[:2] == 'T3':
            fn = self.paths['zone_rates_comps']
            if not fn.exists() or overwrite:
                raise NotImplementedError
                # df.to_csv(fn)
            else:
                df = pd.read_csv(fn, index_col=0)
            return df
        else:
            raise NotImplementedError

    def get_zone_rates_remap(self, overwrite=False, **trial_remap_params):

        params = dict(zr_method='trial',
                      corr_method='kendall',
                      reward_blank=False,
                      not_inzone_blank=True,
                      valid_transitions_blank=True,
                      sp_blank=False,
                      trial_end='tE_2')

        params.update(trial_remap_params)

        fn = self.paths['zone_rates_remap']
        if len(trial_remap_params) > 0:
            fn = append_analysis_mods_2_filename(fn, params)

        if self.task[:2] == 'T3':
            if not fn.exists() or overwrite:

                ta = tmf.TrialAnalyses(self, **params)
                df = ta.all_zone_remapping_analyses(corr_method=params['corr_method'],
                                                    zr_method=params['zr_method'])
                df.to_csv(fn)
            else:
                df = pd.read_csv(fn, index_col=0)
            return df
        else:
            raise NotImplementedError

    def get_pop_zone_rates_remap(self, overwrite=False, **trial_remap_params):
        params = dict(corr_method='kendall',
                      reward_blank=False,
                      not_inzone_blank=True,
                      valid_transitions_blank=True,
                      sp_blank=False,
                      trial_end='tE_2')
        params.update(trial_remap_params)

        fn = self.paths['pop_zone_rates_remap']
        if len(trial_remap_params) > 0:
            fn = append_analysis_mods_2_filename(fn, params)

        if self.task[:2] == 'T3':
            if not fn.exists() or overwrite:
                trial_analyses = tmf.TrialAnalyses(self, **params)
                df = trial_analyses.pop_zone_remapping_analyses(corr_method=params['corr_method'])
                df.to_csv(fn)
            else:
                df = pd.read_csv(fn, index_col=0)
            return df
        else:
            raise NotImplementedError

    def get_bal_conds_seg_rates(self, overwrite=False, **trial_params):
        params = dict(segment_type='bigseg',
                      reward_blank=False,
                      not_inzone_blank=True,
                      valid_transitions_blank=True,
                      sp_blank=False,
                      trial_end='tE_2',
                      n_boot=50)

        params.update(trial_params)

        fn = self.paths['bal_conds_seg_rates']
        if len(trial_params) > 0:
            fn = append_analysis_mods_2_filename(fn, params)

        if self.task[:2] == 'T3':

            if not fn.exists() or overwrite:
                trial_analyses = tmf.TrialAnalyses(self, **params)
                df = trial_analyses.bal_conds_segment_rate_analyses(segment_type=params['segment_type'])
                df.to_csv(fn)
            else:
                df = pd.read_csv(fn, index_col=0)
            return df
        else:
            raise NotImplementedError

    def get_bal_conds_seg_boot_rates(self, overwrite=False, **trial_params):
        params = dict(segment_type='subseg',
                      reward_blank=False,
                      not_inzone_blank=True,
                      valid_transitions_blank=True,
                      sp_blank=False,
                      trial_end='tE_2',
                      n_boot=50)

        params.update(trial_params)

        fn = self.paths['bal_conds_seg_boot_rates']
        if len(trial_params) > 0:
            fn = append_analysis_mods_2_filename(fn, params)

        if self.task[:2] == 'T3':
            if not fn.exists() or overwrite:
                trial_analyses = tmf.TrialAnalyses(self, **params)
                df = trial_analyses.get_avg_seg_rates_boot(segment_type=params['segment_type'],
                                                           n_boot=params['n_boot'])
                df.to_csv(fn)
            else:
                df = pd.read_csv(fn, index_col=0)
            return df
        else:
            raise NotImplementedError

    def get_zone_encoder(self, overwrite=False):
        if self.task[:2] == 'T3':
            fn = self.paths['zone_encoder']
            if not fn.exists() or overwrite:
                exp_sets = [dict(max_lag=lag) for lag in [-50, 0, 50]]

                exp_sets += [dict(cue_type=cue_type, max_lag=0) for cue_type in ['fixed', 'inter']]
                exp_sets += [dict(cue_type=cue_type, max_lag=50) for cue_type in ['fixed', 'inter']]

                exp_sets += [dict(rw_type=rw_type, trial_seg='in', max_lag=0) for rw_type in ['none', 'fixed', 'inter']]
                exp_sets += [dict(rw_type=rw_type, trial_seg='in', max_lag=50) for rw_type in
                             ['none', 'fixed', 'inter']]

                exp_sets += [dict(dir_type=dir_type, trial_seg='all', max_lag=0) for dir_type in
                             ['none', 'fixed', 'inter']]
                exp_sets += [dict(dir_type=dir_type, trial_seg='all', max_lag=50) for dir_type in
                             ['none', 'fixed', 'inter']]

                df = tmf.zone_encoding_analyses(self, exp_sets)
                df.to_csv(fn)
            else:
                df = pd.read_csv(fn, index_col=0)
            return df
        else:
            raise NotImplementedError

    def get_zone_decoder(self, overwrite=False, verbose=False):
        if self.task[:2] == 'T3':
            fn = self.paths['zone_decoder']
            feature_types = ['encoder', 'neural']
            target_types = ['cue', 'dec', 'first_goal', 'rw_goal', 'zones']

            if not fn.exists() or overwrite:
                df = tmf.zone_decoder_analyses(self, feature_types=feature_types,
                                               target_types=target_types, verbose=verbose)
                df.to_csv(fn)
            else:
                df = pd.read_csv(fn, index_col=0)
                tmz = tmf.TreeMazeZones()
                df['zones'] = df['zones'].astype(pd.api.types.CategoricalDtype(tmz.zones2))
            return df
        else:
            raise NotImplementedError


def append_analysis_mods_2_filename(fn, params):
    """
    appends a string extension to a filename that reflects the values in the params dictionary
    :param fn: pathlib path instance
    :param params: dictionary of analyses modifications
    :return: modified fn
    """
    # experiment string
    exp_str = '_'
    #
    # if ~isinstance(params, dict):
    #     print("invalid input")
    #     return fn
    for k, v in params.items():
        if k == 'parallel':
            continue
        if type(v) == 'str':
            exp_str += v
        elif type(v) == bool:
            exp_str += str(int(v))
        else:
            exp_str += str(v)

    if len(params) > 0:
        name = fn.name.split('.')
        name2 = name[0] + exp_str + '.' + name[1]
        fn = fn.parent / name2

    return fn


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
            'occ_num_thr': 3,  # number of occupation times threshold [bins
            'occ_time_thr': time_step * 3,  # time occupation threshold [sec]
            'speed_bin': 2,  # speed bin size [cm/s]

            # filtering parameters
            'spatial_sigma': 2,  # spatial smoothing sigma factor [au]
            'spatial_window_size': 3,  # number of spatial position bins to smooth [bins]
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

            'grid_score_params__': {'ac_thr': 0.01,  # autocorrelation threshold for finding fields
                                    'radix_range': [0.5, 2.0],  # range of radii for grid score in the autocorr
                                    'apply_sigmoid': True,  # apply sigmoid to rate maps
                                    'sigmoid_center': 0.5,  # center for sigmoid
                                    'sigmoid_slope': 10,  # slope for sigmoid
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
        task_params['ang_bin_edges_'] = np.arange(0, 2 * np.pi + task_params['rad_bin'], task_params['rad_bin'])
        task_params['ang_bin_centers_'] = task_params['ang_bin_edges_'][:-1] + task_params['rad_bin'] / 2
        task_params['n_ang_bins'] = len(task_params['ang_bin_centers_'])

        task_params['sp_bin_edges_'] = np.arange(task_params['min_speed_thr'],
                                                 task_params['max_speed_thr'] + task_params['speed_bin'],
                                                 task_params['speed_bin'])
        task_params['sp_bin_centers_'] = task_params['sp_bin_edges_'][:-1] + task_params['speed_bin'] / 2
        task_params['n_sp_bins'] = len(task_params['sp_bin_centers_'])

        task_params['x_bin_edges_'] = np.arange(task_params['x_cm_lims'][0],
                                                task_params['x_cm_lims'][1] + task_params['cm_bin'],
                                                task_params['cm_bin'])
        task_params['x_bin_centers_'] = task_params['x_bin_edges_'][:-1] + task_params['cm_bin'] / 2
        task_params['n_x_bins'] = len(task_params['x_bin_centers_'])
        task_params['n_width_bins'] = task_params['n_x_bins']
        task_params['width'] = task_params['n_x_bins']

        task_params['y_bin_edges_'] = np.arange(task_params['y_cm_lims'][0],
                                                task_params['y_cm_lims'][1] + task_params['cm_bin'],
                                                task_params['cm_bin'])
        task_params['y_bin_centers_'] = task_params['y_bin_edges_'][:-1] + task_params['cm_bin'] / 2
        task_params['n_y_bins'] = len(task_params['y_bin_centers_'])
        task_params['n_height_bins'] = task_params['n_y_bins']
        task_params['height'] = task_params['n_y_bins']

    elif task[:2] == 'T3':
        if subject in ['Li', 'Ne', 'Cl', 'Ca', 'Al']:
            conv_params = {
                # pixel params
                'x_pix_lims': [100, 650],  # camera field of view x limits [pixels]
                'y_pix_lims': [100, 500],  # camera field of view y limits [pixels]
                'x_pix_bias': -380,  # factor for centering the x pixel position
                'y_pix_bias': -280,  # factor for centering the y pixel position
                'vt_rate': 1.0 / 60.0,  # video acquisition frame rate
                'xy_pix_rot_rad': np.pi / 2 + 0.05,  # rotation of original xy pix camera to experimenter xy

                # conversion params
                'x_pix_mm': 1358 / 269.0,  # pixels to mm for the x axis [pix/mm]
                'y_pix_mm': 1308 / 300.0,  # pixels to mm for the y axis [pix/mm]
                'x_neg_warp_factor': 1.05,  # warping for the left side
                'x_pos_warp_factor': 0.93,  # warping for the left side
                'x_mm_bias': 35,  # factor for centering the x mm position
                'y_mm_bias': 650,  # factor for centering the y mm position
                'x_mm_lims': [-650, 650],  # limits on the x axis of the maze [mm]
                'y_mm_lims': [-60, 1400],  # limits on the y axis of the maze [mm]
                'x_cm_lims': [-65, 65],  # limits on the x axis of the maze [cm]
                'y_cm_lims': [-6, 140],  # limits on the y axis of the maze [cm]
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
            'max_speed_thr': 100,  # max speed threshold for allowing valid movement [cm/s]
            'min_speed_thr': 2,  # min speed threshold for allowing valid movement [cm/s]
            'rad_bin': np.deg2rad(10),  # angle radians per bin [rad]
            'occ_num_thr': 3,  # number of occupation times threshold [bins
            'occ_time_thr': time_step * 3,  # time occupation threshold [sec]
            'occ_trial_zone_thr': 2,  # number of instances for a zone to be included
            'speed_bin': 2,  # speed bin size [cm/s]

            # filtering parameters
            'spatial_sigma': 2,  # spatial smoothing sigma factor [au]
            'spatial_window_size': 3,  # number of spatial position bins to smooth [bins]
            'temporal_window_size': 5,  # smoothing temporal window for filtering [bins]
            'temporal_angle_window_size': 5,  # smoothing temporal window for angles [bins]
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

            'grid_score_params__': {'ac_thr': 0.01,  # autocorrelation threshold for finding fields
                                    'radix_range': [0.5, 2.0],  # range of radii for grid score in the autocorr
                                    'apply_sigmoid': True,  # apply sigmoid to rate maps
                                    'sigmoid_center': 0.5,  # center for sigmoid
                                    'sigmoid_slope': 10,  # slope for sigmoid
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
        task_params['ang_bin_edges_'] = np.arange(0, 2 * np.pi + task_params['rad_bin'], task_params['rad_bin'])
        task_params['ang_bin_centers_'] = task_params['ang_bin_edges_'][:-1] + task_params['rad_bin'] / 2
        task_params['n_ang_bins'] = len(task_params['ang_bin_centers_'])

        task_params['sp_bin_edges_'] = np.arange(task_params['min_speed_thr'],
                                                 task_params['max_speed_thr'] + task_params['speed_bin'],
                                                 task_params['speed_bin'])
        task_params['sp_bin_centers_'] = task_params['sp_bin_edges_'][:-1] + task_params['speed_bin'] / 2
        task_params['n_sp_bins'] = len(task_params['sp_bin_centers_'])

        task_params['x_bin_edges_'] = np.arange(task_params['x_cm_lims'][0],
                                                task_params['x_cm_lims'][1] + task_params['cm_bin'],
                                                task_params['cm_bin'])
        task_params['x_bin_centers_'] = task_params['x_bin_edges_'][:-1] + task_params['cm_bin'] / 2
        task_params['n_x_bins'] = len(task_params['x_bin_centers_'])
        task_params['n_width_bins'] = task_params['n_x_bins']
        task_params['width'] = task_params['n_x_bins']

        task_params['y_bin_edges_'] = np.arange(task_params['y_cm_lims'][0],
                                                task_params['y_cm_lims'][1] + task_params['cm_bin'],
                                                task_params['cm_bin'])
        task_params['y_bin_centers_'] = task_params['y_bin_edges_'][:-1] + task_params['cm_bin'] / 2
        task_params['n_y_bins'] = len(task_params['y_bin_centers_'])
        task_params['n_height_bins'] = task_params['n_y_bins']
        task_params['height'] = task_params['n_y_bins']

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
