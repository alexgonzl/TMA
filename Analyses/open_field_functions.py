import sys
import time
import copy

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import filtfilt
from types import SimpleNamespace

from sklearn import linear_model as lm
import Analyses.spatial_functions as spatial_funcs
import Utils.filter_functions as filter_funcs
import Utils.robust_stats as rs


class SpatialMetrics:

    def __init__(self, x, y, speed, ha, hd, fr, spikes, n_jobs=-1, **params):
        self.x = x
        self.y = y
        self.speed = speed
        self.ha = ha
        self.hd = hd
        self.fr = fr
        self.spikes = spikes

        self.n_jobs = n_jobs
        self.n_units = spikes.shape[0]
        self.n_samples = len(x)

        if len(params) > 0:
            for key, val in params.items():
                setattr(self, key, val)

        default_params = default_OF_params()
        for key, val in default_params.items():
            if not hasattr(self, key):
                setattr(self, key, val)

        self.fr_maps = self.get_fr_maps()

        _scores = ['speed_score', 'hd_score', 'ha_score', 'border_score',
                   'grid_score', 'spatial_stability', 'all_scores']
        for s in _scores:
            setattr(self, s, [])

    def get_fr_maps(self):
        fr_maps = np.zeros((self.n_units, self.height, self.width))
        for unit in range(self.n_units):
            fr_maps[unit] = spatial_funcs.firing_rate_2_rate_map(
                self.fr[unit], self.x, self.y,
                x_bin_edges=self.x_bin_edges_, y_bin_edges=self.y_bin_edges_,
                occ_num_thr=self.occ_num_thr,
                spatial_window_size=self.spatial_window_size,
                spatial_sigma=self.spatial_sigma)

        return fr_maps

    def get_speed_score(self):
        score, sig = spatial_funcs.speed_score_traditional(speed=self.speed, fr=self.fr,
                                                           min_speed=self.min_speed_thr,
                                                           max_speed=self.max_speed_thr,
                                                           n_perm=self.n_perm, sig_alpha=self.sig_alpha,
                                                           n_jobs=self.n_jobs)

        out = pd.DataFrame(columns=['speed_score', 'speed_sig'])
        out['speed_score'] = score
        out['speed_sig'] = sig

        self.speed_score = out
        return out

    def get_hd_score(self):
        scores = spatial_funcs.angle_score_traditional(theta=self.hd, fr=self.fr, speed=self.speed,
                                                       min_speed=self.min_speed_thr,
                                                       max_speed=self.max_speed_thr,
                                                       sig_alpha=self.sig_alpha)

        out = scores[['vec_len', 'mean_ang', 'sig']]
        out = out.rename(columns={'sig': 'hd_sig', 'vec_len': 'hd_score', 'mean_ang': 'hd_ang'})

        self.hd_score = out
        return out

    def get_ha_score(self):
        scores = spatial_funcs.angle_score_traditional(theta=self.ha, fr=self.fr, speed=self.speed,
                                                       min_speed=self.min_speed_thr,
                                                       max_speed=self.max_speed_thr,
                                                       sig_alpha=self.sig_alpha)

        out = scores[['vec_len', 'mean_ang', 'sig']]
        out = out.rename(columns={'sig': 'ha_sig', 'vec_len': 'ha_score', 'mean_ang': 'ha_ang'})

        self.ha_score = out

        return out

    def get_border_score(self):

        score, sig = spatial_funcs.permutation_test_border_score(self.fr, self.fr_maps, self.x, self.y,
                                                                 x_bin_edges=self.x_bin_edges_,
                                                                 y_bin_edges=self.y_bin_edges_,
                                                                 n_perm=self.n_perm, sig_alpha=self.sig_alpha,
                                                                 n_jobs=self.n_jobs,
                                                                 **self.border_score_params__)

        out = pd.DataFrame(columns=['border_score', 'border_sig'])
        out['border_score'] = score
        out['border_sig'] = sig

        self.border_score = out

        return out

    def get_grid_score(self):
        """
        Computes grid score.
        :return:
        """

        score, sig, scale, phase = spatial_funcs.permutation_test_grid_score(self.fr, self.fr_maps, self.x, self.y,
                                                                             x_bin_edges=self.x_bin_edges_,
                                                                             y_bin_edges=self.y_bin_edges_,
                                                                             n_perm=self.n_perm,
                                                                             sig_alpha=self.sig_alpha,
                                                                             n_jobs=self.n_jobs,
                                                                             **self.grid_score_params__)

        out = pd.DataFrame(columns=['grid_score', 'grid_sig', 'grid_scale', 'grid_phase'])
        out['grid_score'] = score
        out['grid_sig'] = sig
        out['grid_scale'] = scale
        out['grid_phase'] = phase

        self.grid_score = out

        return out

    def get_spatial_stability(self):

        stability_corr, stability_sig = \
            spatial_funcs.permutation_test_spatial_stability(self.fr, self.x, self.y,
                                                             x_bin_edges=self.x_bin_edges_,
                                                             y_bin_edges=self.y_bin_edges_,
                                                             sig_alpha=self.sig_alpha,
                                                             n_perm=self.n_perm,
                                                             occ_num_thr=self.occ_num_thr,
                                                             spatial_window_size=self.spatial_window_size,
                                                             spatial_sigma=self.spatial_sigma,
                                                             n_jobs=self.n_jobs)

        out = pd.DataFrame(columns=['stability_corr', 'stability_sig'])
        out['stability_corr'] = stability_corr
        out['stability_sig'] = stability_sig

        self.spatial_stability = out

        return out

    def get_all_metrics(self):

        t0 = time.time()
        speed = self.get_speed_score()
        t1 = time.time()
        print(f'Speed Score Completed: {t1 - t0:0.2f}s')

        hd = self.get_hd_score()
        t2 = time.time()
        print(f'Head Dir Score Completed: {t2 - t1:0.2f}s')

        ha = self.get_ha_score()
        t3 = time.time()
        print(f'Head Ang Score Completed: {t3 - t2:0.2f}s')

        border = self.get_border_score()
        t4 = time.time()
        print(f'Border Score Completed: {t4 - t3:0.2f}s')

        grid = self.get_grid_score()
        t5 = time.time()
        print(f'Grid Score Completed: {t5 - t4:0.2f}s')

        spatial_stability = self.get_spatial_stability()
        t6 = time.time()
        print(f'Spatial Stability Score Completed: {t6 - t5:0.2f}s')

        scores = pd.concat([speed, hd, ha, border, grid, spatial_stability], axis=1)

        self.all_scores = scores
        return scores


class SpatialEncodingModelCrossVal:
    """
    Class: Encoding model class with generalized train/test/predict methods for spatial
    """

    def __init__(self, features, response, x_pos, y_pos, n_xval,
                 response_type='fr', reg_type='linear', norm_resp=None,
                 features_by_fold=False, features_by_fold_unit=False, crossval_samp_ids=None, samps_per_split=1000,
                 model_function=None, spatial_map_function=None, random_seed=42, **spatial_map_params):

        np.random.seed(random_seed)

        self.features = features
        self.response = response
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.n_units = response.shape[0]
        self.n_xval = n_xval
        self.reg_type = reg_type
        self.features_by_fold = features_by_fold
        self.features_by_fold_unit = features_by_fold_unit

        self.norm_resp = None
        if norm_resp is not None:
            if norm_resp in ['zscore', 'max']:
                self.norm_resp = norm_resp

        # response info
        self.max_response = np.zeros((n_xval, self.n_units)) * np.nan
        self.mean_response = np.zeros((n_xval, self.n_units)) * np.nan
        self.std_response = np.zeros((n_xval, self.n_units)) * np.nan

        if crossval_samp_ids is not None:
            self.crossval_samp_ids = crossval_samp_ids
        else:
            n_samps = len(x_pos)
            self.crossval_samp_ids = rs.split_timeseries(n_samps=n_samps,
                                                         samps_per_split=samps_per_split,
                                                         n_data_splits=n_xval)

        if features_by_fold:
            self.n_features = features[0, 0].shape[1]
        else:
            self.n_features = features.shape[1]

        if model_function is None:
            if reg_type == 'poisson':
                self.model_function = lm.PoissonRegressor(alpha=0, fit_intercept=False)
            else:
                self.model_function = lm.LinearRegression(fit_intercept=False)
        else:
            self.model_function = model_function

        if spatial_map_function is None:
            map_params_names = ['x_bin_edges', 'y_bin_edges', 'spatial_window_size', 'spatial_sigma']
            default_map_params = default_OF_params()
            for p in map_params_names:
                if p not in spatial_map_params.keys():
                    spatial_map_params[p] = default_map_params[p]
            self.spatial_map_function = spatial_funcs.get_spatial_map_function(response_type, **spatial_map_params)
        else:
            self.spatial_map_function = spatial_map_function

        self.models = np.empty((n_xval, self.n_units), dtype=object)

        self.model_metrics = ['r2', 'ar2', 'err', 'n_err', 'map_r']
        n_idx = self.n_xval * self.n_units * len(self.model_metrics) * 2  # 2-> train/test split
        self.scores = pd.DataFrame(index=range(n_idx), columns=['fold', 'unit_id', 'metric', 'split', 'value'])

    def get_features_fold(self, fold, unit=None):
        if self.features_by_fold:
            train_features = self.features[fold, 0]  # 0 -> train set
            test_features = self.features[fold, 1]  # 1 -> test set
        elif self.features_by_fold_unit:
            train_features = self.features[fold, unit, 0]  # 0 -> train set
            test_features = self.features[fold, unit, 1]  # 1 -> test set
        else:
            train_features = self.features[self.crossval_samp_ids != fold, :]
            test_features = self.features[self.crossval_samp_ids == fold, :]
        return train_features, test_features

    def get_response_fold(self, fold):
        train_response = self.response[:, self.crossval_samp_ids != fold]
        test_response = self.response[:, self.crossval_samp_ids == fold]

        if self.norm_resp == 'zscore':

            self.mean_response[fold, :] = np.nanmean(train_response, axis=1)
            self.std_response[fold, :] = np.nanstd(train_response, axis=1)

            mr = self.mean_response[fold, :][:, np.newaxis]
            sr = self.std_response[fold, :][:, np.newaxis]

            train_response = (train_response - mr) / sr
            test_response = (test_response - mr) / sr

        elif self.norm_resp == 'max':

            self.max_response[fold, :] = np.nanmax(train_response, axis=1)

            mr = self.max_response[fold, :][:, np.newaxis]
            train_response = train_response / mr
            test_response = test_response / mr

        return train_response, test_response

    def train_model_fold(self, fold):
        train_features = None
        if self.features_by_fold_unit:
            pass
        else:
            train_features, _ = self.get_features_fold(fold)
        train_response, _ = self.get_response_fold(fold)

        model_units = np.empty(self.n_units, dtype=object)
        for unit in range(self.n_units):
            model_function = copy.deepcopy(self.model_function)
            if self.features_by_fold_unit:
                train_features, _ = self.get_features_fold(fold, unit)
            if train_features is not None:
                model_units[unit] = model_function.fit(train_features, train_response[unit])
        return model_units

    def predict_model_fold(self, fold):
        train_features = None
        test_features = None
        if self.features_by_fold_unit:
            pass
        else:
            train_features, test_features = self.get_features_fold(fold)

        train_response, test_response = self.get_response_fold(fold)
        train_response_hat = np.empty_like(train_response)
        test_response_hat = np.empty_like(test_response)
        for unit in range(self.n_units):
            if self.features_by_fold_unit:
                train_features, test_features = self.get_features_fold(fold, unit)

            if train_features is not None:
                train_response_hat[unit] = self.models[fold, unit].predict(train_features)
                test_response_hat[unit] = self.models[fold, unit].predict(test_features)
            else:
                train_response_hat[unit] = np.nan
                test_response_hat[unit] = np.nan

        return train_response_hat, test_response_hat

    def get_spatial_map_fold(self, fold):
        # get spatial positions for this fold and by train/test sets
        test_idx = self.crossval_samp_ids == fold
        x_test = self.x_pos[test_idx]
        y_test = self.y_pos[test_idx]

        train_idx = self.crossval_samp_ids != fold
        x_train = self.x_pos[train_idx]
        y_train = self.y_pos[train_idx]

        # get responses
        train_response, test_response = self.get_response_fold(fold)

        # get true spatial maps for this fold
        train_map = np.empty(self.n_units, dtype=object)
        test_map = np.empty(self.n_units, dtype=object)
        for unit in range(self.n_units):
            train_map[unit] = self.spatial_map_function(train_response[unit], x_train, y_train)
            test_map[unit] = self.spatial_map_function(test_response[unit], x_test, y_test)

        return train_map, test_map

    def get_predicted_spatial_map_fold(self, fold):
        # get spatial positions for this fold and by train/test sets
        test_idx = self.crossval_samp_ids == fold
        x_test = self.x_pos[test_idx]
        y_test = self.y_pos[test_idx]

        train_idx = self.crossval_samp_ids != fold
        x_train = self.x_pos[train_idx]
        y_train = self.y_pos[train_idx]

        # get predicted responses
        train_response_hat, test_response_hat = self.predict_model_fold(fold)

        # get predicted spatial maps for this fold
        train_map_hat = np.empty(self.n_units, dtype=object)
        test_map_hat = np.empty(self.n_units, dtype=object)
        for unit in range(self.n_units):
            try:
                train_map_hat[unit] = self.spatial_map_function(train_response_hat[unit], x_train, y_train)
                test_map_hat[unit] = self.spatial_map_function(test_response_hat[unit], x_test, y_test)
            except:
                pass

        return train_map_hat, test_map_hat

    def score_model_fold(self, fold):

        # regression metrics
        train_response, test_response = self.get_response_fold(fold)
        train_response_hat, test_response_hat = self.predict_model_fold(fold)

        train_score = rs.get_regression_metrics(train_response, train_response_hat,
                                                reg_type=self.reg_type, n_params=self.n_features)
        test_score = rs.get_regression_metrics(test_response, test_response_hat,
                                               reg_type=self.reg_type, n_params=self.n_features)

        # spatial map correlations
        train_map, test_map = self.get_spatial_map_fold(fold)
        train_map_hat, test_map_hat = self.get_predicted_spatial_map_fold(fold)

        train_r = np.zeros(self.n_units) * np.nan
        test_r = np.zeros(self.n_units) * np.nan
        for unit in range(self.n_units):
            try:
                train_r[unit] = rs.pearson(train_map[unit].flatten(), train_map_hat[unit].flatten())
                test_r[unit] = rs.pearson(test_map[unit].flatten(), test_map_hat[unit].flatten())
            except:
                pass

        train_score['map_r'] = train_r
        test_score['map_r'] = test_r

        return train_score, test_score

    def train_model(self):
        for fold in range(self.n_xval):
            self.models[fold, :] = self.train_model_fold(fold)

    def score_model(self):
        cnt = 0
        for fold in range(self.n_xval):
            fold_train_scores, fold_test_scores = self.score_model_fold(fold)
            for unit in range(self.n_units):
                for metric in self.model_metrics:
                    self.scores.loc[cnt, ['fold', 'unit_id', 'metric', 'split']] = fold, unit, metric, 'train'
                    self.scores.loc[cnt, 'value'] = fold_train_scores[metric][unit]
                    cnt += 1

                    self.scores.loc[cnt, ['fold', 'unit_id', 'metric', 'split']] = fold, unit, metric, 'test'
                    self.scores.loc[cnt, 'value'] = fold_test_scores[metric][unit]
                    cnt += 1

        self.scores = self.scores.astype({'value': 'float'})
        return self.scores

    def avg_folds(self):
        return self.scores.groupby(['unit_id', 'metric', 'split'], as_index=False).mean()


class AllSpatialEncodingModels:
    def __init__(self, x, y, speed, ha, hd, neural_data, data_type='fr', bias_term=True, n_xval=5,
                 secs_per_split=20.0, n_jobs=-1, random_seed=42, norm_resp='None', **params):
        """
        :param x: array length [n_samps], x position of subject
        :param y: array length [n_samps], y position of subject
        :param speed: array length [n_samps], speed of subject
        :param ha: array length [n_samps],  head angle
        :param hd: array length [n_samps],  heading direction
        :param neural_data: array [n_neurons x n_samps], neural data, can be firing rate (spikes/s)
            or binned spikes (spike counts per sample bin)
        :param data_type: string, indicates the data type in neural_data
        :param bias_term: bool, if True, encoding models will be model with a bias term
        :param n_xval: int, number of cross validation splits
        :param secs_per_split: float, indicates length of time for splitting time series in the x-validation folds
        :param n_jobs: number of cores to use
        :param params: additional model parameters, see specific models for details
        """

        np.random.seed(random_seed)

        self.x = x
        self.y = y
        self.speed = speed
        self.ha = ha
        self.hd = hd

        self.data_type = data_type
        self.bias_term = bias_term
        self.n_xval = n_xval
        self.n_jobs = n_jobs

        self.reg_type = 'poisson' if data_type == 'spikes' else 'linear'

        # add extra dimension if neural data is a 1d of samples (1 unit data)
        if neural_data.ndim == 1:
            neural_data = neural_data[np.newaxis,]

        self.neural_data = neural_data
        self.n_units = neural_data.shape[0]
        self.n_samples = len(x)

        self.norm_resp = norm_resp

        # get parameters
        self.params = params
        if len(params) > 0:
            for key, val in params.items():
                self.params[key] = val

        default_params = default_OF_params()
        for key, val in default_params.items():
            if key not in self.params.keys():
                self.params[key] = val

        self.params['occ_num_thr'] = 1
        self.params['occ_time_thr'] = 0.02

        self.spatial_map_function = spatial_funcs.get_spatial_map_function(self.data_type, **self.params)

        # get time series cross validation splits
        self.secs_per_split = secs_per_split
        self.samps_per_split = np.int(secs_per_split / params['time_step'])

        try:
            self.crossval_samp_ids = rs.split_timeseries(n_samps=self.n_samples,
                                                         samps_per_split=self.samps_per_split,
                                                         n_data_splits=n_xval)
            self.valid_record = True
        except AssertionError:
            print("Insuficient Number of samples to run models.")
            print(sys.exc_info()[1])
            self.crossval_samp_ids = None
            self.valid_record = False

        self.valid_sp_samps = np.ones(self.n_samples, dtype=bool)

        self.models = ['speed', 'hd', 'ha', 'border', 'grid', 'pos']
        self.model_codes = {'speed': 's', 'hd': 'd', 'ha': 'a', 'border': 'b', 'grid': 'g', 'pos': 'p'}
        self.speed_model = None
        self.pos_model = None
        self.ha_model = None
        self.hd_model = None
        self.border_model = None
        self.grid_model = None

        # aggregated models
        self.agg_codes_submodels_dict = {
            'all': ['speed', 'hd', 'border', 'grid', 'pos'],
            'sdp': ['speed', 'hd', 'pos'],
            'sdbg': ['speed', 'hd', 'border', 'grid']}

        self.agg_codes = list(self.agg_codes_submodels_dict.keys())
        self.agg_model_names = ['agg_' + agg_code for agg_code in self.agg_codes]

        self.agg_all_model = None
        self.agg_sdp_model = None  # speed position head-direcation
        self.agg_sdbg_model = None  # speed position head-direction border grid

        self.all_model_names = list(self.models)
        self.all_model_names += self.agg_model_names

        self.scores = None

    def get_speed_model(self):
        features, sp_bin_idx, valid_samps = spatial_funcs.get_speed_encoding_features(self.speed,
                                                                                      self.params['sp_bin_edges_'])
        response = self.neural_data[:, valid_samps]
        self.valid_sp_samps = valid_samps

        crossval_samp_ids = self.crossval_samp_ids[valid_samps]
        x = self.x[valid_samps]
        y = self.y[valid_samps]

        if self.data_type == 'spikes':
            model_function = lm.PoissonRegressor(alpha=0, fit_intercept=False)
        else:
            model_function = lm.LinearRegression(fit_intercept=False)

        self.speed_model = SpatialEncodingModelCrossVal(features, response, x, y,
                                                        crossval_samp_ids=crossval_samp_ids, n_xval=self.n_xval,
                                                        response_type=self.data_type, reg_type=self.reg_type,
                                                        model_function=model_function,
                                                        spatial_map_function=self.spatial_map_function,
                                                        norm_resp=self.norm_resp)
        self.speed_model.train_model()
        self.speed_model.score_model()

        df = self._score_coefs_df('speed')
        self.speed_model.scores = self.speed_model.scores.append(df, ignore_index=True)

    def get_hd_model(self):
        if self.valid_sp_samps is None:
            features, ang_bin_idx, valid_samps = \
                spatial_funcs.get_angle_encoding_features(self.hd, self.params['ang_bin_edges_'],
                                                          speed=self.speed,
                                                          min_speed=self.params['min_speed_thr'],
                                                          max_speed=self.params['max_speed_thr'])
        else:
            features, ang_bin_idx, valid_samps = \
                spatial_funcs.get_angle_encoding_features(self.hd, self.params['ang_bin_edges_'],
                                                          valid_samps=self.valid_sp_samps)

        response = self.neural_data[:, valid_samps]

        crossval_samp_ids = self.crossval_samp_ids[valid_samps]
        x = self.x[valid_samps]
        y = self.y[valid_samps]

        if self.data_type == 'spikes':
            model_function = lm.PoissonRegressor(alpha=0, fit_intercept=False)
        else:
            model_function = lm.LinearRegression(fit_intercept=False)

        self.hd_model = SpatialEncodingModelCrossVal(features, response, x, y,
                                                     crossval_samp_ids=crossval_samp_ids, n_xval=self.n_xval,
                                                     response_type=self.data_type, reg_type=self.reg_type,
                                                     model_function=model_function,
                                                     spatial_map_function=self.spatial_map_function,
                                                     norm_resp=self.norm_resp)
        self.hd_model.train_model()
        self.hd_model.score_model()

        df = self._score_coefs_df('hd')
        self.hd_model.scores = self.hd_model.scores.append(df, ignore_index=True)

    def get_ha_model(self):
        if self.valid_sp_samps is None:
            features, ang_bin_idx, valid_samps = \
                spatial_funcs.get_angle_encoding_features(self.ha, self.params['ang_bin_edges_'],
                                                          speed=self.speed,
                                                          min_speed=self.params['min_speed_thr'],
                                                          max_speed=self.params['max_speed_thr'])
        else:
            features, ang_bin_idx, valid_samps = \
                spatial_funcs.get_angle_encoding_features(self.ha, self.params['ang_bin_edges_'],
                                                          valid_samps=self.valid_sp_samps)

        response = self.neural_data[:, valid_samps]

        crossval_samp_ids = self.crossval_samp_ids[valid_samps]
        x = self.x[valid_samps]
        y = self.y[valid_samps]

        if self.data_type == 'spikes':
            model_function = lm.PoissonRegressor(alpha=0, fit_intercept=False)
        else:
            model_function = lm.LinearRegression(fit_intercept=False)

        self.ha_model = SpatialEncodingModelCrossVal(features, response, x, y,
                                                     crossval_samp_ids=crossval_samp_ids, n_xval=self.n_xval,
                                                     response_type=self.data_type, reg_type=self.reg_type,
                                                     model_function=model_function,
                                                     spatial_map_function=self.spatial_map_function,
                                                     norm_resp=self.norm_resp)
        self.ha_model.train_model()
        self.ha_model.score_model()

        df = self._score_coefs_df('ha')
        self.ha_model.scores = self.ha_model.scores.append(df, ignore_index=True)

    def get_pos_model(self):

        # get feauture parameters
        feature_params = {
            'spatial_window_size': self.params[
                'spatial_window_size'] if 'spatial_window_size' in self.params.keys() else 3,
            'spatial_sigma': self.params['spatial_sigma'] if 'spatial_sigma' in self.params.keys() else 2,
            'feat_type': self.params['feat_type'] if 'feat_type' in self.params.keys() else 'pca'}

        if 'n_components' in self.params.keys():
            feature_params['n_components'] = self.params['n_components']
        else:
            if feature_params['feat_type'] == 'pca':
                feature_params['n_components'] = 0.95
            elif feature_params['feat_type'] == 'nma':
                feature_params['n_components'] = 100

        # get regression params
        if 'reg_alpha' in self.params.keys():
            reg_alpha = self.params['reg_alpha'] if (self.params['alpha'] is not None) else 0.15
        else:
            reg_alpha = 0.15

        if self.data_type == 'spikes':
            model_function = lm.PoissonRegressor(alpha=reg_alpha, fit_intercept=self.bias_term, max_iter=50)
        else:
            l1_ratio = self.params['l1_ratio'] if ('l1_ratio' in self.params.keys()) else 0.15
            model_function = lm.ElasticNet(alpha=reg_alpha, l1_ratio=l1_ratio, fit_intercept=self.bias_term)

        # get features
        features, inverse = spatial_funcs.get_position_encoding_features(self.x, self.y,
                                                                         self.params['x_bin_edges_'],
                                                                         self.params['y_bin_edges_'],
                                                                         **feature_params)

        # get response
        response = self.neural_data

        self.pos_model = SpatialEncodingModelCrossVal(features, response, self.x, self.y,
                                                      crossval_samp_ids=self.crossval_samp_ids, n_xval=self.n_xval,
                                                      response_type=self.data_type, reg_type=self.reg_type,
                                                      model_function=model_function,
                                                      spatial_map_function=self.spatial_map_function,
                                                      norm_resp=self.norm_resp)
        self.pos_model.train_model()
        self.pos_model.score_model()
        setattr(self.pos_model, 'feature_inverse', inverse)

        df = self._score_coefs_df('pos')
        self.pos_model.scores = self.pos_model.scores.append(df, ignore_index=True)

    def get_grid_model(self):

        feature_params = {'thr': self.params['rate_thr'] if ('rate_thr' in self.params.keys()) else 0.1,
                          'min_field_size': self.params['min_field_size'] if (
                                  'min_field_size' in self.params.keys()) else 10,
                          'sigmoid_center': self.params['sigmoid_center'] if (
                                  'sigmoid_center' in self.params.keys()) else 0.5,
                          'sigmoid_slope': self.params['sigmoid_slope'] if (
                                  'sigmoid_slope' in self.params.keys()) else 10,
                          'binary_fields': self.params['binary_fields'] if (
                                  'binary_fields' in self.params.keys()) else False,
                          'x_bin_edges': self.params['x_bin_edges_'],
                          'y_bin_edges': self.params['y_bin_edges_']}

        response = self.neural_data

        # needs to generate feautures by fold/unit
        features = np.empty((self.n_xval, self.n_units, 2), dtype=object)
        fields = np.empty((self.n_xval, self.n_units), dtype=object)
        for unit in range(self.n_units):
            for fold in range(self.n_xval):
                train_ids = self.crossval_samp_ids != fold
                test_ids = self.crossval_samp_ids == fold

                fields[fold, unit] = spatial_funcs.get_grid_fields(response[unit, train_ids], self.x[train_ids],
                                                                   self.y[train_ids],
                                                                   **feature_params)

                if fields[fold, unit] is not None:
                    features[fold, unit, 0] = \
                        spatial_funcs.get_grid_encodign_features(fields[fold, unit],
                                                                 self.x[train_ids],
                                                                 self.y[train_ids],
                                                                 x_bin_edges=feature_params[
                                                                     'x_bin_edges'],
                                                                 y_bin_edges=feature_params[
                                                                     'y_bin_edges'])
                    features[fold, unit, 1] = \
                        spatial_funcs.get_grid_encodign_features(fields[fold, unit],
                                                                 self.x[test_ids],
                                                                 self.y[test_ids],
                                                                 x_bin_edges=feature_params[
                                                                     'x_bin_edges'],
                                                                 y_bin_edges=feature_params[
                                                                     'y_bin_edges'])

        if self.data_type == 'spikes':
            model_function = lm.PoissonRegressor(alpha=0, fit_intercept=self.bias_term)
        else:
            model_function = lm.LinearRegression(fit_intercept=self.bias_term)

        self.grid_model = SpatialEncodingModelCrossVal(features, response, self.x, self.y,
                                                       crossval_samp_ids=self.crossval_samp_ids, n_xval=self.n_xval,
                                                       response_type=self.data_type, reg_type=self.reg_type,
                                                       model_function=model_function,
                                                       spatial_map_function=self.spatial_map_function,
                                                       features_by_fold_unit=True, norm_resp=self.norm_resp)

        self.grid_model.train_model()
        self.grid_model.score_model()
        setattr(self.grid_model, 'grid_fields', fields)

        df = self._score_coefs_df('grid')
        self.grid_model.scores = self.grid_model.scores.append(df, ignore_index=True)

    def get_border_model(self):
        feature_params = {'feat_type': self.params['feat_type'] if ('feat_type' in self.params.keys()) else 'sigmoid',
                          'spatial_window_size': self.params['spatial_window_size'] if (
                                  'spatial_window_size' in self.params.keys()) else 5,
                          'spatial_sigma': self.params['spatial_sigma'] if (
                                  'spatial_sigma' in self.params.keys()) else 2,
                          'border_width_bins': self.params['border_width_bins'] if (
                                  'border_width_bins' in self.params.keys()) else 3,
                          'sigmoid_slope_thr': self.params['sigmoid_slope_thr'] if (
                                  'sigmoid_slope_thr' in self.params.keys()) else 0.1,
                          'center_gaussian_spread': self.params['center_gaussian_spread'] if (
                                  'center_gaussian_spread' in self.params.keys()) else 0.2,
                          'include_center_feature': self.params['include_center_feature'] if (
                                  'include_center_feature' in self.params.keys()) else True,
                          'x_bin_edges': self.params['x_bin_edges_'],
                          'y_bin_edges': self.params['y_bin_edges_']}

        features = spatial_funcs.get_border_encoding_features(self.x, self.y, **feature_params)

        if self.data_type == 'spikes':
            model_function = lm.PoissonRegressor(alpha=0, fit_intercept=self.bias_term)
        else:
            model_function = lm.LinearRegression(fit_intercept=self.bias_term)

        self.border_model = SpatialEncodingModelCrossVal(features, self.neural_data, x_pos=self.x, y_pos=self.y,
                                                         crossval_samp_ids=self.crossval_samp_ids, n_xval=self.n_xval,
                                                         response_type=self.data_type, reg_type=self.reg_type,
                                                         model_function=model_function,
                                                         spatial_map_function=self.spatial_map_function,
                                                         norm_resp=self.norm_resp)

        self.border_model.train_model()
        self.border_model.score_model()

        df = self._score_coefs_df('border')
        self.border_model.scores = self.border_model.scores.append(df, ignore_index=True)

    def get_agg_model(self, agg_code='all'):
        """
        Aggreggate model. combines the outputs of multiple
        :param models:
        :return:
        """

        models = self.agg_codes_submodels_dict[agg_code]
        n_models = len(models)
        agg_model_full_name = f"agg_{agg_code}_model"

        response = self.neural_data
        samples = np.arange(self.n_samples)
        sp_valid_ids_bool = self.valid_sp_samps

        # needs to generate feautures by fold/unit
        features = np.empty((self.n_xval, self.n_units, 2), dtype=object)
        for fold in range(self.n_xval):
            train_ids = self.crossval_samp_ids != fold
            test_ids = self.crossval_samp_ids == fold
            n_fold_train_samps = train_ids.sum()
            n_fold_test_samps = test_ids.sum()

            # get predicted responses for each model, unit, train/test split
            model_resp = {}
            for model in models:
                if getattr(self, f"{model}_model") is None:
                    model_resp[model] = np.zeros((self.n_units, n_fold_train_samps)), \
                                        np.zeros((self.n_units, n_fold_test_samps))
                else:
                    model_resp[model] = getattr(self, f"{model}_model").predict_model_fold(fold)

            # speed & hd responses might have different lengths based on valid moving samples
            fold_train_samples = samples[train_ids]
            fold_train_samples_sp = samples[train_ids & sp_valid_ids_bool]
            within_fold_train_sp_ids_bool = np.in1d(fold_train_samples, fold_train_samples_sp)

            fold_test_samples = samples[test_ids]
            fold_test_samples_sp = samples[test_ids & sp_valid_ids_bool]
            within_fold_test_sp_ids_bool = np.in1d(fold_test_samples, fold_test_samples_sp)

            # create feature vector
            for unit in range(self.n_units):
                X_train = np.zeros((n_fold_train_samps, n_models))
                X_test = np.zeros((n_fold_test_samps, n_models))

                for mm, model in enumerate(models):
                    if model in ['speed', 'hd', 'ha']:
                        X_train[within_fold_train_sp_ids_bool, mm] = model_resp[model][0][unit]
                        X_test[within_fold_test_sp_ids_bool, mm] = model_resp[model][1][unit]
                    else:
                        X_train[:, mm] = model_resp[model][0][unit]
                        X_test[:, mm] = model_resp[model][1][unit]
                X_train[np.isnan(X_train)] = 0
                X_test[np.isnan(X_test)] = 0

                features[fold, unit, 0] = X_train
                features[fold, unit, 1] = X_test

        if self.data_type == 'spikes':
            model_function = lm.PoissonRegressor(alpha=0, fit_intercept=False)
        else:
            model_function = lm.LinearRegression(fit_intercept=False)

        model_obj = SpatialEncodingModelCrossVal(features, response, self.x, self.y,
                                                 crossval_samp_ids=self.crossval_samp_ids, n_xval=self.n_xval,
                                                 response_type=self.data_type, reg_type=self.reg_type,
                                                 model_function=model_function,
                                                 spatial_map_function=self.spatial_map_function,
                                                 features_by_fold_unit=True, norm_resp=self.norm_resp)

        setattr(model_obj, 'included_models', models)
        model_obj.train_model()
        model_obj.score_model()

        setattr(self, agg_model_full_name, model_obj)

        df = self._score_coefs_df('agg_' + agg_code)
        model_obj.scores = model_obj.scores.append(df, ignore_index=True)
        setattr(self, agg_model_full_name, model_obj)

    def _score_coefs_df(self, model, agg_code=None):

        if agg_code is not None:
            metric_name = f"agg_{agg_code}_coef"
            coefs = self.get_coefs(f'agg_{agg_code}')
            models = self.agg_codes_submodels_dict[agg_code]

            def c_func(betas, *args):
                if model in models:
                    model_idx = np.array(models) == model
                    return betas[model_idx]
                else:
                    return np.nan
        else:
            metric_name = 'coef'
            coefs = self.get_coefs(model)
            if model == 'border':
                def c_func(betas):
                    b = np.max(betas[0:4])
                    ce = betas[4]
                    return (b - ce) / np.abs(betas).mean()
            elif model == 'grid':
                def c_func(betas):
                    m = np.nanmean(betas)
                    s = np.nanstd(betas) if len(betas) > 1 else 1
                    return m / s
            elif model == 'pos':
                def c_func(betas, trm):
                    c_map = self.pos_model.feature_inverse(betas)
                    r = rs.pearson(c_map, trm)
                    return np.arctanh(r)
            elif model in ['hd', 'ha']:
                def c_func(betas):
                    cr = rs.circ_corrcl(self.params['ang_bin_edges_'][1:], betas)
                    return np.arctanh(cr)
            elif model == 'speed':
                def c_func(betas):
                    r = rs.spearman(betas, self.params['sp_bin_edges_'][1:])
                    return np.arctanh(r)
            else:
                def c_func(betas):
                    return np.nan

        n_idx = self.n_xval * self.n_units * 2
        df = pd.DataFrame(index=range(n_idx), columns=['fold', 'unit_id', 'metric', 'split', 'value'])
        cnt = 0
        unit_rate_map = None
        for unit in range(self.n_units):
            if (model == 'pos') and (agg_code is None):
                unit_rate_map = self.spatial_map_function(self.neural_data[unit], self.x, self.y).flatten()
            for fold in range(self.n_xval):
                c = coefs[fold, unit]
                if c is not None:
                    if model == 'pos':
                        val = c_func(c, unit_rate_map)
                    elif model == 'grid':
                        if self.grid_model.models[fold, unit] is None:
                            val = np.nan
                        else:
                            val = c_func(c)
                    else:
                        val = c_func(c)
                else:
                    val = np.nan
                df.loc[cnt, ['fold', 'unit_id', 'metric', 'split', 'value']] = fold, unit, metric_name, 'train', val
                cnt += 1
                df.loc[cnt, ['fold', 'unit_id', 'metric', 'split', 'value']] = fold, unit, metric_name, 'test', val
                cnt += 1
        df = df.astype({'value': 'float'})
        return df

    def _add_agg_coefs_model_scores(self):
        '''adds aggregated coefficients score to each submodel: eg. sdp adds the 
        coefficeints of speed, head-direction and position from the agg_sdp_model to the scores of speed_model, hd_model 
        and pos_model with metric name agg_sdp_coefs'''

        for agg_code, included_sub_models in self.agg_codes_submodels_dict.items():
            for model in included_sub_models:
                df = self._score_coefs_df(model, agg_code=agg_code)
                model_obj = getattr(self, f"{model}_model")
                model_scores = model_obj.scores
                model_scores = model_scores.append(df, ignore_index=True)
                setattr(model_obj, 'scores', model_scores)
                # model_scores object updates the model scores tables in self

    def get_models(self, verbose=True):
        if self.valid_record:
            for model in self.models:
                t0 = time.time()
                getattr(self, f"get_{model}_model")()
                if verbose:
                    print(f"{model} model completed. {time.time() - t0:0.2f}secs")

            for agg_code in self.agg_codes:
                t0 = time.time()
                getattr(self, f"get_agg_model")(agg_code)
                if verbose:
                    print(f"aggregate {agg_code} model completed. {time.time() - t0:0.2f}secs")

        return None

    def get_scores(self):
        if self.valid_record:
            scores = pd.DataFrame(columns=['unit_id', 'metric', 'split', 'model', 'value'])
            self._add_agg_coefs_model_scores()

            for model in self.all_model_names:
                if getattr(self, f'{model}_model') is not None:
                    model_tbl = getattr(self, f"{model}_model").avg_folds()
                    model_tbl['model'] = model
                    scores = pd.concat((scores, model_tbl))
                    scores.reset_index(drop=True, inplace=True)
            self.scores = scores
        else:
            self.scores = pd.DataFrame(None)
        return self.scores

    def get_coefs(self, model):
        coefs = np.empty((self.n_xval, self.n_units), dtype=object)
        for fold in range(self.n_xval):
            for unit in range(self.n_units):
                model_obj = getattr(self, f"{model}_model").models[fold][unit]
                if model_obj is not None:
                    coefs[fold, unit] = model_obj.coef_
        return coefs

def cluster_

def get_session_track_data(session_info):
    """
    :param SubjectSessionInfo session_info: instance of class SubjectInfo for a particular subject
    :return: dictionary of open field relevant behavioral variables post processing, including:
            -> np.array t: time bins
            -> np.array x: x position of the animal
            -> np.array y: y position of the animal
            -> np.array sp: speed
            -> np.array ha: head angle
            -> np.array hd: heading direction
            -> np.ndarray pos_map_counts: 2d binned represention of the animals position, values are counts
            -> np.ndarray pos_map_sec: 2d binned represention of the animals position, smoothed, values are seconds
    """

    p = SimpleNamespace(**session_info.task_params)
    time_step = session_info.params['time_step']

    # get session time and track data
    t_rs = session_info.get_time()  # resampled time (binned time)
    t_vt, x_vt, y_vt, ha_vt = session_info.get_raw_track_data()  # position, etc in pixels, ha is in degrees
    ha_vt = np.mod(np.deg2rad(ha_vt), 2 * np.pi)  # convert to radians.

    x, y, ha = _process_track_data(x_vt, y_vt, ha_vt, p)
    speed, hd = spatial_funcs.compute_velocity(x, y, p.vt_rate)

    # convert to cm
    x /= 10
    y /= 10
    speed /= 10  # convert to cm/s
    hd = np.mod(hd, 2 * np.pi)  # convert to 0 to 2pi

    # _rs -> resampled signal
    x_rs = filter_funcs.resample_signal(t_vt, t_rs, x)
    y_rs = filter_funcs.resample_signal(t_vt, t_rs, y)
    ha_rs = filter_funcs.resample_signal(t_vt, t_rs, ha)
    hd_rs = filter_funcs.resample_signal(t_vt, t_rs, hd)
    speed_rs = filter_funcs.resample_signal(t_vt, t_rs, speed)

    # occupation_counts in the spatial matrix
    pos_map_counts = spatial_funcs.histogram_2d(x_rs, y_rs, p.x_bin_edges_, p.y_bin_edges_)
    pos_counts_sm = spatial_funcs.smooth_2d_map(pos_map_counts, n_bins=p.spatial_window_size,
                                                sigma=p.spatial_sigma)

    pos_valid_mask = pos_counts_sm >= p.occ_num_thr

    pos_map_secs = pos_map_counts * time_step
    pos_map_secs = spatial_funcs.smooth_2d_map(pos_map_secs, p.spatial_window_size, p.spatial_sigma)

    # create output dictionary
    of_track_dat = {
        't': t_rs, 'x': x_rs, 'y': y_rs, 'sp': speed_rs, 'ha': ha_rs, 'hd': hd_rs,
        'pos_map_counts': pos_map_counts, 'pos_map_secs': pos_map_secs, 'pos_valid_mask': pos_valid_mask,
        'pos_map_counts_sm': pos_counts_sm}

    return of_track_dat


def _process_track_data(x, y, ha, track_params):
    """
    This function performs the following tasks in order:
        1) masks xy pixel data for out track bound spaces. values set to np.nan
        2) centers the xy pixel data
        3) rotates xy to the experimenters perspective
        4) rescales data from pixels to mm
        5) computes velocity to create a masks for a velocity threshold
        6) masks xy mm data for out of track samples.
        7) apply all the masks
        8) apply median filters to deal with np.nan
        9) if there are any np.nans left, use iterative method to replace with previous non nan value
        10) final smoothing using filtfilt
    :param x: tracking data x position
    :param y: tracking data y position
    :param ha: tracking data head angle
    :param track_params: parameters of the track
    :return: processed x,y and ha
    """

    p = track_params

    # make deep copies
    x = np.array(x)
    y = np.array(y)

    # 1. mask pixels that are out of bounds
    mask_x = np.logical_or(x < p.x_pix_lims[0], x > p.x_pix_lims[1])
    mask_y = np.logical_or(y < p.y_pix_lims[0], y > p.y_pix_lims[1])
    mask = np.logical_or(mask_x, mask_y)

    x[mask] = np.nan
    y[mask] = np.nan

    # 2. centering / pixel translation
    x = x + p.x_pix_bias
    y = y + p.y_pix_bias

    # 3. rotate to experimenter's pov
    x2, y2 = spatial_funcs.rotate_xy(x, y, p.xy_pix_rot_rad)

    # 4. convert to mm / re-scales; bias term re-frames the image
    x2 = x2 * p.x_pix_mm + p.x_mm_bias
    y2 = y2 * p.y_pix_mm + p.y_mm_bias

    with np.errstate(invalid='ignore'):  # avoids warnings about comparing nan values
        # 5. compute velocity to create speed threshold
        dx = np.append(0, np.diff(x2))
        dy = np.append(0, np.diff(y2))
        dr = np.sqrt(dx ** 2 + dy ** 2)
        mask_r = np.abs(dr) > p.max_speed_thr

        # 6. mask creating out of bound zones in mm space
        mask_x = np.logical_or(x2 < p.x_mm_lims[0], x2 > p.x_mm_lims[1])
        mask_y = np.logical_or(y2 < p.y_mm_lims[0], y2 > p.y_mm_lims[1])
        mask = np.logical_or(mask_x, mask_y)
        mask = np.logical_or(mask, mask_r)

    # 7. apply masks
    x2[mask] = np.nan
    y2[mask] = np.nan
    ha2 = np.array(ha)
    ha2[mask] = np.nan

    # 8. median filter the data to deal with nan
    x3 = filter_funcs.median_window_filtfilt(x2, p.temporal_window_size)
    y3 = filter_funcs.median_window_filtfilt(y2, p.temporal_window_size)
    ha3 = filter_funcs.median_window_filtfilt(ha2, p.temporal_angle_window_size)

    # 9. if there are still NaNs assign id to previous value
    nan_ids = np.where(np.logical_or(np.isnan(x3), np.isnan(y3)))[0]
    for ii in nan_ids:
        x3[ii] = filter_funcs.get_last_not_nan_value(x3, ii)
        y3[ii] = filter_funcs.get_last_not_nan_value(y3, ii)
        ha3[ii] = filter_funcs.get_last_not_nan_value(ha3, ii)

    # 10. final filter / smoothing
    x3 = filtfilt(p.filter_coef_, 1, x3)
    y3 = filtfilt(p.filter_coef_, 1, y3)
    ha3 = filter_funcs.angle_filtfilt(ha3, p.filter_coef_angle_)

    return x3, y3, ha3


def get_session_spike_maps(session_info):
    """
    Loops and computes spike count maps for each unit.
    :param SubjectSessionInfo session_info: instance of class SubjectInfo for a particular subject
    :return: np.ndarray spike_maps: shape n_units x n_vertical_bins x n_horizontal_bins
    """

    # get data
    spikes = session_info.get_binned_spikes()
    of_dat = SimpleNamespace(**session_info.get_track_data())
    track_params = SimpleNamespace(**session_info.task_params)
    x_bin_edges = track_params.x_bin_edges_
    y_bin_edges = track_params.y_bin_edges_

    # pre-allocate
    n_units = session_info.n_units
    n_height_bins = track_params.n_height_bins
    n_width_bins = track_params.n_width_bins
    spike_maps = np.zeros((n_units, n_height_bins, n_width_bins))

    # get spike count maps. note that these are un-smooth.
    for unit in range(n_units):
        spike_maps[unit] = spatial_funcs.get_spike_map(spikes[unit], of_dat.x, of_dat.y, x_bin_edges, y_bin_edges)

    return spike_maps


def get_session_fr_maps(session_info):
    """
    Loops and computes smoothed fr maps for each unit.
    :param SubjectSessionInfo session_info: instance of class SubjectInfo for a particular subject
    :return: np.ndarray maps: shape n_units x n_vertical_bins x n_horizontal_bins
    """

    # get data
    spike_maps = session_info.get_spike_maps()
    of_dat = SimpleNamespace(**session_info.get_track_data())
    track_params = SimpleNamespace(**session_info.task_params)
    valid_mask = of_dat.pos_valid_mask

    # mask data
    pos_map_secs2 = np.full_like(of_dat.pos_map_secs, np.nan)
    pos_map_secs2[valid_mask] = of_dat.pos_map_secs[valid_mask]

    # pre-allocate
    n_units = session_info.n_units
    n_vert_bins = of_dat.n_vert_bins
    n_horiz_bins = of_dat.n_horiz_bins
    fr_maps = np.zeros((n_units, n_vert_bins, n_horiz_bins))

    for unit in range(n_units):
        temp_spk_map = np.zeros_like(spike_maps[unit])
        temp_spk_map[valid_mask] = spike_maps[unit][valid_mask]
        temp_fr_map = temp_spk_map / pos_map_secs2
        temp_fr_map[~valid_mask] = 0.0

        fr_maps[unit] = spatial_funcs.smooth_2d_map(temp_fr_map, n_bins=track_params.spatial_window_size,
                                                    sigma=track_params.spatial_sigma)

    return fr_maps


def get_session_fr_maps_cont(session_info):
    """
    Loops and computes smoothed fr maps for each unit. This version uses a weighted 2d histogram, such that each
    x,y sample is weighted by the continuous firing rate of the neuron.
    :param SubjectSessionInfo session_info: instance of class SubjectInfo for a particular subject
    :return: np.ndarray maps: shape n_units x n_vertical_bins x n_horizontal_bins
    """

    # get data
    fr = session_info.get_fr()

    of_dat = SimpleNamespace(**session_info.get_track_data())
    x = of_dat.x
    y = of_dat.y
    valid_mask = of_dat.pos_valid_mask
    pos_counts_sm = of_dat.pos_map_counts_sm

    track_params = SimpleNamespace(**session_info.task_params)
    x_bin_edges = track_params.x_bin_edges_
    y_bin_edges = track_params.y_bin_edges_

    # pre-allocate
    n_units = session_info.n_units
    n_height_bins = track_params.n_height_bins
    n_width_bins = track_params.n_width_bins
    fr_maps = np.zeros((n_units, n_height_bins, n_width_bins))

    for unit in range(n_units):
        temp_fr_map = spatial_funcs.w_histogram_2d(x, y, fr[unit], x_bin_edges, y_bin_edges)
        temp_fr_map[valid_mask] /= pos_counts_sm[valid_mask]

        fr_maps[unit] = spatial_funcs.smooth_2d_map(temp_fr_map, n_bins=track_params.spatial_window_size,
                                                    sigma=track_params.spatial_sigma)

    return fr_maps


def get_session_encoding_models(session_info, models=None):
    """
    Loops and computes traditional scores open-field scores for each unit, these include:
        -> speed score: correlation between firing rate and speed
        -> head angle score: correlation between firing rate and head angle
        -> head directation score: correlation between firing rate and head direction
        -> border score:
        -> grid score:
    :param SubjectSessionInfo session_info: instance of class SubjectInfo for a particular subject
    :return: dict: with all the scores
    """
    # get data
    fr = session_info.get_fr()

    of_dat = SimpleNamespace(**session_info.get_track_data())
    task_params = session_info.task_params

    sem = AllSpatialEncodingModels(x=of_dat.x, y=of_dat.y, speed=of_dat.sp, ha=of_dat.ha,
                                   hd=of_dat.hd, neural_data=fr, n_jobs=10, **task_params)
    if models is None:
        sem.get_models()
    else:
        if isinstance(models, str):
            models = [models]
        for model in models:
            model_method = f"get_{model}_model"
            if hasattr(sem, model_method):
                getattr(sem, model_method)()
            else:
                print(f"Encoding model for {model} does not exists.")

    sem.get_scores()
    return sem


def get_session_scores(session_info):
    """
    Loops and computes traditional scores open-field scores for each unit, these include:
        -> speed score: correlation between firing rate and speed
        -> head angle score: correlation between firing rate and head angle
        -> head directation score: correlation between firing rate and head direction
        -> border score:
        -> spatial information:
        -> grid score:
    :param SubjectSessionInfo session_info: instance of class SubjectInfo for a particular subject
    :return: pandas data frame with scores by neuron
    """
    # get data
    fr = session_info.get_fr()
    spikes = session_info.get_binned_spikes()
    of_dat = SimpleNamespace(**session_info.get_track_data())
    task_params = session_info.task_params

    sm = SpatialMetrics(x=of_dat.x, y=of_dat.y, speed=of_dat.sp, ha=of_dat.ha,
                        hd=of_dat.hd, fr=fr, spikes=spikes, n_jobs=10, **task_params)
    sm.get_all_metrics()

    return sm.all_scores


def default_OF_params():
    params = {
        'time_step': 0.02,  # time step

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

        # binning parameters
        'mm_bin': 30,  # millimeters per bin [mm]
        'cm_bin': 3,  # cm per bin [cm]
        'max_speed_thr': 80,  # max speed threshold for allowing valid movement [cm/s]
        'min_speed_thr': 2,  # min speed threshold for allowing valid movement [cm/s]
        'rad_bin': np.deg2rad(10),  # angle radians per bin [rad]
        'occ_num_thr': 3,  # number of occupation times threshold [bins
        'occ_time_thr': 0.02 * 3,  # time occupation threshold [sec]
        'speed_bin': 2,  # speed bin size [cm/s]

        # filtering parameters
        'spatial_sigma': 2,  # spatial smoothing sigma factor [au]
        'spatial_window_size': 5,  # number of spatial position bins to smooth [bins]
        'temporal_window_size': 11,  # smoothing temporal window for filtering [bins]
        'temporal_angle_window_size': 11,  # smoothing temporal window for angles [bins]
        'temporal_window_type': 'hann',  # window type for temporal window smoothing

        # statistical tests parameters:
        'sig_alpha': 0.02,  # double sided alpha level for significance testing
        'n_perm': 500,  # number of permutations

        # type of encoding model. see spatial_funcs.get_border_encoding_features
        'border_enc_model_type': 'linear',
        # these are ignoed if border_enc_model_type is linear.
        'border_enc_model_feature_params_': {'center_gaussian_spread': 0.2,  # as % of environment
                                             'sigmoid_slope_thr': 0.15,  # value of sigmoid at border width
                                             },
        'reg_type': 'poisson',

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

    # derived parameters
    # -- filter coefficients --
    params['filter_coef_'] = signal.get_window(params['temporal_window_type'],
                                               params['temporal_window_size'],
                                               fftbins=False)
    params['filter_coef_'] /= params['filter_coef_'].sum()

    params['filter_coef_angle_'] = signal.get_window(params['temporal_window_type'],
                                                     params['temporal_angle_window_size'],
                                                     fftbins=False)
    params['filter_coef_angle_'] /= params['filter_coef_angle_'].sum()

    # -- bins --
    params['ang_bin_edges_'] = np.arange(0, 2 * np.pi + params['rad_bin'], params['rad_bin'])
    params['ang_bin_centers_'] = params['ang_bin_edges_'][:-1] + params['rad_bin'] / 2
    params['n_ang_bins'] = len(params['ang_bin_centers_'])

    params['sp_bin_edges_'] = np.arange(params['min_speed_thr'],
                                        params['max_speed_thr'] + params['speed_bin'],
                                        params['speed_bin'])
    params['sp_bin_centers_'] = params['sp_bin_edges_'][:-1] + params['speed_bin'] / 2
    params['n_sp_bins'] = len(params['sp_bin_centers_'])

    params['x_bin_edges_'] = np.arange(params['x_cm_lims'][0],
                                       params['x_cm_lims'][1] + params['cm_bin'],
                                       params['cm_bin'])
    params['x_bin_centers_'] = params['x_bin_edges_'][:-1] + params['cm_bin'] / 2
    params['n_x_bins'] = len(params['x_bin_centers_'])
    params['n_width_bins'] = params['n_x_bins']
    params['width'] = params['n_x_bins']

    params['y_bin_edges_'] = np.arange(params['y_cm_lims'][0],
                                       params['y_cm_lims'][1] + params['cm_bin'],
                                       params['cm_bin'])
    params['y_bin_centers_'] = params['y_bin_edges_'][:-1] + params['cm_bin'] / 2
    params['n_y_bins'] = len(params['y_bin_centers_'])
    params['n_height_bins'] = params['n_y_bins']
    params['height'] = params['n_y_bins']

    return params

