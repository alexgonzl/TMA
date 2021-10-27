
import numpy as np
import pandas as pd
from scipy import ndimage, stats, signal
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from sklearn import linear_model as lm
from sklearn.decomposition import PCA, NMF

from Utils import robust_stats as rs
from skimage import draw
from skimage.transform import rotate
from joblib import delayed, Parallel
from pathlib import Path
import pickle
import warnings


# TODO: 1. move classes out to open field functions
#       2. optimize grid model

# class functions

class Points2D:
    def __init__(self, x, y, polar=False):

        if not isinstance(x, np.ndarray):
            x = np.array([x]).flatten()
        if not isinstance(y, np.ndarray):
            y = np.array([y]).flatten()

        assert len(x) == len(y), 'different lengths'

        self.n = len(x)
        if not polar:
            self.x = np.array(x)
            self.y = np.array(y)
            self.xy = np.column_stack((self.x, self.y))
            self.r, self.ang = self.polar()
        else:
            self.r = x
            self.ang = np.mod(y, 2 * np.pi)
            self.x, self.y = self.eu()
            self.xy = np.column_stack((self.x, self.y))

    def polar(self):
        r = np.sqrt(self.x ** 2 + self.y ** 2)
        ang = np.zeros(self.n)

        for ii in range(self.n):
            ang[ii] = np.math.atan2(self.y[ii], self.x[ii])
        ang = np.mod(ang, 2 * np.pi)
        return r, ang

    def eu(self):
        x = self.r * np.cos(self.ang)
        y = self.r * np.sin(self.ang)
        return x, y

    def __add__(self, b):
        return Points2D(self.x + b.x, self.y + b.y)

    def __sub__(self, b):
        if isinstance(b, (int, float)):
            return Points2D(self.x - b, self.y - b)

        if isinstance(b, (PointsOF, Points2D)):
            return Points2D(self.x - b.x, self.y - b.y)
        else:
            raise NotImplementedError

    def __rsub__(self, b):
        if isinstance(b, (int, float)):
            return Points2D(b - self.x, b - self.y)

        if isinstance(b, (PointsOF, Points2D)):
            return Points2D(b.x - self.x, b.y - self.y)
        else:
            raise NotImplementedError

    def __mul__(self, b):
        if isinstance(b, (int, float, np.float, np.int)):
            return Points2D(b * self.x, b * self.y)

        if isinstance(b, (PointsOF, Points2D)):
            return b.x @ self.x + b.y @ self.y
        else:
            raise NotImplementedError

    def __rmul__(self, b):
        if isinstance(b, (int, float, np.float, np.int)):
            return Points2D(b * self.x, b * self.y)
        elif isinstance(b, (PointsOF, Points2D)):
            if self.n == b.n:
                return Points2D(b.x * self.x, b.y @ self.y)
            if self.n == 1 or b.n == 1:
                return
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        if isinstance(i, (int, np.int, np.ndarray)):
            return Points2D(self.x[i], self.y[i])
        else:
            raise NotImplementedError

    def __len__(self):
        return self.n

    def __str__(self):
        print((self.x, self.y))
        return ''


# open field 2d points
class PointsOF:
    def __init__(self, x, y, height=47, width=42, polar=False):

        if not isinstance(x, np.ndarray):
            x = np.array([x]).flatten()
        if not isinstance(y, np.ndarray):
            y = np.array([y]).flatten()

        assert len(x) == len(y), 'different lengths'

        self.n = len(x)
        self.width = width
        self.height = height

        if not polar:
            self.x = np.round(np.mod(x, self.width))
            self.y = np.round(np.mod(y, self.height))
            self.xy = np.column_stack((x, y))
            self.r, self.ang = self.polar()
        else:
            self.r = x
            self.ang = np.mod(y, 2 * np.pi)
            self.x, self.y = self.eu()
            self.xy = np.column_stack((x, y))

    def polar(self):
        r = np.sqrt(self.x ** 2 + self.y ** 2)
        ang = np.zeros(self.n)

        for ii in range(self.n):
            ang[ii] = np.math.atan2(self.y[ii], self.x[ii])
        ang = np.mod(ang, 2 * np.pi)
        return r, ang

    def eu(self):
        x = np.round(self.r * np.cos(self.ang))
        y = np.round(self.r * np.sin(self.ang))
        return x, y

    def __add__(self, b):
        return PointsOF(self.x + b.x, self.y + b.y)

    def __sub__(self, b):
        return PointsOF(self.x - b.x, self.y - b.y)

    def __getitem__(self, i):
        if isinstance(i, (int, np.int, np.ndarray)):
            return Points2D(self.x[i], self.y[i])
        else:
            raise NotImplementedError

    def __str__(self):
        print((self.x, self.y))
        return ''

    def __len__(self):
        return self.n


# ------------------------------------------------- Spatial Functions --------------------------------------------------

def spatial_information(pos_prob, fr_map):
    """
    returns spatial information in bits/spatial_bin
    :param pos_prob: position probability (position counts divided by sum)
    :param fr_map: resulting firing rate map for each position
    :return:
    spatial information
    """
    nfr = fr_map/np.nansum(fr_map) # normalized map
    info_mat = nfr*np.log2(nfr/pos_prob) # information matrix
    return np.nansum(info_mat)

def smooth_2d_map(bin_map, n_bins=5, sigma=2, apply_median_filt=True, **kwargs):
    """
    :param bin_map: map to be smooth.
        array in which each cell corresponds to the value at that xy position
    :param n_bins: number of smoothing bins
    :param sigma: std for the gaussian smoothing
    :return: sm_map: smoothed map. note that this is a truncated sigma map, meaning that high or
            low values wont affect far away bins
    """
    if apply_median_filt:
        sm_map = ndimage.filters.median_filter(bin_map, n_bins)
    else:
        sm_map = bin_map
    trunc = (((n_bins - 1) / 2) - 0.5) / sigma

    return ndimage.filters.gaussian_filter(sm_map, sigma, mode='constant', truncate=trunc)


def histogram_2d(x, y, x_bin_edges, y_bin_edges):
    """
    :param np.array x: x position of the animal
    :param np.array y: y position of the animal
    :param x_bin_edges: bin edges in the x position
    :param y_bin_edges: bin edges in the y position
    :return: 2d array of position counts, x_bin_edges, and y_bin_edges
    """

    # hist2d converts to a matrix, which reverses x,y
    # inverse order here to preserve visualization.
    pos_counts_2d, _, _ = np.histogram2d(y, x, bins=[y_bin_edges, x_bin_edges])
    return pos_counts_2d


def w_histogram_2d(x, y, w, x_bin_edges, y_bin_edges):
    """
    :param np.array x: x position of the animal
    :param np.array y: y position of the animal
    :param np.array w: weight of each position sample (eg. spike counts or firing rate)
    :param x_bin_edges: bin edges in the x position
    :param y_bin_edges: bin edges in the y position
    :return: 2d array of position counts, x_bin_edges, and y_bin_edges
    """
    # hist2d converts to a matrix, which reverses x,y
    # inverse order here to preserve visualization.
    pos_sum_2d, _, _ = np.histogram2d(y, x, bins=[y_bin_edges, x_bin_edges], weights=w)

    return pos_sum_2d


def firing_rate_2_rate_map(fr, x, y, x_bin_edges, y_bin_edges, pos_counts_map=None, mask=None,
                           occ_num_thr=3, spatial_window_size=5, spatial_sigma=2, **kwargs):
    fr_sum_2d = w_histogram_2d(x, y, fr, x_bin_edges, y_bin_edges)

    if pos_counts_map is None:
        pos_counts_map = histogram_2d(x, y, x_bin_edges, y_bin_edges)
    if mask is None:
        mask = pos_counts_map >= occ_num_thr

    fr_avg_pos = np.zeros_like(fr_sum_2d)
    fr_avg_pos[mask] = fr_sum_2d[mask] / pos_counts_map[mask]

    sm_fr_map = smooth_2d_map(fr_avg_pos, n_bins=spatial_window_size, sigma=spatial_sigma, **kwargs)
    return sm_fr_map


def spikes_2_rate_map(spikes, x, y, x_bin_edges, y_bin_edges, pos_counts_map=None, mask=None,
                      time_step=0.02, occ_time_thr=0.06, spatial_window_size=5, spatial_sigma=2, **kwargs):
    spk_sum_2d = w_histogram_2d(x, y, spikes, x_bin_edges, y_bin_edges)

    if pos_counts_map is None:
        pos_counts_map = histogram_2d(x, y, x_bin_edges, y_bin_edges)
    pos_sec_map = pos_counts_map * time_step
    if mask is None:
        mask = pos_sec_map >= occ_time_thr

    fr_avg_pos = np.zeros_like(spk_sum_2d)
    fr_avg_pos[mask] = spk_sum_2d[mask] / pos_sec_map[mask]

    sm_fr_map = smooth_2d_map(fr_avg_pos, n_bins=spatial_window_size, sigma=spatial_sigma, **kwargs)
    return sm_fr_map


def compute_velocity(x, y, time_step):
    """
    :param np.array x: vector of x position [cm]
    :param np.array y: vector y position [cm]
    :param float time_step: time_step of each bin
    :return: np.arrays speed and angle. lengths are the same as the inputs.
    """
    dx = np.append(0, np.diff(x))
    dy = np.append(0, np.diff(y))

    dr = np.sqrt(dx ** 2 + dy ** 2)

    sp = dr / time_step  # convert delta distance to speed
    an = get_angle_xy(dx, dy)
    return sp, an


def get_movement_samps(speed, speed_lims=None):
    """
    :param np.array speed: speed for each time bin. expects to be in cm/s
    :param speed_lims: 2 element tuple/list/array with min/max for valid movement speeds
    :return: np.array bool: array of time samples that are within the speed limits
    """
    if speed_lims is None:
        speed_lims = [5, 2000]
    return np.logical_and(speed >= speed_lims[0], speed <= speed_lims[1])


def rotate_xy(x, y, angle):
    """
    :param x: x position
    :param y: y position
    :param angle: rotation angle in radians
    :return: rotated coordinates x,y
    """
    x2 = x * np.cos(angle) + y * np.sin(angle)
    y2 = -x * np.sin(angle) + y * np.cos(angle)
    return x2, y2


def get_angle_xy(x, y):
    """
    computes the angle between x/y using np.math.atan2, for all elements.
    :param x: np.array
    :param y: np.array
    :return: np.array arc tangent considering the sign
    """
    n = len(y)
    angle = np.zeros(n)
    for i in range(n):
        angle[i] = np.math.atan2(y[i], x[i])
    return angle


def sigmoid(x, center, slope):
    """
    Sigmoid function
    :param x: array of values
    :param center: center, value at which sigmoid is 0.5
    :param slope: rate of change of the sigmoid
    :return: array of same length as x
    """
    return 1. / (1 + np.exp(-slope * (x - center)))


def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """
    two dimensional gaussian function
    :param x: 2dim ndarray of x values for each y value [as returned by meshgrid]
    :param y: 2dim ndarray of y values for each x value [as returned by meshgrid]
    :param mx: x position of gaussian center
    :param my: y position of gaussian center
    :param sx: std [spread] in x direcation
    :param sy: std [spread] in y direcation
    :return: gaussian 2d array of same dimensions of x and y
    """
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))


# ------------------------------------------------- SPIKE-SPACE FUNCS --------------------------------------------------
def get_bin_spikes_xy(bin_spikes, x, y):
    """
    :param np.array bin_spikes: spike counts by time bin
    :param np.array x: x bin position of animal
    :param np.array y: y bin position of animal
    :return: np.arrays x_spikes, y_spikes: positions for each spike (length=n_spikes)
    """
    max_n_spikes = np.max(bin_spikes)
    x_spikes = []
    y_spikes = []
    for n_spk in np.arange(1, max_n_spikes + 1):
        x_spikes += x[bin_spikes == n_spk].tolist() * int(n_spk)
        y_spikes += y[bin_spikes == n_spk].tolist() * int(n_spk)
    assert len(x_spikes) == np.sum(bin_spikes), 'Spikes To Position Mismatch'
    return x_spikes, y_spikes


def get_bin_spikes_zone(bin_spikes, zones):
    """
    :param np.array bin_spikes: spike counts by time bin
    :param np.array zones: zone bin position of the animal
    :return: np.array zone_spikes: zone positions for each spike (length=n_spikes)
    """
    max_n_spikes = np.max(bin_spikes)
    zone_spikes = []
    for n_spk in np.arange(1, max_n_spikes + 1):
        zone_spikes += zones[bin_spikes == n_spk].tolist() * int(n_spk)
    return zone_spikes


def get_zone_spike_counts(bin_spikes, zones):
    """
    :param np.array bin_spikes: spike counts by time bin
    :param np.array zones: zone bin position of the animal
    :return: np.array zone_spk_counts: number of spikes per zone. length=#zones
    """
    zone_spikes = get_bin_spikes_zone(bin_spikes, zones)
    zone_spk_counts = np.bincount(zone_spikes)
    return zone_spk_counts


def get_spike_map(bin_spikes, x, y, x_bin_edges, y_bin_edges):
    """
    :param np.array bin_spikes: spike counts by time bin
    :param np.array x: x bin position of animal
    :param np.array y: y bin position of animal
    :param x_bin_edges: np.array of edges
    :param y_bin_edges: np.array of edges
    :return: np.ndarray spike_map: number of spikes at each xy position
    """
    x_spk, y_spk = get_bin_spikes_xy(bin_spikes, x, y)
    spike_map = histogram_2d(x_spk, y_spk, x_bin_edges, y_bin_edges)
    return spike_map


def get_fr_map(spike_map, pos_map_secs):
    """
    :param np.ndarray spike_map: number of spikes at each xy position
            -> as returned by get_spike_map()
    :param np.ndarray pos_map_secs: occupation map in seconds
            -> obtained from get_position_map() and normalized by the time_step
    :return: np.ndarray fr_map: same shape as the inputs. firing rate at each xy position
            -> will probably need smoothing after
    """
    pos_map_secs2 = np.array(pos_map_secs, dtype=np.float32)
    pos_map_secs2[pos_map_secs == 0] = np.nan  # convert zero occupation bins to nan
    fr_map = spike_map / pos_map_secs2
    fr_map[np.isnan(fr_map)] = 0  # convert zero occupation bins to 0
    return fr_map


# ---------------------------------------- SPATIAL-STABILITY METRICS ---------------------------------------------------
def permutation_test_spatial_stability(fr, x, y, x_bin_edges, y_bin_edges, sig_alpha=0.02, n_perm=200, occ_num_thr=3,
                                       spatial_window_size=5, spatial_sigma=2, n_jobs=8):
    n_samps = len(x)
    if fr.ndim == 1:
        n_units = 1
        fr = fr[np.newaxis,]
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between samples and fr.'

    # helper function to get slit correlation
    def get_map_split_corr(_fr):

        data = {'x': x, 'y': y, 'neural_data': _fr}
        data_split = rs.split_timeseries_data(data=data, n_splits=2)

        x1 = data_split['x'][0]
        x2 = data_split['x'][1]

        y1 = data_split['y'][0]
        y2 = data_split['y'][1]

        fr1 = data_split['neural_data'][0]
        fr2 = data_split['neural_data'][1]

        fr_map_corr = np.zeros(n_units)

        for _unit in range(n_units):
            fr_map1 = firing_rate_2_rate_map(fr1[_unit], x1, y1,
                                             x_bin_edges=x_bin_edges,
                                             y_bin_edges=y_bin_edges,
                                             occ_num_thr=occ_num_thr,
                                             spatial_window_size=spatial_window_size,
                                             spatial_sigma=spatial_sigma)
            fr_map2 = firing_rate_2_rate_map(fr2[_unit], x2, y2,
                                             x_bin_edges=x_bin_edges,
                                             y_bin_edges=y_bin_edges,
                                             occ_num_thr=occ_num_thr,
                                             spatial_window_size=spatial_window_size,
                                             spatial_sigma=spatial_sigma)
            fr_map_corr[_unit] = rs.pearson(fr_map1.flatten(), fr_map2.flatten())

        return fr_map_corr

    # compute true split half correlation
    true_split_corr = get_map_split_corr(fr)

    # helper function to permute the firing rates
    def p_worker():
        """ helper function for parallelization. Computes a single shuffled border score per unit."""

        perm_fr = np.zeros_like(fr)
        for _unit in range(n_units):
            perm_fr[_unit] = np.roll(fr[_unit], np.random.randint(n_samps))

        split_corr = get_map_split_corr(perm_fr)
        return split_corr

    with Parallel(n_jobs=n_jobs) as parallel:
        perm_split_corr = parallel(delayed(p_worker)() for _ in range(n_perm))
    perm_split_corr = np.array(perm_split_corr)

    sig = np.zeros(n_units, dtype=bool)
    for unit in range(n_units):
        # find location of true corr
        loc = np.array(perm_split_corr[:, unit] >= true_split_corr[unit]).mean()
        # determine if outside distribution @ alpha level
        sig[unit] = np.logical_or(loc <= sig_alpha / 2, loc >= 1 - sig_alpha / 2)

    return true_split_corr, sig


def get_position_encoding_model(x, y, neural_data, x_bin_edges, y_bin_edges, data_type='fr', n_xval=5, **kwargs):
    """
    Discretizes x y positions into binned features to predict firing rate or spikes.
    :param x: array of x positions [n_samps length]
    :param y: array of y positions [n_smaps length]
    :param x_bin_edges: edges of x array
    :param y_bin_edges: edges of y array
    :param neural_data: array floats firing rate n_units x n_samps, also works for one unit
    :param data_type: string ['spikes', 'neural_data'], indicating if the data is firing rate or spike rate.
    :param n_xval: number of x validation folds
    :param feat_type: string. options are ['pca', 'nmf', 'full', 'sparse']
        pca -> pca features
        nmf -> non negative factorization features
        full -> un compressed features, still applies gaussian smoothing around position of the animal
        sparse- > uncompressed feautures, does NOT apply gaussian smoothing around position, results in sparse one hot
                feature for each sample

    ---------kwargs----
     kwargs arguments, need to be input as key=val
     ---- feature params----
    :param feature_design_matrix: n_bins x n_bins array. maps from bin idx to feature array
    :param spatial_window_size: int, spatial extent of smoothing for features [default = 5]
    :param spatial_sigma: float, spatial std. for gaussian smoothing [default = 2]
    :param n_components: number of components to use, if feat_type is pca can be a float [0,1) for var. exp.
            default for pca = 0.95, default for nmf = 100.
    :param pca: object instance of PCA. previously fit PCA instance (saves time); ignored if feat_type != pca.
    :param nmf: object instance of NMF. previously fit NMF instance (saves time); ignored if feat_type != nmf
    --- fitting params ---
    :param regression_penalty: float. alpha parameter for linear models penalty. default = 0.15
    :param bias_term: bool. adds a column of 1s to the features [default = 1]
    :param l1_ratio: float. l1/l2 ratio for elastic net. default 0.15

    :returns:
        model_coef: array n_xval x n_units x n_position_featurs
        train_perf: array n_xval x n_units x 3 [r2, err, map_corr]
        test_perf: array n_xval x n_units x 3
    """

    n_samps = len(x)

    if neural_data.ndim == 1:
        n_units = 1
        neural_data = neural_data[np.newaxis,]
    else:
        n_units, _ = neural_data.shape
    assert n_samps == neural_data.shape[1], 'Mismatch lengths between speed and neural_data.'

    # split data into folds
    xval_samp_ids = rs.split_timeseries(n_samps=n_samps, samps_per_split=1000, n_data_splits=n_xval)

    # get feature parameters
    feature_params = {
        'spatial_window_size': kwargs['spatial_window_size'] if 'spatial_window_size' in kwargs.keys() else 5,
        'spatial_sigma': kwargs['spatial_sigma'] if 'spatial_sigma' in kwargs.keys() else 2,
        'feat_type': kwargs['feat_type'] if 'feat_type' in kwargs.keys() else 'pca'}

    if 'n_components' in kwargs.keys():
        feature_params['n_components'] = kwargs['n_components']
    else:
        if feature_params['feat_type'] == 'pca':
            feature_params['n_components'] = 0.95
        elif feature_params['feat_type'] == 'nma':
            feature_params['n_components'] = 100

    features, inverse = get_position_encoding_features(x, y, x_bin_edges, y_bin_edges, **feature_params)

    n_features = features.shape[1]
    n_pos_bins = (len(x_bin_edges) - 1) * (len(y_bin_edges) - 1)

    # get regression params
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha'] if (kwargs['alpha'] is not None) else 0.15
    else:
        alpha = 0.15

    if 'bias_term' in kwargs.keys():
        bias_term = kwargs['bias_term'] if (kwargs['bias_term'] is not None) else True
    else:
        bias_term = True

    # obtain relevant functions for data type
    map_params = {'x_bin_edges': x_bin_edges, 'y_bin_edges': y_bin_edges,
                  'spatial_window_size': feature_params['spatial_window_size'],
                  'spatial_sigma': feature_params['spatial_sigma']}

    spatial_map_function = get_spatial_map_function(data_type, **map_params)
    if data_type == 'spikes':
        model_function = lm.PoissonRegressor(alpha=0.1, fit_intercept=bias_term, max_iter=50)
        reg_type = 'poisson'
    elif data_type == 'fr':
        l1_ratio = kwargs['l1_ratio'] if ('l1_ratio' in kwargs.keys()) else 0.15
        model_function = lm.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=bias_term)
        reg_type = 'linear'
    else:
        raise NotImplementedError

    # pre-allocate performance metrics
    perf_metrics = ['r2', 'ar2', 'err', 'n_err', 'map_r']
    train_perf = {}
    test_perf = {}
    for mm in perf_metrics:
        train_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
        test_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
    model_coef = np.zeros((n_xval, n_units, n_pos_bins))

    # iterate over x validation folds
    for fold in range(n_xval):
        # test set
        x_test = x[xval_samp_ids == fold]
        y_test = y[xval_samp_ids == fold]
        features_test = features[xval_samp_ids == fold, :]

        # train set
        x_train = x[xval_samp_ids != fold]
        y_train = y[xval_samp_ids != fold]
        features_train = features[xval_samp_ids != fold, :]

        for unit in range(n_units):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    # get responses
                    response_test = neural_data[unit, xval_samp_ids == fold]
                    response_train = neural_data[unit, xval_samp_ids != fold]

                    # train model
                    model = model_function.fit(features_train, response_train)
                    if kwargs['feat_type'] in ['pca', 'nmf', 'kpca']:
                        model_coef[fold, unit] = inverse(model.coef_)
                    else:
                        model_coef[fold, unit] = model.coef_

                    # get predicted responses
                    response_train_hat = model.predict(features_train)
                    response_test_hat = model.predict(features_test)

                    # get true spatial for this fold maps
                    train_map = spatial_map_function(response_train, x_train, y_train)
                    test_map = spatial_map_function(response_test, x_test, y_test)

                    # get predicted maps
                    train_map_hat = spatial_map_function(response_train_hat, x_train, y_train)
                    test_map_hat = spatial_map_function(response_test_hat, x_test, y_test)

                    # train performance
                    temp1 = rs.get_regression_metrics(response_train, response_train_hat, reg_type=reg_type,
                                                      n_params=n_features)

                    train_perf['map_r'][fold, unit] = rs.pearson(train_map.flatten(), train_map_hat.flatten())

                    # test performance
                    temp2 = rs.get_regression_metrics(response_test, response_test_hat, reg_type=reg_type,
                                                      n_params=n_features)
                    test_perf['map_r'][fold, unit] = rs.pearson(test_map.flatten(), test_map_hat.flatten())

                    for metric in ['r2', 'ar2', 'err', 'n_err']:
                        train_perf[metric][fold, unit] = temp1[metric]
                        test_perf[metric][fold, unit] = temp2[metric]

            finally:
                pass

    return model_coef, train_perf, test_perf


def get_position_encoding_features(x, y, x_bin_edges, y_bin_edges, feat_type='pca', n_components=0.95, **params):
    """
    for each sample, creates a 1d feature array that is smoothed around the position of the animal
    :param x: array of x positions [n_samps length]
    :param y: array of y positions [n_smaps length]
    :param x_bin_edges: edges of x array
    :param y_bin_edges: edges of y array
    :param feat_type: string. options are ['pca', 'nmf', 'full', 'sparse']
        pca -> pca features
        nmf -> non negative factorization features
        full -> un compressed features, still applies gaussian smoothing around position of the animal
        sparse- > uncompressed feautures, does NOT apply gaussian smoothing around position, results in sparse one hot
                feature for each sample
    ----
    kwargs arguments, need to be input as key=val
    :param spatial_window_size: int, spatial extent of smoothing for features [default = 5]
    :param spatial_sigma: float, spatial std. for gaussian smoothing [default = 2]
    :param n_components: number of components to use, if feat_type is pca can be a float [0,1) for var. exp.
            default for pca = 0.95, default for nmf = 100.
    :return: array [n_samps x n_feautures].
        n features is the product of x and y bins
    """

    n_samps = len(x)

    # get enviroment dimension
    n_x_bins = len(x_bin_edges) - 1
    n_y_bins = len(y_bin_edges) - 1
    n_spatial_bins = n_x_bins * n_y_bins

    # get x y bin idx
    x_bin = np.digitize(x, x_bin_edges) - 1
    y_bin = np.digitize(y, y_bin_edges) - 1

    # for each sample get the linear bin idx of the xy bins
    yx_bin = np.ravel_multi_index(np.array((y_bin, x_bin)), (n_y_bins, n_x_bins))

    # get or load feature_matrix for given environment size
    feat_mat_fn = Path(f"pos_feat_design_mat_nx{n_x_bins}_ny{n_y_bins}.npy")
    if feat_mat_fn.exists():
        feature_design_matrix = np.load(str(feat_mat_fn))
    else:
        # generate & save locally feature mat
        feature_design_matrix = generate_position_design_matrix(n_x_bins, n_y_bins, **params)
        np.save(str(feat_mat_fn), feature_design_matrix)

    if feat_type in ['pca', 'nmf']:
        # get feature transformation object, if it exists for the environment
        feat_obj_fn = Path(f"pos_feat_{feat_type}_nx{n_x_bins}_ny{n_y_bins}.pickle")
        if feat_obj_fn.exists():
            with open(feat_obj_fn, "rb") as f:
                feat_obj = pickle.load(f)
        else:
            feat_obj = None

        if feat_type == 'pca':
            transform_func, inverse_func, feat_obj = _pca_position_features(feature_design_matrix, n_components,
                                                                            feat_obj)
            features = transform_func(yx_bin)
        else:
            transform_func, inverse_func, feat_obj = _nmf_position_features(feature_design_matrix, n_components,
                                                                            feat_obj)
            features = transform_func(yx_bin)

        with open(feat_obj_fn, "wb") as f:
            pickle.dump(feat_obj, f, pickle.HIGHEST_PROTOCOL)

        return features, inverse_func

    elif feat_type == 'full':
        features = feature_design_matrix[yx_bin]
        return features, None

    elif feat_type == 'sparse':
        features = np.zeros((n_samps, n_spatial_bins))
        features[:, yx_bin] = 1
        return features, None

    elif feat_type == 'splines':
        raise NotImplementedError
    else:
        raise NotImplementedError


def generate_position_design_matrix(n_x_bins, n_y_bins, spatial_window_size=5, spatial_sigma=2):
    """
    for a given geomtry generates an n_bins x n_bins F matrix, in which F[kk] is the kth row corresponding to a
    jj, ii position and applying a gaussian around that jj, ii position.
    :param n_x_bins: edges of x array
    :param n_y_bins: edges of y array
    :param spatial_window_size: int, spatial extent of smoothing for features
    :param spatial_sigma: float, spatial std. for gaussian smoothing
    :return: array [n_features x n_feautures].
        n features is the product of x and y bins
    """

    n_spatial_bins = n_x_bins * n_y_bins

    # get smoothing gaussian kernel. this is applied to each spatial position
    gaussian_coords = np.array((np.arange(-spatial_window_size, spatial_window_size + 1),
                                np.arange(-spatial_window_size, spatial_window_size + 1)))
    xx, yy = np.meshgrid(*gaussian_coords)
    gaussian_vals = gaussian_2d(x=xx, y=yy, sx=spatial_sigma, sy=spatial_sigma)
    gaussian_vals /= gaussian_vals.max()
    gaussian_vals = gaussian_vals.flatten()

    feature_matrix = np.zeros((n_spatial_bins, n_spatial_bins))
    for jj in range(n_y_bins):
        for ii in range(n_x_bins):
            # find where position is in the 1d feature dimension
            linear_jjii = np.ravel_multi_index(np.array((jj, ii)), (n_y_bins, n_x_bins))

            # positions around jj, ii
            jjii_coords = np.array((np.arange(jj - spatial_window_size, jj + spatial_window_size + 1),
                                    np.arange(ii - spatial_window_size, ii + spatial_window_size + 1)))

            jjii_mesh = np.meshgrid(*jjii_coords)
            # get valid coords.
            valid_coords = ((jjii_mesh[0] >= 0) & (jjii_mesh[0] < n_y_bins)) & (
                    (jjii_mesh[1] >= 0) & (jjii_mesh[1] < n_x_bins))
            valid_coords = valid_coords.flatten()

            # convert those position to 1d feature dimension
            feature_idx = np.ravel_multi_index(jjii_mesh, (n_y_bins, n_x_bins), mode='clip').flatten()

            feature_matrix[linear_jjii, feature_idx[valid_coords]] = gaussian_vals[valid_coords]

    return feature_matrix


def _pca_position_features(feature_matrix, n_components=0.95, pca=None):
    """
    Utility function to be used in generating position encoding features. It provides three outputs that are tied to the
    feature design matrix input. See below for details.
    :param feature_matrix: output from generate_position_design_matrix
    :param n_components: argument for sklearn.decomposition.PCA function
        if float in [0,1), it interepreted as to get the # of components such that there's at least that
        % varianced explained. if an int, uses that many components.
    :param pca: previous instantiation of this function, simply uses that instance to make the transformation functions.
    :return:
        transform: function, maps: feature index -> pca feature components
        inverse_transfrom: function, maps: feature components (in pca space) -> original feature space
        pca -> isntance of pca.
    """
    if pca is None:
        pca = PCA(n_components=n_components)
        pca.fit(feature_matrix)

    def transform(feature_linear_idx):
        # linear feature idx to component_space features
        return pca.transform(feature_matrix[feature_linear_idx])

    def inverse_transform(component_features):
        # component_space features to original space
        return pca.inverse_transform(component_features)

    return transform, inverse_transform, pca


def _nmf_position_features(feature_matrix, n_components=100, nmf=None):
    """
    :param feature_matrix:
    :param n_components:
    :param nmf:
    :return:
        transform: function, maps: feature index -> nmf feature components
        inverse_transfrom: function, maps: feature components (in nmf space) -> original feature space
        nmf -> isntance of nmf.
    """
    if nmf is None:
        nmf = NMF(n_components=n_components, alpha=0.01, init='nndsvda', max_iter=500)
        nmf.fit(feature_matrix)

    def transform(feature_linear_idx):
        # linear feature idx to component_space features
        return nmf.transform(feature_matrix[feature_linear_idx])

    def inverse_transform(component_features):
        # component_space features to original space
        return nmf.inverse_transform(component_features)

    return transform, inverse_transform, nmf


# ------------------------------------------------- SPEED METRICS ------------------------------------------------------
def speed_score_traditional(speed, fr, min_speed=2, max_speed=80, sig_alpha=0.02, n_perm=100, n_jobs=-1):
    """
    Traditional method of computing speed score. simple correlation of speed & firing rate
    :param speed: array floats vector of speed n_samps
    :param fr: array floats firing rate of the neuron
    :param max_speed: float
    :param min_speed: float
    :param sig_alpha: float, significant level to evaluate the permutation test
    :param n_perm: int, number of permutations to perform.
    :param n_jobs: int, number of cpus to use
    :returns: score: speed score per unit
              sig: if the speed score reached significance after permutation test
    """

    n_samps = len(speed)
    if fr.ndim == 1:
        n_units = 1
        fr = fr.reshape(1, -1)
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and neural_data.'

    # get valid samples and assign new variables for fitting
    valid_samps = np.logical_and(speed >= min_speed, speed <= max_speed)
    speed_valid = speed[valid_samps]
    fr_valid = fr[:, valid_samps]

    # traditional correlation method
    score = np.zeros(n_units)
    sig = np.zeros(n_units, dtype=bool)
    for unit in range(n_units):
        score[unit] = rs.spearman(speed_valid, fr_valid[unit])
        sig[unit], _ = rs.permutation_test(function=rs.pearson, x=speed_valid, y=fr_valid[unit],
                                           n_perm=n_perm, alpha=sig_alpha, n_jobs=n_jobs)

    return score, sig


def get_speed_encoding_model_old(speed, fr, speed_bin_edges, compute_sp_score=True, sig_alpha=0.02, n_perm=100):
    """
    Discretizes the speed into the speed_bin_edges to predict firing rate. Essentially an OLS, implemented by taking the
    mean per speed bin.
    :param speed: array floats vector of speed n_samps
    :param fr: array floats firing rate n_units x n_samps, also works for one unit
    :param speed_bin_edges: edges to bin the speed
    :param compute_sp_score: bool, if true, correlates the model coefficients to the speed_bins and gets significance
    :param sig_alpha: float, significant level to evaluate the permutation test of the model speed score
    :param n_perm: int, number of permutations to perform for the model speed score
    :returns: scores: pd.Dataframe with columns ['score', 'sig', 'aR2', 'rmse', 'nrmse'], rows are n_units
              model_coef: array n_units x n_bins mean firing rate at each bin
              model_coef_sem: array n_units x n_bins sem for each bin
              valid_samps: samples that were used in the estimation (fell withing the speed_bin_edges)

    Note on implementation:
    There are several ways of doing this that are equivalent:

    [1] ols: [using stats.linear_model]
    model = lm.LinearRegression(fit_intercept=False).fit(design_matrix, fr_valid.T)
    model_coef = model.coef_

    [2] mean neural_data per speed bin: (could use trim mean, median, or other robust methods here);
    - equivalency with ols is only true for mean
    implemented below. this method allows to get standard errors per bin easily and fast

    [3] weighted histogram:
    neural_data weighted speed histogram then normalization by speed bin occupancy.
    -needs to be performed by unit
    sp_occ,_ = np.histogram(speed, sp_bins)
    model_coef[unit],_ = np.histogram(speed_valid, sp_bins, weights=fr_valid[unit])
    model_coef[unit] /= sp_occ

    [4] ols: (as above) using statsmodels.api
    this is probably the most powerful, but slowest.
    - needs to be perfomed by unit
    model = sm.OLS(fr_valid[unit],design_matrix)
    results = model.fit()
    model_coef[unit] = results.params

    """

    n_samps = len(speed)
    n_sp_bins = len(speed_bin_edges) - 1

    if fr.ndim == 1:
        n_units = 1
        fr = fr.reshape(1, -1)
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and neural_data.'

    # discretize speed / get features
    sp_design_matrix, sp_bin_idx, valid_samps = get_speed_encoding_features(speed, speed_bin_edges)
    fr_valid = fr[:, valid_samps]

    # compute model coefficients
    model_coef = np.zeros((n_units, n_sp_bins))
    model_coef_s = np.zeros((n_units, n_sp_bins))
    for i in range(n_sp_bins):
        fr_sp_bin_i = fr_valid[:, sp_bin_idx == i]
        model_coef[:, i] = np.mean(fr_sp_bin_i, axis=1)
        model_coef_s[:, i] = stats.sem(fr_sp_bin_i, axis=1)

    # use model coefficients and correlate to speed
    score = np.zeros(n_units)
    score_sig = np.zeros(n_units, dtype=bool)
    if compute_sp_score:
        for unit in range(n_units):
            score[unit] = rs.spearman(model_coef[unit], speed_bin_edges[:-1])
            score_sig[unit], _ = rs.permutation_test(function=rs.spearman, x=speed_bin_edges[:-1], y=model_coef[unit],
                                                     n_perm=n_perm, alpha=sig_alpha)

    # get prediction
    # -> basically assigns to each sample its corresponding mean value
    fr_hat = model_coef @ sp_design_matrix.T

    # get scores arrange into a data frame
    scores = pd.DataFrame(index=range(n_units), columns=['score', 'sig', 'r2', 'rmse', 'nrmse'])
    scores['score'] = score
    scores['sig'] = score_sig
    scores['r2'] = rs.get_ar2(fr_valid, fr_hat, n_sp_bins)
    scores['rmse'] = rs.get_rmse(fr_valid, fr_hat)
    scores['nrmse'] = rs.get_nrmse(fr_valid, fr_hat)

    return scores, model_coef, model_coef_s, valid_samps


def get_speed_encoding_model(speed, neural_data, speed_bin_edges, data_type='spikes', n_xval=5):
    """
    Discretizes the speed into the speed_bin_edges to predict firing rate.
    :param x: array of floats vector of position of x the animal, n_samps
    :param y: array of floats vector of position of y the animal, n_samps
    :param speed: array floats vector of speed n_samps
    :param neural_data: array floats firing rate n_units x n_samps, also works for one unit
    :param speed_bin_edges: edges to bin the speed
    :param data_type: string ['spikes', 'neural_data'], indicating if the data is firing rate or spike rate.
    :param n_xval: number of x validation folds
    :returns:
        model_coef: array n_xval x n_units x n_bins of model coefficients.
        train_perf: dict of metrics ['r2', 'err', 'map_r'], each an array of array n_xval x n_units
        test_perf**: ['r2', 'err', 'map_r'], each an array of array n_xval x n_units
        ** NOTE that map_r for train and test are the same as it is the correlation between speed bins and
        training model coefficients
    """

    n_samps = len(speed)

    if neural_data.ndim == 1:
        n_units = 1
        neural_data = neural_data[np.newaxis,]
    else:
        n_units, _ = neural_data.shape
    assert n_samps == neural_data.shape[1], 'Mismatch lengths between speed and neural_data.'

    # discretize speed / get features
    features, sp_bin_idx, valid_samps = get_speed_encoding_features(speed, speed_bin_edges)
    neural_data = neural_data[:, valid_samps]
    n_valid_samps = int(valid_samps.sum())

    n_features = features.shape[1]

    # split data into folds
    xval_samp_ids = rs.split_timeseries(n_samps=n_valid_samps, samps_per_split=1000, n_data_splits=n_xval)

    # pre-allocate performance metrics
    perf_metrics = ['r2', 'ar2', 'err', 'n_err', 'map_r']
    train_perf = {}
    test_perf = {}
    for mm in perf_metrics:
        train_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
        test_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
    model_coef = np.zeros((n_xval, n_units, n_features)) * np.nan

    # obtain relevant functions for data type
    if data_type == 'spikes':
        model_function = lm.PoissonRegressor(alpha=0, fit_intercept=False)
        reg_type = 'poisson'
    elif data_type == 'fr':
        model_function = lm.LinearRegression(fit_intercept=False)
        reg_type = 'linear'
    else:
        raise NotImplementedError

    for fold in range(n_xval):
        # test set
        features_test = features[xval_samp_ids == fold, :]
        # train set
        features_train = features[xval_samp_ids != fold, :]

        for unit in range(n_units):
            try:
                # get responses
                response_test = neural_data[unit, xval_samp_ids == fold]
                response_train = neural_data[unit, xval_samp_ids != fold]

                # train model
                model = model_function.fit(features_train, response_train)
                model_coef[fold, unit] = model.coef_

                # get predicted responses
                response_train_hat = model.predict(features_train)
                response_test_hat = model.predict(features_test)

                # train performance
                temp1 = rs.get_regression_metrics(response_train, response_train_hat, reg_type=reg_type,
                                                  n_params=n_features)
                train_perf['map_r'][fold, unit] = rs.pearson(speed_bin_edges[1:], model.coef_)

                # test performance
                temp2 = rs.get_regression_metrics(response_test, response_test_hat, reg_type=reg_type,
                                                  n_params=n_features)
                test_perf['map_r'][fold, unit] = rs.pearson(speed_bin_edges[1:], model.coef_)

                for metric in ['r2', 'ar2', 'err', 'n_err']:
                    train_perf[metric][fold, unit] = temp1[metric]
                    test_perf[metric][fold, unit] = temp2[metric]

            finally:
                pass

    return model_coef, train_perf, test_perf


def get_speed_encoding_features(speed, speed_bin_edges):
    """
    Obtains the features for speed encoding model. A wrapper for the robust stats get_discrete_data_mat function,
    that thersholds the speed to the limiits of the bins as a pre-step. these valid samples are also return
    :param speed: array n_samps , floats vector of speed n_samps
    :param speed_bin_edges: array n_bins, edges to bin the speed
    :return: sp_design_matrix [n_valid_samps x n_bins], binary mat indicating what bin each sample is in
            sp_bin_idx: array ints [n_valid_samps], as above, but indicating the bins by an integer in order
            valid_samps: array bool [n_samps], sum(valid_samps)=n_valid_samps
    """
    min_speed = speed_bin_edges[0]
    max_speed = speed_bin_edges[-1]

    # get valid samples and assign new variables for fitting
    valid_samps = np.logical_and(speed >= min_speed, speed <= max_speed)
    speed_valid = speed[valid_samps]

    sp_design_matrix, sp_bin_idx = rs.get_discrete_data_mat(speed_valid, speed_bin_edges)

    return sp_design_matrix, sp_bin_idx, valid_samps


# ------------------------------------------------- ANGLE METRICS ------------------------------------------------------
def get_angle_stats(theta, step, weights=None):
    """
    Computes several circular statistics based on the histogram of the data.
    expects radians. Then uses the Rayleigh test for
    :param theta: original theta vector [radians]
    :param weights: weights for the each angle observation (e.g. spikes/ neural_data)
    :param step: angular bin size [radians]
    :return: dictionary with descriptive stats:
            {
                vec_len -> resulting vector length
                mean_ang -> resulting mean angle
                rayleigh -> Rayleigh's R [statistic]
                p_val -> two sided statistical test
                var_ang -> variance of the estimates
                std_ang -> standard deviation
            }
            w_counts: weighted counts
            bin_centers: bin centers in radians
            bin_edges: bin edges in radians
    """

    counts, bin_edges = np.histogram(theta, np.arange(0, 2 * np.pi + step, step))
    bin_centers = bin_edges[:-1] + step / 2

    if weights is None:
        w_counts = counts
    else:
        w_counts, _ = np.histogram(theta, bin_edges, weights=weights)
        w_counts /= counts

    # add the weighted vectors to obtain the complex mean vector, its components, and descriptive stats
    vec_len, mean_ang, var_ang, std_ang, = rs.resultant_vector_length(bin_centers, w=w_counts, d=step)

    # rayleigh statistical test
    p_val, rayleigh = rs.rayleigh(bin_centers, w=w_counts, d=step)

    out_dir = {'vec_len': vec_len, 'mean_ang': mean_ang, 'rayleigh': rayleigh, 'p_val': p_val, 'var': var_ang,
               'std': std_ang}
    return out_dir, w_counts, bin_centers, bin_edges


def angle_score_traditional(theta, fr, speed=None, min_speed=None, max_speed=None, sig_alpha=0.02, n_perm=200,
                            n_jobs=8):
    """"
    computes angle by firing rate without binning
    :param theta: array n_samps of angles in radians
    :param fr: array n_units x n_samps of firing rates
    :param speed: array of n_samps of speed to threshold the computations
    :param min_speed: minimum speed threshold
    :param max_speed: max speed threshold
    :param n_jobs: int number of cpus to use for permutation
    :param n_perm: int number of permutations
    :param sig_alpha: parametric alpha for significance of Rayleigh test.
    :return:  scores: pd.Dataframe n_units x columns
                ['vec_len', 'mean_ang', 'p_val', 'sig', 'rayleigh', 'var_ang', 'std_ang']
    """

    n_samps = len(theta)
    if fr.ndim == 1:
        n_units = 1
        fr = fr.reshape(1, -1)
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and neural_data.'

    if (speed is not None) and (min_speed is not None) and (max_speed is not None):
        valid_samps = np.logical_and(speed >= min_speed, speed <= max_speed)
        theta = theta[valid_samps]
        fr = fr[:, valid_samps]

    scores = pd.DataFrame(index=range(n_units),
                          columns=['vec_len', 'mean_ang', 'p_val', 'sig', 'rayleigh', 'var_ang', 'std_ang'])

    def p_worker(_unit):
        p_fr = np.random.permutation(fr[unit])
        p_vec_len, _, _, _ = rs.resultant_vector_length(alpha=theta, w=p_fr)
        return p_vec_len

    with Parallel(n_jobs=n_jobs) as parallel:
        for unit in range(n_units):
            vec_len, mean_ang, var_ang, std_ang, = rs.resultant_vector_length(alpha=theta, w=fr[unit])
            p_val, rayleigh = rs.rayleigh(alpha=theta, w=fr[unit])

            # permutation
            perm_vec_len = parallel(delayed(p_worker)(unit) for _ in range(n_perm))
            loc = np.array(perm_vec_len >= vec_len).mean()
            # determine if outside distribution @ alpha level
            sig = np.logical_or(loc <= sig_alpha / 2, loc >= 1 - sig_alpha / 2)

            out_dir = {'vec_len': vec_len, 'mean_ang': np.mod(mean_ang, 2 * np.pi), 'rayleigh': rayleigh,
                       'rayleigh_p_val': p_val, 'sig': sig, 'var_ang': var_ang, 'std_ang': std_ang}

            for key, val in out_dir.items():
                scores.loc[unit, key] = val

    return scores


def get_angle_encoding_model_old(theta, fr, ang_bin_edges, speed=None, min_speed=None, max_speed=None, sig_alpha=0.02):
    """
    :param theta: array n_samps of angles in radians
    :param fr: array n_units x n_samps of firing rates
    :param ang_bin_edges: bin edges in radians
    :param speed: array of n_samps of speed to threshold the computations
    :param min_speed: minimum speed threshold
    :param max_speed: max speed threshold
    :param sig_alpha: parametric alpha for significance of Rayleigh test.
    :return:  scores: pd.Dataframe n_units x columns ['vec_len', 'mean_ang', 'sig', 'r2', 'rmse', 'nrmse']
              model_coef: array n_units x n_bins mean firing rate at each bin
              model_coef_sem: array n_units x n_bins sem for each bin.
              angle_bins: array of centered bins in radians
    """

    n_samps = len(speed)
    if fr.ndim == 1:
        n_units = 1
        fr = fr.reshape(1, -1)
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and neural_data.'

    # binning of the angle
    ang_bin_spacing = ang_bin_edges[1] - ang_bin_edges[0]
    ang_bin_centers = ang_bin_edges[:-1] + ang_bin_spacing / 2
    n_ang_bins = len(ang_bin_centers)

    # get discrete design matrix and valid samples
    ang_design_matrix, ang_bin_idx, valid_samps = \
        get_angle_encoding_features(theta, ang_bin_edges, speed=speed, min_speed=min_speed, max_speed=max_speed)
    fr = fr[:, valid_samps]

    # get model coefficients (mean neural_data per bin) and se of the mean
    model_coef = np.zeros((n_units, n_ang_bins))
    model_coef_s = np.zeros((n_units, n_ang_bins))
    for i in range(n_ang_bins):
        fr_ang_bin_i = fr[:, ang_bin_idx == i]
        model_coef[:, i] = np.mean(fr_ang_bin_i, axis=1)
        model_coef_s[:, i] = stats.sem(fr_ang_bin_i, axis=1)

    # get prediction
    # -> basically assigns to each sample its corresponding mean value
    fr_hat = model_coef @ ang_design_matrix.T

    # pre-allocate score outputs
    scores = pd.DataFrame(index=range(n_units),
                          columns=['score', 'mean_ang', 'p_val', 'sig', 'r2', 'rmse', 'nrmse'])

    # loop to get circular stats scores
    for unit in range(n_units):
        # get vector length and mean angle
        vec_len, mean_ang, _, _, = rs.resultant_vector_length(ang_bin_centers, w=model_coef[unit], d=ang_bin_spacing)
        # rayleigh statistical test
        p_val, _ = rs.rayleigh(ang_bin_centers, w=model_coef[unit], d=ang_bin_spacing)

        # store results
        scores.at[unit, 'score'] = vec_len
        scores.at[unit, 'mean_ang'] = np.mod(mean_ang, 2 * np.pi)
        scores.at[unit, 'sig'] = p_val < sig_alpha

    scores['r2'] = rs.get_ar2(fr, fr_hat, n_ang_bins)
    scores['rmse'] = rs.get_rmse(fr, fr_hat)
    scores['nrmse'] = scores['rmse'] / fr.mean(axis=1)

    return scores, model_coef, model_coef_s


def get_angle_encoding_model(theta, neural_data, ang_bin_edges, speed=None, min_speed=None, max_speed=None,
                             data_type='spikes', n_xval=5):
    """
    :param theta: array n_samps of angles in radians
    :param neural_data: array n_units x n_samps of firing rates
    :param ang_bin_edges: bin edges in radians
    :param speed: array of n_samps of speed to threshold the computations
    :param min_speed: minimum speed threshold
    :param max_speed: max speed threshold
    :param data_type: string ['spikes', 'neural_data'], indicating if the data is firing rate or spike rate.
    :param n_xval: int number of xvalidation folds
    :returns:
        model_coef: array n_xval x n_units x n_bins of model coefficients.
        train_perf: dict of metrics ['r2', 'err', 'map_r'], each an array of array n_xval x n_units
        test_perf**: ['r2', 'err', 'map_r'], each an array of array n_xval x n_units
        ** NOTE that map_r for train and test are the same as it is the correlation between speed bins and
        training model coefficients
    """

    n_samps = len(speed)
    if neural_data.ndim == 1:
        n_units = 1
        neural_data = neural_data.reshape(1, -1)
    else:
        n_units, _ = neural_data.shape
    assert n_samps == neural_data.shape[1], 'Mismatch lengths between speed and neural_data.'

    # get discrete design matrix and valid samples
    features, ang_bin_idx, valid_samps = \
        get_angle_encoding_features(theta, ang_bin_edges, speed=speed, min_speed=min_speed, max_speed=max_speed)
    neural_data = neural_data[:, valid_samps]
    n_valid_samps = int(valid_samps.sum())
    n_features = len(ang_bin_edges) - 1

    # split data into folds
    xval_samp_ids = rs.split_timeseries(n_samps=n_valid_samps, samps_per_split=1000, n_data_splits=n_xval)

    # pre-allocate performance metrics
    perf_metrics = ['r2', 'ar2', 'err', 'n_err', 'map_r']
    train_perf = {}
    test_perf = {}
    for mm in perf_metrics:
        train_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
        test_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
    model_coef = np.zeros((n_xval, n_units, n_features)) * np.nan

    # obtain relevant functions for data type
    if data_type == 'spikes':
        model_function = lm.PoissonRegressor(alpha=0, fit_intercept=False)
        reg_type = 'poisson'
    elif data_type == 'fr':
        model_function = lm.LinearRegression(fit_intercept=False)
        reg_type = 'linear'
    else:
        raise NotImplementedError

    for fold in range(n_xval):
        # test set
        features_test = features[xval_samp_ids == fold, :]
        # train set
        features_train = features[xval_samp_ids != fold, :]

        for unit in range(n_units):
            try:
                # get responses
                response_test = neural_data[unit, xval_samp_ids == fold]
                response_train = neural_data[unit, xval_samp_ids != fold]

                # train model
                model = model_function.fit(features_train, response_train)
                model_coef[fold, unit] = model.coef_

                # get predicted responses
                response_train_hat = model.predict(features_train)
                response_test_hat = model.predict(features_test)

                # train performance
                temp1 = rs.get_regression_metrics(response_train, response_train_hat, reg_type=reg_type,
                                                  n_params=n_features)
                train_perf['map_r'][fold, unit] = rs.circ_corrcl(ang_bin_edges[1:], model.coef_.flatten())

                # test performance
                temp2 = rs.get_regression_metrics(response_test, response_test_hat, reg_type=reg_type,
                                                  n_params=n_features)
                test_perf['map_r'][fold, unit] = rs.circ_corrcl(ang_bin_edges[1:], model.coef_.flatten())

                for metric in ['r2', 'ar2', 'err', 'n_err']:
                    train_perf[metric][fold, unit] = temp1[metric]
                    test_perf[metric][fold, unit] = temp2[metric]

            finally:
                pass

    return model_coef, train_perf, test_perf


def get_angle_encoding_features(theta, ang_bin_edges, speed=None, min_speed=None, max_speed=None, valid_samps=None):
    if valid_samps is None:
        # get valid samples and overwrite for fitting
        if (speed is not None) and (min_speed is not None) and (max_speed is not None):
            valid_samps = np.logical_and(speed >= min_speed, speed <= max_speed)
        else:
            valid_samps = np.ones(len(theta), dtype=bool)

    theta = theta[valid_samps]
    # binning of the angle / get discrete design matrix
    ang_design_matrix, ang_bin_idx = rs.get_discrete_data_mat(theta, ang_bin_edges)

    return ang_design_matrix, ang_bin_idx, valid_samps


# ------------------------------------------------- BORDER METRICS -----------------------------------------------------
def get_border_encoding_model(x, y, neural_data, x_bin_edges, y_bin_edges, feat_type='sigmoid', data_type='spikes',
                              bias_term=True, spatial_window_size=3, spatial_sigma=2, n_xval=5):
    """
    Obtains the solstad border score and creates an encoding model based on proximity to the borders.
    :param x: array n_samps of x positions of the animal
    :param y: array n_samps of y positions of the animal
    :param neural_data: ndarray n_units x n_samps of firing rate,
    :param x_bin_edges: x bin edges
    :param y_bin_edges: y bin edges
    :param feat_type: str ['linear', 'sigmoid']. linear or sigmoid proximity features for encoding model
    :param data_type: string ['spikes', 'neural_data'], indicating if the data is firing rate or spike rate.
    :param bias_term: bool. if True, includes a bias term in the encoding features (recommended).
    :param spatial_window_size: int, spatial extent of smoothing for features
    :param spatial_sigma: float, spatial std. for gaussian smoothing
    :param n_xval: int. number of crovalidation folds.
    :return:
        model_coef: array n_xval x n_units x n_bins mean firing rate at each bin
        train_perf: array n_xval x n_units x n_metrics [metrics = r2, err, map_corr]
        test_perf: array n_xval x n_units x n_metrics [metrics = r2, err, map_corr]
    """
    n_samps = len(x)
    if neural_data.ndim == 1:
        n_units = 1
        neural_data = neural_data.reshape(1, -1)
    else:
        n_units, _ = neural_data.shape
    assert n_samps == neural_data.shape[1], 'Mismatch lengths between speed and neural_data.'

    # split data into folds
    xval_samp_ids = rs.split_timeseries(n_samps=n_samps, samps_per_split=1000, n_data_splits=n_xval)

    # pre-allocate data
    features = get_border_encoding_features(x, y, x_bin_edges, y_bin_edges, feat_type=feat_type)
    if bias_term:
        features = np.append(np.ones((n_samps, 1), dtype=features.dtype), features, axis=1)
    n_features = features.shape[1]  # number of columns

    # obtain relevant functions for data type
    map_params = {'x_bin_edges': x_bin_edges, 'y_bin_edges': y_bin_edges,
                  'spatial_window_size': spatial_window_size, 'spatial_sigma': spatial_sigma}
    spatial_map_function = get_spatial_map_function(data_type, **map_params)
    if data_type == 'spikes':
        model_function = lm.PoissonRegressor(alpha=0, fit_intercept=False)
        reg_type = 'poisson'
    elif data_type == 'fr':
        model_function = lm.LinearRegression(fit_intercept=False)
        reg_type = 'linear'
    else:
        raise NotImplementedError

    # pre-allocate performance metrics
    perf_metrics = ['r2', 'ar2', 'err', 'n_err', 'map_r']
    train_perf = {}
    test_perf = {}
    for mm in perf_metrics:
        train_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
        test_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
    model_coef = np.zeros((n_xval, n_units), dtype=object)  # variable number of features per fold/unit depending on fit

    for fold in range(n_xval):
        # train set
        train_idx = xval_samp_ids != fold
        x_train = x[train_idx]
        y_train = y[train_idx]

        # test set
        test_idx = xval_samp_ids == fold
        x_test = x[test_idx]
        y_test = y[test_idx]

        # get features
        features_train = get_border_encoding_features(x_train, y_train, x_bin_edges, y_bin_edges,
                                                      feat_type=feat_type)
        features_test = get_border_encoding_features(x_test, y_test, x_bin_edges, y_bin_edges,
                                                     feat_type=feat_type)
        if bias_term:
            features_train = np.append(np.ones((train_idx.sum(), 1), dtype=features_train.dtype),
                                       features_train, axis=1)
            features_test = np.append(np.ones((test_idx.sum(), 1), dtype=features_test.dtype),
                                      features_test, axis=1)

        for unit in range(n_units):
            # get responses
            response_train = neural_data[unit, train_idx]
            response_test = neural_data[unit, test_idx]

            # train model
            model = model_function.fit(features_train, response_train)
            model_coef[fold, unit] = model.coef_

            # get predicted responses
            response_train_hat = model.predict(features_train)
            response_test_hat = model.predict(features_test)

            # get true spatial for this fold maps
            train_map = spatial_map_function(response_train, x_train, y_train)
            test_map = spatial_map_function(response_test, x_test, y_test)

            # get predicted maps
            train_map_hat = spatial_map_function(response_train_hat, x_train, y_train)
            test_map_hat = spatial_map_function(response_test_hat, x_test, y_test)

            # train performance
            temp1 = rs.get_regression_metrics(response_train, response_train_hat, reg_type=reg_type,
                                              n_params=n_features)

            train_perf['map_r'][fold, unit] = rs.pearson(train_map.flatten(), train_map_hat.flatten())

            # test performance
            temp2 = rs.get_regression_metrics(response_test, response_test_hat, reg_type=reg_type,
                                              n_params=n_features)
            test_perf['map_r'][fold, unit] = rs.pearson(test_map.flatten(), test_map_hat.flatten())

            for metric in ['r2', 'ar2', 'err', 'n_err']:
                train_perf[metric][fold, unit] = temp1[metric]
                test_perf[metric][fold, unit] = temp2[metric]

    return model_coef, train_perf, test_perf


def get_border_encoding_features(x, y, x_bin_edges, y_bin_edges, feat_type='linear', **non_linear_params):
    """
    Returns proximity vectos given x y positions. 3 vectors, east, north, and center
    :param y: array of y positions in cm
    :param x_bin_edges: x bin edges
    :param y_bin_edges: y bin edges
    :param feat_type: str, 2 posibilities: ['linear'. sigmoid']. indicating linear or sigmoid
    :param non_linear_params: dictionary of parameters for smooth proximity matrix calculation.
        include border_width_bin, sigmoid_slope_thr, center_gaussian_spread,
        see get_non_linear_border_proximity_mats for details.

    :return: 3 arrays of proximity (1-distance) for each xy position to the east wall, north wall and center.
    """
    x_bin_idx, y_bin_idx = get_xy_samps_pos_bins(x, y, x_bin_edges, y_bin_edges)

    width = len(x_bin_edges) - 1
    height = len(y_bin_edges) - 1

    if feat_type == 'linear':  # linear features
        prox_mats = get_linear_border_proximity_mats(width=width, height=height)
    else:
        prox_mats = get_sigmoid_border_proximity_mats(width=width, height=height, **non_linear_params)

    return prox_mats[:, y_bin_idx, x_bin_idx].T


def compute_border_score_solstad(fr_maps, fr_thr=0.3, min_field_size_bins=20, width_bins=3, return_all=False):
    """
    Border score method from Solstad et al Science 2008. Returns the border score along with the max coverage by a field
    and the weighted firing rate. This works for a single fr_map or multiple.
    :param fr_maps: np.ndarray, (dimensions can be 2 or 3), if 3 dimensions, first dimensions must
                    correspond to the # of units, other 2 dims are height and width of the map
    :param fr_thr: float, proportion of the max firing rate to threshold the data
    :param min_field_size_bins: int, # of bins that correspond to the total area of the field. fields found
                    under this threshold are discarded
    :param width_bins: wall width by which the coverage is determined.
    :param return_all: bool, if False only returns the border_score
    :return: border score, max coverage, distanced weighted neural_data for each unit in maps.

    -> code based of the description on Solstad et al, Science 2008
    """
    n_walls = 4
    # add a singleton dimension in case of only one map to find fields.
    if fr_maps.ndim == 2:
        fr_maps = fr_maps[np.newaxis,]
    n_units, map_height, map_width = fr_maps.shape

    # get fields
    field_maps, n_fields = get_map_fields(fr_maps, thr=fr_thr, min_field_size=min_field_size_bins)

    if field_maps.ndim == 2:
        field_maps = field_maps[np.newaxis,]
        n_fields = n_fields[np.newaxis,]

    # get border distance matrix
    distance_mat = get_center_border_distance_mat(map_height, map_width)  # linear distance to closest wall [bins]

    # get wall labels
    wall_labels_mask = get_wall_masks(map_height, map_width, width_bins)

    # pre-allocate scores
    border_score = np.zeros(n_units) * np.nan
    border_max_cov = np.zeros(n_units) * np.nan
    border_w_fr = np.zeros(n_units) * np.nan

    def _border_score_solstad(_field_map, _fr_map, _distance_mat, _wall_labels_mask):
        """
        computes the border scores given the field id map, firing rate and wall_mask
        :param _fr_map: 2d firing rate map
        :param _field_map: as obtained from get_map_fields
        :param _wall_labels_mask: as obtained from get_wall_masks
        :return: border_score, max_coverage, weighted_fr
        """
        _n_fields = int(np.max(_field_map)) + 1

        wall_coverage = np.zeros((_n_fields, n_walls))
        for field in range(_n_fields):
            for wall in range(n_walls):
                wall_coverage[field, wall] = np.sum(
                    (_field_map == field) * (_wall_labels_mask[wall] == wall)) / np.sum(
                    _wall_labels_mask[wall] == wall)
        c_m = np.max(wall_coverage)

        # get normalized distanced weighted firing rate
        field_fr_map = _fr_map * (_field_map >= 0)
        d_m = np.sum(field_fr_map * _distance_mat) / np.sum(field_fr_map)

        # get border score
        b = (c_m - d_m) / (c_m + d_m)
        return b, c_m, d_m

    # loop and get scores
    for unit in range(n_units):
        fr_map = fr_maps[unit]
        field_map = field_maps[unit]
        n_fields_unit = n_fields[unit]
        if n_fields_unit > 0:
            border_score[unit], border_max_cov[unit], border_w_fr[unit] = \
                _border_score_solstad(field_map, fr_map, distance_mat, wall_labels_mask)

    if return_all:
        return border_score, border_max_cov, border_w_fr
    else:
        return border_score


def permutation_test_border_score(fr, fr_maps, x, y, x_bin_edges, y_bin_edges, n_perm=200, sig_alpha=0.02,
                                  true_bs=None, n_jobs=8, **border_score_params):
    n_samps = len(x)
    if fr.ndim == 1:
        n_units = 1
        fr = fr[np.newaxis,]
        fr_maps = fr_maps[np.newaxis,]
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between samples and neural_data.'

    if true_bs is None:
        true_bs = compute_border_score_solstad(fr_maps, **border_score_params)

    def p_worker(unit_id):
        """ helper function for parallelization. Computes a single shuffled border score per unit."""
        fr_unit = fr[unit_id]
        # roll firing rate
        p_fr = np.roll(fr_unit, np.random.randint(n_samps))
        # get rate map
        p_fr_map = firing_rate_2_rate_map(p_fr, x=x, y=y, x_bin_edges=x_bin_edges, y_bin_edges=y_bin_edges)
        # get single border score
        p_bs = compute_border_score_solstad(p_fr_map, **border_score_params)
        return p_bs

    sig = np.zeros(n_units, dtype=bool)
    with Parallel(n_jobs=n_jobs) as parallel:
        for unit in range(n_units):
            if not np.isnan(true_bs[unit]):
                # get border score shuffle dist
                perm_bs = parallel(delayed(p_worker)(unit) for _ in range(n_perm))
                # find location of true gs
                loc = np.array(perm_bs >= true_bs[unit]).mean()
                # determine if outside distribution @ alpha level
                sig[unit] = np.logical_or(loc <= sig_alpha / 2, loc >= 1 - sig_alpha / 2)

    return true_bs, sig


# -border aux
def get_center_border_distance_mat(h, w):
    """
    creates a pyramid like matrix of distances to border walls.
    :param h: height
    :param w: width
    :return: normalized matrix of distances, center =1, borders=0
    """
    a = np.arange(h)
    b = np.arange(w)

    r_h = np.minimum(a, a[::-1])
    r_w = np.minimum(b, b[::-1])
    pyr = np.minimum.outer(r_h, r_w)
    return pyr / np.max(pyr)


def get_wall_masks(map_height, map_width, wall_width):
    """
    returns a mask for each wall. *assumes [0,0] is on lower left corner.*
    :param map_height:
    :param map_width:
    :param wall_width: size of the border wall
    :return: mask, ndarray size 4 x map_height x map_width, 4 maps each containing a mask for each wall
    """

    mask = np.ones((4, map_height, map_width), dtype=int) * -1

    mask[0][:, map_width:(map_width - wall_width - 1):-1] = 0  # right / East
    mask[1][map_height:(map_height - wall_width - 1):-1, :] = 1  # top / north
    mask[2][:, 0:wall_width] = 2  # left / West
    mask[3][0:wall_width, :] = 3  # bottom / south

    return mask


def get_map_fields(maps, thr=0.3, min_field_size=20, filt_structure=None):
    """
    gets labeled firing rate maps. works on either single maps or an array of maps.
    returns an array of the same dimensions as fr_maps with
    :param maps: np.ndarray, (dimensions can be 2 or 3), if 3 dimensions, first dimensions must
                    correspond to the # of units, other 2 dims are height and width of the map
    :param thr: float, proportion of the max firing rate to threshold the data
    :param min_field_size: int, # of bins that correspond to the total area of the field. fields found
                    under this threshold are discarded
    :param filt_structure: 3x3 array of connectivity. see ndimage for details
    :return field_labels (same dimensions as input), -1 values are background, each field has an int label

    """
    if filt_structure is None:
        filt_structure = np.ones((3, 3))

    # add a singleton dimension in case of only one map to find fields.
    if maps.ndim == 2:
        maps = maps[np.newaxis, :, :]
    elif maps.ndim == 1:
        print('maps is a one dimensional variable.')
        return None

    n_units, map_height, map_width = maps.shape

    # create border mask to avoid elimating samples during the image processing step
    border_mask = np.ones((map_height, map_width), dtype=bool)
    border_mask[[0, -1], :] = False
    border_mask[:, [0, -1]] = False

    # determine thresholds
    max_fr = maps.max(axis=1).max(axis=1)

    # get fields
    field_maps = np.zeros_like(maps)
    n_fields = np.zeros(n_units, dtype=int)
    for unit in range(n_units):
        # threshold the maps
        thr_map = maps[unit] >= max_fr[unit] * thr

        # eliminates small/noisy fields, fills in gaps
        thr_map = ndimage.binary_closing(thr_map, structure=filt_structure, mask=border_mask)
        thr_map = ndimage.binary_dilation(thr_map, structure=filt_structure)

        # get fields ids
        field_map, n_fields_unit = ndimage.label(thr_map, structure=filt_structure)

        # get the area of the fields in bins
        field_sizes = np.zeros(n_fields_unit)
        for f in range(n_fields_unit):
            field_sizes[f] = np.sum(field_map == f)

        # check for small fields and re-do field identification if necessary
        if np.any(field_sizes < min_field_size):
            small_fields = np.where(field_sizes < min_field_size)[0]
            for f in small_fields:
                thr_map[field_map == f] = 0
            field_map, n_fields_unit = ndimage.label(thr_map, structure=filt_structure)

        # store
        field_maps[unit] = field_map
        n_fields[unit] = n_fields_unit

    field_maps -= 1  # make background -1, labels start at zero

    # if only one unit, squeeze to match input dimensions
    if n_units == 1:
        field_maps = field_maps.squeeze()
        n_fields = n_fields.squeeze()

    return field_maps, n_fields


def get_xy_samps_pos_bins(x, y, x_bin_edges, y_bin_edges, ):
    """
    Converts x y position samples to the corresponding bin ids based on the limits and step.
    This essentially discretizes the x,y positions into bin ids.
    :param x: array of x positions in cm
    :param y: array of y positions in cm
    :param x_bin_edges: x bin edges
    :param y_bin_edges: y bin edges
    :returns:
        x_bin_ids: array of integers idx of the x bins
        y_bin_ids: array of integers idx of the y bins
        x_bin_centers: array of x bin centers
        y_bin_centers: array of y bin centers
    """
    _, x_bin_idx = rs.get_discrete_data_mat(x, x_bin_edges)
    _, y_bin_idx = rs.get_discrete_data_mat(y, y_bin_edges)

    return x_bin_idx, y_bin_idx


def get_linear_border_proximity_mats(width, height, border_width_bins=3):
    """
     Computes linear proximity to environment walls and the center. It will be linear from the center to the wall-border
     width, zero ow. values on the border for that wall are 1.
     Returns 5 proximity matrices, east, north, west, south, center
     :param width: width of the environment [bins]
     :param height: height of the environment [bins]
     :param border_width_bins: border width
     :returns: prox_mats: ndarray 3 x height x width, in order: east, north and center proximities.
     """

    east_prox = np.tile(_get_lin_proximity_array(width, border_width_bins), height).reshape(height, width)
    west_prox = np.fliplr(east_prox)

    north_prox = np.repeat(_get_lin_proximity_array(height, border_width_bins), width).reshape(height, width)
    south_prox = np.flipud(north_prox)

    center_prox = get_center_border_distance_mat(height, width)

    prox_mats = np.zeros((5, height, width))
    prox_mats[0] = east_prox
    prox_mats[1] = north_prox
    prox_mats[2] = west_prox
    prox_mats[3] = south_prox
    prox_mats[4] = center_prox

    return prox_mats


def _get_lin_proximity_array(dim_size, border_width):
    dim_size2 = dim_size // 2

    out_array = np.zeros(dim_size)
    out_array[dim_size - border_width:] = 1

    n_lin_bins = dim_size2 - border_width + np.mod(dim_size, 2)
    out_array[dim_size2:(dim_size - border_width)] = np.arange(n_lin_bins) / n_lin_bins

    return out_array


def get_sigmoid_border_proximity_mats(width, height, border_width_bins=3,
                                      sigmoid_slope_thr=0.1, center_gaussian_spread=0.2, include_center_feature=True,
                                      **kwargs):
    """
    Computes normalized and smoothed proximity to the east wall, north wall, and to the center.
    For the walls it uses a sigmoid function, for which the wall_width determines when it saturates
    For the center it uses a normalized gaussian.
    :param include_center_feature:
    :param width: width of the environment [bins]
    :param height: height of the environment [bins]
    :param border_width_bins: number of bins from the border for the sigmoid to saturate
    :param sigmoid_slope_thr: value of the sigmoid at the first bin of the border_width (symmetric)
    :param center_gaussian_spread: this gets multiplied by the dimensions of the environment to get the spread.
    :returns: prox_mats: ndarray 5 x height x width, in order: east, north, west, south and center proximities.
    """

    sigmoid_slope_w = _get_optimum_sigmoid_slope(border_width_bins, width / 4, sigmoid_slope_thr=sigmoid_slope_thr)
    sigmoid_slope_h = _get_optimum_sigmoid_slope(border_width_bins, height / 4, sigmoid_slope_thr=sigmoid_slope_thr)
    center_w = width / 2
    center_h = height / 2

    west_prox = np.tile(1 - sigmoid(np.arange(width), width / 4, sigmoid_slope_w), height).reshape(height, width)
    east_prox = np.fliplr(west_prox)
    south_prox = np.repeat(1 - sigmoid(np.arange(height), height / 4, sigmoid_slope_h), width).reshape(height, width)
    north_prox = np.flipud(south_prox)

    x, y = np.meshgrid(np.arange(width), np.arange(height))  # get 2D variables instead of 1D

    if include_center_feature:
        center_prox = gaussian_2d(y=y, x=x, my=center_h, mx=center_w, sx=width * center_gaussian_spread,
                                  sy=height * center_gaussian_spread)
        center_prox = center_prox / np.max(center_prox)
        prox_mats = np.zeros((5, height, width))
        prox_mats[4] = center_prox
    else:
        prox_mats = np.zeros((4, height, width))

    prox_mats[0] = east_prox
    prox_mats[1] = north_prox
    prox_mats[2] = west_prox
    prox_mats[3] = south_prox

    return prox_mats


# ------------------------------------------------- GRID METRICS -----------------------------------------------------
def get_grid_encoding_model_old(x, y, fr, fr_maps, x_bin_edges, y_bin_edges, grid_fit='auto_corr', reg_type='linear',
                                compute_gs_sig=False, sig_alpha=0.02, n_perm=200, verbose=False, **kwargs):
    """
    Grid encoding model. Also obtains grid score.
    :param x: array n_samps of x positions of the animal
    :param y: array n_samps of y positions of the animal
    :param fr: ndarray n_units x n_samps of firing rate,
    :param fr_maps: ndarray n_units x height x width of smoothed firing rate position maps
    :param x_bin_edges: x bin edges
    :param y_bin_edges: y bin edges
    :param sig_alpha: significance alpha for permutation test
    :param n_perm: number of permutations
    :param grid_fit: two types ['auto_corr', 'moire']. if auto_corr, uses the scale/angle obtain from the autocorr to
    generate encoding feature. otherwise, uses a grid-search of different moire patterns
    :param reg_type: string ['linear', 'poisson'], use linear for firing rate, poisson for binned spikes
    :param compute_gs_sig: bool. if True, performs permutations to determine grid score significance
    :param verbose: bool.
    :param kwargs: grid_score parameters
    :return: scores: pd.Dataframe with columns ['grid_score', 'grid_sig', 'scale', 'phase', 'aR2', 'rmse', 'nrmse'],
          model_coef: array n_units x 2 of encoding coefficients [bias, east, north, west, south, center]
          model_coef_sem: array n_units x 4 sem for the coefficients
    """

    # get analyses constants and make sure they are consistent
    n_samps = len(x)
    if fr.ndim == 1:
        n_units = 1
        fr = fr.reshape(1, -1)
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and neural_data.'

    n_units2, height, width = fr_maps.shape
    assert n_units2 == n_units, 'inconsistent number of units'
    del n_units2

    # pre-allocated outputs
    coefs = np.zeros((n_units, 2)) * np.nan  # 2 coefficients, 1 for moire fit + bias
    coefs_sem = np.zeros((n_units, 2)) * np.nan
    scores = pd.DataFrame(index=range(n_units),
                          columns=['score', 'sig', 'scale', 'phase', 'r2', 'rmse', 'nrmse'])

    # compute grid score
    for unit in range(n_units):
        if verbose:
            print(f'Computing Grid Score unit # {unit}')
        temp = compute_grid_score(fr_maps[unit], **kwargs)
        scores.at[unit, 'score'] = temp[0]
        scores.at[unit, 'scale'] = temp[1]
        scores.at[unit, 'phase'] = temp[2]

        if np.isnan(temp[0]):  # failed grid score computation
            if verbose:
                print('Grid Score compt. Failed.')
                print('Finding scale and phase by finding best fitting moire grid')
            temp = fit_moire_grid(fr_maps[unit])
            scores.at[unit, 'scale'] = temp[0]
            scores.at[unit, 'phase'] = temp[1]

    if compute_gs_sig:
        scores['sig'] = permutation_test_grid_score(fr, fr_maps, x, y, x_bin_edges, y_bin_edges,
                                                    n_perm=n_perm, alpha=sig_alpha, true_gs=scores['score'],
                                                    n_jobs=8)

    # environment grid
    if grid_fit == 'auto_corr':

        for unit in range(n_units):

            if ~np.isnan(scores.at[unit, 'scale']):
                fr_map = fr_maps[unit]

                # max field location becomes the spatial phase of the moire grid / center of it.
                max_field_loc = np.unravel_index(np.argmax(fr_map), fr_map.shape)

                moire_mat = generate_moire_grid(width, height, [max_field_loc[1], max_field_loc[0]],
                                                scores.at[unit, 'scale'], scores.at[unit, 'phase'])

                _, coef_temp, scores.at[unit, 'r2'], scores.at[unit, 'rmse'], scores.at[unit, 'nrmse'] = \
                    get_encoding_map_fit(fr[unit], moire_mat, x, y, x_bin_edges=x_bin_edges, y_bin_edges=y_bin_edges,
                                         reg_type=reg_type)
                coefs[unit, :] = coef_temp.flatten()

    else:
        raise NotImplementedError

    return scores, coefs, coefs_sem


def get_grid_encoding_model(x, y, neural_data, x_bin_edges, y_bin_edges, data_type='spikes', bias_term=True, n_xval=5,
                            spatial_window_size=3, spatial_sigma=2, **kwargs):
    """
    Grid encoding model. Also obtains grid score.
    :param x: array n_samps of x positions of the animal
    :param y: array n_samps of y positions of the animal
    :param neural_data: ndarray n_units x n_samps of firing rate,
    :param x_bin_edges: x bin edges
    :param y_bin_edges: y bin edges
    :param bias_term: bool. if true adds a column of ones to features.
    :param data_type: string ['spikes', 'neural_data'], indicating if the data is firing rate or spike rate.
    :param n_xval: int. number of x validation
    :param spatial_window_size: int, spatial extent of smoothing for features
    :param spatial_sigma: float, spatial std. for gaussian smoothing
    :param kwargs: grid_score parameters
    :return:
        model_coef: array n_xval x n_units x n_bins mean firing rate at each bin
        train_perf: array n_xval x n_units x n_metrics [metrics = r2, err, map_corr]
        test_perf: array n_xval x n_units x n_metrics [metrics = r2, err, map_corr]
    """

    # get analyses constants and make sure they are consistent
    n_samps = len(x)
    if neural_data.ndim == 1:
        n_units = 1
        neural_data = neural_data.reshape(1, -1)
    else:
        n_units, _ = neural_data.shape
    assert n_samps == neural_data.shape[1], 'Mismatch lengths between speed and neural_data.'

    grid_encoding_features_params = {}
    grid_encoding_features_params_list = ['thr', 'min_field_size', 'sigmoid_center', 'sigmoid_slope']
    for k, v in kwargs:
        if k in grid_encoding_features_params_list:
            grid_encoding_features_params[k] = v

    # split data into folds
    xval_samp_ids = rs.split_timeseries(n_samps=n_samps, samps_per_split=1000, n_data_splits=n_xval)

    # obtain relevant functions for data type
    map_params = {'x_bin_edges': x_bin_edges, 'y_bin_edges': y_bin_edges,
                  'spatial_window_size': spatial_window_size, 'spatial_sigma': spatial_sigma}
    spatial_map_function = get_spatial_map_function(data_type, **map_params)
    if data_type == 'spikes':
        model_function = lm.PoissonRegressor(alpha=0, fit_intercept=False)
        reg_type = 'poisson'
    elif data_type == 'fr':
        model_function = lm.LinearRegression(fit_intercept=False)
        reg_type = 'linear'
    else:
        raise NotImplementedError

    # pre-allocate performance metrics
    perf_metrics = ['r2', 'ar2', 'err', 'n_err', 'map_r']
    train_perf = {}
    test_perf = {}
    for mm in perf_metrics:
        train_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
        test_perf[mm] = np.zeros((n_xval, n_units)) * np.nan
    model_coef = np.zeros((n_xval, n_units), dtype=object)  # variable number of features per fold/unit depending on fit

    for fold in range(n_xval):
        # test set
        test_idx = xval_samp_ids == fold
        x_test = x[test_idx]
        y_test = y[test_idx]

        # train set
        train_idx = xval_samp_ids != fold
        x_train = x[train_idx]
        y_train = y[train_idx]

        for unit in range(n_units):
            try:
                # train response
                response_train = neural_data[unit, train_idx]

                # get grid fields
                fields_train = get_grid_fields(response_train, x_train, y_train, x_bin_edges, y_bin_edges,
                                               **grid_encoding_features_params)

                # can only create model if fields enough are found.
                if fields_train is not None:
                    # test response
                    response_test = neural_data[unit, test_idx]

                    # convert to fields to features
                    features_train = get_grid_encodign_features(fields_train, x_train, y_train,
                                                                x_bin_edges, y_bin_edges)
                    features_test = get_grid_encodign_features(fields_train, x_test, y_test,
                                                               x_bin_edges, y_bin_edges)

                    if bias_term:
                        features_train = np.append(np.ones((train_idx.sum(), 1), dtype=features_train.dtype),
                                                   features_train, axis=1)
                        features_test = np.append(np.ones((test_idx.sum(), 1), dtype=features_test.dtype),
                                                  features_test, axis=1)

                    # train model
                    model = model_function.fit(features_train, response_train)
                    model_coef[fold, unit] = model.coef_

                    # note that # of features changes depending on grid fit
                    n_features = len(model.coef_)

                    # get predicted responses
                    response_train_hat = model.predict(features_train)
                    response_test_hat = model.predict(features_test)

                    # get true spatial for this fold maps
                    train_map = spatial_map_function(response_train, x_train, y_train)
                    test_map = spatial_map_function(response_test, x_test, y_test)

                    # get predicted maps
                    train_map_hat = spatial_map_function(response_train_hat, x_train, y_train)
                    test_map_hat = spatial_map_function(response_test_hat, x_test, y_test)

                    # train performance
                    temp1 = rs.get_regression_metrics(response_train, response_train_hat, reg_type=reg_type,
                                                      n_params=n_features)

                    train_perf['map_r'][fold, unit] = rs.pearson(train_map.flatten(), train_map_hat.flatten())

                    # test performance
                    temp2 = rs.get_regression_metrics(response_test, response_test_hat, reg_type=reg_type,
                                                      n_params=n_features)
                    test_perf['map_r'][fold, unit] = rs.pearson(test_map.flatten(), test_map_hat.flatten())

                    for metric in ['r2', 'ar2', 'err', 'n_err']:
                        train_perf[metric][fold, unit] = temp1[metric]
                        test_perf[metric][fold, unit] = temp2[metric]
            finally:
                pass

    return model_coef, train_perf, test_perf


def get_grid_encodign_features(fields, x, y, x_bin_edges, y_bin_edges):
    x_bin_idx, y_bin_idx = get_xy_samps_pos_bins(x, y, x_bin_edges, y_bin_edges)
    return fields[:, y_bin_idx, x_bin_idx].T


def get_grid_fields(fr, x, y, x_bin_edges, y_bin_edges, thr=0.1, min_field_size=10,
                    sigmoid_center=0.5, sigmoid_slope=10, binary_fields=False):
    height = len(y_bin_edges) - 1
    width = len(x_bin_edges) - 1

    fr_map = firing_rate_2_rate_map(fr, x, y, x_bin_edges, y_bin_edges)
    nl_fr_map = sigmoid(fr_map / fr_map.max(), center=sigmoid_center, slope=sigmoid_slope)
    fields_map, n_fields = get_map_fields(nl_fr_map, thr=thr, min_field_size=min_field_size)
    thr_fr_map = (fields_map >= 0) * nl_fr_map

    # if sufficient fields for gs computation:
    if n_fields >= 3:
        _, scale, phase, _ = compute_grid_score(thr_fr_map)
        if np.isnan(scale):  # if auto correlation finding of scale/phase fails, fit moire grid
            temp = fit_moire_grid(thr_fr_map)
            moire_fit = temp[2]
        else:
            max_field_loc = np.unravel_index(np.argmax(thr_fr_map), fr_map.shape)
            moire_fit = generate_moire_grid(width, height, [max_field_loc[1], max_field_loc[0]],
                                            scale=scale, theta=phase)

        fields, n_fields = get_map_fields(moire_fit)

        field_maps = np.zeros((n_fields, height, width))
        for field_id in range(n_fields):
            if binary_fields:
                field_maps[field_id] = fields == field_id
            else:
                field_maps[field_id] = (fields == field_id) * moire_fit

        return field_maps
    else:
        return None


def get_encoding_map_fit(fr, maps, x, y, x_bin_edges, y_bin_edges, reg_type='linear', bias_term=False):
    """
    From spikes, an amplitude matrix map corresponding to locations, and the locations of the animals obtain encoding
    model predicting the firing or spiking as function of location.
    :param fr: n_units x n_samps array of firing rate or binned spikes
    :param maps: n_maps x height x width representing the amplitude of the map to be tested
    :param x: xlocation of the animal
    :param y: y location of the animal
    :param x_bin_edges:
    :param y_bin_edges:
    :param reg_type:  str, regression type ['poisson', 'linear']
    :param bias_term: boolean, add a bias term to the fit
    :return: predictions [fr_hat], coeficcients [coefficients], variance exp. [r2/d2], error [rmse], norm. err. [nrmse]
    """
    n_samps = len(x)
    if fr.ndim == 1:
        n_units = 1
        fr = fr[np.newaxis,]
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between samples and neural_data.'

    # if only one map, add a singleton axis
    if maps.ndim == 2:
        maps = maps[np.newaxis,]

    # fit model
    _, x_bin_idx = rs.get_discrete_data_mat(x, bin_edges=x_bin_edges)
    _, y_bin_idx = rs.get_discrete_data_mat(y, bin_edges=y_bin_edges)

    n_maps, height, width = maps.shape

    # encoding vectors
    if bias_term:
        bias = 1
    else:
        bias = 0

    X = np.ones((n_samps, n_maps + bias))  # + bias vector
    for mm in range(n_maps):
        X[:, mm + bias] = maps[mm, y_bin_idx, x_bin_idx]

    # get model and fit
    if reg_type == 'poisson':
        coef = np.zeros((n_units, n_maps + bias))
        fr_hat = np.zeros_like(fr)
        for unit in range(n_units):
            model = lm.PoissonRegressor(alpha=0, fit_intercept=False, max_iter=5000).fit(X, fr[unit])
            coef[unit] = model.coef_
            fr_hat[unit] = model.predict(X)
    elif reg_type == 'linear':
        model = lm.LinearRegression(fit_intercept=False).fit(X, fr.T)
        coef = model.coef_.T
        fr_hat = model.predict(X).T
    else:
        print(f'method {reg_type} not implemented.')
        raise NotImplementedError

    # get_scores
    if reg_type == 'poisson':
        r2 = rs.get_poisson_ad2(fr, fr_hat, n_maps)
        err = rs.get_poisson_deviance(fr, fr_hat)
        nerr = rs.get_poisson_pearson_chi2(fr, fr_hat)
    elif reg_type == 'linear':
        r2 = rs.get_ar2(fr, fr_hat, n_maps)
        err = rs.get_rmse(fr, fr_hat)
        nerr = rs.get_nrmse(fr, fr_hat)
    else:
        print(f'method {reg_type} not implemented.')
        raise NotImplementedError

    return fr_hat, coef, r2, err, nerr


def get_encoding_map_predictions(fr, maps, coefs, x, y, x_bin_edges, y_bin_edges, reg_type='linear', bias_term=False):
    """
    Test for 2d map models. Given a set of coefficients and data, obtain predicted firing rate or spikes, along with
    metrics of performance. Note that the given neural_data, x, y should be from a held out test set.
    :param fr: n_units x n_samps array of firing rate or binned spikes
    :param maps: n_maps x height x width representing the amplitude of the map to be tested
    :param coefs: n_units x n_coefs,  coefficients of the model. type of coefficients most match regression type
    :param x: xlocation of the animal
    :param y: y location of the animal
    :param x_bin_edges:
    :param y_bin_edges:
    :param reg_type:  str, regression type ['poisson', 'linear']
    :param bias_term: boolean, if there is bias term on the coefficients
    :returns: predictions [fr_hat], coeficcients [coefficients], variance exp. [r2/d2], error [rmse], norm. err. [nrmse]
    """

    n_samps = len(x)
    if fr.ndim == 1:
        n_units = 1
        fr = fr[np.newaxis,]
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between samples and neural_data.'

    # if only one map, add a singleton axis
    if maps.ndim == 2:
        maps = maps[np.newaxis,]

    # prepare data
    _, x_bin_idx = rs.get_discrete_data_mat(x, bin_edges=x_bin_edges)
    _, y_bin_idx = rs.get_discrete_data_mat(y, bin_edges=y_bin_edges)
    n_maps, height, width = maps.shape

    if bias_term:
        bias = 1
    else:
        bias = 0

    X = np.ones((n_samps, n_maps + bias))  # + bias vector
    for mm in range(n_maps):
        X[:, mm + bias] = maps[mm, y_bin_idx, x_bin_idx]

    # get model predictions
    if reg_type == 'linear':
        fr_hat = (X @ coefs.T).T
    elif reg_type == 'poisson':
        fr_hat = np.exp(X @ coefs.T).T
    else:
        print(f'Method {reg_type} not implemented.')
        raise NotImplementedError

    # get_scores
    if reg_type == 'poisson':
        r2 = rs.get_poisson_ad2(fr, fr_hat, n_maps)
        err = rs.get_poisson_deviance(fr, fr_hat)
        nerr = rs.get_poisson_pearson_chi2(fr, fr_hat)
    else:
        r2 = rs.get_ar2(fr, fr_hat, n_maps)
        err = rs.get_rmse(fr, fr_hat)
        nerr = rs.get_nrmse(fr, fr_hat)

    return fr_hat, r2, err, nerr


def compute_grid_score(rate_map, ac_thr=0.01, radix_range=None,
                       apply_sigmoid=True, sigmoid_center=None, sigmoid_slope=None,
                       find_fields=True,
                       verbose=False, ):
    """
    Function to compute grid score as detailed in Moser 07. Code inspired on version from Matt Nolans lab:
    https://github.com/MattNolanLab/grid_cell_analysis

    :param rate_map: original rate map. 2dim
    :param ac_thr: cut threshold to find fields in the autocorrelation in relation to max
    :param radix_range: ring size dimensions in relation to the spacing/scale
        (as computed by the mean distance to the six closest autocorrelation fields).
    :param apply_sigmoid: bool. uses a sigmoid non linearity to amplify rate map SNR
    :param sigmoid_center: float. center of sigmoid for amplify_rate, ignored if amplify_rate_map is False
    :param sigmoid_slope: float. slope of sigmoid for amplify_rate, ignored if amplify_rate_map is False
    :param find_fields: bool.
    :param verbose: bool.
    :return: 4 elements:
        1. grid score, float
        2. scale (grid spacing), float
        3. angle (angle from horizontal; phase of grid), float
        4. locations of auto correlation grid fields [x,y], np.ndarray

    """

    if radix_range is None:
        radix_range = [0.5, 2.0]

    # normalize rate map
    max_rate = rate_map.max()
    n_rate_map = rate_map / max_rate

    if apply_sigmoid:
        if sigmoid_center is None:
            sigmoid_center = 0.5
        if sigmoid_slope is None:
            sigmoid_slope = 10
        sigmoid_params = {'center': sigmoid_center,
                          'slope': sigmoid_slope}
        n_rate_map = sigmoid(n_rate_map, **sigmoid_params)

    if find_fields:
        mean_rate = n_rate_map.mean()

        while_counter = 0
        found_three_fields_flag = False
        thr_factor = 1
        fields_map = np.zeros_like(n_rate_map)
        while (not found_three_fields_flag) and (while_counter <= 4):
            fields_map, n_fields = get_map_fields(n_rate_map, thr=mean_rate * thr_factor)
            if n_fields >= 3:
                found_three_fields_flag = True
                break
            else:
                thr_factor *= 0.5
                while_counter += 1

        if not found_three_fields_flag:
            if verbose:
                print('Not enought rate fields found to have a reliable computation.')
            return np.nan, np.nan, np.nan, np.nan

        n_rate_map = (fields_map >= 0) * n_rate_map

    # get auto-correlation
    ac_map = rs.compute_autocorr_2d(n_rate_map)
    # ac_map = signal.correlate2d(n_rate_map, n_rate_map, boundary='wrap')
    ac_map = (ac_map / np.abs(ac_map.max()))

    ac_map_w = ac_map.shape[1]
    ac_map_h = ac_map.shape[0]

    # # get fields
    # map_fields, n_fields = get_map_fields(ac_map, thr=ac_thr)
    # n_fields = int(n_fields)
    #
    # # get field positions
    # map_fields = np.array(map_fields, dtype=int)
    # field_locs = ndimage.measurements.center_of_mass(ac_map, map_fields, np.arange(n_fields))
    #
    ac_p = detect_img_peaks(ac_map, background_thr=ac_thr)
    labeled_ac_p, n_fields = ndimage.label(ac_p)
    labeled_ac_p -= 1
    field_locs = ndimage.measurements.center_of_mass(ac_p, labeled_ac_p, np.arange(n_fields))
    field_locs = np.array(field_locs)

    # field_mass = [np.sum(map_fields == field_id) for field_id in np.arange(n_fields)]
    # field_mass = np.array(field_mass)

    field_locs2 = Points2D(field_locs[:, 1], field_locs[:, 0])
    center = Points2D(ac_map_w / 2, ac_map_h / 2)
    field_distances = field_locs2 - center

    dist_sorted_field_idx = np.argsort(field_distances.r)
    # get closest 6 fields idx
    if n_fields >= 7:
        closest_six_fields_idx = dist_sorted_field_idx[1:7]
    elif n_fields >= 3:
        closest_six_fields_idx = dist_sorted_field_idx[1:n_fields]
    else:
        if verbose:
            print('Did not find enough auto correlation fields.')
        return np.nan, np.nan, np.nan, np.nan
    #
    # # maske the closest fields
    # masked_fields = np.array(map_fields)
    # for field in range(int(n_fields)):
    #     if field not in closest_six_fields_idx:
    #         masked_fields[map_fields == field] = -1

    # select fields
    field_distances2 = field_distances[closest_six_fields_idx]

    mean_field_dist = np.mean(field_distances2.r)
    angs = np.array(field_distances2.ang)
    angs[angs > np.pi] = np.mod(angs[angs > np.pi], np.pi) - np.pi

    grid_phase = np.min(np.abs(angs))  # min angle corresponds to closest autocorr from x axis

    radix_range = np.array(radix_range) * mean_field_dist

    # mask the region
    mask_radix_out = np.zeros_like(ac_map)
    mask_radix_in = np.zeros_like(ac_map)

    r, c = draw.disk((center.xy[0, 1], center.xy[0, 0]), radix_range[1], shape=(ac_map_h, ac_map_w))
    mask_radix_out[r, c] = 1

    r, c = draw.disk((center.xy[0, 1], center.xy[0, 0]), radix_range[0], shape=(ac_map_h, ac_map_w))
    mask_radix_in[r, c] = 1

    mask_ac = mask_radix_out - mask_radix_in

    # rotate fields and get grid score
    unrotated_masked_ac = ac_map[mask_ac == 1]
    rotations = np.arange(30, 151, 30)  # rotations 30, 60, 90, 120, 150
    corrs = np.zeros(len(rotations))
    for i, angle in enumerate(rotations):
        rotated_masked_ac = rotate(ac_map, angle)[mask_ac == 1]
        corrs[i] = rs.pearson(unrotated_masked_ac, rotated_masked_ac)

    gs = np.mean(corrs[1::2]) - np.mean(corrs[::2])

    return gs, mean_field_dist, grid_phase, field_distances2.xy


def permutation_test_grid_score(fr, fr_maps, x, y, x_bin_edges, y_bin_edges,
                                n_perm=200, sig_alpha=0.02, n_jobs=8, **grid_score_params):
    n_samps = len(x)
    if fr.ndim == 1:
        n_units = 1
        fr = fr[np.newaxis,]
        fr_maps = fr_maps[np.newaxis,]
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between samples and neural_data.'

    true_gs = np.zeros(n_units) * np.nan
    true_scale = np.zeros(n_units) * np.nan
    true_phase = np.zeros(n_units) * np.nan
    for unit in range(n_units):
        true_gs[unit], true_scale[unit], true_phase[unit], _ = compute_grid_score(fr_maps[unit], **grid_score_params)

    def p_worker(unit_id):
        """ helper function for parallelization. Computes a single shuffled grid score per unit."""
        fr_unit = fr[unit_id]
        # roll firing rate
        p_fr = np.roll(fr_unit, np.random.randint(n_samps))
        # get rate map
        p_fr_map = firing_rate_2_rate_map(p_fr, x=x, y=y, x_bin_edges=x_bin_edges, y_bin_edges=y_bin_edges)
        # get single grid score
        p_gs, _, _, _ = compute_grid_score(p_fr_map, **grid_score_params)
        return p_gs

    sig = np.zeros(n_units, dtype=bool)
    with Parallel(n_jobs=n_jobs) as parallel:
        for unit in range(n_units):
            if not np.isnan(true_gs[unit]):
                # get grid score shuffle dist
                perm_gs = parallel(delayed(p_worker)(unit) for perm in range(n_perm))
                # find location of true gs
                loc = np.array(perm_gs >= true_gs[unit]).mean()
                # determine if outside distribution @ alpha level
                sig[unit] = np.logical_or(loc <= sig_alpha / 2, loc >= 1 - sig_alpha / 2)

    return true_gs, sig, true_scale, true_phase


def _get_optimum_sigmoid_slope(border_width, center, sigmoid_slope_thr=0.1):
    """
    Finds the optimal sigmoid slope for a sigmoid function given the parameters.
    :param border_width: number of bins at which the sigmoid should saturate
    :param center: center of the relevant dimension (e.g. for width=40, center=20)
    :param sigmoid_slope_thr: value of the sigmoid at the first bin of the border_width (symmetric)
    :return: slope value for sigmoid
    """
    slopes = np.linspace(0, 50, 1000)
    z = sigmoid(border_width, center / 2, slopes)
    return slopes[np.argmin((z - sigmoid_slope_thr) ** 2)]


def generate_moire_grid(width, height, center, scale=30, theta=0, a=1):
    """
    This function creates a Moire 2 dimensional grid. This is an idealized grid.
    :param width: float width of environment
    :param height: float heigth of environment
    :param center: [x,y] location of the center of the grid
    :param scale: distance between grid noes
    :param theta: phase of the grid in radians
    :param a: field gain
    :return: amplitude of the moire grid as a matrix
    """
    n_gratings = 3

    c = PointsOF(center[0], center[1])
    x_mat, y_mat = np.meshgrid(np.arange(width), np.arange(height))
    r = PointsOF(x_mat.flatten(), y_mat.flatten())

    w = 1 / scale * 4 * np.pi / np.sqrt(3) * np.ones(n_gratings)  # 3 vecs w same length

    angs = theta - np.pi / 3 * (1 / 2 - np.arange(n_gratings))  # the angles
    wk = Points2D(w, angs, polar=True)  # create object with the 3 vectos

    ph_k = (r.xy - c.xy) @ wk.xy.T
    cos_k = np.cos(ph_k)
    g = gain_func(cos_k.sum(axis=1), a=a, xmin=-1.5, xmax=3)

    return g.reshape(height, width)


def gain_func(x, a=5 / 9, xmin=None, xmax=None):
    """
    Exponential gain function for moire grid
    :param x: array of values to be evaluated
    :param a: gain of the exponential
    :param xmin: minimum value possible for x (for scaling)
    :param xmax: maximum value possible for x (for scaling)
    if xmin, xmax not provied, it is determined from the data.
    :return: np.array of same dimensions of x after passing it through the gain function.
    """
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    c = a * (xmax - xmin)
    return (np.exp(a * (x - xmin)) - 1) / (np.exp(c) - 1)


def moire_grid_fit_params(**kwargs):
    fit_params_default = {'func': rs.get_mse, 'find_max': False,
                          'scale_range': np.arange(10, 40), 'angle_range': np.linspace(0, np.pi / 3, 30), 'gain': 1}

    fit_params = {}
    for k in fit_params_default.keys():
        if k in kwargs.keys():
            fit_params[k] = kwargs[k]
        else:
            fit_params[k] = fit_params_default[k]

    return fit_params


def fit_moire_grid(fr_map, **kwargs):
    n_jobs = 6
    fit_params = moire_grid_fit_params(**kwargs)

    # normalize the map
    fr_map = fr_map / fr_map.max()

    height, width = fr_map.shape
    c = np.unravel_index(np.argmax(fr_map), fr_map.shape)

    ls = fit_params['scale_range']
    thetas = fit_params['angle_range']
    gain = fit_params['gain']
    func = fit_params['func']
    find_max = fit_params['find_max']

    def worker(scale, th):
        moire_grid_ = generate_moire_grid(width, height, center=c, scale=l, theta=th, a=gain)
        score = func(fr_map.flatten(), moire_grid_.flatten())
        return score

    score_mat = np.zeros((len(ls), len(thetas)))
    with Parallel(n_jobs=n_jobs) as parallel:
        for ii, l in enumerate(ls):
            score_mat[ii] = parallel(delayed(worker)(l, th) for th in thetas)
    # score_mat = np.zeros((len(ls), len(thetas)))
    #

    #     for ii, l in enumerate(ls):
    #         for jj, th in enumerate(thetas):
    #             moire_grid = generate_moire_grid(width, height, center=c, scale=l, theta=th, a=gain)
    #             score_mat[ii, jj] = func(fr_map.flatten(), moire_grid.flatten())
    if find_max:
        fit_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
    else:
        fit_idx = np.unravel_index(np.argmin(score_mat), score_mat.shape)

    fit_l = ls[fit_idx[0]]
    fit_theta = thetas[fit_idx[1]]
    moire_grid = generate_moire_grid(width, height, center=[c[1], c[0]], scale=fit_l, theta=fit_theta, a=gain)

    return fit_l, fit_theta, moire_grid, score_mat


# ------------------------------------------------- Auxiliary Functions ----------------------------------------------#
def get_spatial_map_function(data_type, **params):
    map_params = ['x_bin_edges', 'y_bin_edges', 'spatial_window_size', 'spatial_sigma', 'apply_median_filt']
    if 'x_bin_edges' not in params.keys():
        if 'x_bin_edges_' in params.keys():
            params['x_bin_edges'] = params['x_bin_edges_']
    if 'y_bin_edges' not in params.keys():
        if 'y_bin_edges_' in params.keys():
            params['y_bin_edges'] = params['y_bin_edges_']

    for p in map_params:
        if p not in params.keys():
            if p == 'apply_median_filt':
                params[p] = False
            elif p == 'spatial_window_size':
                params[p] = 5
            elif p == 'spatial_sigma':
                params[p] = 2
            else:
                raise ValueError(f"Missing {p} Param")

    if data_type == 'spikes':
        def spatial_map_function(_spikes, _x, _y):
            out_map = spikes_2_rate_map(_spikes, _x, _y, **params)
            return out_map

    elif data_type == 'fr':
        def spatial_map_function(_fr, _x, _y):
            out_map = firing_rate_2_rate_map(_fr, _x, _y, **params)
            return out_map
    else:
        raise NotImplementedError
    return spatial_map_function


def detect_img_peaks(image, background_thr=0.01):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    ** modified from: https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    """
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image < background_thr)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max * ~eroded_background

    return detected_peaks
