import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn import linear_model as lm
from Utils import robust_stats as rs


# spatial manipulation functions
def get_smoothed_map(bin_map, n_bins=8, sigma=2):
    """
    :param bin_map: map to be smooth.
        array in which each cell corresponds to the value at that xy position
    :param n_bins: number of smoothing bins
    :param sigma: std for the gaussian smoothing
    :return: sm_map: smoothed map. note that this is a truncated sigma map, meaning that high or
            low values wont affect far away bins
    """
    sm_map = ndimage.filters.median_filter(bin_map, n_bins)
    trunc = (((n_bins - 1) / 2) - 0.5) / sigma

    return ndimage.filters.gaussian_filter(sm_map, sigma, mode='constant', truncate=trunc)


def get_position_map_counts(x, y, x_lims, y_lims, spacing):
    """
    :param np.array x: x position of the animal
    :param np.array y: y position of the animal
    :param x_lims: 2 element list or array of x limits
    :param y_lims: 2 element list or array of y limits
    :param spacing: how much to bin the space
    :return: 2d array of position counts, x_edges, and y_edges
    """
    x_edges = np.arange(x_lims[0], x_lims[1] + spacing, spacing)
    y_edges = np.arange(y_lims[0], y_lims[1] + spacing, spacing)

    # hist2d converts to a matrix, which reverses x,y
    # inverse order here to preserve visualization.
    pos_counts_2d, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    return pos_counts_2d, y_edges, x_edges


def get_weighted_position_map(x, y, w, x_lims, y_lims, spacing):
    """
    :param np.array x: x position of the animal
    :param np.array y: y position of the animal
    :param np.array w: weight of each position sample (eg. spike counts or firing rate)
    :param x_lims: 2 element list or array of x limits
    :param y_lims: 2 element list or array of y limits
    :param spacing: how much to bin the space
    :return: 2d array of position counts, x_edges, and y_edges
    """
    x_edges = np.arange(x_lims[0], x_lims[1] + spacing, spacing)
    y_edges = np.arange(y_lims[0], y_lims[1] + spacing, spacing)

    # hist2d converts to a matrix, which reverses x,y
    # inverse order here to preserve visualization.
    pos_counts_2d, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=w)
    return pos_counts_2d, y_edges, x_edges


def get_velocity(x, y, time_step):
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


# spike-space functions
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


def get_spike_map(bin_spikes, x, y, x_lims, y_lims, spacing):
    """
    :param np.array bin_spikes: spike counts by time bin
    :param np.array x: x bin position of animal
    :param np.array y: y bin position of animal
    :param x_lims: 2 element list or array of x limits
    :param y_lims: 2 element list or array of y limits
    :param spacing: how much to bin the space
    :return: np.ndarray spike_map: number of spikes at each xy position
    """
    x_spk, y_spk = get_bin_spikes_xy(bin_spikes, x, y)
    spike_map, _, _ = get_position_map_counts(x_spk, y_spk, x_lims, y_lims, spacing)
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


def get_speed_score_traditional(speed, fr, min_speed, max_speed, alpha=0.05, n_perm=100):
    """
    Traditional method of computing speed score. simple correlation of speed & firing rate
    :param speed: array floats vector of speed n_samps
    :param fr: array floats firing rate of the neuron
    :param max_speed: float
    :param min_speed: float
    :param alpha: float, significant level to evaluate the permutation test
    :param n_perm: int, number of permutations to perform.
    :returns: scores: pd.Dataframe with columns ['score', 'sig', 'r2', 'rmse', 'nrmse'], rows are n_units
              speed_bins: array of speed bins
              model_coef: array n_units x n_bins mean firing rate at each bin
              model_coef_sem: array n_units x n_bins sem for each bin.
    """

    n_samps = len(speed)
    if fr.ndim == 1:
        n_units = 1
        fr = fr.reshape(1, -1)
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and fr.'

    # get valid samples and assign new variables for fitting
    speed_valid_idx = np.logical_and(speed >= min_speed, speed <= max_speed)
    n_samps = np.sum(speed_valid_idx)
    speed_valid = speed[speed_valid_idx]
    fr_valid = fr[:, speed_valid_idx]

    # traditional correlation method
    score = np.zeros(n_units)
    sig = np.zeros(n_units, dtype=bool)
    for unit in range(n_units):
        score[unit] = rs.spearman(speed_valid, fr_valid[unit])
        sig[unit], _ = rs.permutation_test(speed_valid, fr_valid[unit], rs.spearman, n_perm=n_perm, alpha=alpha)

    # additional linear model fit and metrics. Note that the code below is equivalent to utilizing statsmodels.api.OLS
    # but much faster, as I only compute relevant parameters.
    sp2_valid = np.column_stack((np.ones(n_samps), speed_valid))
    model = lm.LinearRegression(fit_intercept=False).fit(sp2_valid, fr_valid.T)
    model_coef = model.coef_
    fr_hat = model.predict(sp2_valid).T

    ar2 = rs.get_ar2(fr_valid, fr_hat, 1)
    rmse = rs.get_rmse(fr_valid, fr_hat)
    nrmse = rs.get_nrmse(fr_valid, fr_hat)

    # get standard errors:
    model_coef_s = np.zeros((n_units, 2))
    for unit in range(n_units):
        model_coef_s[unit] = rs.get_simple_regression_se(speed_valid, fr_valid[unit], fr_hat[unit])

    # arrange into a data frame
    scores = pd.DataFrame(index=range(n_units), columns=['score', 'sig', 'aR2', 'rmse', 'nrmse'])
    scores['score'] = score
    scores['sig'] = sig
    scores['aR2'] = ar2
    scores['rmse'] = rmse
    scores['nrmse'] = nrmse

    return scores, model_coef, model_coef_s


def get_speed_score_bins(speed, fr, speed_bin_spacing, min_speed, max_speed, alpha, n_perm):
    """
    :param speed: array floats vector of speed n_samps
    :param fr: array floats firing rate n_units x n_samps, also works for one unit
    :param speed_bin_spacing: float bin spacing cm/s
    :param max_speed: float max speed to threshold data
    :param min_speed: float min speed to threshold data
    :param alpha: float, significant level to evaluate the permutation test
    :param n_perm: int, number of permutations to perform.
    :returns: scores: pd.Dataframe with columns ['score', 'sig', 'r2', 'rmse', 'nrmse'], rows are n_units
              model_coef: array n_units x n_bins mean firing rate at each bin
              model_coef_sem: array n_units x n_bins sem for each bin.
              speed_bins: array of speed bins
    """

    n_samps = len(speed)
    if fr.ndim == 1:
        n_units = 1
        fr = fr.reshape(1, -1)
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and fr.'

    # get valid samples and assign new variables for fitting
    speed_valid_idx = np.logical_and(speed >= min_speed, speed <= max_speed)
    speed_valid = speed[speed_valid_idx]
    fr_valid = fr[:, speed_valid_idx]

    # binning of speed
    design_matrix, sp_bin_idx, sp_bin_centers, sp_bin_edges = \
        get_discrete_data_mat(speed_valid, min_speed, max_speed, speed_bin_spacing)
    n_sp_bins = len(sp_bin_centers)

    # Model additional details / observations.
    # There are several ways of doing this that are equivalent:
    #
    # weighted histogram:
    # fr weighted speed histogram then normalization by speed bin occupancy.
    # sp_occ,_ = np.histogram(speed_valid, sp_bins)
    # model_coef[unit],_ = np.histogram(speed_valid, sp_bins, weights=fr_valid[unit])
    # model_coef[unit] /= sp_occ
    #
    # linear regression: [using stats.linear_model]
    # model = lm.LinearRegression(fit_intercept=False).fit(speed_valid, fr_valid[unit])
    # model_coef[unit] = model.coef_
    #
    # ols: ordinary least squares (as above) using statsmodels.api
    # this is probably the most powerful, but slowest.
    # model = sm.OLS(fr_valid[unit],design_matrix)
    # results = model.fit()
    # model_coef[unit] = results.params
    #
    # mean fr per speed bin: (could use a trim mean or other robust methods here)
    # implemented below. this method further allows to get standard errors

    model_coef = np.zeros((n_units, n_sp_bins))
    model_coef_s = np.zeros((n_units, n_sp_bins))
    for i in range(n_sp_bins):
        model_coef[:, i] = np.mean(fr_valid[:, sp_bin_idx == i], axis=1)
        model_coef_s[:, i] = stats.sem(fr_valid[:, sp_bin_idx == i], axis=1)

    # get prediction
    # -> basically assigns to each sample its corresponding mean value
    fr_hat = model_coef @ design_matrix.T

    # pre-allocate scores
    score = np.zeros(n_units)
    score_sig = np.zeros(n_units, dtype=bool)
    ar2 = rs.get_ar2(fr_valid, fr_hat, n_sp_bins)
    rmse = rs.get_rmse(fr_valid, fr_hat)
    nrmse = rs.get_nrmse(fr_valid, fr_hat)

    # get scores
    for unit in range(n_units):
        score[unit] = rs.spearman(model_coef[unit], sp_bin_centers)
        score_sig[unit], _ = rs.permutation_test(model_coef[unit], sp_bin_centers, rs.spearman,
                                                 n_perm=n_perm, alpha=alpha)

    # arrange into a data frame
    scores = pd.DataFrame(index=range(n_units), columns=['score', 'sig', 'aR2', 'rmse', 'nrmse'])
    scores['score'] = score
    scores['sig'] = score_sig
    scores['aR2'] = ar2
    scores['rmse'] = rmse
    scores['nrmse'] = nrmse

    return scores, model_coef, model_coef_s, sp_bin_centers


def get_discrete_data_mat(data, min_val, max_val, step):
    """
    :param data: array float of data to be discretrize into a matrix
    :param min_val: float minimum value of the data to consider
    :param max_val: float max value of data to consider
    :param step: float bin step size
    :return: design_matrix: ndarray n_samps x n_bins of binary data
                            entry ij indicates that data sample i is in the jth bin
            data_bin_ids: array n_samps of ints,
                          ith value is the bin_center that gets assign to data[i]
            bin_centers: array float of bin centers
            bin_edges: array float of bin edges
    """

    n_samps = len(data)
    bin_edges = np.arange(min_val, max_val + step, step)
    bin_centers = bin_edges[:-1] + step / 2
    n_bins = len(bin_centers)
    data_bin_ids = np.digitize(data, bins=bin_edges) - 1  # shift ids so that first bin center corresponds to id 0

    design_matrix = np.zeros((n_samps, n_bins))
    for i in range(n_bins):
        design_matrix[:, i] = data_bin_ids == i

    return design_matrix, data_bin_ids,  bin_centers, bin_edges


# angular stats functions
def get_angle_hist(th, step):
    """
    get angle histogram for a given step. expects radians.
    :param th: angle
    :param step: radian bin step
    :return: counts per bin, bin_centers, bins
    """
    counts, bins = np.histogram(th, np.arange(0, 2 * np.pi + 0.01, step))
    bin_centers = bins[:-1] + step / 2
    return counts, bin_centers, bins


def get_angle_stats(theta, step):
    """
    Computes several circular statistics based on the histogram of the data.
    expects radians.
    :param theta: original theta vector [radians]
    :param step: angular bin size [radians]
    :return: dictionary with various stats
    """
    counts, bin_centers, bins = get_angle_hist(theta, step)
    z = np.mean(counts * np.exp(1j * bin_centers))
    ang = np.angle(z)
    r = np.abs(z)
    p, t = rs.rayleigh(theta)

    stats = {'r': r, 'ang': ang, 'R': t, 'p_val': p}

    return stats



