import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn import linear_model as lm
import statsmodels.api as sm
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


def get_position_map_counts(x, y, x_edges, y_edges):
    """
    :param np.array x: x position of the animal
    :param np.array y: y position of the animal
    :param x_edges: bin edges in the x position
    :param y_edges: bin edges in the y position
    :return: 2d array of position counts, x_edges, and y_edges
    """

    # hist2d converts to a matrix, which reverses x,y
    # inverse order here to preserve visualization.
    pos_counts_2d, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    return pos_counts_2d


def get_weighted_position_map(x, y, w, x_edges, y_edges):
    """
    :param np.array x: x position of the animal
    :param np.array y: y position of the animal
    :param np.array w: weight of each position sample (eg. spike counts or firing rate)
    :param x_edges: bin edges in the x position
    :param y_edges: bin edges in the y position
    :return: 2d array of position counts, x_edges, and y_edges
    """
    # hist2d converts to a matrix, which reverses x,y
    # inverse order here to preserve visualization.
    pos_counts_2d, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=w)
    return pos_counts_2d


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


# speed scores
def get_speed_score_traditional(speed, fr, min_speed=2, max_speed=80, sig_alpha=0.02, n_perm=100):
    """
    Traditional method of computing speed score. simple correlation of speed & firing rate
    :param speed: array floats vector of speed n_samps
    :param fr: array floats firing rate of the neuron
    :param max_speed: float
    :param min_speed: float
    :param sig_alpha: float, significant level to evaluate the permutation test
    :param n_perm: int, number of permutations to perform.
    :returns: score: speed score per unit
              sig: if the speed score reached significance after permutation test
    """

    n_samps = len(speed)
    if fr.ndim == 1:
        n_units = 1
        fr = fr.reshape(1, -1)
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and fr.'

    # get valid samples and assign new variables for fitting
    valid_samps = np.logical_and(speed >= min_speed, speed <= max_speed)
    speed_valid = speed[valid_samps]
    fr_valid = fr[:, valid_samps]

    # traditional correlation method
    score = np.zeros(n_units)
    sig = np.zeros(n_units, dtype=bool)
    for unit in range(n_units):
        score[unit] = rs.spearman(speed_valid, fr_valid[unit])
        sig[unit], _ = rs.permutation_test(function=rs.spearman, x=speed_valid, y=fr_valid[unit],
                                           n_perm=n_perm, alpha=sig_alpha)

    return score, sig


def get_speed_encoding_model(speed, fr, speed_bin_edges, compute_sp_score=True, sig_alpha=0.02, n_perm=100):
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

    [2] mean fr per speed bin: (could use trim mean, median, or other robust methods here);
    - equivalency with ols is only true for mean
    implemented below. this method allows to get standard errors per bin easily and fast

    [3] weighted histogram:
    fr weighted speed histogram then normalization by speed bin occupancy.
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
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and fr.'

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
    scores = pd.DataFrame(index=range(n_units), columns=['score', 'sig', 'aR2', 'rmse', 'nrmse'])
    scores['score'] = score
    scores['sig'] = score_sig
    scores['aR2'] = rs.get_ar2(fr_valid, fr_hat, n_sp_bins)
    scores['rmse'] = rs.get_rmse(fr_valid, fr_hat)
    scores['nrmse'] = rs.get_nrmse(fr_valid, fr_hat)

    return scores, model_coef, model_coef_s, valid_samps


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


# angle scores
def get_angle_stats(theta, step, weights=None):
    """
    Computes several circular statistics based on the histogram of the data.
    expects radians. Then uses the Rayleigh test for
    :param theta: original theta vector [radians]
    :param weights: weights for the each angle observation (e.g. spikes/ fr)
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


def get_angle_score_traditional(theta, fr, speed=None, min_speed=None, max_speed=None, sig_alpha=0.02):
    """"
    computes angle by firing rate without binning
    :param theta: array n_samps of angles in radians
    :param fr: array n_units x n_samps of firing rates
    :param speed: array of n_samps of speed to threshold the computations
    :param min_speed: minimum speed threshold
    :param max_speed: max speed threshold
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
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and fr.'

    if (speed is not None) and (min_speed is not None) and (max_speed is not None):
        valid_samps = np.logical_and(speed >= min_speed, speed <= max_speed)
        theta = theta[valid_samps]
        fr = fr[:, valid_samps]

    scores = pd.DataFrame(index=range(n_units),
                          columns=['vec_len', 'mean_ang', 'p_val', 'sig', 'rayleigh', 'var_ang', 'std_ang'])

    for unit in range(n_units):
        vec_len, mean_ang, var_ang, std_ang, = rs.resultant_vector_length(alpha=theta, w=fr[unit])
        p_val, rayleigh = rs.rayleigh(alpha=theta, w=fr[unit])
        out_dir = {'vec_len': vec_len, 'mean_ang': np.mod(mean_ang, 2 * np.pi), 'rayleigh': rayleigh, 'p_val': p_val,
                   'sig': p_val < sig_alpha, 'var_ang': var_ang,
                   'std_ang': std_ang}

        for key, val in out_dir.items():
            scores.loc[unit, key] = val

    return scores


def get_angle_encoding_model(theta, fr, ang_bin_edges, speed=None, min_speed=None, max_speed=None, sig_alpha=0.02):
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
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and fr.'

    # binning of the angle
    ang_bin_spacing = ang_bin_edges[1] - ang_bin_edges[0]
    ang_bin_centers = ang_bin_edges[:-1] + ang_bin_spacing / 2
    n_ang_bins = len(ang_bin_centers)

    # get discrete design matrix and valid samples
    ang_design_matrix, ang_bin_idx, valid_samps = \
        get_angle_encoding_features(theta, ang_bin_edges, speed=speed, min_speed=min_speed, max_speed=max_speed)
    fr = fr[:, valid_samps]

    # get model coefficients (mean fr per bin) and se of the mean
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
                          columns=['vec_len', 'mean_ang', 'p_val', 'sig', 'aR2', 'rmse', 'nrmse'])

    # loop to get circular stats scores
    for unit in range(n_units):
        # get vector length and mean angle
        vec_len, mean_ang, _, _, = rs.resultant_vector_length(ang_bin_centers, w=model_coef[unit], d=ang_bin_spacing)
        # rayleigh statistical test
        p_val, _ = rs.rayleigh(ang_bin_centers, w=model_coef[unit], d=ang_bin_spacing)

        # store results
        scores.at[unit, 'vec_len'] = vec_len
        scores.at[unit, 'mean_ang'] = np.mod(mean_ang, 2 * np.pi)
        scores.at[unit, 'sig'] = p_val < sig_alpha

    scores['aR2'] = rs.get_ar2(fr, fr_hat, n_ang_bins)
    scores['rmse'] = rs.get_rmse(fr, fr_hat)
    scores['nrmse'] = scores['rmse'] / fr.mean(axis=1)

    return scores, model_coef, model_coef_s


def get_angle_encoding_features(theta, ang_bin_edges, speed=None, min_speed=None, max_speed=None):
    # get valid samples and overwrite for fitting
    if (speed is not None) and (min_speed is not None) and (max_speed is not None):
        valid_samps = np.logical_and(speed >= min_speed, speed <= max_speed)
        theta = theta[valid_samps]
    else:
        valid_samps = np.ones(len(theta), dtype=bool)

    # binning of the angle / get discrete design matrix
    ang_design_matrix, ang_bin_idx = rs.get_discrete_data_mat(theta, ang_bin_edges)

    return ang_design_matrix, ang_bin_idx, valid_samps


# border scores
def get_border_encoding_model(x, y, fr, fr_maps, x_bin_edges, y_bin_edges,
                              compute_solstad=True, border_fr_thr=0.3, min_field_size_bins=20, border_width_bins=3,
                              sig_alpha=0.02, n_perm=100, non_linear=True):
    """
    Obtains the solstad border score and creates an encoding model based on proximity to the borders.
    :param x: array n_samps of x positions of the animal
    :param y: array n_samps of y positions of the animal
    :param fr: ndarray n_units x n_samps of firing rate,
    :param fr_maps: ndarray n_units x height x width of smoothed firing rate position maps
    :param x_bin_edges: x bin edges
    :param y_bin_edges: y bin edges
    :param sig_alpha: significance alpha for permutation test
    :param n_perm: number of permutations
    :param border_fr_thr: firing rate threshold for border score
    :param min_field_size_bins: minimum field size threshold for border score
    :param border_width_bins: size of the border in bins
    :param compute_solstad: bool. compute Solstad border score.
    :param non_linear: if true uses non-linear functions for the border proximity functions.
    :return: scores: pd.Dataframe with columns ['score', 'sig', 'aR2', 'rmse', 'nrmse'], rows are n_units
          model_coef: array n_units x 4 of encoding coefficients [bias, east, north, center]
          model_coef_sem: array n_units x 4 sem for the coeffieicents
    """
    n_samps = len(x)
    if fr.ndim == 1:
        n_units = 1
        fr = fr.reshape(1, -1)
    else:
        n_units, _ = fr.shape
    assert n_samps == fr.shape[1], 'Mismatch lengths between speed and fr.'

    # get solstad border score
    if compute_solstad:
        border_score_solstad_params = {'border_fr_thr': border_fr_thr,
                                       'min_field_size_bins': min_field_size_bins,
                                       'border_width_bins': border_width_bins,
                                       'sig_alpha': sig_alpha,
                                       'n_perm': n_perm}
        border_score, score_sig = get_border_score_solstad(fr_maps, **border_score_solstad_params)
    else:
        border_score = np.zeros(n_units)*np.nan
        score_sig = np.zeros(n_units)*np.nan

    # border encoding model
    # pre-allocate
    n_predictors = 3
    model_coef = np.zeros((n_units, n_predictors + 1))  # + bias term
    model_coef_s = np.zeros((n_units, n_predictors + 1))
    fr_hat = np.zeros_like(fr)

    # get proximity vectors
    X = get_border_encoding_features(x, y, x_bin_edges, y_bin_edges, non_linear=non_linear)
    X = sm.add_constant(X)

    # obtain model for each unit and extract coefficients.
    for unit in range(n_units):
        model = sm.OLS(fr[unit], X).fit()
        fr_hat[unit] = model.predict(X)
        model_coef[unit] = model.summary2().tables[1]['Coef.'].values
        model_coef_s[unit] = model.summary2().tables[1]['Std.Err.'].values

    # get performance scores
    scores = pd.DataFrame(index=range(n_units), columns=['solstad_score', 'solstad_sig', 'aR2', 'rmse', 'nrmse'])
    scores['solstad_score'] = border_score
    scores['solstad_sig'] = score_sig
    scores['aR2'] = rs.get_ar2(fr, fr_hat, n_predictors)
    scores['rmse'] = rs.get_rmse(fr, fr_hat)
    scores['nrmse'] = rs.get_nrmse(fr, fr_hat)

    return scores, model_coef, model_coef_s


def get_border_encoding_features(x, y, x_bin_edges, y_bin_edges, non_linear=True, **non_linear_params):
    """
    Returns proximity vectos given x y positions. 3 vectors, east, north, and center
    :param y: array of y positions in cm
    :param x_bin_edges: x bin edges
    :param y_bin_edges: y bin edges
    :param non_linear: if True, computes the proximity matrices with non_linear functions, otherwise uses linear
    :param non_linear_params: dictionary of parameters for smooth proximity matrix calculation.
        include border_width_bin, sigmoid_slope_thr, center_gaussian_spread,
        see get_non_linear_border_proximity_mats for details.

    :return: 3 arrays of proximity (1-distance) for each xy position to the east wall, north wall and center.
    """
    x_bin_idx, y_bin_idx = get_xy_samps_pos_bins(x, y, x_bin_edges, y_bin_edges)

    width = len(x_bin_edges) - 1
    height = len(y_bin_edges) - 1

    if non_linear:
        prox_mats = get_non_linear_border_proximity_mats(width=width, height=height, **non_linear_params)
    else:
        prox_mats = get_linear_border_proximity_mats(width=width, height=height)

    return prox_mats[:, y_bin_idx, x_bin_idx].T


def get_border_score_solstad(fr_maps, border_fr_thr=0.3, min_field_size_bins=20, border_width_bins=3,
                             sig_alpha=0.02, n_perm=100, return_all=False):
    """
    Border score method from Solstad et al Science 2008. Returns the border score along with the max coverage by a field
    and the weighted firing rate. This works for a single fr_map or multiple.
    :param fr_maps: np.ndarray, (dimensions can be 2 or 3), if 3 dimensions, first dimensions must
                    correspond to the # of units, other 2 dims are height and width of the map
    :param border_fr_thr: float, proportion of the max firing rate to threshold the data
    :param min_field_size_bins: int, # of bins that correspond to the total area of the field. fields found
                    under this threshold are discarded
    :param border_width_bins: wall width by which the coverage is determined.
    :param sig_alpha. float. permutation significance alpha
    :param n_perm: int. number of permuations for significance
    :param return_all: bool, if False only returns the border_score
    :return: border score, max coverage, distanced weighted fr for each unit in fr_maps.

    -> code based of the description on Solstad et al, Science 2008
    """
    n_walls = 4
    # add a singleton dimension in case of only one map to find fields.
    if fr_maps.ndim == 2:
        fr_maps = fr_maps[np.newaxis,]
    n_units, map_height, map_width = fr_maps.shape

    # get fields
    field_maps, n_fields = get_map_fields(fr_maps, fr_thr=border_fr_thr, min_field_size=min_field_size_bins)

    if field_maps.ndim == 2:
        field_maps = field_maps[np.newaxis,]

    # get border distance matrix
    distance_mat = get_center_border_distance_mat(map_height, map_width)  # linear distance to closest wall [bins]

    # get wall labels
    wall_labels_mask = get_wall_masks(map_height, map_width, border_width_bins)

    # pre-allocate scores
    border_score = np.zeros(n_units) * np.nan
    border_sig = np.zeros(n_units) * np.nan
    border_max_cov = np.zeros(n_units) * np.nan
    border_w_fr = np.zeros(n_units) * np.nan

    # loop and get scores
    for unit in range(n_units):
        fr_map = fr_maps[unit]
        field_map = field_maps[unit]
        n_fields_unit = n_fields[unit]
        if n_fields_unit > 0:
            border_score[unit], border_max_cov[unit], border_w_fr[unit] = \
                _border_score_solstad(field_map, fr_map, distance_mat, wall_labels_mask)
            border_sig[unit] = _permutation_test_border_score(field_map, fr_map, distance_mat, wall_labels_mask,
                                                              n_perm=n_perm, sig_alpha=sig_alpha,
                                                              seed=int(n_fields_unit*unit))
    if return_all:
        return border_score, border_sig, border_max_cov, border_w_fr
    else:
        return border_score, border_sig


def _border_score_solstad(field_map, fr_map, distance_mat, wall_labels_mask):
    """
    computes the border scores given the field id map, firing rate and wall_mask
    :param fr_map: 2d firing rate map
    :param field_map: as obtained from get_map_fields
    :param wall_labels_mask: as obtained from get_wall_masks
    :return: border_score, max_coverage, weighted_fr
    """
    n_walls = 4
    n_fields = int(np.max(field_map)) + 1

    wall_coverage = np.zeros((n_fields, n_walls))
    for field in range(n_fields):
        for wall in range(n_walls):
            wall_coverage[field, wall] = np.sum(
                (field_map == field) * (wall_labels_mask[wall] == wall)) / np.sum(
                wall_labels_mask[wall] == wall)
    c_m = np.max(wall_coverage)

    # get normalized distanced weighted firing rate
    field_fr_map = fr_map * (field_map >= 0)
    d_m = np.sum(field_fr_map * distance_mat) / np.sum(field_fr_map)

    # get border score
    b = (c_m - d_m) / (c_m + d_m)
    return b, c_m, d_m


def _shuffle_fields(field_map, fr_map):
    """
    shuffles fields in the map
    :param field_map: as obtained from get_map_fields
    :param fr_map: 2d firing rate map
    :return: shuffled_field_map, shuffled_fr_map
    """
    height, width = field_map.shape
    n_fields = int(np.max(field_map)) + 1

    shuffled_field_map = np.zeros_like(field_map)
    shufflued_fr_map = np.zeros_like(fr_map)
    xy_shift = np.array([np.random.randint(dim) for dim in [height, width]])
    for field in range(n_fields):
        # find field idx and shift (circularly)
        fields_idx = np.argwhere(field_map == field)
        shift_fields = fields_idx + xy_shift
        shift_fields[:, 0] = np.mod(shift_fields[:, 0], height)
        shift_fields[:, 1] = np.mod(shift_fields[:, 1], width)

        # get shuffled field map and fr map
        shuffled_field_map[shift_fields[:, 0], shift_fields[:, 1]] = field
        shufflued_fr_map[shift_fields[:, 0], shift_fields[:, 1]] = fr_map[fields_idx[:, 0], fields_idx[:, 1]]
    return shuffled_field_map, shufflued_fr_map


def _permutation_test_border_score(field_map, fr_map, distance_mat, wall_labels_mask, n_perm=100, sig_alpha=0.02, seed=0):
    """
    permuation test for border score. shuffles using the _shuffle_fields function that moves the field ids and the
    corresponding firing rates
    :param field_map: as obtained from get_map_fields
    :param fr_map: 2d firing rate map
    :param wall_labels_mask: as obtained from get_wall_masks
    :param n_perm: number of permutations
    :param sig_alpha: significance level
    :param seed: random seed
    :returns: bool, is the border score outside [=1] or within [=0] of the shuffled distribution
    """

    np.random.seed(seed)
    b, _, _ = _border_score_solstad(field_map, fr_map, distance_mat, wall_labels_mask)
    sh_b = np.zeros(n_perm)
    for perm in range(n_perm):
        sh_field_map, sh_fr_map = _shuffle_fields(field_map, fr_map)
        sh_b[perm], _, _ = _border_score_solstad(sh_field_map, sh_fr_map, distance_mat, wall_labels_mask)

    loc = (sh_b >= b).mean()
    outside_dist = loc <= sig_alpha / 2 or loc >= 1 - sig_alpha / 2
    return outside_dist


# border score auxiliary functions
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

    mask[0][:, map_width:(map_width - wall_width):-1] = 0  # right / East
    mask[1][map_height:(map_height - wall_width):-1, :] = 1  # top / north
    mask[2][:, 0:wall_width] = 2  # left / West
    mask[3][0:wall_width, :] = 3  # bottom / south

    return mask


def get_map_fields(fr_maps, fr_thr=0.3, min_field_size=20, filt_structure=None):
    """
    gets labeled firing rate maps. works on either single maps or an array of maps.
    returns an array of the same dimensions as fr_maps with
    :param fr_maps: np.ndarray, (dimensions can be 2 or 3), if 3 dimensions, first dimensions must
                    correspond to the # of units, other 2 dims are height and width of the map
    :param fr_thr: float, proportion of the max firing rate to threshold the data
    :param min_field_size: int, # of bins that correspond to the total area of the field. fields found
                    under this threshold are discarded
    :param filt_structure: 3x3 array of connectivity. see ndimage for details
    :return field_labels (same dimensions as input), -1 values are background, each field has an int label

    -> code based of the description on Solstad et al, Science 2008
    """
    if filt_structure is None:
        filt_structure = np.ones((3, 3))

    # add a singleton dimension in case of only one map to find fields.
    if fr_maps.ndim == 2:
        fr_maps = fr_maps[np.newaxis, :, :]
    elif fr_maps.ndim == 1:
        print('fr_maps is a one dimensional variable.')
        return None

    n_units, map_height, map_width = fr_maps.shape

    # create border mask to avoid elimating samples during the image processing step
    border_mask = np.ones((map_height, map_width), dtype=bool)
    border_mask[[0, -1], :] = False
    border_mask[:, [0, -1]] = False

    # determine thresholds
    max_fr = fr_maps.max(axis=1).max(axis=1)

    # get fields
    field_maps = np.zeros_like(fr_maps)
    n_fields = np.zeros(n_units, dtype=int)
    for unit in range(n_units):
        # threshold the maps
        thr_map = fr_maps[unit] >= max_fr[unit] * fr_thr

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


def get_linear_border_proximity_mats(width, height):
    """
     Computes linear proximity to the east wall, north wall and the center. That is, 1-closest, 0 farthest.
     Note that the reciprocal [1-x], of each is the distance to west wall, south wall, center-to-wall.
     :param width: width of the environment [bins]
     :param height: height of the environment [bins]
     :returns: prox_mats: ndarray 3 x height x width, in order: east, north and center proximities.
     """
    east_prox = np.tile(np.arange(width), height).reshape(height, width)
    east_prox = east_prox / np.max(east_prox)

    north_prox = np.repeat(np.arange(height), width).reshape(height, width)
    north_prox = north_prox / np.max(north_prox)

    center_prox = get_center_border_distance_mat(height, width)

    prox_mats = np.zeros((3, height, width))
    prox_mats[0] = east_prox
    prox_mats[1] = north_prox
    prox_mats[2] = center_prox

    return prox_mats


def get_non_linear_border_proximity_mats(width, height, border_width_bins=3,
                                         sigmoid_slope_thr=0.01, center_gaussian_spread=0.2):
    """
    Computes normalized and smoothed proximity to the east wall, north wall, and to the center.
    For the walls it uses a sigmoid function, for which the wall_width determines when it saturates
    For the center it uses a normalized gaussian.
    Note that the reciprocal of each is the distance to west wall, south wall, center-to-wall.
    :param width: width of the environment [bins]
    :param height: height of the environment [bins]
    :param border_width_bins: number of bins from the border for the sigmoid to saturate
    :param sigmoid_slope_thr: value of the sigmoid at the first bin of the border_width (symmetric)
    :param center_gaussian_spread: this gets multiplied by the dimensions of the environment to get the spread.
    :returns: prox_mats: ndarray 3 x height x width, in order: east, north and center proximities.
    """

    sigmoid_slope_w = _get_optimum_sigmoid_slope(border_width_bins, width, sigmoid_slope_thr=sigmoid_slope_thr)
    sigmoid_slope_h = _get_optimum_sigmoid_slope(border_width_bins, height, sigmoid_slope_thr=sigmoid_slope_thr)
    center_w = width / 2
    center_h = height / 2

    east_prox = np.tile(sigmoid(np.arange(width), center_w, sigmoid_slope_w), height).reshape(height, width)
    north_prox = np.repeat(sigmoid(np.arange(height), center_h, sigmoid_slope_h), width).reshape(height, width)

    x, y = np.meshgrid(np.arange(width), np.arange(height))  # get 2D variables instead of 1D
    center_prox = gaussian_2d(y=y, x=x, my=center_h, mx=center_w, sx=width * center_gaussian_spread,
                              sy=height * center_gaussian_spread)
    center_prox = center_prox / np.max(center_prox)

    prox_mats = np.zeros((3, height, width))
    prox_mats[0] = east_prox
    prox_mats[1] = north_prox
    prox_mats[2] = center_prox

    return prox_mats


def _get_optimum_sigmoid_slope(border_width, width, sigmoid_slope_thr=0.01):
    """
    Finds the optimal sigmoid slope for a sigmoid function given the parameters.
    :param border_width: number of bins at which the sigmoid should saturate
    :param width: number of bins of the environment
    :param sigmoid_slope_thr: value of the sigmoid at the first bin of the border_width (symmetric)
    :return: slope value for sigmoid
    """
    slopes = np.linspace(0, 2, 200)
    z = sigmoid(border_width, width / 2, slopes)
    return slopes[np.argmin((z - sigmoid_slope_thr) ** 2)]


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
