import numpy as np
import pandas as pd
from scipy import stats
from joblib import delayed, Parallel
import warnings


def mad(x):
    """ Computes median absolute deviation for an array.
        Defined as: median(abs(x-median(x)))

    Parameters
    ----------
    x: input numpy array (1D)

    Returns
    -------
    median absolute deviation (ignoring nan values)

    """
    medx = np.nanmedian(x)
    return np.nanmedian(np.abs(x - medx))


def mov_mad(x, window):
    """ Computes the moving median absolute deviation for a 1D array. Returns
        an array with the same length of the input array.
        Defined as: median(abs(Ai-median(x)))
        where Ai is a segment of the array x of length window.
        Small window length provides a finer description of deviation
        Longer window coarser (faster to compute).

        By default, each segment is centered, going L/2 to L/2-1 around Ai.
        For example for window = 4 and x= [1,2,1,2,5,2,5,2]
        A1=[0,1,2,3], A2=[4,5,6,7], the return array will be
        [1,1,1,1,3,3,3,3]

    Parameters
    ----------
    x       : input numpy array (1D)
    window  : integer for the evaluation window,
    Returns
    -------
    median absolute deviation (ignoring nan values)

    """
    if not type(x) == np.ndarray:
        x = np.array(x)

    if window % 2:
        window = window - 1
    win2 = np.int(window / 2)

    N = len(x)
    medx = np.median(x)
    madx = mad(x)

    y = np.full(N, madx)  # prefill with madx

    for ii in np.arange(win2, N - win2 + 1, window):
        try:
            idx = (np.arange(-win2, win2) + ii).astype(np.int)
            y[idx] = np.median(np.abs((x[idx] - medx)))
        except:
            pass
    return y


def mov_std(x, window):
    """ Computes the moving standard deviation for a 1D array. Returns
        an array with the same length of the input array.

        Small window length provides a finer description of deviation
        Longer window coarser (faster to compute).

        By default, each segment is centered, going L/2 to L/2-1 around Ai.


    Parameters
    ----------
    x       : input numpy array (1D)
    window  : integer for the evaluation window,
    Returns
    -------
    1d vector of standard deviations

    """
    if not type(x) == np.ndarray:
        x = np.array(x)

    if window % 2:
        window = window - 1

    win2 = np.floor(window / 2)
    N = len(x)
    sx = np.nanstd(x)
    y = np.full(N, sx)
    for ii in np.arange(win2, N - win2 + 1, window):
        try:
            idx = (np.arange(-win2, win2) + ii).astype(np.int)
            y[idx] = np.nanstd(x[idx])
        except:
            pass
    return y


def robust_zscore(x, axis=None):
    """ robust_zscore
        function that uses median and median absolute deviation to standard the
        input vector

    Parameters
    ----------
    x       : input numpy array (1D)
    axis    : axis in which to operate

    Returns
    -------
    z       : standarized vector with zero median and std ~1 (without outliers)

    """
    m = np.nanmedian(x, axis=axis)

    if axis is not None:
        m = np.expand_dims(m, axis=axis)

    mad = np.nanmedian(np.abs(x - m))

    return (x - m) / (mad * 1.4826)


def zscore(signal, axis=None):
    mu = np.nanmean(signal, axis=axis)
    s = np.nanstd(signal, axis=axis)
    z = np.zeros_like(signal)

    if signal.ndim == 1:
        return (signal - mu) / s

    for ii, s_ii in enumerate(s):
        if s_ii > 0:
            z[ii] = (signal[ii] - mu[ii]) / s_ii
        else:
            z[ii] = signal[ii] - mu[ii]
    return z


def rzscore(x, axis=None):
    m = np.nanmedian(x, axis=axis)

    if axis is not None:
        m = np.expand_dims(m, axis=axis)

    mad = np.nanmedian(np.abs(x - m))

    return (x - m) / (mad * 1.4826)


def mannwhitney_z(x, y, return_all=False):
    "only operates on axis=1"

    assert x.ndim == y.ndim
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    n_comps = x.shape[1]
    u = np.zeros(n_comps)
    p = np.zeros(n_comps)
    n1 = np.zeros(n_comps)
    n2 = np.zeros(n_comps)

    for ii in range(n_comps):
        xii = x.iloc[:, ii].dropna()
        yii = y.iloc[:, ii].dropna()
        n1[ii] = len(xii)
        n2[ii] = len(yii)
        u[ii], p[ii] = stats.mannwhitneyu(xii, yii)

    m = n1 * n2 / 2
    s = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    z = (u - m) / s
    if return_all:
        return z, u, p, n1, n2
    else:
        return z


def mannwhitney_u(x, y, axis=None):
    return stats.mannwhitneyu(x, y, axis=axis)[0]


def sig_stats(signal):
    """ sig_stats
        function that returns various signal statistics:
        std, mad, min, max

    Parameters
    ----------
    signal   : input numpy array (1D)

    Returns
    -------
    out      : a dictionary with the above signal statistics

    """
    out = {}
    out['std'] = "{0:0.3e}".format(np.std(signal))
    out['mad'] = "{0:0.3e}".format(mad(signal))
    out['min'] = "{0:0.3e}".format(np.nanmin(signal))
    out['max'] = "{0:0.3e}".format(np.nanmax(signal))
    return out


def get_simple_regression_se(x, y, y_hat):
    """
    method to obtain the standard error of an OLS
    :param x: array, length(n) indepedent var
    :param y: array, length(n) dependent var
    :param y_hat: array, length(n) predicted var
    :return: se_b0, se_b1: floats with the standard errors from a linear regression of the form y_hat = b0 + b1*x

    source: https://people.duke.edu/~rnau/mathreg.htm
    the values match what I get using statsmodels.api.OLS for a single parameter regresion.
    """

    # pre allocation of some variables to be used
    n = len(x)
    n_sr = np.sqrt(n)
    var_x = np.var(x)

    # standard error of the model
    s = np.std(y - y_hat) * np.sqrt((n - 1) / (n - 2))

    # standard error of the coefficients
    se_b0 = s / n_sr * np.sqrt(1 + x.mean() ** 2 / var_x)
    se_b1 = s / n_sr * 1 / np.sqrt(var_x)

    return se_b0, se_b1


def get_r2(y, y_hat):
    """
    Coefficient of determination for vectors.
    :param y: array n_samps of dependent variable;
        can also work with matrices organized as n_samps x n_predictions
    :param y_hat: array with result of a prediction, same shape as y
    :return: R2. float. or array if y,y_hat are matrices. in that case r2 is an array of length n_predictions
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)
    y_bar = y.mean(axis=1)
    y_bar = y_bar.reshape(-1, 1)

    return 1 - ((y - y_hat) ** 2).sum(axis=1) / ((y - y_bar) ** 2).sum(axis=1)


def get_ar2(y, y_hat, p, r2=None):
    """
    Adjusted coefficient of determination for number of parameters used in the prediction.
    :param y: array n_samps of dependent variable;
        can also work with matrices organized as n_samps x n_predictions
    :param y_hat: array with result of a prediction, same shape as y
    :param p: number of parameters used in the estimation, excluding bias
    :return: aR2: flaot, adjusted R2. or array if y,y_hat are matrices.
                in that case r2 is an array of length n_predictions
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)
    n = y.shape[1]

    if r2 is None:
        r2 = get_r2(y, y_hat)

    return 1 - (n - 1) / (n - p - 1) * (1 - r2)


def get_poisson_deviance(y, y_hat):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)

    if np.all(y >= 0):
        try:
            # the implementation below is for numerical stability.
            with np.errstate(divide='ignore'):
                dev_per_samp = np.log(y ** y) - np.log(y_hat ** y) - y + y_hat

            # for invalid samples [y_hat<=0], make deviance equal to y
            invalid_samps = y_hat <= 0
            dev_per_samp[invalid_samps] = y[invalid_samps]

            return 2 * np.nanmean(dev_per_samp, axis=1)
        except:
            return np.nan
    else:
        return np.nan


def get_poisson_d2(y, y_hat):
    """
    Pseudo coefficient of determination for a poisson glm. This implementation is the deviance square.
    :param y: array n_samps of dependent variable;
        can also work with matrices organized as n_samps x n_predictions
    :param y_hat: array with result of a prediction, same shape as y
    :return: D2. float. or array if y,y_hat are matrices. in that case r2 is an array of length n_predictions
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)

    n_samps = y.shape[1]
    y = y.astype(int)  # y should be integers, otherwise will run into numerical errors
    y_bar = np.tile(y.mean(axis=1), (n_samps, 1)).T
    d = get_poisson_deviance(y, y_hat)
    d_null = get_poisson_deviance(y, y_bar)
    d2 = 1 - d / d_null
    return d2


def get_poisson_pearson_chi2(y, y_hat):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)
    return np.nansum((y - y_hat) ** 2 / y_hat, axis=1)


def get_poisson_ad2(y, y_hat, p, d2=None):
    """
    Pseudo adjusted coefficient of determination for a poisson glm. This implementation is the deviance square.
    :param y: array n_samps of dependent variable;
        can also work with matrices organized as n_samps x n_predictions
    :param y_hat: array with result of a prediction, same shape as y
    :param p: number of parameters used in the estimation, excluding bias
    :return: aD2. float. or array if y,y_hat are matrices. in that case r2 is an array of length n_predictions
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)
    n = y.shape[1]

    if d2 is None:
        d2 = get_poisson_d2(y, y_hat)

    ad2 = 1 - (n - 1) / (n - p - 1) * (1 - d2)

    return ad2


def get_mse(y, y_hat):
    """
    Mean Square Error Calculation
    :param y: array n_samps of dependent variable;
        can also work with matrices organized as n_samps x n_predictions
    :param y_hat: array with result of a prediction, same shape as y
    :return: mse: float. mean square error. or array of length n_predictions
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)
    return np.mean((y - y_hat) ** 2, axis=1)


def get_rmse(y, y_hat):
    """
    Root Mean Square Error Calculation
    :param y: array n_samps of dependent variable;
        can also work with matrices organized as n_samps x n_predictions
    :param y_hat: array with result of a prediction, same shape as y
    :return: rmse: float. root mean square error, or array of length n_predictions
    """
    return np.sqrt(get_mse(y, y_hat))


def get_nrmse(y, y_hat):
    """
    Normalized Root Mean Square Error Calculation.
    Divides the RMSE by the mean of variable y
    :param y: array n_samps of dependent variable
    :param y_hat: array of n_samps as a result of a prediction
    :return: nrmse: float. normalize root mean square error, or array of length n_predictions
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)
    return get_rmse(y, y_hat) / np.mean(y, axis=1)


def permutation_test(function, x, y=None, n_perm=500, alpha=0.02, seed=0, n_jobs=-1, **function_params):
    """
    Permutation test.
    :param function: of the form func(x) -> float, or func(x,y) -> : eg. rs.spearman, np.mean_fr_map##
    :param x: array. first variable
    :param y: array. second variable (must be same length as x)
    :param n_perm: number of permutations
    :param alpha: double sided alpha level
    :param seed: random seed
    :return: outside_dist: bool, indicates if the real value is outside the
        permutated distribution at level alpha
        perm_out: array, length n_perm. distribution of func(x_p,y), where x_p is a random
        permuation of x.
    """
    np.random.seed(seed)
    perm_out = np.zeros(n_perm)

    # create dummy function to absorb the possibility of a bivariate function. note that only variable x is permuted.
    if y is None:
        def func2(x2, **kwargs):
            return function(x2, **kwargs)
    else:
        def func2(x2, y2=y, **kwargs):
            return function(x2, y2, **kwargs)

    if x.ndim > 1:
        def p_worker():
            x_p = np.random.permutation(x.flatten()).reshape(*x.shape)
            res = func2(x_p, **function_params)
            return res
    else:
        def p_worker():
            x_p = np.random.permutation(x)
            res = func2(x_p, **function_params)
            return res

    real_out = func2(x, **function_params)

    with Parallel(n_jobs=n_jobs) as parallel:
        perm_out = parallel(delayed(p_worker)() for _ in range(n_perm))

    perm_out = np.array(perm_out)
    loc = (perm_out >= real_out).mean()

    outside_dist = loc <= alpha / 2 or loc >= 1 - alpha / 2
    return outside_dist, perm_out


def get_discrete_data_mat(data, bin_edges):
    """
    :param data: array float of data to be discretrize into a matrix
    :param bin_edges: bin edges to discretize the data
    :return: design_matrix: ndarray n_samps x n_bins of binary data
                            entry ij indicates that data sample i is in the jth bin
            data_bin_ids: array n_samps of ints,
                          ith value is the bin_center that gets assign to data[i]
    """

    n_samps = len(data)
    n_bins = len(bin_edges) - 1
    data_bin_ids = np.digitize(data, bins=bin_edges) - 1  # shift ids so that first bin center corresponds to id 0

    design_matrix = np.zeros((n_samps, n_bins))
    for i in range(n_bins):
        design_matrix[:, i] = data_bin_ids == i

    return design_matrix, data_bin_ids


def spearman(x, y):
    """spearman correlation"""
    return stats.spearmanr(x, y, nan_policy='omit')[0]


def kendall(x, y):
    """kendall correlation"""
    return stats.kendalltau(x, y, nan_policy='omit')[0]


def pearson(x, y):
    """pearsons correlation"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        r = stats.pearsonr(x, y)[0]
    return r


def circ_mean(x):
    """
    circular mean
    :param x: array of angles in radians
    :return:
    """
    return np.angle(np.nansum(np.exp(x*1j)))

def circ_corr(x,y):
    """
    circular correlation
    :param x:
    :param y:
    :return:
    """
    sx = np.sin(x-circ_mean(x))
    sy = np.sin(y-circ_mean(y))
    r = np.nansum(sx*sy) / np.sqrt( np.nansum(sx**2) * np.nansum(sy**2))
    return r


def circ_corrcl(x, y):
    """Correlation coefficient between one circular and one linear variable
    random variables.

    Parameters
    ----------
    x : 1-D array_like
        First circular variable (expressed in radians).
        The range of ``x`` must be either :math:`[0, 2\\pi]` or
        :math:`[-\\pi, \\pi]`. If ``angles`` is not
        expressed in radians (e.g. degrees or 24-hours), please use the
        :py:func:`pingouin.convert_angles` function prior to using the present
        function.
    y : 1-D array_like
        Second circular variable (linear)
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.

    Returns
    -------
    r : float
        Correlation coefficient
    pval : float
        Uncorrected p-value

    Notes
    -----
    Please note that NaN are automatically removed from datasets.

    Examples
    --------
    Compute the r and p-value between one circular and one linear variables.
    #
    # >>> from pingouin import circ_corrcl
    # >>> x = [0.785, 1.570, 3.141, 0.839, 5.934]
    # >>> y = [1.593, 1.291, -0.248, -2.892, 0.102]
    # >>> r, pval = circ_corrcl(x, y)
    # >>> print(round(r, 3), round(pval, 3))
    0.109 0.971

    # modified from pingouin
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.size == y.size, 'x and y must have the same length.'

    # Remove NA
    valid_samps = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[valid_samps]
    y = y[valid_samps]
    n = x.size

    # Compute correlation coefficent for sin and cos independently
    rxs = pearson(y, np.sin(x))
    rxc = pearson(y, np.cos(x))
    rcs = pearson(np.sin(x), np.cos(x))

    # Compute angular-linear correlation (equ. 27.47)
    r = np.sqrt((rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2))

    # Compute p-value
    # pval = chi2.sf(n * r**2, 2)
    # pval = pval / 2 if tail == 'one-sided' else pval
    return r


def get_regression_metrics(y, y_hat, n_params=1, reg_type='linear'):
    if y.ndim > 1:
        y_bar = np.nanmean(y, axis=1)
    else:
        y_bar = np.nanmean(y)

    if reg_type == 'poisson':
        r2 = get_poisson_d2(y, y_hat)
        ar2 = get_poisson_ad2(y, y_hat, n_params, d2=r2)
        err = get_poisson_deviance(y, y_hat)
        nerr = err / y_bar

    elif reg_type == 'linear':
        r2 = get_r2(y, y_hat)
        ar2 = get_ar2(y, y_hat, n_params, r2=r2)
        err = get_rmse(y, y_hat)
        nerr = err / y_bar

    else:
        print(f'method {reg_type} not implemented.')
        raise NotImplementedError

    out = {'r2': r2, 'ar2': ar2, 'err': err, 'n_err': nerr}

    return out


def resultant_vector_length(alpha, w=None, d=None, axis=None, axial_correction=1, ci=None, bootstrap_iter=None):
    # source: https://github.com/circstat/pycircstat/blob/master/pycircstat/descriptive.py
    """
    Computes mean resultant vector length for circular data.
    This statistic is sometimes also called vector strength.
    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length
    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain resultant vector length
    r = np.abs(cmean)
    # obtain mean
    mean = np.mod(np.angle(cmean), 2 * np.pi)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    # obtain variance
    variance = 1 - r
    std = np.sqrt(-2 * np.log(r))
    return r, mean, variance, std


def rayleigh(alpha, w=None, d=None, axis=None):
    """
    Computes Rayleigh test for non-uniformity of circular data.
    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle
    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!
    :param alpha: sample of angles in radian
    :param w:       number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return z:    value of the z-statistic
    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    # if axis is None:
    # axis = 0
    #     alpha = alpha.ravel()

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    r, mean, variance, std = resultant_vector_length(alpha, w=w, d=d, axis=axis)
    n = np.sum(w, axis=axis)

    # compute Rayleigh's R (equ. 27.1)
    R = n * r

    # compute Rayleigh's z (equ. 27.2)
    z = R ** 2 / n

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return pval, z


def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    # REQUIRED for mean vector length calculation
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
                                   str(w.shape) + " do not match!"

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            cmean = ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) / np.sum(w, axis=axis))
        except Warning as e:
            print('Could not compute complex mean for MVL calculation', e)
            cmean = np.nan
    return cmean


def compute_autocorr_2d(X):
    """
    Normalized 2D autocorrelation using the FFT approach.
    :param X: 2d np.ndarray
    :return: autocorrelation
    """

    assert isinstance(X, np.ndarray), 'input needs to be a numpy 2d array'
    assert X.ndim == 2, 'input needs to be a 2d array'

    n, m = X.shape

    H = np.fft.fft2(X, [2 * n - 1, 2 * m - 1])  # 2d FFT w zero padding
    H /= np.sqrt((H ** 2).sum())  # variance normalization

    acc = np.fft.fftshift(np.fft.ifft2(H * np.conjugate(H)))  # HxH* and shifts fft quadrants to the center
    acc = np.real(acc)  # takes care of numerical errors

    return acc


def split_timeseries(n_samps, samps_per_split=1500, n_data_splits=10, ):
    """
    Function that returns a time series of length of n_samps, and every samps_per_splits has a different id up to
    n_data_splits. That is, it segments a time series by integers. Useful for crossvalidation and wanting to maintain
    some temporal integrety between samples. If want more randomization, use the np.roll function to move the split ids.
    :param n_samps: number of samples in the time series
    :param samps_per_split: samples per split
    :param n_data_splits: number of splits/ids in the time series
    :return: time series of integers, each with the id of the split.
    """

    assert n_data_splits * samps_per_split <= n_samps, f"Error. Needs at least {n_data_splits * samps_per_split} samples" \
                                                       f" provided only {n_samps} "
    split_edges = np.append(np.arange(0, n_samps, samps_per_split), n_samps)
    n_ts_segments = len(split_edges) - 1

    ts_seg_id = np.zeros(n_samps, dtype=int)
    for ii in range(n_ts_segments):
        split_id = np.mod(ii, n_data_splits)
        ts_seg_id[split_edges[ii]:split_edges[ii + 1]] = split_id

    return ts_seg_id


def split_timeseries_data(data, n_splits=2, samp_rate=0.02, split_interval=30):
    """
    Function that divides time every split_interval samples
    :param data: dictionary of the time series to be split. if the time series do not have all the same the same number
        of samples the function will exit with an error.
    :param n_splits: number of splits to divide the data
    :param samp_rate: sampling rate [samps / seconds]
    :param split_interval: interval for each split in [seconds]
    :return: split_data: dictionary with the same keys as data, now each entry is the same ts as the original data but
     it is a numpy object indexed by the splits.
    """

    # check all time series match
    n_ts = len(data)
    ts_lengths = np.zeros(n_ts, dtype=int)
    cnt = 0
    for key, ts in data.items():
        ts_lengths[cnt] = max(ts.shape)  # time series dimension expected to be the longest^*
        cnt += 1
    assert np.all(ts_lengths[0] == ts_lengths), "Time series have different lengths."
    n_total_samps = ts_lengths[0]

    # obtain time series segment splits
    samps_per_split = int(split_interval / samp_rate)
    split_samps = split_timeseries(n_total_samps, samps_per_split=samps_per_split, n_data_splits=n_splits)

    split_data = {}
    for key, ts in data.items():  # for every timeseries in the data
        split_data[key] = np.empty(n_splits, dtype=object)
        dim = np.where(np.array(ts.shape) == n_total_samps)[0]  # find what dimension the time series is on

        if dim > 0:
            # if the ts is in dimension other than the first, reshape the ts temporarely to boolean find the the correct
            # segment indices
            ts_temp = np.moveaxis(ts, dim, 0)
            for split in range(n_splits):
                split_data[key][split] = np.moveaxis(ts_temp[split_samps == split], 0, dim)
        else:
            for split in range(n_splits):
                split_data[key][split] = ts[split_samps == split]

    return split_data


def kendall2pearson(tau):
    return np.sin(np.pi * tau * 0.5)


def fisher_r2z(r):
    return np.arctanh(r)


def bootstrap_corr(x, y, n_boot=100, corr_method='kendall'):
    n = len(x)
    assert n == len(y)

    if corr_method == 'kendall':
        r_func = kendall
    elif corr_method == 'spearman':
        r_func = spearman()
    else:
        r_func = pearson

    r = np.zeros(n_boot)
    for b in range(n_boot):
        b_idx = np.random.choice(n, n)
        r[b] = r_func(x[b_idx], y[b_idx])

    return r


def bootstrap_diff(x, y, n_boot=1000, seed=0):
    """
    bootstrap difference Algorithm 16.1 based on Efron,Tibshirani 1993
    :param x: 1d array
    :param y: 1d array
    :param n_boot: number of bootstrap samples to take
    :param seed: random seed
    :return:
        (1) true difference,
        (2) ASL (Achieved significance level)
        (3) distribution of bootstrap differences
    """
    np.random.seed(seed)
    assert x.ndim == 1
    assert x.ndim == y.ndim

    def _dif(_x, _y):
        return np.nanmean(_x) - np.nanmean(_y)

    n = len(x)
    m = len(y)
    p = n+m

    true_d = _dif(x, y)
    z = np.concatenate((x, y))
    boot_d = np.zeros(n_boot)
    for boot in range(n_boot):
        zb = np.random.choice(z, p)
        xb = zb[:n]
        yb = zb[n:]
        boot_d[boot] = _dif(xb, yb)

    ASL = (boot_d >= true_d).sum()/n_boot

    return true_d, ASL, boot_d


def bootstrap_tdiff(x, y, n_boot=500, seed=0):
    """
    bootstrap difference Algorithm 16.2 based on Efron,Tibshirani 1993
    :param x: 1d array
    :param y: 1d array
    :param n_boot: number of bootstrap samples to take
    :param seed: random seed
    :return:
        (1) true statistic
        (2) ASL (Achieved significance level)
        (3) distribution of bootstrap statistic
    """
    np.random.seed(seed)
    assert x.ndim == 1
    assert x.ndim == y.ndim

    n = len(x)
    m = len(y)

    def _dif(_x, _y):
        return np.nanmean(_x) - np.nanmean(_y)

    def _tstat(_x, _y):
        num = _dif(_x, _y)
        den = np.sqrt(np.nanvar(_x)/n + np.nanvar(_y)/m)
        return num/den

    true_t = _tstat(x, y)
    z = np.concatenate((x, y))

    xbar = np.nanmean(x)
    ybar = np.nanmean(y)
    zbar = np.nanmean(z)

    xtilde = x - xbar + zbar
    ytilde = y - ybar + zbar

    boot_t = np.zeros(n_boot)

    for boot in range(n_boot):
        xb = np.random.choice(xtilde, n)
        yb = np.random.choice(ytilde, m)

        boot_t[boot] = _tstat(xb, yb)

    ASL = 2*min((boot_t >= true_t).mean(), (boot_t < true_t).mean())

    return true_t, ASL, boot_t


def bootstrap_udiff(x,y, n_boot=500, seed=0):
    """
        bootstrap difference Algorithm 16.2 based on Efron,Tibshirani 1993
        - version using the mannwhitney u, instead of t-test.
        :param x: 1d array
        :param y: 1d array
        :param n_boot: number of bootstrap samples to take
        :param seed: random seed
        :return:
            (1) true statistic
            (2) ASL (Achieved significance level)
            (3) distribution of bootstrap statistic
        """
    np.random.seed(seed)
    assert x.ndim == 1
    assert x.ndim == y.ndim

    n = len(x)
    m = len(y)

    def _dif(_x, _y):
        return np.nanmean(_x) - np.nanmean(_y)

    true_t = mannwhitney_u(x, y)
    z = np.concatenate((x, y))

    xbar = np.nanmean(x)
    ybar = np.nanmean(y)
    zbar = np.nanmean(z)

    xtilde = x - xbar + zbar
    ytilde = y - ybar + zbar

    boot_t = np.zeros(n_boot)

    for boot in range(n_boot):
        xb = np.random.choice(xtilde, n)
        yb = np.random.choice(ytilde, m)

        boot_t[boot] = mannwhitney_u(xb, yb)

    ASL = (boot_t >= true_t).sum() / n_boot

    return true_t, ASL, boot_t


def compare_corrs(r1, r2, n1, n2, corr_method='kendall'):
    """
    takes pairs of correlations sets and compures a z statistic
    :param r1: array, first correlation array
    :param r2: array, second correlation array
    :param n1: int, # of entries used to compute r1
    :param n2: int, # of entries used to compute r2
    :param corr_method: str, correlation method
    """
    if corr_method == 'kendall':
        r1 = kendall2pearson(r1)
        r2 = kendall2pearson(r2)

    zd = (fisher_r2z(r1) - fisher_r2z(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    pd = pval_from_z(zd)
    return zd, pd


def combine_pvals(pvals, axis):
    """uses fisher's method of combining p vals that test for the same hypothesis"""
    n = len(pvals)
    return stats.chi2.sf(np.nansum(-2*np.log(pvals), axis=axis), df=n*2)


def pval_from_z(zval):
    return 2*stats.norm.sf(np.abs(zval))


def multiple_testing_correction(pvalues, correction_type="FDR"):
    """
    Consistent with R - print
    correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05,
                                          0.069, 0.07, 0.071, 0.09, 0.1])
    taken from:
    https://github.com/CoBiG2/cobig_misc_scripts/blob/master/FDR.py
    """
    pvalues = np.array(pvalues)
    sample_size = pvalues.shape[0]
    qvalues = np.empty(sample_size)
    if correction_type == "Bonferroni":
        # Bonferroni correction
        qvalues = sample_size * pvalues
    elif correction_type == "Bonferroni-Holm":
        # Bonferroni-Holm correction
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            qvalues[i] = (sample_size-rank) * pvalue
    elif correction_type == "FDR":
        # Benjamini-Hochberg, AKA - FDR test
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = sample_size - i
            pvalue, index = vals
            new_values.append((sample_size/rank) * pvalue)
        for i in range(0, int(sample_size)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            qvalues[index] = new_values[i]
    return qvalues

# def getDirZoneSpikeMaps(spikes, PosDat, sp_thr=[5, 2000]):
#     SegSeq = PosDat['SegDirSeq']
#
#     # subsample to get moving segments
#     valid_moving = getValidMovingSamples(PosDat['Speed'])
#     valid_samps = np.logical_and(valid_moving, SegSeq > 0)
#
#     MovSegSeq = np.array(SegSeq)
#     dir_seq = MovSegSeq[valid_samps] - 1
#     seqInfo = getSeqInfo(dir_seq, PosDat['step'])
#
#     dir_spikes = spikes[valid_samps]
#     spikesByZoneDir = getZoneSpikeMaps(dir_spikes, dir_seq)
#     return spikesByZoneDir, seqInfo
#
#
# def getSeqInfo(Seq, step=0.02):
#     fields = ['counts', 'time', 'prob']
#     seqInfo = pd.DataFrame(np.full((3, TMF.nZones), np.nan), columns=TMF.ZonesNames, index=fields)
#
#     counts = np.bincount(Seq.astype(int), minlength=TMF.nZones)
#     seqInfo.loc['counts'] = counts
#     seqInfo.loc['time'] = counts * step
#     seqInfo.loc['prob'] = counts / np.sum(counts)
#
#     return seqInfo
#
#
# def getTM_OccupationInfo(PosDat, spacing=25, occ_time_thr=0.1):
#     occ_counts, xed, yed = getPositionMat(PosDat['x'], PosDat['y'], TMF.x_limit, TMF.y_limit, spacing)
#     occ_time = occ_counts * PosDat['step']
#     occ_mask = occ_time >= occ_time_thr
#
#     OccInfo = {}
#     OccInfo['time'] = occ_time * occ_mask
#     OccInfo['counts'] = occ_counts * occ_mask
#     OccInfo['prob'] = occ_counts / np.nansum(occ_counts)
#     return OccInfo
