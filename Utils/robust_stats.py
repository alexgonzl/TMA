import numpy as np
from scipy import stats
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


def robust_zscore(signal):
    """ robust_zscore
        function that uses median and median absolute deviation to standard the
        input vector

    Parameters
    ----------
    x       : input numpy array (1D)

    Returns
    -------
    z       : standarized vector with zero median and std ~1 (without outliers)

    """
    return (signal - np.nanmedian(signal)) / (mad(signal) * 1.4826)


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
    s = np.std(y-y_hat) * np.sqrt((n - 1) / (n - 2))

    # standard error of the coefficients
    se_b0 = s/n_sr * np.sqrt(1 + x.mean() ** 2 / var_x)
    se_b1 = s/n_sr * 1 / np.sqrt(var_x)

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


def get_ar2(y, y_hat, p):
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
    return 1 - (1 - get_r2(y, y_hat)) * (n - 1) / (n - p - 1)


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


def permutation_test(x, y, func, n_perm=500, alpha=0.02, seed=0):
    """
    Permutation test.
    :param x: array. first variable
    :param y: array. second variable (must be same length as x)
    :param func: of the form func(x,y) -> float
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
    real_out = func(x, y)

    for p in range(n_perm):
        x_p = np.random.permutation(x)
        perm_out[p] = func(x_p, y)

    loc = (perm_out >= real_out).mean()

    outside_dist = loc <= alpha / 2 or loc >= 1 - alpha / 2
    return outside_dist, perm_out


def spearman(x, y):
    """spearman correlation"""
    return stats.spearmanr(x, y)[0]


def kendall(x, y):
    """kendall correlation"""
    return stats.kendalltau(x, y)[0]


def pearson(x, y):
    """pearsons correlation"""
    return stats.pearsonr(x, y)[0]


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
    mean = np.angle(cmean)

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
