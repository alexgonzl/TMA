import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, get_window


def median_window_filter(x, window):
    """ moving median filter that can take np.nan as entries.
        note that the filter is non-causal, output of sample ii is the median
        of samples of the corresponding window centered around ii.
    Inputs:
        x       ->  signal to filtered,
        window  ->  number of samples to use for median estimation.

    Output:
        y       <-  median filtered signal
    """
    if window % 2:
        window = window - 1
    win2 = np.int(window / 2)
    n = len(x)
    y = np.array(x)
    for ii in np.arange(win2, n - win2 + 1):
        try:
            idx = (np.arange(-win2, win2) + ii).astype(np.int)
            y[ii] = np.nanmedian(y[idx])
        except:
            pass
    return y


def median_window_filtfilt(x, window):
    """ moving median filter that can take np.nan as entries.
        note that the filter is non-causal, output of sample ii is the median
        of samples of the corresponding window centered around ii. This repeats
        function repeats the median operation in the reverse direction.
    Inputs:
        x       ->  signal to filtered,
        window  ->  number of samples to use for median estimation.

    Output:
        y       <-  median filtered signal
    """
    if window % 2:
        window = window - 1
    win2 = np.int(window / 2)
    N = len(x)
    y = np.array(x)
    for ii in np.arange(win2, N - win2 + 1):
        try:
            idx = (np.arange(-win2, win2) + ii).astype(np.int)
            y[ii] = np.nanmedian(y[idx])
        except:
            pass
    for ii in np.arange(N - win2, win2 - 1, -1):
        try:
            idx = (np.arange(-win2, win2) + ii).astype(np.int)
            y[ii] = np.nanmedian(y[idx])
        except:
            pass
    return y


def angle_filtfilt(angle, filter_coef, radians=False):
    """
    :param np.array angle: 1d array of angles
    :param filter_coef: filter coefficients
    :param bool radians:
    :return: filter angle signal
    """
    if not radians:
        angle = np.deg2rad(angle)

    # get components of the angle
    x = np.cos(angle)
    y = np.sin(angle)

    # make sure window conserves energy in the signal
    filter_coef /= filter_coef.sum()
    xf = filtfilt(filter_coef, 1, x)
    yf = filtfilt(filter_coef, 1, y)

    # normalize to unit length
    c = np.sqrt(xf ** 2 + yf ** 2)
    yf /= c
    xf /= c

    angle_f = angle_xy(xf, yf)

    if radians:
        return angle_f
    else:
        return np.mod(np.rad2deg(angle_f), 360)


def resample_signal(t_orig, t_new, x):
    """
    Nearest neighbor interpolation for resampling.
    :param np.array t_orig: original time series corresponding to x
    :param np.array t_new: resampled time series for the output
    :param np.array x: signal to be resampled (same length as t_orig)
    :return: np.array of x_new. estimation for signal at t_news
    """
    sig_ip = interp1d(t_orig, x, kind="nearest", fill_value="extrapolate")
    return sig_ip(t_new)


def angle_xy(x, y):
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


def get_last_not_nan_value(x, i):
    """
    Recursively looks back on an array x until it finds a not-nan value
    :param x: np.array
    :param i: index of array
    :return: last_non nan value. if none, returns zero. ow. recursively runs again
    """
    if i == 0:
        return 0
    elif np.isnan(x[i]):
        return get_last_not_nan_value(x, i - 1)
    else:
        return x[i]
