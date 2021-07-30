import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
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

def median_window_filter_causal(x, window):
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
        window = window + 1
    n = len(x)
    y = x.copy()
    window_range = np.arange(0, window).astype(int)
    for ii in np.arange(window, n - window + 1):
        try:
            idx = window_range+ii
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

def fill_nan_vals(x:np.ndarray, method='last'):

    x_out = x.copy()
    nan_ids = np.where(np.isnan(x_out))[0]

    if method =='last':
        for ii in nan_ids:
            x_out[ii] = get_last_not_nan_value(x_out, ii)
    elif method=='next':
        r_nan_ids = np.flip(nan_ids)
        for ii in r_nan_ids:
            x_out[ii] = get_next_non_nan_value(x_out, ii)
    else:
        x_l = fill_nan_vals(x, 'last')
        x_n = fill_nan_vals(x, 'next')
        x_out = (x_l+x_n)/2
    return x_out

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

def get_next_non_nan_value(x,i):

    if i == len(x):
        return 0
    elif np.isnan(x[i]):
        return get_next_non_nan_value(x,i+1)
    else:
        return x[i]



def get_sos_filter_bank(f_types, fs=32000.0, hp_edge_freq=None, lp_edge_freq=None, sp_edge_freq=None, notch_freq=None,
                        notch_harmonics=2, notch_q=20, gpass=0.2, gstop=60.0):
    """
    Function that creates default filters
    fTypes -> list of filters that have to be ['HP', 'LP', 'Notch', 'Sp'].
    fs -> integer, sampling frequency in samps/s defualt to 32000
    *_EdgeFreq -> edge frequencies for each filter type.
        Defaults are: HP-2Hz, LP-5000Hz, Sp->300Hz [High Pass], Notch->60Hz (3 Harmonics)
    Notch_Harmonics -> int, # of harmonics from Notch_Freq to filter [as default]
    gpass -> allowed oscillation gain in the pass bands [def. 0.2dB ~ to up to 1.03 multiplication of the pass band  ]
    gstop -> required suppresion in the stopband [def. 60dB ~ at least to 0.001 multiplication of the stop band  - ]
    returns SOS a N sections x 6 second order sections filter matrix.
    """

    SOS = np.zeros((0, 6))
    for f in f_types:
        if f not in ['HP', 'LP', 'Notch', 'Sp']:
            print('filter type {} not supported.'.format(f))
            print('skipping filter.')

        # settings for low pass and bandpass
        if f in ['LP', 'HP']:
            if f is 'LP':
                if lp_edge_freq is None:
                    cut_freq = 5000.0
                    cut_buffer = 5500.0
                else:
                    cut_freq = lp_edge_freq
                    cut_buffer = lp_edge_freq + lp_edge_freq * 0.1
            elif f is 'HP':
                if hp_edge_freq is None:
                    cut_freq = 2.0
                    cut_buffer = 0.2
                else:
                    cut_freq = hp_edge_freq
                    cut_buffer = hp_edge_freq * 0.1

            sos = scipy.signal.iirdesign(cut_freq / (fs / 2), cut_buffer / (fs / 2), gpass, gstop, output='sos')
            SOS = np.vstack((SOS, sos))

        if f is 'Notch':

            n_notches = notch_harmonics + 1

            if notch_freq is None:
                cut_freq = np.arange(1, n_notches + 1) * 60.0
            else:
                cut_freq = np.arange(1, n_notches + 1) * notch_freq

            if notch_q is None:
                q = np.array(cut_freq)  # changing Quality factor to keep notch bandwidth constant.

            elif type(notch_q) is np.ndarray:
                if len(notch_q) >= n_notches:
                    q = np.array(notch_q)
                # if length of quality factor array don't match the number of harmonics default to the first one
                elif len(notch_q) < n_notches:
                    q = np.ones(n_notches) * notch_q[0]
            else:
                # Q = np.ones(nNotches)*Notch_Q
                q = np.arange(1, n_notches + 1) * notch_q

            for i, notch in enumerate(cut_freq):
                b, a = scipy.signal.iirnotch(notch, q[i], fs=fs)
                sos = scipy.signal.tf2sos(b, a)
                SOS = np.vstack((SOS, sos))

        if f is 'Sp':
            if sp_edge_freq is None:
                cut_freq = 350.0
                cut_buffer = 300.0
            else:
                cut_freq = sp_edge_freq
                cut_buffer = sp_edge_freq - sp_edge_freq * 0.1
            sos = scipy.signal.iirdesign(cut_freq / (fs / 2), cut_buffer / (fs / 2), gpass, gstop, output='sos')
            SOS = np.vstack((SOS, sos))

    zi = scipy.signal.sosfilt_zi(SOS)
    return SOS, zi
