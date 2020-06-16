import numpy as np


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


def movmad(x, window):
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


def movstd(x, window):
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
