import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

import warnings

from pathlib import Path
import h5py
import sys


def get_smoothed_map(fr_map, n_bins=4, sigma=1):
    """
    :param fr_map: firing rate matrix. array in which each cell corresponds to the firing at that xy position
    :param n_bins: number of smoothing bins
    :param sigma: std for the gaussian smoothing
    :return: fr_map: smoothed map. note that this is a truncated sigma map, meaning that high or
            low fr wont affect far away bins
    """
    fr_smoothed = ndimage.filters.median_filter(fr_map, n_bins)
    trunc = (((n_bins - 1) / 2) - 0.5) / sigma

    return ndimage.filters.gaussian_filter(fr_smoothed, sigma, mode='constant', truncate=trunc)


def get_position_mat(x, y, x_lims, y_lims, spacing):
    """
    :param np.array x: x position of the animal
    :param np.array y: y position of the animal
    :param x_lims: 2 element list or array of x limits
    :param y_lims: 2 element list or array of y limits
    :param spacing: how much to bin the space
    :return: 2d array of position counts, x_edges, and y_edges
    """
    x_edges = np.arange(x_lims[0], x_lims[1] + 1, spacing)
    y_edges = np.arange(y_lims[0], y_lims[1] + 1, spacing)
    position_2d, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    return position_2d, x_edges, y_edges


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


def get_spike_map(bin_spikes, x, y, x_lims, y_lims, spacing=3):
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
    spike_map, _, _ = get_position_mat(x_spk, y_spk, x_lims, y_lims, spacing)
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


def get_movement_samps(speed, speed_lims=None):
    """
    :param np.array speed: speed for each time bin. expects to be in cm/s
    :param speed_lims: 2 element tuple/list/array with min/max for valid movement speeds
    :return: np.array bool: array of time samples that are within the speed limits
    """
    if speed_lims is None:
        speed_lims = [5, 2000]
    return np.logical_and(speed >= speed_lims[0], speed <= speed_lims[1])


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


def rotate_xy(x, y, angle):
    """
    :param x:
    :param y:
    :param angle:
    :return: rotated coordinates x,y
    """
    x2 = x * np.cos(angle) + y * np.sin(angle)
    y2 = -x * np.sin(angle) + y * np.cos(angle)
    return x2, y2


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


def get_angle_hist(th, step, th_units='deg'):
    if th_units == 'deg':
        th = np.deg2rad(th)
        step = np.deg2rad(step)

    counts, bins = np.histogram(th, np.arange(-np.pi, np.pi + 0.01, step))
    bin_centers = bins[:-1] + step
    return counts, bin_centers, bins


def angle_stats(angle, step=10):
    counts, bin_centers, bins = get_angle_hist(angle, step)
    z = np.mean(counts * np.exp(1j * bin_centers))
    ang = np.angle(z)
    r = np.abs(z)
    p, t = rayleigh(angle)

    stats = {'r': r, 'ang': ang, 'R': t, 'p_val': p, 'counts': counts, 'bins': bins, 'bin_centers': bin_centers}

    return stats


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