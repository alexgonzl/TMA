import numpy as np
from scipy.signal import filtfilt
from types import SimpleNamespace
import pandas as pd

import Analyses.spatial_functions as spatial_funcs
import Utils.filter_functions as filter_funcs


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
    speed, hd = spatial_funcs.get_velocity(x, y, p.vt_rate)

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
    pos_map_counts = spatial_funcs.get_position_map_counts(x_rs, y_rs, p.x_bin_edges_, p.y_bin_edges_)
    pos_counts_sm = spatial_funcs.get_smoothed_map(pos_map_counts, n_bins=p.spatial_window_size,
                                                   sigma=p.spatial_sigma)

    pos_valid_mask = pos_counts_sm >= p.occ_num_thr

    pos_map_secs = pos_map_counts * time_step
    pos_map_secs = spatial_funcs.get_smoothed_map(pos_map_secs, p.spatial_window_size, p.spatial_sigma)

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

    for unit in range(n_units):
        x_spk, y_spk = spatial_funcs.get_bin_spikes_xy(spikes[unit], of_dat.x, of_dat.y)
        spike_maps[unit] = spatial_funcs.get_position_map_counts(x_spk, y_spk, x_bin_edges, y_bin_edges)

    return spike_maps


def get_session_fr_maps(session_info):
    """
    Loops and computes smoothed fr maps for each unit.
    :param SubjectSessionInfo session_info: instance of class SubjectInfo for a particular subject
    :return: np.ndarray fr_maps: shape n_units x n_vertical_bins x n_horizontal_bins
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

        fr_maps[unit] = spatial_funcs.get_smoothed_map(temp_fr_map, n_bins=track_params.spatial_window_size,
                                                       sigma=track_params.spatial_sigma)

    return fr_maps


def get_session_fr_maps_cont(session_info):
    """
    Loops and computes smoothed fr maps for each unit. This version uses a weighted 2d histogram, such that each
    x,y sample is weighted by the continuous firing rate of the neuron.
    :param SubjectSessionInfo session_info: instance of class SubjectInfo for a particular subject
    :return: np.ndarray fr_maps: shape n_units x n_vertical_bins x n_horizontal_bins
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
        temp_fr_map = spatial_funcs.get_weighted_position_map(x, y, fr[unit], x_bin_edges, y_bin_edges)
        temp_fr_map[valid_mask] /= pos_counts_sm[valid_mask]

        fr_maps[unit] = spatial_funcs.get_smoothed_map(temp_fr_map, n_bins=track_params.spatial_window_size,
                                                       sigma=track_params.spatial_sigma)

    return fr_maps


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
    :return: dict: with all the scores
    """
    # get data
    fr = session_info.get_fr()
    spike_maps = session_info.get_spike_maps()
    fr_maps = session_info.get_fr_maps()

    of_dat = SimpleNamespace(**session_info.get_track_data())
    track_params = SimpleNamespace(**session_info.task_params)

    # preallocate output dictionary
    output_dir = {x: {} for x in ['sp', 'ha', 'hd', 'border', 'grid', 'si']}

    # speed scores
    n_units = session_info.n_units
    scores, model_coef, model_coef_s = \
        spatial_funcs.get_speed_score_discrete(of_dat.sp, fr,
                                               track_params.sp_bin_edges_,
                                               sig_alpha=track_params.sig_alpha,
                                               n_perm=track_params.n_perm)

    output_dir['sp']['scores'] = scores
    output_dir['sp']['model_coef'] = model_coef
    output_dir['sp']['model_coef_s'] = model_coef_s

    # head direction scores
    scores, model_coef, model_coef_s = \
        spatial_funcs.get_angle_score(of_dat.hd, fr,
                                      track_params.ang_bin_edges_,
                                      speed=of_dat.sp,
                                      min_speed=track_params.min_speed_thr,
                                      max_speed=track_params.max_speed_thr,
                                      sig_alpha=track_params.sig_alpha)

    output_dir['hd']['scores'] = scores
    output_dir['hd']['model_coef'] = model_coef
    output_dir['hd']['model_coef_s'] = model_coef_s

    # head angle scores
    scores, model_coef, model_coef_s, ang_bin_centers = \
        spatial_funcs.get_angle_score(of_dat.ha, fr,
                                      track_params.ang_bin_edges_,
                                      speed=of_dat.sp,
                                      min_speed=track_params.min_speed_thr,
                                      max_speed=track_params.max_speed_thr,
                                      sig_alpha=track_params.sig_alpha)

    output_dir['ha']['scores'] = scores
    output_dir['ha']['model_coef'] = model_coef
    output_dir['ha']['model_coef_s'] = model_coef_s

    # border scores
    scores, model_coef, model_coef_s, = \
        spatial_funcs.get_border_score(of_dat.x, of_dat.y, fr, fr_maps,
                                       track_params.x_cm_lims,
                                       track_params.y_cm_lims, track_params.cm_bin,
                                       sig_alpha=track_params.sig_alpha,
                                       n_perm=track_params.n_perm,
                                       border_fr_thr=track_params.border_fr_thr,
                                       min_field_size_bins=track_params.border_min_field_size_bins,
                                       border_width_bins=track_params.border_width_bins,
                                       non_linear=True)

    output_dir['border']['scores'] = scores
    output_dir['border']['model_coef'] = model_coef
    output_dir['border']['model_coef_s'] = model_coef_s

    return NotImplementedError

# def allOFBehavPlots(OFBehavDat):
#     sp = OFBehavDat['sp']
#     se = OFBehavDat['se']
#
#     f = plt.figure(figsize=(16, 18))
#
#     gsTop = mpl.gridspec.GridSpec(2, 3)
#     axTop = np.full((2, 3), type(gsTop[0, 0]))
#     for i in np.arange(2):
#         for j in np.arange(3):
#             axTop[i, j] = f.add_subplot(gsTop[i, j])
#     gsTop.tight_layout(f, rect=[0, 0.25, 1, 0.70])
#
#     # xy traces
#     axTop[0, 0].plot(OFBehavDat['x'], OFBehavDat['y'], linewidth=1, color='k', alpha=0.5)
#     axTop[0, 0].set_aspect('equal', adjustable='box')
#     axTop[0, 0].set_axis_off()
#
#     axTop[0, 1] = sns.heatmap(OFBehavDat['Occ_Counts'], xticklabels=[], yticklabels=[], cmap='magma', square=True,
#                               robust=True, cbar=False, ax=axTop[0, 1])
#     axTop[0, 1].invert_yaxis()
#
#     axTop[0, 2] = sns.heatmap(OFBehavDat['Occ_SmSecs'], xticklabels=[], cmap='magma', yticklabels=[], square=True,
#                               robust=True, cbar=False, ax=axTop[0, 2])
#     axTop[0, 2].invert_yaxis()
#
#     axTop[1, 0] = sns.heatmap(OFBehavDat['Occ_SmSecs'] > secThr, cmap='Greys', xticklabels=[], yticklabels=[],
#                               square=True, cbar=False, ax=axTop[1, 0])
#     axTop[1, 0].invert_yaxis()
#
#     axTop[1, 1] = sns.distplot(sp, ax=axTop[1, 1])
#     axTop[1, 1].set_yticklabels([])
#
#     axTop[1, 2] = sns.distplot(OFBehavDat['HAo'], ax=axTop[1, 2])
#     axTop[1, 2].set_yticklabels([])
#
#     titles = ['xy', 'mm/bin={}; maxCnt={}'.format(mm2bin, OFBehavDat['Occ_Counts'].max()),
#               'max s/bin = {0:0.2f}'.format(OFBehavDat['Occ_Counts'].max()),
#               'secThr = {}'.format(secThr), 'Speed [cm/s]', 'HA orig [deg]']
#
#     cnt = 0
#     for a in axTop.flatten():
#         a.set_title(titles[cnt])
#         cnt += 1
#
#     gsBot = mpl.gridspec.GridSpec(1, 4)
#     axBot = np.full(4, type(gsBot))
#     for i in np.arange(4):
#         axBot[i] = f.add_subplot(gsBot[i], projection='polar')
#     gsBot.tight_layout(f, rect=[0, 0, 1, 0.25])
#
#     for i in [0, 1]:
#         if i == 0:
#             txt = 'HA'
#         else:
#             txt = 'HD'
#
#         th = OFBehavDat[txt][sp > spThr]
#         stats = OFBehavDat[txt + '_Stats']
#         counts = stats['counts']
#         bins = stats['bins']
#         ang = stats['ang']
#         r = stats['r']
#         R = stats['R']
#         pval = stats['pval']
#
#         axBot[0 + i * 2].plot(bins, np.append(counts, counts[0]), linewidth=4)
#         axBot[0 + i * 2].plot([0, ang], [0, r], color='k', linewidth=4)
#         axBot[0 + i * 2].scatter(ang, r, s=50, color='red')
#         # axBot[0+i*2].set_title('r={0:0.1f},th={1:0.1f},R={2:0.2f},p={3:0.2f}'.format(r,np.rad2deg(ang),R,pval) )
#         axBot[0 + i * 2].set_xticklabels(['$0^o$', '', '$90^o$', '', '$180^o$'])
#         axBot[0 + i * 2].set_yticks([])
#         axBot[0 + i * 2].text(0, -0.1,
#                               'r={0:0.1f},th={1:0.1f},R={2:0.2f},p={3:0.2f}'.format(r, np.rad2deg(ang), R, pval),
#                               transform=axBot[0 + i * 2].transAxes)
#
#         counts2 = counts * timeStep
#         colors = plt.cm.magma(counts2 / counts2.max())
#         axBot[1 + i * 2].bar(bins[:-1], counts2, width=ang2bin, color=colors, bottom=counts2.min())
#         axBot[1 + i * 2].set_axis_off()
#         cax = getColBar(axBot[1 + i * 2], counts2)
#         cax.yaxis.set_label('sec')
#
#     ax = plt.axes([0, 0.23, 1, 0.1])
#     ax.text(0, 0, 'HD', {'fontsize': 20})
#     ax.plot([0, 0.45], [-0.05, -0.05], color='k', linewidth=4)
#     ax.text(0.5, 0, 'HA', {'fontsize': 20})
#     ax.plot([0.5, 0.95], [-0.05, -0.05], color='k', linewidth=4)
#     ax.set_axis_off()
#     ax.set_xlim([0, 1])
#     ax.set_ylim([-.1, 1])
#
#     ax = plt.axes([0, 0.7, 1, 0.05])
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.text(0, 0, se, {'fontsize': 30}, transform=ax.transAxes)
#
#     ax.set_axis_off()
#     return f
#
#
# def getColBar(ax, values, cmap='magma'):
#     pos = ax.get_position()
#     cax = plt.axes([pos.x0 + pos.width * 0.85, pos.y0, 0.05 * pos.width, 0.2 * pos.height])
#
#     cMap = mpl.colors.ListedColormap(sns.color_palette(cmap, 50))
#     vmax = values.max()
#     vmin = values.min()
#     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
#     mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cMap)
#
#     mapper.set_array([])
#     cbar = plt.colorbar(mapper, ticks=[vmin, vmax], cax=cax)
#     cax.yaxis.set_tick_params(right=False)
#     return cax
