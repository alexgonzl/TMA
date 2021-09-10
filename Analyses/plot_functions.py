import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats

import Pre_Processing.pre_process_functions as pp_funcs
import Utils.filter_functions as filt_funcs

import seaborn as sns
from seaborn.utils import remove_na
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.text import Text
from matplotlib import transforms, lines
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties

from shapely.geometry import Point
from shapely.geometry.polygon import LinearRing, Polygon
from collections import Counter
from descartes import PolygonPatch

from Analyses import experiment_info as si
from Analyses import tree_maze_functions as tmf


# import spatial_tuning as ST
# import stats_functions as StatsF

################################################################################
# Figure Classes
################################################################################

class Fig1:
    fontsize = 7

    fig_w = 6.2
    fig_ratio = 1.7

    wspace = 0.02
    hspace = 0.08

    a_x0 = 0.02
    abc_w = 0.3
    abc_h = 0.45

    c_x0 = 0.05 + a_x0
    c_h = 0.35
    c_y0 = 0.1

    de_h = 0.33
    de_w = 0.22
    d_y0 = 0.12
    de_x0 = abc_w * 2 + wspace * 5

    panel_locs = dict(a=[0, abc_h + hspace, abc_w, abc_h],
                      b=[abc_w + wspace, abc_h + hspace, abc_w, abc_h],
                      c=[c_x0, c_y0, abc_w * 2 - c_x0, c_h],
                      d=[de_x0, abc_h + hspace * 2, de_w, de_h],
                      e=[de_x0, d_y0, de_w, de_h])

    label_fontsize = 9
    label_base_loc_x = 0.02
    label_base_loc_y = 0.98
    label_row2_y = 0.5
    subject_palette = 'deep'

    params = dict(figsize=(fig_w, fig_w / fig_ratio), dpi=1000, fontsize=fontsize,
                  panel_a={'lw': 0.4},
                  panel_b={'lw': 0.2, 'line_alpha': 1, 'line_color': '0.2', 'sub_seg_lw': 0.05,
                           'n_trials': 5, 'max_dur': 700, 'marker_alpha': 1, 'marker_size': 8,
                           'fontsize': fontsize, 'seed': 1, 'leg_fontsize': fontsize - 1,
                           'leg_markersize': 2, 'leg_lw': 0.8,
                           'cue': np.array([['L', 'R'], ['L', 'R']]), 'dec': np.array([['L', 'R'], ['L', 'L']]),
                           'goal': np.array([[3, 2], [4, -1]]), 'long_trial': np.array([[0, 0], [1, -1]])},
                  panel_c={'h_lw': 0.3, 'v_lw': 1, 'samp_buffer': 20, 'trial_nums': np.arange(13, 17),
                           'fontsize': fontsize},
                  panel_d={'min_n_units': 1, 'min_n_trials': 50, 'fontsize': fontsize, 'palette': subject_palette,
                           'marker_alpha': 0.7, 'marker_swarm_size': 1.5,
                           'y_ticks': np.array([0, .25, .50, .75, 1]),
                           'box_plot_lw': 0.75, 'box_plot_median_lc': '0.4', 'box_plot_median_lw': 0.75,
                           'summary_marker_size': 5, 'scale': 0.7},
                  panel_e={'min_n_units': 1, 'min_n_trials': 50, 'fontsize': fontsize, 'palette': subject_palette,
                           'marker_alpha': 0.9, 'marker_swarm_size': 1.5,
                           'y_ticks': [0, 15, 30, 45],
                           'box_plot_lw': 0.75, 'box_plot_median_lc': '0.4', 'box_plot_median_lw': 0.75,
                           'summary_marker_size': 5, 'scale': 0.7}
                  )

    def __init__(self, session='Li_T3g_070618', **kargs):
        self.params.update(kargs)
        self.tree_maze = tmf.TreeMazeZones()

        # session info and data for panels b and c.
        subject = session.split('_')[0]
        self.session_info = si.SubjectSessionInfo(subject, session)
        self.session_behav = self.session_info.get_event_behavior()
        self.session_track_data = self.session_info.get_track_data()
        self.session_pos_zones_mat = self.session_info.get_pos_zones_mat()

        # summary data for panels d and c.
        self.summary_info = si.SummaryInfo()

    def fig_layout(self):
        f = plt.figure(constrained_layout=False,
                       figsize=self.params['figsize'],
                       dpi=self.params['dpi'])

        # define figure layout
        f_ax = np.zeros(5, dtype=object)

        labels = ['a', 'b', 'c', 'd', 'e']
        for ii, label in enumerate(labels):
            f_ax[ii] = f.add_axes(self.panel_locs[label])

        label_ax = f.add_axes([0, 0, 1, 1])
        label_locs = dict(a=(self.label_base_loc_x, self.label_base_loc_y),
                          b=(self.label_base_loc_x + self.abc_w + self.wspace, self.label_base_loc_y),
                          c=(self.label_base_loc_x, self.label_row2_y),
                          d=(self.label_base_loc_x + self.abc_w * 2 + self.wspace, self.label_base_loc_y),
                          e=(self.label_base_loc_x + self.abc_w * 2 + self.wspace, self.label_row2_y)
                          )
        for label in labels:
            label_ax.text(label_locs[label][0], label_locs[label][1], label, transform=label_ax.transAxes,
                          fontsize=self.label_fontsize, fontweight='bold', va='top', ha='left')
        label_ax.axis("off")
        return f, f_ax

    def plot_all(self):

        f, ax = self.fig_layout()

        panels = ['a', 'b', 'c', 'd', 'e']
        for ii, p in enumerate(panels):
            obj = getattr(self, f"panel_{p}")
            obj(ax[ii])

        return f

    def panel_a(self, ax=None, **params):
        fig_params = dict(seg_color=None, zone_labels=True, sub_segs=None, tm_layout=True, plot_cue=True,
                          fontsize=self.fontsize,
                          lw=self.params['panel_a']['lw'])
        fig_params.update(params)

        if ax is None:
            f, ax = plt.subplots(figsize=(2, 2), dpi=600)
        else:
            f = ax.figure

        ax = self.tree_maze.plot_maze(axis=ax, **fig_params)

        # legend axis
        leg_ax = f.add_axes(ax.get_position())
        leg_ax.axis("off")

        cue_w = 0.1
        cue_h = 0.1

        cues_p0 = dict(right=np.array([0.65, 0.15]),
                       left=np.array([0.25, 0.15]))

        text_strs = dict(right=r"$H \rightarrow D \rightarrow G_{1,2}$",
                         left=r"$G_{3,4} \leftarrow D \leftarrow H$")

        txt_ha = dict(right='left', left='right')

        txt_hspace = 0.05
        txt_pos = dict(right=cues_p0['right'] + np.array((0, -txt_hspace)),
                       left=cues_p0['left'] + np.array((cue_w, -txt_hspace)))

        for cue in ['right', 'left']:
            cue_p0 = cues_p0[cue]
            cue_coords = np.array([cue_p0, cue_p0 + np.array((0, cue_h)),
                                   cue_p0 + np.array((cue_w, cue_h)), cue_p0 + np.array((cue_w, 0)), ])
            cue_poly = Polygon(cue_coords)
            plot_poly(cue_poly, ax=leg_ax, lw=0, alpha=0.9, color=self.tree_maze.split_colors[cue])

            leg_ax.text(txt_pos[cue][0], txt_pos[cue][1], text_strs[cue], fontsize=fig_params['fontsize'],
                        horizontalalignment=txt_ha[cue], verticalalignment='center')
        leg_ax.set_xlim(0, 1)
        leg_ax.set_ylim(0, 1)

        leg_ax.text(cues_p0['right'][0] + cue_w, cues_p0['right'][1] + cue_h // 2,
                    'Right Cue', fontsize=fig_params['fontsize'],
                    horizontalalignment='left', verticalalignment='bottom')
        leg_ax.text(cues_p0['left'][0], cues_p0['left'][1] + cue_h // 2,
                    'Left Cue', fontsize=fig_params['fontsize'],
                    horizontalalignment='right', verticalalignment='bottom')

    def panel_b(self, ax=None, **params):
        behav = self.session_behav
        track_data = self.session_track_data

        if ax is None:
            f, a1 = plt.subplots(2, 2, figsize=(1, 1), dpi=600)
        else:
            f = ax.figure

            p = ax.get_position()
            x0, y0, w, h = p.x0, p.y0, p.width, p.height
            w2 = w / 2
            h2 = h / 2

            a1 = np.zeros((2, 2), dtype=object)
            a1[1, 0] = f.add_axes([x0, y0, w2, h2])
            a1[1, 1] = f.add_axes([x0 + w2, y0, w2, h2])
            a1[0, 0] = f.add_axes([x0, y0 + h2, w2, h2])
            a1[0, 1] = f.add_axes([x0 + w2, y0 + h2, w2, h2])
            ax.axis("off")

        fig_params = self.params['panel_b']
        fig_params.update(params)

        np.random.seed(fig_params['seed'])

        cue = fig_params['cue']
        goal = fig_params['goal']
        dec = fig_params['dec']
        long_trial = fig_params['long_trial']

        n_trials = fig_params['n_trials']
        max_dur = fig_params['max_dur']
        lw = fig_params['lw']
        line_alpha = fig_params['line_alpha']
        marker_alpha = fig_params['marker_alpha']
        marker_size = fig_params['marker_size']

        H_loc = self.tree_maze.well_coords['H']
        for row in range(2):
            for col in range(2):
                dec_full = 'left' if dec[row, col] == 'L' else 'right'

                _ = self.tree_maze.plot_maze(axis=a1[row, col],
                                             seg_color=None, zone_labels=False, seg_alpha=0.1,
                                             plot_cue=True, cue_color=cue[row, col],
                                             fontsize=self.fontsize, lw=fig_params['lw'],
                                             line_color=fig_params['line_color'],
                                             sub_segs='all', sub_seg_color='None', sub_seg_lw=fig_params['sub_seg_lw'])

                sub_table = behav.trial_table[(behav.trial_table.dec == dec[row, col]) &
                                              (behav.trial_table.cue == cue[row, col]) &
                                              (behav.trial_table.dur <= max_dur) &
                                              (behav.trial_table.long == long_trial[row, col]) &
                                              (behav.trial_table.goal == goal[row, col])
                                              ]

                # noinspection PyTypeChecker
                sel_trials = np.random.choice(sub_table.index, size=n_trials, replace=False)

                a1[row, col].scatter(H_loc[0], H_loc[1], s=marker_size, marker='o', lw=0, color='k')
                marker_end_color = 'b' if dec[row, col] == cue[row, col] else 'r'

                if goal[row, col] > 0:
                    G_loc = self.tree_maze.well_coords[f"G{goal[row, col]}"]
                    a1[row, col].scatter(G_loc[0], G_loc[1], s=marker_size, marker='d',
                                         lw=0, alpha=marker_alpha, color=marker_end_color)
                else:
                    incorrect_wells = self.tree_maze.split_segs[dec_full]['goals']
                    for iw in incorrect_wells:
                        iw_loc = self.tree_maze.well_coords[iw]
                        a1[row, col].scatter(iw_loc[0], iw_loc[1], s=marker_size, marker='d',
                                             lw=0, alpha=marker_alpha, color=marker_end_color)

                for trial in sel_trials:
                    t0 = sub_table.loc[trial, 't0']
                    tE = sub_table.loc[trial, 'tE']
                    a1[row, col].plot(track_data.loc[t0:tE, 'x'], track_data.loc[t0:tE, 'y'],
                                      color=self.tree_maze.split_colors[dec_full], lw=lw, alpha=line_alpha)

        # plt.subplots_adjust(hspace=0.02, wspace=0, left=0.02, right=0.96, top=0.98, bottom=0)

        legend_elements = [mpl.lines.Line2D([0.1], [0.1], color='g', lw=fig_params['leg_lw'], label='L Dec'),
                           mpl.lines.Line2D([0], [0], color='purple', lw=fig_params['leg_lw'], label='R Dec'),
                           mpl.lines.Line2D([0], [0], marker='o', color='k', lw=0, label='Start',
                                            markerfacecolor='k', markersize=fig_params['leg_markersize']),
                           mpl.lines.Line2D([0], [0], marker='d', color='b', lw=0, label='Correct',
                                            markerfacecolor='b', markersize=fig_params['leg_markersize']),
                           mpl.lines.Line2D([0], [0], marker='d', color='r', lw=0, label='Incorrect',
                                            markerfacecolor='r', markersize=fig_params['leg_markersize'])]

        ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=[0, 0.05, 1, 1], frameon=False,
                  fontsize=fig_params['leg_fontsize'], labelspacing=0.2)

    def panel_c(self, ax=None, **params):
        pos_zones_mat = self.session_info.get_pos_zones_mat()
        behav = self.session_behav
        track_data = self.session_track_data

        if ax is None:
            f, ax = plt.subplots(figsize=(2, 1), dpi=600)
        else:
            f = ax.figure

        c_params = self.params['panel_c']
        c_params.update(params)

        self.tree_maze.plot_zone_ts_window(pos_zones_mat, trial_table=behav.trial_table,
                                           t=track_data.t.values - track_data.t[0],
                                           trial_nums=c_params['trial_nums'], samp_buffer=c_params['samp_buffer'],
                                           h_lw=c_params['h_lw'], v_lw=c_params['v_lw'], fontsize=self.fontsize,
                                           ax=ax)

    def panel_d(self, ax=None, **params):

        perf = self.summary_info.get_behav_perf()
        subjects = self.summary_info.subjects

        subset = perf[
            (perf.n_units >= self.summary_info.min_n_units) & (perf.n_trials >= self.summary_info.min_n_trials)]

        fig_params = self.params['panel_d']
        fig_params.update(params)
        fontsize = fig_params['fontsize']

        if ax is None:
            f, ax = plt.subplots(figsize=(1, 1), dpi=600)
        else:
            f = ax.figure

        # ax = reduce_ax(ax, fig_params['scale'])
        ax_pos = ax.get_position()

        x0, y0, w, h = ax_pos.x0, ax_pos.y0, ax_pos.width, ax_pos.height
        x_split = w * 0.75
        ax.set_position([x0, y0, x_split, h])

        sns.set_theme(context='paper', style="whitegrid", font_scale=1, palette=fig_params['palette'])
        sns.set_style(rc={"axes.edgecolor": '0.3',
                          'xtick.bottom': True,
                          'ytick.left': True})

        sns.boxplot(ax=ax, x='subject', y='pct_correct', data=subset, color='w', linewidth=fig_params['box_plot_lw'],
                    whis=100)
        sns.swarmplot(ax=ax, x='subject', y='pct_correct', data=subset, size=fig_params['marker_swarm_size'],
                      **{'alpha': fig_params['marker_alpha']})

        for line in ax.get_lines()[4::len(subjects)]:
            line.set(**{'color': fig_params['box_plot_median_lc'],
                        'lw': fig_params['box_plot_median_lw']})

        ax.set_ylim(0, 1.01)
        ax.set_yticks(fig_params['y_ticks'])
        ax.set_yticklabels((fig_params['y_ticks'] * 100).astype(int), fontsize=fontsize)
        ax.set_ylabel('% Correct Decision', fontsize=fontsize)

        ax.set_xticklabels([f"s$_{{{ii}}}$" for ii in range(1, len(subjects) + 1)], fontsize=fontsize)
        ax.set_xlabel('Subjects', fontsize=fontsize)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        ax.tick_params(axis="both", direction="in", length=2, width=0.8, color="0.5", which='major', pad=1)
        ax.xaxis.set_label_coords(0.625, -0.1)

        # summary
        ax = f.add_axes([x0 + x_split, y0, w - x_split, h])
        subset2 = subset.groupby('subject').median()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        subset2['color'] = 0

        for ii, s in enumerate(subjects):
            subset2.loc[s, 'color'] = mpl.colors.to_hex(colors[ii])

        sns.boxplot(ax=ax, y='pct_correct', data=subset2.loc[subjects], color='w', whis=100, width=0.5,
                    linewidth=fig_params['box_plot_lw'])

        # hard-coded x location for no overalp
        x_locs = np.array([0, -1, 1, 0, -1, 0]) * 0.1
        ax.scatter(x_locs, subset2.loc[subjects, 'pct_correct'], lw=0, zorder=10,
                   s=fig_params['summary_marker_size'],
                   c=subset2.loc[subjects, 'color'],
                   alpha=fig_params['marker_alpha'])

        for line in ax.get_lines()[4::len(subjects)]:
            line.set(**{'color': fig_params['box_plot_median_lc'],
                        'lw': fig_params['box_plot_median_lw']})

        ax.set_ylim(0, 1.01)
        ax.set_yticks(fig_params['y_ticks'])
        ax.set_yticklabels('')
        ax.set_ylabel('')

        ax.set_xticklabels([r" $\bar s$ "], fontsize=fontsize)

        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

        ax.tick_params(axis="both", direction="in", length=2, width=0.8, color="0.5", which='major', pad=1)
        ax.tick_params(axis='y', left=False)

    def panel_e(self, ax=None, **params):
        perf = self.summary_info.get_behav_perf()
        subjects = self.summary_info.subjects

        subset = perf[
            (perf.n_units >= self.summary_info.min_n_units) & (perf.n_trials >= self.summary_info.min_n_trials)]

        fig_params = self.params['panel_e']
        fig_params.update(params)
        fontsize = fig_params['fontsize']

        if ax is None:
            f, ax = plt.subplots(figsize=(1, 1), dpi=600)
        else:
            f = ax.figure

        # ax = reduce_ax(ax, fig_params['scale'])
        ax_pos = ax.get_position()
        x0, y0, w, h = ax_pos.x0, ax_pos.y0, ax_pos.width, ax_pos.height
        x_split = w * 0.75
        ax.set_position([x0, y0, x_split, h])

        sns.set_theme(context='paper', style="whitegrid", font_scale=1, palette=fig_params['palette'])
        sns.set_style(rc={"axes.edgecolor": '0.3',
                          'xtick.bottom': True,
                          'ytick.left': True})

        sns.boxplot(ax=ax, x='subject', y='n_units', data=subset, color='w', linewidth=fig_params['box_plot_lw'],
                    whis=100)
        sns.swarmplot(ax=ax, x='subject', y='n_units', data=subset, size=fig_params['marker_swarm_size'],
                      **{'alpha': fig_params['marker_alpha']})

        for line in ax.get_lines()[4::len(subjects)]:
            line.set(**{'color': fig_params['box_plot_median_lc'],
                        'lw': fig_params['box_plot_median_lw']})

        ax.set_ylim(0, 50)
        ax.set_yticks(fig_params['y_ticks'])
        ax.set_yticklabels(fig_params['y_ticks'], fontsize=fontsize)
        ax.set_ylabel('# units ', fontsize=fontsize)

        ax.set_xticklabels([f"s$_{{{ii}}}$" for ii in range(1, len(subjects) + 1)], fontsize=fontsize)
        ax.set_xlabel('Subjects', fontsize=fontsize)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        ax.tick_params(axis="both", direction="in", length=2, width=0.8, color="0.5", which='major', pad=1)
        ax.xaxis.set_label_coords(0.625, -0.1)

        # summary
        ax = f.add_axes([x0 + x_split, y0, w - x_split, h])
        subset2 = subset.groupby('subject').median()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        subset2['color'] = 0

        for ii, s in enumerate(subjects):
            subset2.loc[s, 'color'] = mpl.colors.to_hex(colors[ii])

        sns.boxplot(ax=ax, y='n_units', data=subset2.loc[subjects], color='w', whis=100, width=0.5,
                    linewidth=fig_params['box_plot_lw'])

        # hard-coded x location for no overalp
        x_locs = np.array([0, 0, 0, 0, 0, -1]) * 0.1
        ax.scatter(x_locs, subset2.loc[subjects, 'n_units'], lw=0, zorder=10,
                   s=fig_params['summary_marker_size'],
                   c=subset2.loc[subjects, 'color'],
                   alpha=fig_params['marker_alpha'])

        for line in ax.get_lines()[4::len(subjects)]:
            line.set(**{'color': fig_params['box_plot_median_lc'],
                        'lw': fig_params['box_plot_median_lw']})

        ax.set_ylim(0, 50)
        ax.set_yticks(fig_params['y_ticks'])
        ax.set_yticklabels('')
        ax.set_ylabel('')

        ax.set_xticklabels([r" $\bar s$ "], fontsize=fontsize)

        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

        ax.tick_params(axis="both", direction="in", length=2, width=0.8, color="0.5", which='major', pad=1)
        ax.tick_params(axis='y', left=False)


################################################################################
# Plot Functions
################################################################################
def plot_poly(poly, ax, alpha=0.3, color='g', lw=1.5, line_alpha=1, line_color='0.5'):
    p1x, p1y = poly.exterior.xy
    ax.plot(p1x, p1y, color=line_color, linewidth=lw, alpha=line_alpha)
    ring_patch = PolygonPatch(poly, fc=color, ec='none', alpha=alpha)
    ax.add_patch(ring_patch)


def reduce_ax(ax, scale):
    " reduces the axes dimensions by multiplying by scale"

    assert scale < 1, "Scale needs to be < 1"
    ax_pos = ax.get_position()
    x0, y0, w, h = ax_pos.x0, ax_pos.y0, ax_pos.width, ax_pos.height
    wp = w * scale
    hp = h * scale
    wspace = w - wp
    hspace = h - hp

    x0p = x0 + wspace / 2
    y0p = y0 + hspace / 2

    ax.set_position([x0p, y0p, wp, hp])
    return ax
# def plotCounts(counts, names,ax):
#     nX = len(names)
#     ab=sns.barplot(x=np.arange(nX),y=counts,ax=ax, ci=[],facecolor=(0.4, 0.6, 0.7, 1))
#     ax.set_xticks(np.arange(nX))
#     ax.set_xticklabels(names)
#     sns.set_style("whitegrid")
#     sns.despine(left=True)
#     for tick in ax.get_xticklabels():
#         tick.set_rotation(45)
#     return ax
#
# def plotBehWindow(time,dat,names,ax):
#     sns.heatmap(dat,ax=ax,yticklabels=names,cbar=0,cmap='Greys_r',vmax=1.1)
#     ax.hlines(np.arange(len(names)+1), *ax.get_xlim(),color=(0.7,0.7,0.7,1))
#     x=ax.get_xticks().astype(int)
#     x=np.linspace(x[0],x[-1], 6, endpoint=False).astype(int)
#     x=x[1::]
#     ax.set_xticks(x)
#     ax.vlines(x,*ax.get_ylim(),color=(0.3,0.3,0.3,1),linestyle='-.')
#     _=ax.set_xticklabels(np.round(time[x]).astype(int))
#     for tick in ax.get_xticklabels():
#         tick.set_rotation(45)
#     return ax
#
# def plotBehavZonesWindowAndSpikes(time,behav,bin_spikes):
#     f,(a1,a2)=plt.subplots(2,1, figsize=(12,12))
#     a1.set_position([0.125, 0.4, 0.775, 0.4])
#     a1=plotBehWindow(time,behav,TMF.ZonesNames,a1)
#     a1.set_xticks([])
#
#     nCells = bin_spikes.shape[0]
#     a2=plotBehWindow(time,bin_spikes,np.arange(nCells).astype(str),a2)
#     a2.set_xlabel('Time[s]')
#     a2.set_ylabel('Cell Number')
#
#     a2.set_position([0.125, 0.2, 0.775, 0.18])
#
#     yt=np.linspace(0,nCells,5, endpoint=False).astype(int)
#     a2.set_yticks(yt)
#     a2.set_yticklabels(yt.astype(str))
#     for tick in a2.get_yticklabels():
#         tick.set_rotation(0)
#     return f,a1,a2
#
# def plotTM_Trace(ax,x,y,bin_spikes=[], plot_zones=1, plot_raw_traces=0):
#     if plot_zones:
#         for zo in TMF.MazeZonesGeom.keys():
#             plotPoly(TMF.MazeZonesGeom[zo],ax,alpha=0.2)
#         for spine in plt.gca().spines.values():
#             spine.set_visible(False)
#     if plot_raw_traces:
#         ax.scatter(x,y,0.3,marker='D',color=[0.3,0.3,0.4],alpha=0.05)
#     if len(bin_spikes)==len(x):
#         ax.scatter(x,y,s=bin_spikes, alpha=0.1, color = 'r')
#     ax.set_axis_off()
#     ax.set_xlim(TMF.x_limit)
#     ax.set_ylim(TMF.y_limit)
#
#     return ax
#
# def plotZonesHeatMap(ax,cax,data,zones=TMF.ZonesNames,cmap='div',alpha=1,colArray=[]):
#     if len(colArray)==0:
#         if cmap=='div':
#             cDat,colArray =  getDatColorMap(data)
#             cMap = plt.get_cmap('RdBu_r')
#         else:
#             cDat,colArray =  getDatColorMap(data,col_palette='YlOrBr_r',div=False)
#             cMap = plt.get_cmap('YlOrBr_r')
#     else:
#         if cmap=='div':
#             cDat,_ =  getDatColorMap(data,colArray=colArray)
#             cMap = plt.get_cmap('RdBu_r')
#         else:
#             cDat,_ =  getDatColorMap(data,colArray=colArray,col_palette='YlOrBr_r',div=False)
#             cMap = plt.get_cmap('YlOrBr_r')
#     cnt=0
#     for zo in zones:
#         plotPoly(TMF.MazeZonesGeom[zo],ax,color=cDat[cnt],alpha=alpha)
#         cnt+=1
#     for spine in plt.gca().spines.values():
#         spine.set_visible(False)
#     ax.set_axis_off()
#     ax.set_xlim(TMF.x_limit)
#     ax.set_ylim(TMF.y_limit)
#     ax.axis('equal')
#
#     cNorm = mpl.colors.Normalize(vmin=colArray[0],vmax=colArray[-1])
#     sm = plt.cm.ScalarMappable(cmap=cMap,norm=cNorm)
#     sm.set_array([])
#
#     cbar = plt.colorbar(sm,cax=cax)
#     cax.yaxis.set_tick_params(right=False)
#
#     return ax,cax
#
# def getDatColorMap(data, nBins = 25, col_palette="RdBu_r",div=True,colArray=[]):
#
#     if len(colArray)>0:
#         nBins = len(colArray)
#     else:
#         if div:
#             maxV = np.ceil(np.max(np.abs(data))*100)/100
#             colArray = np.linspace(-maxV,maxV,nBins-1)
#         else:
#             maxV = np.ceil(np.max(data)*100)/100
#             minV = np.ceil(np.min(data)*100)/100
#             colArray = np.linspace(minV,maxV,nBins-1)
#
#     x = np.digitize(data,colArray).astype(int)
#     colMap = np.array(sns.color_palette(col_palette, nBins))
#     return colMap[x],colArray
#
# def plotHeatMap(ax,cax,img,cmap='viridis', colbar_label = 'FR [sp/s]', smooth=True,robust=False,w=4,s=1):
#     if smooth:
#         img = ST.getSmoothMap(img,w,s)
#     with sns.plotting_context(font_scale=2):
#         ax=sns.heatmap(img.T,xticklabels=[],yticklabels=[],ax=ax,square=False, robust=robust, cbar_ax=cax, cmap=cmap,cbar_kws={'label': colbar_label})
#         ax.invert_yaxis()
#     return ax
#
# def plotMaze_XY(x,y):
#     f,a1=plt.subplots(1,1, figsize=(10,10))
#     sns.set_style("white")
#     for zo in TMF.MazeZonesGeom.keys():
#         plotPoly(TMF.MazeZonesGeom[zo],a1)
#
#     for spine in plt.gca().spines.values():
#         spine.set_visible(False)
#     #a1.plot(PosDat['x'],PosDat['y'],alpha=0.1,color='k',linewidth=0.1)
#     a1.scatter(x,y,20, alpha=0.005,color='k')
#     a1.set_xlabel('x-position [mm]')
#     a1.set_ylabel('y-position [mm]')
#     a1.axis('equal')
#     #a1.grid()
#     a1.hlines(-150,-800,800,color=(0.7,0.7,0.7),linewidth=2)
#     a1.vlines(-850,0,1400,color=(0.7,0.7,0.7),linewidth=2)
#     a1.set_xlim([-800,800])
#     a1.set_xticks([-800,0,800])
#     a1.set_yticks([0,700,1400])
#     a1.set_ylim([-160,1500])
#     return f
#
# def plotMazeZoneCounts(PosMat):
#     f,a1=plt.subplots(1,1, figsize=(12,6))
#     with sns.axes_style("whitegrid"):
#         counts = np.sum(PosMat)
#         a1 = plotCounts(counts/1000,TMF.ZonesNames,a1)
#         a1.set_xlabel('Animal Location')
#         a1.set_ylabel('Sample Counts [x1000]')
#     return f
#
# def plotEventCounts(EventMat):
#     f,a1=plt.subplots(1,1, figsize=(12,6))
#     ev_subset = ['RH','RC','R1','R2','R3','R4','DH','DC','D1','D2','D3','D4','CL','CR']
#     counts = np.sum(EventMat[ev_subset]/1000,0)
#     with sns.axes_style("whitegrid"):
#         a1 = plotCounts(counts,ev_subset,a1)
#         a1.set_yscale('log')
#         a1.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
#         a1.set_yticks([1,10])
#         a1.set_yticklabels([1,10])
#         a1.set_xlabel('Event Type')
#         a1.set_ylabel('Sample Counts [x1000]')
#
#     return f
#
# def plotSpikeWFs(wfi,plotStd=0,ax=None):
#     wfm = wfi['mean']
#     wfstd = wfi['std']
#
#     if (ax is None):
#         f,ax = plt.subplots(1,figsize=(6,4))
#
#     nSamps,nChan = wfm.shape
#     x = np.arange(nSamps)
#     ax.plot(x,wfm,lw=3,alpha=0.9)
#     ax.get_yaxis().set_ticklabels([])
#     if plotStd:
#         for ch in np.arange(nChan):
#             plt.fill_between(x,wfm[:,ch]-wfstd[:,ch],wfm[:,ch]+wfstd[:,ch],alpha=0.1)
#
#     plt.legend(['ch'+str(ch) for ch in np.arange(nChan)],loc='best',frameon=False)
#     if nSamps==64:
#         ax.get_xaxis().set_ticks([0,16,32,48,64])
#         ax.get_xaxis().set_ticklabels(['0','','1','','2'])
#         ax.set_xlabel('Time [ms]')
#     #ax.text(0.65,0.1,'mFR={0:.2f}[sp/s]'.format(wfi['mFR']),transform=ax.transAxes)
#     ax.set_title('WaveForms mFR={0:.2f}[sp/s]'.format(wfi['mFR']))
#     return ax
#
# def plotRateMap(binSpikes, PosDat, OccInfo, cbar = False, ax=None):
#     spikesByPos = ST.getPosBinSpikeMaps(binSpikes,PosDat)
#     FR_ByPos = ST.getPosBinFRMaps(spikesByPos,OccInfo['time'])
#
#     if (ax is None):
#         f,ax = plt.subplots(1,figsize=(4,4))
#     cmap = 'viridis'
#     colbar_label = 'FR [sp/s]'
#     smooth =  True
#     robust = False
#     w =4
#     s=1
#     ax.axis('equal')
#     pos = ax.get_position()
#     if cbar:
#         cax = plt.axes([pos.x0+pos.width,pos.y0,0.05*pos.width,0.3*pos.height])
#     if smooth:
#         FR_ByPos = ST.getSmoothMap(FR_ByPos,w,s)
#     maxFR = np.max(FR_ByPos)
#     with sns.plotting_context(font_scale=1):
#         if cbar:
#             ax=sns.heatmap(FR_ByPos.T,xticklabels=[],yticklabels=[],ax=ax,square=False, robust=robust, cbar_ax=cax, cmap=cmap,cbar_kws={'label': colbar_label})
#         else:
#             #ax=sns.heatmap(FR_ByPos.T,xticklabels=[],yticklabels=[],ax=ax,square=False, robust=robust, cbar=False, cmap=cmap)
#             #ax=sns.heatmap(FR_ByPos.T,xticklabels=[],yticklabels=[],ax=ax,square=False, robust=robust, cbar=False, cmap=cmap)
#             ax=sns.heatmap(FR_ByPos.T,xticklabels=[],yticklabels=[],ax=ax,square=True, robust=robust, cbar=False, cmap=cmap, vmin=0, vmax=maxFR*0.9)
#             #ax.text(0.7,0.12,'{0:.2f}[Hz]'.format(maxFR),color='w',transform=ax.transAxes)
#
#         ax.invert_yaxis()
#     ax.set_title('Rate Map: maxFR {0:.2f}Hz'.format(maxFR))
#     return ax
#
# def plotISIh(wfi,ax=None):
#     x = wfi['isi_h'][1][1:]
#     h = wfi['isi_h'][0]
#     #h = h/np.sum(h)
#
#     if (ax is None):
#         f,ax = plt.subplots(1,figsize=(4,3))
#
#
#     ax.bar(x,h,color=[0.3,0.3,0.4],alpha=0.8)
#     ax.set_xlabel('ISI [ms]')
#     #ax.text(0.7,0.7,'CV={0:.2f}'.format(wfi['cv']),transform=ax.transAxes)
#     ax.set_yticklabels([''])
#     ax.set_title('ISI Hist')
#     return ax
#
# def plotTracesSpikes(PosDat,spikes,ax=None):
#     if (ax is None):
#         f,ax = plt.subplots(1,figsize=(4,4))
#
#     x = PosDat['x']
#     y = PosDat['y']
#     ax.scatter(x,y,0.2,marker='D',color=np.array([0.3,0.3,0.3])*2,alpha=0.05)
#     if len(spikes)==len(x):
#         ax.scatter(x,y,s=spikes, alpha=0.1, color = 'r')
#     ax.set_axis_off()
#     ax.set_xlim(TMF.x_limit)
#     ax.set_ylim(TMF.y_limit)
#     ax.set_title('Spike Traces')
#     return ax
#
# def plotZoneAvgMaps(ZoneAct,vmax = None,ax=None):
#     if (ax is None):
#         f,ax = plt.subplots(1,figsize=(6,6))
#
#     ax.axis('equal')
#     pos = ax.get_position()
#     cax = plt.axes([pos.x0+pos.width*0.78,pos.y0,0.05*pos.width,0.3*pos.height])
#
#     #cDat,colArray =  PF.getDatColorMap(ZoneAct)
#     #cMap = plt.get_cmap('RdBu_r')
#     cMap=mpl.colors.ListedColormap(sns.diverging_palette(250, 10, s=90, l=50,  n=50, center="dark"))
#     if vmax is None:
#         minima = np.min(ZoneAct)
#         maxima = np.max(ZoneAct)
#         vmax = np.max(np.abs([minima,maxima]))
#     norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax, clip=True)
#     mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cMap)
#
#     cnt=0
#     for zo in TMF.ZonesNames:
#         #PF.plotPoly(TMF.MazeZonesGeom[zo],ax,color=cDat[cnt],alpha=1)
#         plotPoly(TMF.MazeZonesGeom[zo],ax,color=mapper.to_rgba(ZoneAct[cnt]),alpha=1)
#
#         cnt+=1
#     for spine in plt.gca().spines.values():
#         spine.set_visible(False)
#
#     ax.set_axis_off()
#     ax.set_xlim(TMF.x_limit)
#     ax.set_ylim(TMF.y_limit)
#     ax.axis('equal')
#
# #     cNorm = mpl.colors.Normalize(vmin=colArray[0],vmax=colArray[-1])
# #     sm = plt.cm.ScalarMappable(cmap=cMap,norm=cNorm)
#     mapper.set_array([])
#
#     cbar = plt.colorbar(mapper,cax=cax)
#     cax.yaxis.set_tick_params(right=False)
#     #cax.get_yticklabels().set_fontsize(10)
#
#     return ax,cax
#
# def plotTrial_IO(frVector,trDat,ax=None):
#     if (ax is None):
#         f,ax = plt.subplots(1,figsize=(4,4))
#
#     cellDat = trDat.copy()
#     cellDat.loc[:,'zFR'] = frVector
#     subset = cellDat['Co']=='Co'
#
#     dat =[]
#     dat = cellDat[subset].groupby(['trID','IO','Cue','Desc']).mean()
#     dat = dat.reset_index()
#
#     pal = sns.xkcd_palette(['spring green','light purple'])
#     with sns.color_palette(pal):
#         ax=sns.violinplot(y='zFR',x='IO',hue='Desc',data=dat,split=True, ax=ax,
#                           scale='count',inner='quartile',hue_order=['L','R'],saturation=0.5,order=['Out','In','O_I'])
#     pal = sns.xkcd_palette(['emerald green','medium purple'])
#     with sns.color_palette(pal):
#         ax=sns.stripplot(y='zFR',x='IO',hue='Desc',data=dat,dodge=True,hue_order=['L','R'],alpha=0.7,ax=ax,
#                          edgecolor='gray',order=['Out','In','O_I'])
#
#     l=ax.get_legend()
#     l.set_visible(False)
#     ax.set_xlabel('Direction')
#
#     return ax
#
# def plotTrial_Desc(frVector,trDat,ax=None):
#     if (ax is None):
#         f,ax = plt.subplots(1,figsize=(4,4))
#
#     cellDat = trDat.copy()
#     cellDat.loc[:,'zFR'] = frVector
#     subset= cellDat['IO']=='Out'
#
#     dat = []
#     dat = cellDat[subset].groupby(['trID','Cue','Co','Desc']).mean()
#     dat = dat.reset_index()
#
#     pal = sns.xkcd_palette(['spring green','light purple'])
#     with sns.color_palette(pal):
#         ax=sns.violinplot(y='zFR',x='Desc',hue='Cue',data=dat,split=True,scale='width',ax=ax,
#                           inner='quartile',order=['L','R'],hue_order=['L','R'],saturation=0.5)
#     pal = sns.xkcd_palette(['emerald green','medium purple'])
#     with sns.color_palette(pal):
#         ax=sns.stripplot(y='zFR',x='Desc',hue='Cue',data=dat,dodge=True,order=['L','R'],ax=ax,
#                             hue_order=['L','R'],alpha=0.7,edgecolor='gray')
#
#     #
#     ax.set_xlabel('Decision')
#     #ax.set_ylabel('')
#
#     l=ax.get_legend()
#     handles, labels = ax.get_legend_handles_labels()
#     l.set_visible(False)
#     #plt.legend(handles[2:],labels[2:],bbox_to_anchor=(1.05, 0), borderaxespad=0.,frameon=False,title='Cue')
#
#     #plt.legend(handles[2:],labels[2:],loc=(1,1), borderaxespad=0.,frameon=False,title='Cue')
#
#     return ax,
#
# def plotLinearTraj(TrFRData,TrLongMat,savePath):
#
#     cellColIDs =  [i for i,item in enumerate(TrFRData.columns.values) if 'cell' in item]
#     nCells = len(cellColIDs)
#     muaColIDs =  [i for i,item in enumerate(TrFRData.columns.values) if 'mua' in item]
#     nMua = len(muaColIDs)
#     nTotalUnits = nCells+nMua
#     nUnits = {'cell':nCells,'mua':nMua}
#
#     cellCols = TrFRData.columns[cellColIDs]
#     muaCols = TrFRData.columns[muaColIDs]
#     unitCols = {'cell':cellCols,'mua':muaCols}
#
#     nMaxPos = 11
#     nMinPos = 7
#
#     pal = sns.xkcd_palette(['green','purple'])
#
#     cellDat = TrLongMat.copy()
#     cnt =0
#     for ut in ['cell','mua']:
#         for cell in np.arange(nUnits[ut]):
#             print('\nPlotting {} {}'.format(ut,cell))
#
#             cellDat.loc[:,'zFR'] = TrFRData[unitCols[ut][cell]]
#
#             f,ax = plt.subplots(2,3, figsize=(15,6))
#             w = 0.25
#             h = 0.43
#             ratio = 6.5/10.5
#             hsp = 0.05
#             vsp = 0.05
#             W = [w,w*ratio,w*ratio]
#             yPos = [vsp,2*vsp+h]
#             xPos = [hsp,1.5*hsp+W[0],2.5*hsp+W[1]+W[0]]
#             xlims = [[-0.25,10.25],[3.75,10.25],[-0.25,6.25]]
#             for i in [0,1]:
#                 for j in np.arange(3):
#                     ax[i][j].set_position([xPos[j],yPos[i],W[j],h])
#                     ax[i][j].set_xlim(xlims[j])
#
#             xPosLabels = {}
#             xPosLabels[0] = ['Home','SegA','Center','SegBE','Int','CDFG','Goals','CDFG','Int','CDFG','Goals']
#             xPosLabels[2] = ['Home','SegA','Center','SegBE','Int','CDFG','Goals']
#             xPosLabels[1] = xPosLabels[2][::-1]
#
#             plotAll = False
#             alpha=0.15
#             mlw = 1
#             with sns.color_palette(pal):
#                 coSets = ['InCo','Co']
#                 for i in [0,1]:
#                     if i==0:
#                         leg=False
#                     else:
#                         leg='brief'
#
#                     if plotAll:
#                         subset = (cellDat['IO']=='Out') & (cellDat['Co']==coSets[i]) & (cellDat['Valid'])
#                         ax[i][0] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',ci=None,data=cellDat[subset],
#                                  ax=ax[i][0],legend=False,lw=3,hue_order=['L','R'],style_order=['1','2','3','4'])
#                         ax[i][0] = sns.lineplot(x='Pos',y='zFR',hue='Desc',estimator=None,units='trID',data=cellDat[subset],
#                                 ax=ax[i][0],legend=False,lw=mlw,alpha=alpha,hue_order=['L','R'])
#
#                         subset = (cellDat['IO']=='In') & (cellDat['Co']==coSets[i]) & (cellDat['Pos']>=4) & (cellDat['Valid'])
#                         ax[i][1] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',ci=None,data=cellDat[subset],
#                                  ax=ax[i][1],legend=False,lw=3,hue_order=['L','R'],style_order=['1','2','3','4'])
#                         ax[i][1] = sns.lineplot(x='Pos',y='zFR',hue='Cue',estimator=None,units='trID',data=cellDat[subset],
#                                 ax=ax[i][1],legend=False,lw=mlw,alpha=alpha,hue_order=['L','R'])
#
#                         subset = (cellDat['IO']=='O_I') & (cellDat['Co']==coSets[i])& (cellDat['Valid'])
#                         ax[i][2] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',ci=None,data=cellDat[subset],
#                                     ax=ax[i][2],legend=leg,lw=3,hue_order=['L','R'],style_order=['1','2','3','4'])
#                         ax[i][2] = sns.lineplot(x='Pos',y='zFR',hue='Cue',estimator=None,units='trID',data=cellDat[subset],
#                                      ax=ax[i][2],legend=False,lw=mlw,alpha=alpha,hue_order=['L','R'])
#
#                     else:
#                         subset = (cellDat['IO']=='Out') & (cellDat['Co']==coSets[i]) & (cellDat['Valid'])
#                         ax[i][0] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',data=cellDat[subset],
#                                               ax=ax[i][0],lw=2,legend=False,hue_order=['L','R'],style_order=['1','2','3','4'])
#                         subset = (cellDat['IO']=='In') & (cellDat['Co']==coSets[i]) & (cellDat['Pos']>=4) & (cellDat['Valid'])
#                         ax[i][1] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',data=cellDat[subset],
#                                              ax=ax[i][1],lw=2,legend=False,hue_order=['L','R'],style_order=['1','2','3','4'])
#                         subset = (cellDat['IO']=='O_I') & (cellDat['Co']==coSets[i])& (cellDat['Valid'])
#                         ax[i][2] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',data=cellDat[subset],
#                                              ax=ax[i][2],legend=leg,lw=2,hue_order=['L','R'],style_order=['1','2','3','4'])
#
#                     ax[i][1].set_xticks(np.arange(4,nMaxPos))
#                     ax[i][0].set_xticks(np.arange(nMaxPos))
#                     ax[i][2].set_xticks(np.arange(nMinPos))
#
#                     for j in np.arange(3):
#                         ax[i][j].set_xlabel('')
#                         ax[i][j].set_ylabel('')
#                         ax[i][j].tick_params(axis='x', rotation=60)
#
#                     ax[i][0].set_ylabel('{} zFR'.format(coSets[i]))
#                     ax[i][1].set_yticklabels('')
#
#                     if i==0:
#                         for j in np.arange(3):
#                             ax[i][j].set_xticklabels(xPosLabels[j])
#                     else:
#                         ax[i][0].set_title('Out')
#                         ax[i][1].set_title('In')
#                         ax[i][2].set_title('O-I')
#                         for j in np.arange(3):
#                             ax[i][j].set_xticklabels('')
#                 l =ax[1][2].get_legend()
#                 plt.legend(bbox_to_anchor=(1.05, 0), loc=6, borderaxespad=0.,frameon=False)
#                 l.set_frame_on(False)
#
#                 # out/in limits
#                 lims = np.zeros((4,2))
#                 cnt =0
#                 for i in [0,1]:
#                     for j in [0,1]:
#                         lims[cnt]=np.array(ax[i][j].get_ylim())
#                         cnt+=1
#                 minY = np.floor(np.min(lims[:,0])*20)/20
#                 maxY = np.ceil(np.max(lims[:,1]*20))/20
#                 for i in [0,1]:
#                     for j in [0,1]:
#                         ax[i][j].set_ylim([minY,maxY])
#
#                 # o-i limits
#                 lims = np.zeros((2,2))
#                 cnt =0
#                 for i in [0,1]:
#                     lims[cnt]=np.array(ax[i][2].get_ylim())
#                     cnt+=1
#                 minY = np.floor(np.min(lims[:,0])*20)/20
#                 maxY = np.ceil(np.max(lims[:,1]*20))/20
#                 for i in [0,1]:
#                     ax[i][2].set_ylim([minY,maxY])
#
#             f.savefig(savePath/('LinearizedTr_{}ID-{}.pdf'.format(ut,cell)),dpi=300, bbox_inches='tight',pad_inches=0.2)
#             plt.close(f)
#
# def plotTrialConds(savePath,TrFRData,TrLongMat):
#     cellColIDs =  [i for i,item in enumerate(TrFRData.columns.values) if 'cell' in item]
#     nCells = len(cellColIDs)
#     muaColIDs =  [i for i,item in enumerate(TrFRData.columns.values) if 'mua' in item]
#     nMua = len(muaColIDs)
#     nTotalUnits = nCells+nMua
#     nUnits = {'cell':nCells,'mua':nMua}
#
#     cellCols = TrFRData.columns[cellColIDs]
#     muaCols = TrFRData.columns[muaColIDs]
#     unitCols = {'cell':cellCols,'mua':muaCols}
#
#     sns.set()
#     sns.set(style="whitegrid",context='notebook',font_scale=1.5,rc={
#         'axes.spines.bottom': False,
#         'axes.spines.left': False,
#         'axes.spines.right': False,
#         'axes.spines.top': False,
#         'axes.edgecolor':'0.5'})
#
#     cellDat = TrLongMat.copy()
#     for ut in ['cell','mua']:
#         for cell in np.arange(nUnits[ut]):
#             print('\nPlotting {} {}'.format(ut,cell))
#
#             cellDat.loc[:,'zFR'] = TrFRData[unitCols[ut][cell]]
#
#             f,ax = plt.subplots(1,2, figsize=(10,4))
#
#             # Correct Trials Out/In O_I
#             subset = cellDat['Co']=='Co'
#             dat =[]
#             dat = cellDat[subset].groupby(['trID','IO','Cue','Desc']).mean()
#             dat = dat.reset_index()
#
#             pal = sns.xkcd_palette(['spring green','light purple'])
#             with sns.color_palette(pal):
#                 ax[0]=sns.violinplot(y='zFR',x='IO',hue='Desc',data=dat,split=True, ax=ax[0],
#                                   scale='count',inner='quartile',hue_order=['L','R'],saturation=0.5,order=['Out','In','O_I'])
#             pal = sns.xkcd_palette(['emerald green','medium purple'])
#             with sns.color_palette(pal):
#                 ax[0]=sns.swarmplot(y='zFR',x='IO',hue='Desc',data=dat,dodge=True,hue_order=['L','R'],alpha=0.7,ax=ax[0],
#                                  edgecolor='gray',order=['Out','In','O_I'])
#             l=ax[0].get_legend()
#             l.set_visible(False)
#             ax[0].set_xlabel('Direction')
#
#             #
#             subset= cellDat['IO']=='Out'
#             dat = []
#             dat = cellDat[subset].groupby(['trID','Cue','Co','Desc']).mean()
#             dat = dat.reset_index()
#
#             pal = sns.xkcd_palette(['spring green','light purple'])
#             with sns.color_palette(pal):
#                 ax[1]=sns.violinplot(y='zFR',x='Desc',hue='Cue',data=dat,split=True,scale='width',ax=ax[1],
#                                   inner='quartile',order=['L','R'],hue_order=['L','R'],saturation=0.5)
#             pal = sns.xkcd_palette(['emerald green','medium purple'])
#             with sns.color_palette(pal):
#                 ax[1]=sns.swarmplot(y='zFR',x='Desc',hue='Cue',data=dat,dodge=True,order=['L','R'],ax=ax[1],
#                                     hue_order=['L','R'],alpha=0.7,edgecolor='gray')
#
#             #
#             ax[1].set_xlabel('Decision')
#             ax[1].set_ylabel('')
#             l=ax[1].get_legend()
#             handles, labels = ax[1].get_legend_handles_labels()
#             l.set_visible(False)
#             plt.legend(handles[2:],labels[2:],bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.,frameon=False,title='Cue')
#
#             f.savefig(savePath/('TrialConds_{}ID-{}.pdf'.format(ut,cell)),dpi=300, bbox_inches='tight',pad_inches=0.2)
#             plt.close(f)
#
# def plot_TrLinearized(fr,trDat,pos=None,f=None):
#
#     alpha=0.15
#     mlw = 1
#     nMaxPos = 11
#     nMinPos = 7
#     plotAll = False
#
#     cellDat = trDat.copy()
#     cellDat.loc[:,'zFR'] = fr
#
#     if (pos is None) or (f is None):
#         f,ax = plt.subplots(2,3, figsize=(16,6))
#         yorig = 0
#         xorig = 0
#         yscale = 1
#         xscale = 1
#     else:
#         xorig,yorig,xscale,yscale = pos
#         ax={}
#         #f.add_subplot()
#         cnt=0
#         for i in [0,1]:
#             ax[i]={}
#             for j in [0,1,2]:
#                 #ax[i][j] = f.add_axes([0,0,1,1,])
#                 ax[i][j] = f.add_subplot(231+cnt)
#                 cnt+=1
#
#
#     w = 0.25*xscale
#     ratio = 6.5/10.5
#     hsp = 0.05*xscale
#     vsp = 0.05*yscale
#     h = 0.43*yscale
#     W = [w,w*ratio,w*ratio]
#     yPos = np.array([vsp,2*vsp+h])+yorig
#     xPos = np.array([hsp,1.5*hsp+W[0],2.5*hsp+W[1]+W[0]])+xorig
#     #print(xPos,yPos)
#
#     xlims = [[-0.25,10.25],[3.75,10.25],[-0.25,6.25]]
#     for i in [0,1]:
#         for j in np.arange(3):
#             ax[i][j].set_position([xPos[j],yPos[i],W[j],h])
#             ax[i][j].set_xlim(xlims[j])
#
#     xPosLabels = {}
#     xPosLabels[0] = ['Home','SegA','Center','SegBE','Int','CDFG','Goals','CDFG','Int','CDFG','Goals']
#     xPosLabels[2] = ['Home','SegA','Center','SegBE','Int','CDFG','Goals']
#     xPosLabels[1] = xPosLabels[2][::-1]
#
#     pal = sns.xkcd_palette(['green','purple'])
#
#     with sns.color_palette(pal):
#         coSets = ['InCo','Co']
#         for i in [0,1]:
#             if i==0:
#                 leg=False
#             else:
#                 leg='brief'
#
#             if plotAll:
#                 subset = (cellDat['IO']=='Out') & (cellDat['Co']==coSets[i]) & (cellDat['Valid'])
#                 ax[i][0] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',ci=None,data=cellDat[subset],
#                          ax=ax[i][0],legend=False,lw=3,hue_order=['L','R'],style_order=['1','2','3','4'])
#                 ax[i][0] = sns.lineplot(x='Pos',y='zFR',hue='Desc',estimator=None,units='trID',data=cellDat[subset],
#                         ax=ax[i][0],legend=False,lw=mlw,alpha=alpha,hue_order=['L','R'])
#
#                 subset = (cellDat['IO']=='In') & (cellDat['Co']==coSets[i]) & (cellDat['Pos']>=4) & (cellDat['Valid'])
#                 ax[i][1] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',ci=None,data=cellDat[subset],
#                          ax=ax[i][1],legend=False,lw=3,hue_order=['L','R'],style_order=['1','2','3','4'])
#                 ax[i][1] = sns.lineplot(x='Pos',y='zFR',hue='Cue',estimator=None,units='trID',data=cellDat[subset],
#                         ax=ax[i][1],legend=False,lw=mlw,alpha=alpha,hue_order=['L','R'])
#
#                 subset = (cellDat['IO']=='O_I') & (cellDat['Co']==coSets[i])& (cellDat['Valid'])
#                 ax[i][2] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',ci=None,data=cellDat[subset],
#                             ax=ax[i][2],legend=leg,lw=3,hue_order=['L','R'],style_order=['1','2','3','4'])
#                 ax[i][2] = sns.lineplot(x='Pos',y='zFR',hue='Cue',estimator=None,units='trID',data=cellDat[subset],
#                              ax=ax[i][2],legend=False,lw=mlw,alpha=alpha,hue_order=['L','R'])
#
#             else:
#                 subset = (cellDat['IO']=='Out') & (cellDat['Co']==coSets[i]) & (cellDat['Valid'])
#                 ax[i][0] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',data=cellDat[subset],
#                                       ax=ax[i][0],lw=2,legend=False,hue_order=['L','R'])#,style_order=['1','2','3','4'])
#                 subset = (cellDat['IO']=='In') & (cellDat['Co']==coSets[i]) & (cellDat['Pos']>=4) & (cellDat['Valid'])
#                 ax[i][1] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',data=cellDat[subset],
#                                      ax=ax[i][1],lw=2,legend=False,hue_order=['L','R'])#,style_order=['1','2','3','4'])
#                 subset = (cellDat['IO']=='O_I') & (cellDat['Co']==coSets[i])& (cellDat['Valid'])
#                 ax[i][2] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',data=cellDat[subset],
#                                      ax=ax[i][2],legend=leg,lw=2,hue_order=['L','R'])#,style_order=['1','2','3','4'])
#
#             ax[i][1].set_xticks(np.arange(4,nMaxPos))
#             ax[i][0].set_xticks(np.arange(nMaxPos))
#             ax[i][2].set_xticks(np.arange(nMinPos))
#
#             for j in np.arange(3):
#                 ax[i][j].set_xlabel('')
#                 ax[i][j].set_ylabel('')
#                 ax[i][j].tick_params(axis='x', rotation=60)
#
#             ax[i][0].set_ylabel('{} zFR'.format(coSets[i]))
#             ax[i][1].set_yticklabels('')
#
#             if i==0:
#                 for j in np.arange(3):
#                     ax[i][j].set_xticklabels(xPosLabels[j])
#             else:
#                 ax[i][0].set_title('Out')
#                 ax[i][1].set_title('In')
#                 ax[i][2].set_title('O-I')
#                 for j in np.arange(3):
#                     ax[i][j].set_xticklabels('')
#         l =ax[1][2].get_legend()
#         plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.,frameon=False)
#         l.set_frame_on(False)
#
#         # out/in limits
#         lims = np.zeros((4,2))
#         cnt =0
#         for i in [0,1]:
#             for j in [0,1]:
#                 lims[cnt]=np.array(ax[i][j].get_ylim())
#                 cnt+=1
#         minY = np.floor(np.min(lims[:,0])*20)/20
#         maxY = np.ceil(np.max(lims[:,1]*20))/20
#         for i in [0,1]:
#             for j in [0,1]:
#                 ax[i][j].set_ylim([minY,maxY])
#
#         # o-i limits
#         lims = np.zeros((2,2))
#         cnt =0
#         for i in [0,1]:
#             lims[cnt]=np.array(ax[i][2].get_ylim())
#             cnt+=1
#         minY = np.floor(np.min(lims[:,0])*20)/20
#         maxY = np.ceil(np.max(lims[:,1]*20))/20
#         for i in [0,1]:
#             ax[i][2].set_ylim([minY,maxY])
#
#         return f,ax
# ################################################################################
# ################################################################################
# ################################################################################
# ##### Stat annot code!, this is a copy of the statannot github. ################
# ################################################################################
# ################################################################################
# ################################################################################
# def stat_test(box_data1, box_data2, test):
#     testShortName = ''
#     formattedOutput = None
#     if test == 'Mann-Whitney':
#         u_stat, pval = stats.mannwhitneyu(box_data1, box_data2, alternative='two-sided')
#         testShortName = 'M.W.W.'
#         formattedOutput = "MWW RankSum two-sided P_val={:.3e} U_stat={:.3e}".format(pval, u_stat)
#     elif test == 't-test_ind':
#         stat, pval = stats.ttest_ind(a=box_data1, b=box_data2)
#         testShortName = 't-test_ind'
#         formattedOutput = "t-test independent samples, P_val={:.3e} stat={:.3e}".format(pval, stat)
#     elif test == 't-test_paired':
#         stat, pval = stats.ttest_rel(a=box_data1, b=box_data2)
#         testShortName = 't-test_rel'
#         formattedOutput = "t-test paired samples, P_val={:.3e} stat={:.3e}".format(pval, stat)
#
#     return pval, formattedOutput, testShortName
#
# def pvalAnnotation_text(x, pvalueThresholds):
#     singleValue = False
#     if type(x) is np.array:
#         x1 = x
#     else:
#         x1 = np.array([x])
#         singleValue = True
#     # Sort the threshold array
#     pvalueThresholds = pd.DataFrame(pvalueThresholds).sort_values(by=0, ascending=False).values
#     xAnnot = pd.Series(["" for _ in range(len(x1))])
#     for i in range(0, len(pvalueThresholds)):
#         if (i < len(pvalueThresholds)-1):
#             condition = (x1 <= pvalueThresholds[i][0]) & (pvalueThresholds[i+1][0] < x1)
#             xAnnot[condition] = pvalueThresholds[i][1]
#         else:
#             condition = x1 < pvalueThresholds[i][0]
#             xAnnot[condition] = pvalueThresholds[i][1]
#
#     return xAnnot if not singleValue else xAnnot.iloc[0]
#
# def add_stat_annotation(ax,
#                         data=None, x=None, y=None, hue=None, order=None, hue_order=None,
#                         boxPairList=None,
#                         test='Mann-Whitney', textFormat='star', loc='inside',
#                         pvalueThresholds=[[1,"ns"], [0.05,"*"], [1e-2,"**"], [1e-3,"***"], [1e-4,"****"]],
#                         useFixedOffset=False, lineYOffsetToBoxAxesCoord=None, lineYOffsetAxesCoord=None,
#                         lineHeightAxesCoord=0.02, textYOffsetPoints=1,
#                         color='0.2', linewidth=1.5, fontsize='medium', verbose=1):
#     """
#     User should use the same argument for the data, x, y, hue, order, hue_order as the seaborn boxplot function.
#
#     boxPairList can be of either form:
#     For non-grouped boxplot: [(cat1, cat2), (cat3, cat4)]
#     For boxplot grouped by hue: [((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))]
#     """
#
#     def find_x_position_box(boxPlotter, boxName):
#         """
#         boxName can be either a name "cat" or a tuple ("cat", "hue")
#         """
#         if boxPlotter.plot_hues is None:
#             cat = boxName
#             hueOffset = 0
#         else:
#             cat = boxName[0]
#             hue = boxName[1]
#             hueOffset = boxPlotter.hue_offsets[boxPlotter.hue_names.index(hue)]
#
#         groupPos = boxPlotter.group_names.index(cat)
#         boxPos = groupPos + hueOffset
#         return boxPos
#
#
#     def get_box_data(boxPlotter, boxName):
#         """
#         boxName can be either a name "cat" or a tuple ("cat", "hue")
#
#         Here we really have to duplicate seaborn code, because there is not direct access to the
#         box_data in the BoxPlotter class.
#         """
#         if boxPlotter.plot_hues is None:
#             cat = boxName
#         else:
#             cat = boxName[0]
#             hue = boxName[1]
#
#         i = boxPlotter.group_names.index(cat)
#         group_data = boxPlotter.plot_data[i]
#
#         if boxPlotter.plot_hues is None:
#             # Draw a single box or a set of boxes
#             # with a single level of grouping
#             box_data = remove_na(group_data)
#         else:
#             hue_level = hue
#             hue_mask = boxPlotter.plot_hues[i] == hue_level
#             box_data = remove_na(group_data[hue_mask])
#
#         return box_data
#
#     fig = plt.gcf()
#
#     validList = ['inside', 'outside']
#     if loc not in validList:
#         raise ValueError("loc value should be one of the following: {}.".format(', '.join(validList)))
#     validList = ['t-test_ind', 't-test_paired', 'Mann-Whitney']
#     if test not in validList:
#         raise ValueError("test value should be one of the following: {}.".format(', '.join(validList)))
#
#     if verbose >= 1 and textFormat == 'star':
#         print("pvalue annotation legend:")
#         pvalueThresholds = pd.DataFrame(pvalueThresholds).sort_values(by=0, ascending=False).values
#         for i in range(0, len(pvalueThresholds)):
#             if (i < len(pvalueThresholds)-1):
#                 print('{}: {:.2e} < p <= {:.2e}'.format(pvalueThresholds[i][1], pvalueThresholds[i+1][0], pvalueThresholds[i][0]))
#             else:
#                 print('{}: p <= {:.2e}'.format(pvalueThresholds[i][1], pvalueThresholds[i][0]))
#         print()
#
#     # Create the same BoxPlotter object as seaborn's boxplot
#     boxPlotter = sns.categorical._BoxPlotter(x, y, hue, data, order, hue_order,
#                                              orient=None, width=.8, color=None, palette=None, saturation=.75,
#                                              dodge=True, fliersize=5, linewidth=None)
#     plotData = boxPlotter.plot_data
#
#     xtickslabels = [t.get_text() for t in ax.xaxis.get_ticklabels()]
#     ylim = ax.get_ylim()
#     yRange = ylim[1] - ylim[0]
#
#     if lineYOffsetAxesCoord is None:
#         if loc == 'inside':
#             lineYOffsetAxesCoord = 0.05
#             if lineYOffsetToBoxAxesCoord is None:
#                 lineYOffsetToBoxAxesCoord = 0.06
#         elif loc == 'outside':
#             lineYOffsetAxesCoord = 0.03
#             lineYOffsetToBoxAxesCoord = lineYOffsetAxesCoord
#     else:
#         if loc == 'inside':
#             if lineYOffsetToBoxAxesCoord is None:
#                 lineYOffsetToBoxAxesCoord = 0.06
#         elif loc == 'outside':
#             lineYOffsetToBoxAxesCoord = lineYOffsetAxesCoord
#     yOffset = lineYOffsetAxesCoord*yRange
#     yOffsetToBox = lineYOffsetToBoxAxesCoord*yRange
#
#     yStack = []
#     annList = []
#     for box1, box2 in boxPairList:
#
#         valid = None
#         groupNames = boxPlotter.group_names
#         hueNames = boxPlotter.hue_names
#         if boxPlotter.plot_hues is None:
#             cat1 = box1
#             cat2 = box2
#             hue1 = None
#             hue2 = None
#             label1 = '{}'.format(cat1)
#             label2 = '{}'.format(cat2)
#             valid = cat1 in groupNames and cat2 in groupNames
#         else:
#             cat1 = box1[0]
#             hue1 = box1[1]
#             cat2 = box2[0]
#             hue2 = box2[1]
#             label1 = '{}_{}'.format(cat1, hue1)
#             label2 = '{}_{}'.format(cat2, hue2)
#             valid = cat1 in groupNames and cat2 in groupNames and hue1 in hueNames and hue2 in hueNames
#
#
#         if valid:
#             # Get position of boxes
#             x1 = find_x_position_box(boxPlotter, box1)
#             x2 = find_x_position_box(boxPlotter, box2)
#             box_data1 = get_box_data(boxPlotter, box1)
#             box_data2 = get_box_data(boxPlotter, box2)
#             ymax1 = box_data1.max()
#             ymax2 = box_data2.max()
#
#             pval, formattedOutput, testShortName = stat_test(box_data1, box_data2, test)
#             if verbose >= 2: print ("{} v.s. {}: {}".format(label1, label2, formattedOutput))
#
#             if textFormat == 'full':
#                 text = "{} p < {:.2e}".format(testShortName, pval)
#             elif textFormat is None:
#                 text = None
#             elif textFormat is 'star':
#                 text = pvalAnnotation_text(pval, pvalueThresholds)
#
#             if loc == 'inside':
#                 yRef = max(ymax1, ymax2)
#             elif loc == 'outside':
#                 yRef = ylim[1]
#
#             if len(yStack) > 0:
#                 yRef2 = max(yRef, max(yStack))
#             else:
#                 yRef2 = yRef
#
#             if len(yStack) == 0:
#                 y = yRef2 + yOffsetToBox
#             else:
#                 y = yRef2 + yOffset
#             h = lineHeightAxesCoord*yRange
#             lineX, lineY = [x1, x1, x2, x2], [y, y + h, y + h, y]
#             if loc == 'inside':
#                 ax.plot(lineX, lineY, lw=linewidth, c=color)
#             elif loc == 'outside':
#                 line = lines.Line2D(lineX, lineY, lw=linewidth, c=color, transform=ax.transData)
#                 line.set_clip_on(False)
#                 ax.add_line(line)
#
#             if text is not None:
#                 ann = ax.annotate(text, xy=(np.mean([x1, x2]), y + h),
#                                   xytext=(0, textYOffsetPoints), textcoords='offset points',
#                                   xycoords='data', ha='center', va='bottom', fontsize=fontsize,
#                                   clip_on=False, annotation_clip=False)
#                 annList.append(ann)
#
#             ax.set_ylim((ylim[0], 1.1*(y + h)))
#
#             if text is not None:
#                 plt.draw()
#                 yTopAnnot = None
#                 gotMatplotlibError = False
#                 if not useFixedOffset:
#                     try:
#                         bbox = ann.get_window_extent()
#                         bbox_data = bbox.transformed(ax.transData.inverted())
#                         yTopAnnot = bbox_data.ymax
#                     except RuntimeError:
#                         gotMatplotlibError = True
#
#                 if useFixedOffset or gotMatplotlibError:
#                     if verbose >= 1:
#                         print("Warning: cannot get the text bounding box. Falling back to a fixed y offset. Layout may be not optimal.")
#                     # We will apply a fixed offset in points, based on the font size of the annotation.
#                     fontsizePoints = FontProperties(size='medium').get_size_in_points()
#                     offsetTrans = mtransforms.offset_copy(ax.transData, fig=fig,
#                                                           x=0, y=1.0*fontsizePoints + textYOffsetPoints, units='points')
#                     yTopDisplay = offsetTrans.transform((0, y + h))
#                     yTopAnnot = ax.transData.inverted().transform(yTopDisplay)[1]
#             else:
#                 yTopAnnot = y + h
#
#             yStack.append(yTopAnnot)
#         else:
#             raise ValueError("boxPairList contains an unvalid box pair.")
#             pass
#
#
#     yStackMax = max(yStack)
#     if loc == 'inside':
#         ax.set_ylim((ylim[0], 1.03*yStackMax))
#     elif loc == 'outside':
#         ax.set_ylim((ylim[0], ylim[1]))
#
#     return ax
