import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats

import Pre_Processing.pre_process_functions as pp_funcs
import Utils.filter_functions as filt_funcs

import seaborn as sns
from seaborn.utils import remove_na
import matplotlib as mpl
import matplotlib.pyplot as plt
import Utils.robust_stats as rs
import matplotlib.ticker as ticker
from matplotlib.text import Text
from matplotlib import transforms, lines
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties

from shapely.geometry import Point
from shapely.geometry.polygon import LinearRing, Polygon
from collections import Counter
from descartes import PolygonPatch

from Analyses import experiment_info as ei
from Analyses import tree_maze_functions as tmf


# import spatial_tuning as ST
# import stats_functions as StatsF

################################################################################
# Figure Classes
################################################################################
class Figure:
    _wspace = 0.02
    _hspace = 0.03

    _fontsizes = dict(default=7,
                      legend=6,
                      panel_label=9)

    label_axes_loc = (0, 1)

    def __init__(self, fig_size=(6, 4), font_modifier=1, dpi=500):

        self.fig_size = fig_size
        self._fig_h = fig_size[1]
        self._fig_w = fig_size[0]
        self._fig_ratio = fig_size[0] / fig_size[1]
        self.dpi = dpi
        self.font_modifier = 1

        self.fig = plt.figure(figsize=fig_size, dpi=dpi)

        if font_modifier != 1:
            for k, v in self._fontsizes.items():
                self._fontsizes[k] = v + font_modifier

        self._all_axes = []
        self._panels = {}
        self._panel_pos = {}
        self._panel_axes = {}

    def add_axes(self, pos):
        ax = self.fig.add_axes(pos)
        # ax.margins(0)
        self._all_axes.append(ax)
        return ax

    def add_panel(self, label, pos, label_txt=True, label_fontsize=None):
        ax = self.add_axes(pos)
        self._panels[label] = ax
        if label_txt:
            if label_fontsize is None:
                label_fontsize = self._fontsizes['panel_label']
            ax.text(*self.label_axes_loc, label,
                    fontsize=label_fontsize,
                    fontweight='bold',
                    transform=ax.transAxes,
                    ha='left',
                    va='center')
        ax.axis('off')
        self._panel_pos[label] = pos
        self._panel_axes[label] = []
        return ax

    def add_panel_axes(self, label, axes_pos):
        panel_pos = self._panel_pos[label]
        panel_axes_coords = self._convert_axes_to_panel_pos(panel_pos, axes_pos)
        panel_axes = self.add_axes(panel_axes_coords)
        self._panel_axes[label].append(panel_axes)
        return panel_axes

    @staticmethod
    def _convert_axes_to_panel_pos(panel_pos, axes_pos):
        panel_x0, panel_y0, panel_w, panel_h = panel_pos
        axes_x0, axes_y0, axes_w, axes_h = axes_pos

        panel_axes_x0 = panel_x0 + axes_x0 * panel_w
        panel_axes_y0 = panel_y0 + axes_y0 * panel_h
        panel_axes_w = axes_w * panel_w
        panel_axes_h = axes_h * panel_h

        return panel_axes_x0, panel_axes_y0, panel_axes_w, panel_axes_h

    def savefig(self, filename, **params):
        self.fig.savefig(filename, **params)

    def __del__(self):
        plt.close(self.fig)


class Fig1(Figure):
    fig_w = 6.2
    fig_h = 3.7
    dpi = 1000
    fig_ratio = fig_w / fig_h
    fig_size = (fig_w, fig_h)

    wspace = 0.02
    hspace = 0.03

    panels = ['a', 'b', 'c', 'd', 'e']

    a_x0 = 0
    ab_w = 0.3
    ab_h = 0.45

    b_x0 = a_x0 + ab_w + wspace

    c_x0 = 0
    c_y0 = 0.1
    c_w = ab_w * 2
    c_h = 0.33

    ab_y0 = hspace + c_y0 + c_h

    de_h = c_h
    de_w = 0.22
    e_y0 = 0.1

    de_x0 = ab_w * 2 + wspace * 3
    d_y0 = ab_y0 + ab_h - de_h

    panel_locs = dict(a=[a_x0, ab_y0, ab_w, ab_h],
                      b=[b_x0, ab_y0, ab_w, ab_h],

                      c=[c_x0, c_y0, c_w, c_h],
                      d=[de_x0, d_y0, de_w, de_h],
                      e=[de_x0, e_y0, de_w, de_h])

    subject_palette = 'deep'

    params = dict(fig_size=fig_size, dpi=dpi,
                  panel_a={'lw': 0.4},
                  panel_b={'lw': 0.2, 'line_alpha': 1, 'line_color': '0.2', 'sub_seg_lw': 0.05,
                           'n_trials': 5, 'max_dur': 700, 'marker_alpha': 1, 'marker_size': 8,
                           'seed': 1, 'leg_markersize': 2, 'leg_lw': 0.8,
                           'cue': np.array([['L', 'R'], ['L', 'R']]),
                           'dec': np.array([['L', 'R'], ['L', 'L']]),
                           'goal': np.array([[3, 2], [4, 'NULL']], dtype=object),
                           'long_trial': np.array([[0, 0], [1, 'NULL']], dtype=object)},

                  panel_c={'h_lw': 0.3, 'v_lw': 1, 'samp_buffer': 20, 'trial_nums': np.arange(13, 17)},

                  panel_d={'min_n_units': 1, 'min_n_trials': 50, 'palette': subject_palette,
                           'marker_alpha': 0.7, 'marker_swarm_size': 1.5,
                           'y_ticks': np.array([0, .25, .50, .75, 1]),
                           'box_plot_lw': 0.75, 'box_plot_median_lc': '0.4', 'box_plot_median_lw': 0.75,
                           'summary_marker_size': 5, 'scale': 0.7},

                  panel_e={'min_n_units': 1, 'min_n_trials': 50, 'palette': subject_palette,
                           'marker_alpha': 0.9, 'marker_swarm_size': 1.5,
                           'y_ticks': [0, 15, 30, 45],
                           'box_plot_lw': 0.75, 'box_plot_median_lc': '0.4', 'box_plot_median_lw': 0.75,
                           'summary_marker_size': 5, 'scale': 0.7}
                  )

    def __init__(self, session='Li_T3g_070618', **kargs):

        # setup
        self.params.update(kargs)
        # inherit figure methods
        super().__init__(fig_size=self.params['fig_size'], dpi=self.params['dpi'])
        # update fontsizes
        self.params['fontsizes'] = self._fontsizes
        self.params.update(kargs)

        self.tree_maze = tmf.TreeMazeZones()

        self.fontsize = self.params['fontsizes']['default']
        self.label_fontsize = self.params['fontsizes']['panel_label']
        self.legend_fontsize = self.params['fontsizes']['legend']

        # session info and data for panels b and c.
        subject = session.split('_')[0]
        self.session_info = ei.SubjectSessionInfo(subject, session)
        self.session_behav = self.session_info.get_event_behavior()
        self.session_track_data = self.session_info.get_track_data()
        self.session_pos_zones_mat = self.session_info.get_pos_zones_mat()

        # summary data for panels d and c.
        self.summary_info = ei.SummaryInfo()

    def fig_layout(self):

        for label, loc in self.panel_locs.items():
            self.add_panel(label, loc)

    def plot_all(self):

        self.fig_layout()

        for p in self.panels:
            obj = getattr(self, f"panel_{p}")
            obj(ax=self._panels[p])

        return self.fig

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
        trial_table = self.session_behav.trial_table.copy()
        trial_table = trial_table.fillna('NULL')

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
        cue_coords = self.tree_maze.cue_label_coords
        for row in range(2):
            for col in range(2):
                dec_full = 'left' if dec[row, col] == 'L' else 'right'

                _ = self.tree_maze.plot_maze(axis=a1[row, col],
                                             seg_color=None, zone_labels=False, seg_alpha=0.1,
                                             plot_cue=True, cue_color=cue[row, col],
                                             fontsize=self.fontsize, lw=fig_params['lw'],
                                             line_color=fig_params['line_color'],
                                             sub_segs='all', sub_seg_color='None', sub_seg_lw=fig_params['sub_seg_lw'])

                sub_table = trial_table[(trial_table.dec == dec[row, col]) &
                                        (trial_table.cue == cue[row, col]) &
                                        (trial_table.dur <= max_dur) &
                                        (trial_table.long == long_trial[row, col]) &
                                        (trial_table.goal == goal[row, col])
                                        ]

                # noinspection PyTypeChecker
                sel_trials = np.random.choice(sub_table.index, size=n_trials, replace=False)

                a1[row, col].scatter(H_loc[0], H_loc[1], s=marker_size, marker='o', lw=0, color='k')
                marker_end_color = 'b' if dec[row, col] == cue[row, col] else 'r'

                if goal[row, col] != 'NULL':
                    G_loc = self.tree_maze.well_coords[f"G{goal[row, col]}"]
                    a1[row, col].scatter(G_loc[0], G_loc[1], s=marker_size, marker='d',
                                         lw=0, alpha=marker_alpha, color=marker_end_color)
                else:
                    incorrect_wells = [v for v in self.tree_maze.split_segs[dec_full] if v[0] == 'G']
                    for iw in incorrect_wells:
                        iw_loc = self.tree_maze.well_coords[iw]
                        a1[row, col].scatter(iw_loc[0], iw_loc[1], s=marker_size, marker='d',
                                             lw=0, alpha=marker_alpha, color=marker_end_color)

                for trial in sel_trials:
                    t0 = sub_table.loc[trial, 't0']
                    tE = sub_table.loc[trial, 'tE']
                    a1[row, col].plot(track_data.loc[t0:tE, 'x'], track_data.loc[t0:tE, 'y'],
                                      color=self.tree_maze.split_colors[dec_full], lw=lw, alpha=line_alpha)

                a1[row, col].text(cue_coords[0], cue_coords[1], cue[row, col] + 'C', fontsize=self.legend_fontsize,
                                  horizontalalignment='center', verticalalignment='center', color='w')

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
                  fontsize=self.legend_fontsize, labelspacing=0.2)

    def panel_c(self, ax=None, **params):
        pos_zones_mat = self.session_info.get_pos_zones_mat()
        pos_zones_mat = self.tree_maze.subseg_pz_mat_transform(pos_zones_mat, 'seg')
        behav = self.session_behav
        track_data = self.session_track_data

        if ax is None:
            f, s_ax = plt.subplots(figsize=(2, 1), dpi=600)
        else:
            s_ax = self.add_panel_axes('c', [0.05, 0, 0.95, 0.9])

        c_params = self.params['panel_c']
        c_params.update(params)

        self.tree_maze.plot_zone_ts_window(pos_zones_mat, trial_table=behav.trial_table,
                                           t=track_data.t.values - track_data.t[0],
                                           trial_nums=c_params['trial_nums'], samp_buffer=c_params['samp_buffer'],
                                           h_lw=c_params['h_lw'], v_lw=c_params['v_lw'], fontsize=self.fontsize,
                                           ax=s_ax)

    def panel_d(self, ax=None, **params):

        perf = self.summary_info.get_behav_perf()
        subjects = self.summary_info.subjects

        subset = perf[
            (perf.n_units >= self.summary_info.min_n_units) & (perf.n_trials >= self.summary_info.min_n_trials)]

        fig_params = self.params['panel_d']
        fig_params.update(params)
        fontsize = self.fontsize

        if ax is None:
            f, ax = plt.subplots(figsize=(1.5, 1.5), dpi=600)
            ax_pos = ax.get_position()
            x0, y0, w, h = ax_pos.x0, ax_pos.y0, ax_pos.width, ax_pos.height
            x_split = w * 0.75
            ax.set_position([x0, y0, x_split, h])
            ax2 = f.add_axes([x0 + x_split, y0, w - x_split, h])

            return_flag = True
        else:
            ax = self.add_panel_axes('d', [0.2, 0, 0.65, 0.92])
            ax2 = self.add_panel_axes('d', [0.85, 0, 0.15, 0.92])
            return_flag = False

        # ax = reduce_ax(ax, fig_params['scale'])

        sns.set_theme(context='paper', style="whitegrid", font_scale=1, palette=fig_params['palette'])
        sns.set_style(rc={"axes.edgecolor": '0.3',
                          'xtick.bottom': True,
                          'ytick.left': True})

        sns.boxplot(ax=ax, x='subject', y='pct_correct', data=subset, color='w', linewidth=fig_params['box_plot_lw'],
                    whis=100)
        sns.stripplot(ax=ax, x='subject', y='pct_correct', data=subset, size=fig_params['marker_swarm_size'],
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
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1)
            ax.spines[spine].set_color('0.2')

        ax.tick_params(axis="both", direction="out", length=2, width=1, color="0.2", which='major', pad=0.5)
        ax.xaxis.set_label_coords(0.625, -0.15)

        # summary

        subset2 = subset.groupby('subject').median()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        subset2['color'] = 0

        for ii, s in enumerate(subjects):
            subset2.loc[s, 'color'] = mpl.colors.to_hex(colors[ii])

        sns.boxplot(ax=ax2, y='pct_correct', data=subset2.loc[subjects], color='w', whis=100, width=0.5,
                    linewidth=fig_params['box_plot_lw'])

        # hard-coded x location for no overalp
        x_locs = np.array([0, -1, 1, 0, -1, 0]) * 0.1
        ax2.scatter(x_locs, subset2.loc[subjects, 'pct_correct'], lw=0, zorder=10,
                    s=fig_params['summary_marker_size'],
                    c=subset2.loc[subjects, 'color'],
                    alpha=fig_params['marker_alpha'])

        for line in ax2.get_lines()[4::len(subjects)]:
            line.set(**{'color': fig_params['box_plot_median_lc'],
                        'lw': fig_params['box_plot_median_lw']})

        ax2.set_ylim(0, 1.01)
        ax2.set_yticks(fig_params['y_ticks'])
        ax2.set_yticklabels('')
        ax2.set_ylabel('')

        ax2.set_xticklabels([r" $\bar s$ "], fontsize=fontsize)

        for spine in ['top', 'right', 'left']:
            ax2.spines[spine].set_visible(False)
        ax2.spines['bottom'].set_linewidth(1)
        ax2.spines['bottom'].set_color('0.2')

        ax2.tick_params(axis="both", direction="out", length=2, width=1, color="0.2", which='major', pad=0.5)
        ax2.tick_params(axis='y', left=False)

        ax.grid(linewidth=0.5)
        ax2.grid(linewidth=0.5)
        if return_flag:
            return f


    def panel_e(self, ax=None, **params):
        perf = self.summary_info.get_behav_perf()
        subjects = self.summary_info.subjects

        subset = perf[
            (perf.n_units >= self.summary_info.min_n_units) & (perf.n_trials >= self.summary_info.min_n_trials)]

        fig_params = self.params['panel_e']
        fig_params.update(params)
        fontsize = self.fontsize

        if ax is None:
            f, ax = plt.subplots(figsize=(1, 1), dpi=600)
            ax_pos = ax.get_position()
            x0, y0, w, h = ax_pos.x0, ax_pos.y0, ax_pos.width, ax_pos.height
            x_split = w * 0.75
            ax.set_position([x0, y0, x_split, h])
            ax2 = f.add_axes([x0 + x_split, y0, w - x_split, h])
        else:
            ax = self.add_panel_axes('e', [0.2, 0, 0.65, 0.92])
            ax2 = self.add_panel_axes('e', [0.85, 0, 0.15, 0.92])

        sns.set_theme(context='paper', style="whitegrid", font_scale=1, palette=fig_params['palette'])
        sns.set_style(rc={"axes.edgecolor": '0.3',
                          'xtick.bottom': True,
                          'ytick.left': True})

        sns.boxplot(ax=ax, x='subject', y='n_units', data=subset, color='w', linewidth=fig_params['box_plot_lw'],
                    whis=100)
        sns.stripplot(ax=ax, x='subject', y='n_units', data=subset, size=fig_params['marker_swarm_size'],
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
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1)
            ax.spines[spine].set_color('k')

        ax.tick_params(axis="both", direction="in", length=2, width=0.8, color="0.5", which='major', pad=1)
        ax.xaxis.set_label_coords(0.625, -0.1)

        # summary
        # ax = f.add_axes([x0 + x_split, y0, w - x_split, h])
        subset2 = subset.groupby('subject').median()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        subset2['color'] = 0

        for ii, s in enumerate(subjects):
            subset2.loc[s, 'color'] = mpl.colors.to_hex(colors[ii])

        sns.boxplot(ax=ax2, y='n_units', data=subset2.loc[subjects], color='w', whis=100, width=0.5,
                    linewidth=fig_params['box_plot_lw'])

        # hard-coded x location for no overalp
        x_locs = np.array([0, 0, 0, 0, 0, -1]) * 0.1
        ax2.scatter(x_locs, subset2.loc[subjects, 'n_units'], lw=0, zorder=10,
                    s=fig_params['summary_marker_size'],
                    c=subset2.loc[subjects, 'color'],
                    alpha=fig_params['marker_alpha'])

        for line in ax2.get_lines()[4::len(subjects)]:
            line.set(**{'color': fig_params['box_plot_median_lc'],
                        'lw': fig_params['box_plot_median_lw']})

        ax2.set_ylim(0, 50)
        ax2.set_yticks(fig_params['y_ticks'])
        ax2.set_yticklabels('')
        ax2.set_ylabel('')

        ax2.set_xticklabels([r" $\bar s$ "], fontsize=fontsize)

        for spine in ['top', 'right', 'left']:
            ax2.spines[spine].set_visible(False)
        for spine in ['bottom']:
            ax.spines[spine].set_linewidth(1)
            ax.spines[spine].set_color('k')

        ax2.tick_params(axis="both", direction="in", length=2, width=0.8, color="0.5", which='major', pad=1)
        ax2.tick_params(axis='y', left=False)


class Fig2(Figure):
    seed = 4

    fig_w = 5
    fig_h = 4
    dpi = 1000

    fig_ratio = fig_w / fig_h
    fig_size = (fig_w, fig_h)

    wspace = 0.02
    hspace = 0.02

    panels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    a_x0, a_w, a_h = 0.0, 1 / 3, 1 / 3
    b_x0, b_y0, b_w, b_h = 0.0, 0.0, 1 / 2, 1 - a_h
    a_y0 = b_y0 + b_h

    c_x0, c_y0 = a_x0 + a_w, a_y0
    c_w, c_h = b_w - c_x0, a_h

    d_x0, d_y0 = c_x0 + c_w, a_y0
    d_w, d_h = 1 - 1.5 * (c_x0 + c_w), a_h

    e_x0, e_w, e_h = d_x0, d_w, d_h
    e_y0 = 1 - e_h - d_h

    f_x0, f_y0, f_w, f_h = e_x0, b_y0, e_w, e_h

    g_x0, g_y0 = d_x0 + d_w, d_y0
    g_w, g_h = d_w, d_h

    h_x0, h_y0, h_w, h_h = g_x0, e_y0, e_w, e_h
    i_x0, i_y0, i_w, i_h = g_x0, f_y0, f_w, f_h

    assert np.isclose(a_w + c_w + d_w + g_w, 1)
    assert np.isclose(a_h + b_h, 1)
    assert np.isclose(d_h + e_h + f_h, 1)

    subject_palette = 'deep'

    params = dict(fig_size=fig_size, dpi=dpi, seed=seed,
                  panel_a=dict(unit_idx_track=0,
                               maze_params=dict(lw=0.2, line_color='0.6', sub_segs='all',
                                                sub_seg_color='None', sub_seg_lw=0.1),
                               legend_params=dict(lw=0.7, markersize=1),
                               trajectories_params=dict(lw=0.15, alpha=0.1),
                               well_marker_size=3),
                  panel_b=dict(cond_pair=['CR_bo-CL_bo'] * 4,
                               cm_params=dict(color_map='viridis',
                                              n_color_bins=25, nans_2_zeros=True, div=False,
                                              label='FR'),
                               zone_rates_lw=0.05, zone_rates_lc='0.75',
                               dist_n_boot=20, dist_test_color='b', dist_null_color='0.5',
                               dist_lw=0.5, dist_v_lw=0.4),
                  panel_c=dict(test_var_name='CR_bo-CL_bo', measure='corr_m', z_measure='corr_zm',
                               test_color='#bcbd22', null_color='#17becf', z_color='0.3',
                               rug_height=0.08, rug_alpha=0.75, rug_lw=0.1,
                               kde_lw=0.8, kde_alpha=0.8, kde_smoothing=0.5, kde_v_lw=0.5,
                               ),
                  panel_d=dict(behav_score='pct_correct', corr_method='kendall',
                               remap_score='CR_bo-CL_bo-Even_bo-Odd_bo-corr_zm',
                               dot_sizes=(1, 5), dot_alpha=0.75, dot_lw=0.1, dot_ec='k',
                               reg_line_params=dict(lw=1, color='#e46c5c'), plot_ci=True,
                               ci_params=dict(alpha=0.4, color='#e46c5c', linewidth=0, zorder=10)),
                  panel_e=dict(behav_score='pct_correct', corr_method='kendall',
                               remap_score='CR_bo-CL_bo-Even_bo-Odd_bo-corr_zm',
                               rescale_behav=False,
                               dot_sizes=(2, 8), dot_alpha=0.75, dot_lw=0.1, dot_ec='k',
                               dot_scale=1, legend_markersize=3,
                               reg_line_params=dict(lw=1, color='0.3'), plot_ci=True,
                               ci_params=dict(alpha=0.4, color='#86b4cb', linewidth=0, zorder=-1),
                               color_pal=sns.color_palette(subject_palette)),
                  panel_f=dict(behav_score='pct_correct', corr_method='kendall',
                               remap_score='CR_bo-CL_bo-Even_bo-Odd_bo-corr_zm',
                               rescale_behav=True,
                               point_plot_params=dict(estimator=np.mean, linewidth=0.2, join=False,
                                                      scale=0.4, errwidth=1, ci='sd'),
                               scatter_summary_params=dict(s=12, alpha=0.8, lw=0, zorder=10),
                               summary_lw=1.2, summary_c='0.2', summary_xrange_scale=0.2,
                               color_pal=sns.color_palette(subject_palette)))

    params['panel_g'] = params['panel_d'].copy()
    params['panel_g']['remap_score'] = 'Co_bi-Inco_bi-Even_bi-Odd_bi-corr_zm'

    params['panel_h'] = params['panel_e'].copy()
    params['panel_h']['remap_score'] = 'Co_bi-Inco_bi-Even_bi-Odd_bi-corr_zm'

    params['panel_i'] = params['panel_f'].copy()
    params['panel_i']['remap_score'] = 'Co_bi-Inco_bi-Even_bi-Odd_bi-corr_zm'

    def __init__(self, unit_ids=None, unit_type='cell', trial_end='tE_2', task='T3', **kargs):
        # setup
        self.params.update(kargs)
        # inherit figure methods
        super().__init__(fig_size=self.params['fig_size'], dpi=self.params['dpi'])
        # update fontsizes
        self.params['fontsizes'] = self._fontsizes
        self.params.update(kargs)
        np.random.seed(self.params['seed'])

        assert unit_type in ['all', 'cell', 'mua']
        self.unit_type = unit_type
        assert 'T3' in task
        self.task = task
        self.tree_maze = tmf.TreeMazeZones()

        self.fontsize = self.params['fontsizes']['default']
        self.label_fontsize = self.params['fontsizes']['panel_label']
        self.legend_fontsize = self.params['fontsizes']['legend']

        # summary data for panels d and c.
        self.summary_info = ei.SummaryInfo()
        zrc = self.summary_info.get_zone_rates_remap(trial_end=trial_end, **kargs)
        b_table = self.summary_info.get_behav_perf()
        self.zrc_b = self._combine_tables(zrc, b_table)

        self.n_total_units = len(self.zrc_b)
        if unit_ids is None:
            unit_ids = np.random.choice(self.zrc_b.unit_id, 4)
        else:
            assert len(unit_ids) == 4

        self.unit_ids = unit_ids

        self.unit_sessions = {}
        self.unit_subjects = {}
        self.unit_session_info = {}
        self.unit_session_ta = {}
        _loaded_sessions = []
        _session_units = {}
        self.unit_session_unit_idx = {}
        for unit in unit_ids:
            session = self.zrc_b.loc[unit].session
            self.unit_session_unit_idx[unit] = self.zrc_b.loc[unit].session_unit_id
            self.unit_sessions[unit] = session
            subject = session.split("_")[0]
            self.unit_subjects[unit] = subject

            if session not in _loaded_sessions:
                _loaded_sessions.append(session)
                _session_units[session] = []
                _session_units[session].append(unit)

                self.unit_session_info[unit] = ei.SubjectSessionInfo(subject, session)
                self.unit_session_ta[unit] = tmf.TrialAnalyses(session_info=self.unit_session_info[unit],
                                                               trial_end=trial_end, **kargs)
            else:
                _session_units[session].append(unit)
                _first_unit = _session_units[session][0]
                self.unit_session_info[unit] = self.unit_session_info[_first_unit]
                self.unit_session_ta[unit] = self.unit_session_ta[_first_unit]

        self.panel_locs = {}
        for pp in self.panels:
            self.panel_locs[pp] = []
            for kk in ['x0', 'y0', 'w', 'h']:
                self.panel_locs[pp].append(getattr(self, f"{pp}_{kk}"))

        self.unit_session_boot_trials = {}
        self.unit_session_boot_seg_rates = {}
        self.unit_cond_pair = {}

        seg_rates_params = dict(reward_blank=False, trial_end=trial_end)
        if 'reward_blank' in kargs.keys():
            seg_rates_params['reward_blank'] = kargs['reward_blank']

        for ii, unit in enumerate(unit_ids):
            ta = self.unit_session_ta[unit]
            session_unit_id = self.unit_session_unit_idx[unit]
            self.unit_cond_pair[unit] = self.params['panel_b']['cond_pair'][ii]

            cond_pair = self.unit_cond_pair[unit].split('-')
            cond_sets = {}
            cond_sets.update(ta.bal_cond_sets[cond_pair[0]]['cond_set'])
            cond_sets.update(ta.bal_cond_sets[cond_pair[1]]['cond_set'])

            self.unit_session_boot_trials[unit] = ta.get_trials_boot_cond_set(cond_sets)

            boot_seg_rates = self.unit_session_info[unit].get_bal_conds_seg_boot_rates(**seg_rates_params)
            self.unit_session_boot_seg_rates[unit] = boot_seg_rates.loc[
                boot_seg_rates.unit == session_unit_id].reset_index(drop=True)

    def fig_layout(self):
        for label, loc in self.panel_locs.items():
            self.add_panel(label, loc)

    def plot_all(self):
        for p in self.panels:
            try:
                obj = getattr(self, f"panel_{p}")
                obj(fig_template=True)
            except:
                pass

    def panel_a(self, fig_template=None, **panel_params):

        if fig_template:
            self.add_panel(label='a', pos=self.panel_locs['a'])
            ax = np.zeros(3, dtype=object)
            ax[0] = self.add_panel_axes('a', [0.0, 0.1, 0.47, 1])
            ax[1] = self.add_panel_axes('a', [0.49, 0.1, 0.47, 1])
            ax[2] = self.add_panel_axes('a', [0, 0, 1, 1])
            ax[2].axis('off')
            leg_pos = [0, 0.2, 1, 1]
        else:
            f, ax = plt.subplots(1, 3, figsize=(2, 1), dpi=500)
            ax[0].set_position([0, 0, 0.5, 1])
            ax[1].set_position([0.5, 0, 0.5, 1])
            ax[2].set_position([0, 0, 1, 1])
            ax[2].axis('off')
            leg_pos = [0, 0, 1, 1]

        params = self.params['panel_a']
        params.update(panel_params)

        unit_idx = params['unit_idx_track']
        unit_id = self.unit_ids[unit_idx]
        ta = self.unit_session_ta[unit_id]

        x, y,_ = ta.get_trial_track_pos()
        cue_trial_sets = self.unit_session_boot_trials[unit_id]

        _ = self.tree_maze.plot_maze(axis=ax[0],
                                     seg_color=None, zone_labels=False, seg_alpha=0.1,
                                     plot_cue=True, cue_color='L', **params['maze_params'])
        _ = self.tree_maze.plot_maze(axis=ax[1],
                                     seg_color=None, zone_labels=False, seg_alpha=0.1,
                                     plot_cue=True, cue_color='R', **params['maze_params'])

        well_coords = self.tree_maze.well_coords
        correct_cue_goals = {'CR': ['G1', 'G2'], 'CL': ['G3', 'G4']}
        cue_coords = self.tree_maze.cue_label_coords

        for ii, cue in enumerate(['CL', 'CR']):
            for tr in cue_trial_sets[cue][:, 0]:
                dec = ta.trial_table.loc[tr, 'dec']
                valid_dur = ta.trial_table.loc[tr, 'dur'] <= 1000
                if (dec in ['L', 'R']) & valid_dur:
                    col = ta.tmz.split_colors[dec]
                    ax[ii].plot(x[tr], y[tr], zorder=1, color=col, **params['trajectories_params'])

            ax[ii].scatter(well_coords['H'][0], well_coords['H'][1], s=params['well_marker_size'], marker='o', lw=0,
                           color='k',
                           zorder=10)

            for jj in range(4):
                goal_id = f"G{jj + 1}"
                coords = well_coords[goal_id]
                marker_end_color = 'b' if (goal_id in correct_cue_goals[cue]) else 'r'

                ax[ii].scatter(coords[0], coords[1], s=params['well_marker_size'], marker='d', lw=0,
                               color=marker_end_color,
                               zorder=10)

            ax[ii].text(cue_coords[0], cue_coords[1], cue[1] + 'C', fontsize=self.legend_fontsize,
                        horizontalalignment='center', verticalalignment='center', color='w')

        for ii in range(2):
            ax[ii].axis("square")
            ax[ii].axis("off")
            # ax[ii].set_ylim(ta.y_edges[0], ta.y_edges[-1])
            ax[ii].set_xlim(ta.x_edges[0] * 1.24, ta.x_edges[-1] * 1.24)

        legend_params = params['legend_params']
        legend_elements = [mpl.lines.Line2D([0.1], [0.1], color='g', lw=legend_params['lw'], label='L Decision'),
                           mpl.lines.Line2D([0], [0], color='purple', lw=legend_params['lw'], label='R Decision'),
                           mpl.lines.Line2D([0], [0], marker='o', color='k', lw=0, label='Start',
                                            markerfacecolor='k', markersize=legend_params['markersize']),
                           mpl.lines.Line2D([0], [0], marker='d', color='b', lw=0, label='Correct',
                                            markerfacecolor='b', markersize=legend_params['markersize']),
                           mpl.lines.Line2D([0], [0], marker='d', color='r', lw=0, label='Incorrect',
                                            markerfacecolor='r', markersize=legend_params['markersize'])]

        ax[2].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=leg_pos, frameon=False,
                     fontsize=self.legend_fontsize - 1, labelspacing=0.1, handlelength=0.5, handletextpad=0.4)

        return ax

    def panel_b(self, fig_template=None, **panel_params):

        params = self.params['panel_b']
        params.update(panel_params)

        sub_panel_labels = ['i', 'ii', 'iii', 'iv']
        if fig_template is None:
            f = Figure(fig_size=(3, 3), dpi=500)
            x0, y0 = 0, 0
            w2, h2 = 0.5, 0.5
        else:
            f = self
            sub_panel_labels = ['b.' + _ for _ in sub_panel_labels]
            x0, y0 = f.b_x0, f.b_y0
            w2, h2 = f.b_w / 2, f.b_h / 2

        sub_panel_pos = [[x0, y0 + h2, w2, h2],
                         [x0, y0, w2, h2],
                         [x0 + w2, y0 + h2, w2, h2],
                         [x0 + w2, y0, w2, h2]]

        for ii, label in enumerate(sub_panel_labels):
            f.add_panel(pos=sub_panel_pos[ii], label=label)
            self.sub_panel_b(fig=f, unit=self.unit_ids[ii], sub_panel_label=label)

        if fig_template is None:
            return f

    def sub_panel_b(self, unit, fig=None, sub_panel_label=None, **panel_params):

        x0, y0 = 0, 0.1
        w, h = 0.4, 0.4
        wi = 2.1 / 5 * w
        sub_panels_pos_splits = [[x0, y0 + h, w, h],
                                 [x0 + w, y0 + h, w, h],
                                 [x0, y0, w, h],
                                 [x0 + w, y0, w, h],
                                 [x0 + w - wi / 2, y0 + 0.02 + h, wi, 0.08]]

        params = self.params['panel_b']
        params.update(panel_params)

        ax = np.zeros(5, dtype=object)
        if fig is None:
            fig = Figure(fig_size=(2, 2))

        if sub_panel_label is not None:
            for ii in range(5):
                ax[ii] = fig.add_panel_axes(sub_panel_label, sub_panels_pos_splits[ii])
        else:
            for ii in range(5):
                ax[ii] = fig.add_axes(sub_panels_pos_splits[ii])

        ta = self.unit_session_ta[unit]
        session_unit_id = self.unit_session_unit_idx[unit]

        # ---- traces + spikes ---- #
        x, y, _ = ta.get_trial_track_pos()
        trial_sets = self.unit_session_boot_trials[unit]
        spikes = ta.get_trial_neural_data(data_type='spikes')[session_unit_id]

        cond_pair_name = self.unit_cond_pair[unit]
        cond_pair = cond_pair_name.split('-')
        right_cond = cond_pair[0].split('_')[0]
        left_cond = cond_pair[1].split('_')[0]

        if cond_pair_name == 'CR_bo-CL_bo':
            r_name = 'RC'
            l_name = 'LC'
        else:
            r_name = right_cond
            l_name = left_cond

        l_trials = trial_sets[left_cond][:, 0]
        r_trials = trial_sets[right_cond][:, 0]

        self.tree_maze.plot_spk_trajectories(x=x[l_trials], y=y[l_trials],
                                             spikes=spikes[l_trials], ax=ax[0])
        ax[0].set_title(l_name, fontsize=self.fontsize, pad=1)

        self.tree_maze.plot_spk_trajectories(x=x[r_trials], y=y[r_trials],
                                             spikes=spikes[r_trials], ax=ax[1])
        ax[1].set_title(r_name, fontsize=self.fontsize, pad=1)

        # ---- zone rate mean heat maps ---- #
        lw = params['zone_rates_lw']
        line_color = params['zone_rates_lc']
        cm_params = params['cm_params']
        cm_params['tick_fontsize'] = self.legend_fontsize - 1
        cm_params['label_fontsize'] = self.legend_fontsize - 1

        seg_rates = self.unit_session_boot_seg_rates[unit].groupby(['cond', 'unit', 'seg']).mean()
        seg_rates = seg_rates.reset_index()
        max_val = seg_rates[(seg_rates.cond.isin(cond_pair))].m.max()

        for ii, ax_ii in enumerate([3, 2]):
            zr = seg_rates[seg_rates.cond == cond_pair[ii]][['m', 'seg']]
            zr = zr.pivot_table(columns='seg', aggfunc=lambda xx: xx)
            zr = zr.reset_index().drop('index', axis=1)
            self.tree_maze.plot_zone_activity(zr, ax=ax[ax_ii], legend=(ii + 1) % 2,
                                              min_value=0, max_value=max_val,
                                              lw=lw, line_color=line_color, **cm_params)

        # ---- distributions of correlations  ---- #
        test_cond_pair = cond_pair_name
        null_cond_pair = ta.test_null_bal_cond_pairs[test_cond_pair]
        n_boot = params['dist_n_boot']
        test_boot_corr_dist = ta.unit_zrm_boot_corr(unit_id=session_unit_id,
                                                    bal_cond_pair=test_cond_pair, n_boot=n_boot)
        null_boot_corr_dist = ta.unit_zrm_boot_corr(unit_id=session_unit_id,
                                                    bal_cond_pair=null_cond_pair, n_boot=n_boot)

        plot_kde_dist(data=test_boot_corr_dist, v_lines=np.nanmean(test_boot_corr_dist), label=f"{r_name}-{l_name}",
                      color=params['dist_test_color'], lw=params['dist_lw'], v_lw=params['dist_v_lw'], ax=ax[4])
        plot_kde_dist(data=null_boot_corr_dist, v_lines=np.nanmean(null_boot_corr_dist), label=f"Null",
                      color=params['dist_null_color'], lw=params['dist_lw'], v_lw=params['dist_v_lw'], ax=ax[4])

        ax[4].get_yaxis().set_ticks([])
        ax[4].tick_params(axis="both", direction="out", length=1, width=0.5, color='0.2', which='major', pad=0.05)
        ax[4].grid(False)
        ax[4].set_xticks([0, 1])
        ax[4].set_xticklabels([0, 1], fontsize=self.legend_fontsize - 2)
        ax[4].set_xlabel(r"$\tau$", fontsize=self.legend_fontsize - 1, labelpad=0)
        ax[4].xaxis.set_label_coords(0.5, -0.1, transform=ax[4].transAxes)

        sns.despine(ax=ax[4], left=True)
        ax[4].spines['bottom'].set_linewidth(0.4)
        ax[4].spines['bottom'].set_color('k')

        ax[4].set_ylabel('')
        legend_elements = [plt.Line2D([0], [0], color=params['dist_test_color'],
                                      label=f"{r_name}-{l_name}", lw=params['dist_lw']),
                           plt.Line2D([0], [0], color=params['dist_null_color'],
                                      label="Null", lw=params['dist_lw']),
                           ]
        legend_params = dict(handlelength=0.25, handletextpad=0.2, bbox_to_anchor=[0.5, 1.15], loc='lower center',
                             frameon=False, fontsize=self.legend_fontsize - 2, markerscale=0.6, labelspacing=0.1,
                             ncol=2,
                             columnspacing=0.7)
        ax[4].legend(handles=legend_elements, **legend_params)

        z_var = f"{test_cond_pair}-{null_cond_pair}-corr_zm"
        z_val = self.zrc_b.loc[unit, z_var]
        ax[4].text(0.5, 1.2, r"$\bar{z}_{\Delta \tau}$=" + str(np.around(z_val, 1)), fontsize=self.legend_fontsize - 2,
                   ha='center', va='center', transform=ax[4].transAxes)

    def panel_c(self, fig_template=None, **panel_params):

        x0, y0 = 0.2, 0.65
        x1, y1 = x0, 0.25
        w, h = 0.65, 0.25
        panel_axes_pos = [[x0, y0, w, h],
                          [x1, y1, w, h]]

        if fig_template is None:
            f = Figure(fig_size=(1.5, 1.5), dpi=500)
            p_ax = f.add_panel(label='c', pos=[0, 0, 1, 1], label_txt=False)
        else:
            f = self
            p_ax = self.add_panel(label='c', pos=self.panel_locs['c'])

        ax = np.zeros(2, dtype=object)
        for ii, pos in enumerate(panel_axes_pos):
            ax[ii] = f.add_panel_axes(label='c', axes_pos=pos)

        legend_params = dict(handlelength=0.25, handletextpad=0.2, bbox_to_anchor=[-0.1, 0.4], loc='lower left',
                             frameon=False, fontsize=self.legend_fontsize, markerscale=0.6, labelspacing=0.1)

        params = self.params['panel_c']
        params.update(panel_params)

        test_var_name = params['test_var_name']
        measure = params['measure']
        z_measure = params['z_measure']

        null_var_name = self.unit_session_ta[self.unit_ids[0]].test_null_bal_cond_pairs[test_var_name]

        test_var = f"{test_var_name}-{measure}"
        null_var = f"{null_var_name}-{measure}"
        z_var = f"{test_var_name}-{null_var_name}-{z_measure}"

        id_vars = ['unit_id', 'unit_type']
        table = self.zrc_b.melt(id_vars=id_vars, value_vars=[test_var, null_var, z_var], var_name='comparison',
                                value_name='score').copy()
        table.replace([np.inf, -np.inf], np.nan, inplace=True)

        if self.unit_type != 'all':
            table = table[table.unit_type == self.unit_type]
        table = table.assign(x='vars')

        kde_params = dict(alpha=params['kde_alpha'], cut=0)

        # test/null dists
        null_dat = table[table.comparison == null_var]
        plot_kde_dist(data=null_dat.score, color=params['null_color'],
                      lw=params['kde_lw'], v_lines=null_dat.score.mean(), v_lw=params['kde_lw'],
                      ax=ax[0], **kde_params)
        test_dat = table[table.comparison == test_var]
        plot_kde_dist(data=test_dat.score, color=params['test_color'],
                      lw=params['kde_lw'], v_lines=test_dat.score.mean(), v_lw=params['kde_lw'],
                      ax=ax[0], **kde_params)

        # plot individual ticks
        sns.rugplot(ax=ax[0], data=null_dat, x='score', height=-params['rug_height'], alpha=params['rug_alpha'],
                    color=params['null_color'], clip_on=False, linewidth=params['rug_lw'])

        sns.rugplot(ax=ax[0], data=test_dat, x='score', height=params['rug_height'], alpha=params['rug_alpha'],
                    color=params['test_color'], clip_on=False, linewidth=params['rug_lw'])

        if 'corr_m' in measure:
            ax[0].set_xlim([-1.01, 1.01])
            ax[0].set_xticks([-1, 0, 1])
            ax[0].set_xticklabels([-1, 0, 1], fontsize=self.fontsize - 1)
            ax[0].set_xlabel(r"$\bar{\tau}$", labelpad=0, fontsize=self.fontsize)

        if test_var_name == 'CR_bo-CL_bo':
            test_var_name_short = 'RC-LC'
            null_var_name_short = 'Null'
        elif test_var_name == 'Co_bi-Inco_bi':
            test_var_name_short = 'Co-Inco'
            null_var_name_short = 'Null'
        else:
            test_var_name_short = test_var_name
            null_var_name_short = null_var_name

        legend_elements = [plt.Line2D([0], [0], color=params['test_color'], label=test_var_name_short),
                           plt.Line2D([0], [0], color=params['null_color'], label=null_var_name_short),
                           ]
        ax[0].legend(handles=legend_elements, **legend_params)

        # z data
        z_dat = table[table.comparison == z_var]
        plot_kde_dist(data=z_dat.score, color=params['z_color'],
                      lw=params['kde_lw'], v_lines=z_dat.score.mean(), v_lw=params['kde_lw'],
                      ax=ax[1], **kde_params)
        sns.rugplot(ax=ax[1], data=z_dat, x='score', height=params['rug_height'], alpha=params['rug_alpha'],
                    color=params['z_color'], clip_on=False, linewidth=params['rug_lw'])

        xticks = np.floor(min(z_dat.score)), np.ceil(max(z_dat.score))
        xticks = np.sort(np.append(xticks, 0)).astype(int)
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels(xticks, fontsize=self.fontsize - 1)

        if 'CR' in z_var:
            ax[1].set_xlabel(r"$\bar{z}_{\Delta \tau} \: Cue$", fontsize=self.fontsize, labelpad=0)
        elif 'Co' in z_var:
            ax[1].set_xlabel(r"$\bar{z}_{\Delta \tau} \: Rw$", fontsize=self.fontsize, labelpad=0)
        else:
            ax[1].set_xlabel(r"$\bar{z}_{\Delta \tau}$", fontsize=self.fontsize, labelpad=0)

        for ax_ii in ax:
            sns.despine(ax=ax_ii, left=True)
            ax_ii.get_yaxis().set_ticks([])
            ax_ii.set_ylabel("")
            ax_ii.spines['bottom'].set_linewidth(1)
            ax_ii.spines['bottom'].set_color('k')
            ax_ii.grid(False)
            ax_ii.tick_params(axis="x", direction="out", bottom=True, length=2, width=0.8, color='0.2', which='major',
                              pad=0.2)
            ax_ii.set_facecolor('none')

        # y label for both axes
        p_ax.text(0, 0.6, f"Units (n={len(z_dat)})", rotation='vertical', ha='left', va='center',
                  fontsize=self.fontsize,
                  transform=p_ax.transAxes)

        if fig_template is None:
            return f

    def panel_d(self, fig_template=None, **panel_params):
        x0, y0 = 0.28, 0.3
        w, h = 0.6, 0.6
        ax_pos = [x0, y0, w, h]

        if fig_template is None:
            f = Figure(fig_size=(1, 1), dpi=500)
            ax = f.add_axes(ax_pos)
        else:
            f = self
            p_ax = self.add_panel(label='d', pos=self.panel_locs['d'])
            ax = self.add_panel_axes(label='d', axes_pos=ax_pos)

        params = self.params['panel_d']
        params.update(panel_params)

        r_score = params['remap_score']
        b_score = params['behav_score']

        self.plot_unit_remap_v_beh(ax=ax, r_score=r_score, b_score=b_score, **params)

        if fig_template is None:
            return f

    def panel_e(self, fig_template=None, **panel_params):

        if fig_template is None:
            x0, y0 = 0.1, 0.1
            w, h = 0.8, 0.8
            ax_pos = [x0, y0, w, h]
            f = Figure(fig_size=(1, 1), dpi=500)
            ax = f.add_axes(ax_pos)
        else:
            x0, y0 = 0.28, 0.3
            w, h = 0.6, 0.6
            ax_pos = [x0, y0, w, h]

            f = self
            p_ax = self.add_panel(label='e', pos=self.panel_locs['e'])
            ax = self.add_panel_axes(label='e', axes_pos=ax_pos)

        params = self.params['panel_e']
        params.update(panel_params)
        r_score = params['remap_score']
        b_score = params['behav_score']

        self.plot_session_remap_v_beh(ax=ax, r_score=r_score, b_score=b_score, **params)

        if fig_template is None:
            return f

    def panel_f(self, fig_template=None, **panel_params):

        label='f'
        x_split = 0.75
        if fig_template is None:
            x0, y0 = 0.1, 0.1
            w, h = 0.8, 0.8

            f = Figure(fig_size=(1, 1), dpi=500)
            p_ax = f.add_panel(label=label, pos=[0, 0, 1, 1], label_txt=False)
        else:
            x0, y0 = 0.28, 0.3
            w, h = 0.6, 0.6
            f = self
            p_ax = self.add_panel(label=label, pos=self.panel_locs[label])

        ax_pos = [[x0, y0, w * x_split, h],
                  [x0 + w * x_split, y0, w * (1 - x_split), h]]

        ax = np.zeros(2, dtype=object)
        for ii in range(2):
            ax[ii] = f.add_panel_axes(label=label, axes_pos=ax_pos[ii])

        params = self.params['panel_'+label]
        params.update(panel_params)
        r_score = params['remap_score']
        b_score = params['behav_score']

        self.plot_subject_remap_beh_slope(ax=ax, r_score=r_score, b_score=b_score, **params)

        if fig_template is None:
            return f

    def panel_g(self, fig_template=None, **panel_params):
        label = 'g'
        x0, y0 = 0.32, 0.3
        w, h = 0.6, 0.6
        ax_pos = [x0, y0, w, h]

        if fig_template is None:
            f = Figure(fig_size=(1, 1), dpi=500)
            ax = f.add_axes(ax_pos)
        else:
            f = self
            p_ax = self.add_panel(label=label, pos=self.panel_locs[label])
            ax = self.add_panel_axes(label=label, axes_pos=ax_pos)

        params = self.params['panel_' + label]
        params.update(panel_params)

        r_score = params['remap_score']
        b_score = params['behav_score']

        self.plot_unit_remap_v_beh(ax=ax, r_score=r_score, b_score=b_score, **params)

        if fig_template is None:
            return f

    def panel_h(self, fig_template=None, **panel_params):

        label = 'h'
        if fig_template is None:
            x0, y0 = 0.1, 0.1
            w, h = 0.8, 0.8
            ax_pos = [x0, y0, w, h]
            f = Figure(fig_size=(1, 1), dpi=500)
            ax = f.add_axes(ax_pos)
        else:
            x0, y0 = 0.32, 0.3
            w, h = 0.6, 0.6
            ax_pos = [x0, y0, w, h]
            f = self
            p_ax = self.add_panel(label=label, pos=self.panel_locs[label])
            ax = self.add_panel_axes(label=label, axes_pos=ax_pos)

        params = self.params['panel_' + label]
        params.update(panel_params)
        r_score = params['remap_score']
        b_score = params['behav_score']

        self.plot_session_remap_v_beh(ax=ax, r_score=r_score, b_score=b_score, **params)

        if fig_template is None:
            return f

    def panel_i(self, fig_template=None, **panel_params):

        label = 'i'
        x_split = 0.75
        if fig_template is None:
            x0, y0 = 0.1, 0.1
            w, h = 0.8, 0.8

            f = Figure(fig_size=(1, 1), dpi=500)
            p_ax = f.add_panel(label=label, pos=[0, 0, 1, 1], label_txt=False)
        else:
            x0, y0 = 0.32, 0.3
            w, h = 0.6, 0.6
            f = self
            p_ax = self.add_panel(label=label, pos=self.panel_locs[label])

        ax_pos = [[x0, y0, w * x_split, h],
                  [x0 + w * x_split, y0, w * (1 - x_split), h]]

        ax = np.zeros(2, dtype=object)
        for ii in range(2):
            ax[ii] = f.add_panel_axes(label=label, axes_pos=ax_pos[ii])

        params = self.params['panel_' + label]
        params.update(panel_params)
        r_score = params['remap_score']
        b_score = params['behav_score']

        self.plot_subject_remap_beh_slope(ax=ax, r_score=r_score, b_score=b_score, **params)

        if fig_template is None:
            return f

    def plot_unit_remap_v_beh(self, ax, r_score, b_score, **params):
        corr_method = params['corr_method']

        table = self.zrc_b.copy()
        table['r_score'] = table[r_score]
        table['r_nscore'] = -table['r_score']
        table['b_score'] = table[b_score]

        r = np.around(table[['b_score', 'r_score']].corr(method=corr_method).iloc[0, 1], 2)
        if r < 0:
            size_sign = 'r_nscore'
        else:
            size_sign = 'r_score'

        sns.scatterplot(ax=ax, x='r_score', y='b_score', data=table, hue='r_score', size=size_sign,
                        palette='crest_r', legend=False, alpha=params['dot_alpha'], sizes=params['dot_sizes'],
                        **{'linewidth': params['dot_lw'], 'edgecolor': params['dot_ec']})

        # regression line
        temp = table[['b_score', 'r_score']].dropna()
        x = temp['r_score'].values
        y = temp['b_score'].values

        if corr_method == 'pearson':
            m, b = np.polyfit(x, y, 1)
        else:
            m, b = stats.siegelslopes(y, x)

        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, m * xx + b, **params['reg_line_params'])

        if params['plot_ci']:
            if corr_method == 'pearson':
                yb, yu, xx = get_reg_ci(x, y, reg_type='linear', eval_x=xx)
            else:
                yb, yu, xx = get_reg_ci(x, y, eval_x=xx)
            ax.fill_between(xx, yb, yu, **params['ci_params'])

        # aesthetics
        ax.set_ylabel(r"$p_{se} (\%)$", fontsize=self.fontsize, labelpad=0)
        if 'corr_m' in r_score:
            ax.set_xlabel(r"$\bar \tau$", fontsize=self.fontsize, labelpad=0)
            xticks = [-1, 0, 1]
        elif 'zm' in r_score:
            xticks = np.floor(min(x)), np.ceil(max(x))
            xticks = np.sort(np.append(xticks, 0)).astype(int)

            if 'CR' in r_score:
                ax.set_xlabel(r"$\bar{z}_{\Delta \tau} \: Cue$", fontsize=self.fontsize, labelpad=0)
            elif 'Co' in r_score:
                ax.set_xlabel(r"$\bar{z}_{\Delta \tau} \: Rw$", fontsize=self.fontsize, labelpad=0)
            else:
                ax.set_xlabel(r"$\bar{z}_{\Delta \tau}$", fontsize=self.fontsize, labelpad=0)
        else:
            xticks = ax.get_xticks()

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=self.fontsize - 1)
        ax.tick_params(axis="both", direction="out", length=2, width=1, color='0.2', which='major',
                       pad=0.5, labelsize=self.fontsize)

        ax.set_ylim(0.25, 1.01)
        yticks = ax.get_yticks()
        ax.set_yticklabels((yticks * 100).astype(int))
        sns.despine(ax=ax)

        for sp in ['bottom', 'left']:
            ax.spines[sp].set_linewidth(1)
            ax.spines[sp].set_color('k')

        xt = ax.get_xlim()[0]
        xt = xt + np.abs(xt) * 0.1
        if corr_method == 'kendall':
            ax.text(xt, 0.3, r"$\tau={}$".format(r), fontsize=self.legend_fontsize)
        else:
            ax.text(xt, 0.3, r"$\rho={}$".format(r), fontsize=self.legend_fontsize)

        ax.grid(linewidth=0.5)

    def plot_session_remap_v_beh(self, ax, r_score, b_score, **params):

        corr_method = params['corr_method']
        table = self.zrc_b.copy()
        table['r_score'] = table[r_score]
        table['b_score'] = table[b_score]

        table = table[['subject', 'task', 'session', 'unit_type', 'r_score', 'b_score']].copy()
        table['x'] = table['r_score']
        table['y'] = table['b_score']
        table = table.dropna()

        if params['rescale_behav']:
            table['y'] = self._logit(table['y'])

        session_means = table.groupby(['subject', 'session']).mean()
        session_means['n'] = table.groupby(['subject', 'session']).size()

        sns.scatterplot(ax=ax, x='x', y='y', hue='subject', size='n', data=session_means,
                        legend=True, hue_order=self.summary_info.subjects,
                        alpha=params['dot_alpha'], sizes=params['dot_sizes'],
                        **{'linewidth': params['dot_lw'], 'edgecolor': params['dot_ec']})

        l = ax.get_legend()
        # hack to get seaborn scaled markers:
        o = []
        on_numerics = False
        for l_handle in l.legendHandles:
            if on_numerics:
                label = l_handle.properties()['label']
                l_handle.set_label = "n=" + label
                o.append(l_handle)
            elif l_handle.properties()['label'] == 'n':
                on_numerics = True
        legend_size_marker_handles = [o[ii] for ii in [0, -1]]
        l.remove()

        # regression line
        x = session_means['x']
        y = session_means['y']

        if corr_method == 'pearson':
            m, b = np.polyfit(x, y, 1)
        else:
            m, b = stats.siegelslopes(y, x)

        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, m * xx + b, **params['reg_line_params'])

        if params['plot_ci']:
            if corr_method == 'pearson':
                yb, yu, xx = get_reg_ci(x, y, reg_type='linear', eval_x=xx)
            else:
                yb, yu, xx = get_reg_ci(x, y, eval_x=xx)

            ax.fill_between(xx, yb, yu, **params['ci_params'])

        ax.grid(linewidth=0.5)
        ax.tick_params(axis="both", direction="out", length=3, width=1, color='0.2', which='major', pad=1,
                       labelsize=self.fontsize)

        if not params['rescale_behav']:
            ax.set_ylim(0.25, 1.01)
            yticks = ax.get_yticks()
            ax.set_yticklabels((yticks * 100).astype(int))
            ax.set_ylabel(r"$p_{se} (\%)$", fontsize=self.fontsize, labelpad=0)
        else:
            ax.set_ylabel(r"$p_{se}$ [logit]", fontsize=self.fontsize, labelpad=0)

        if 'zm' in r_score:
            xticks = np.floor(min(x)), np.ceil(max(x))
            xticks = np.sort(np.append(xticks, 0)).astype(int)
        else:
            xticks = ax.get_xticks()

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=self.fontsize - 1)
        ax.tick_params(axis="both", direction="out", length=2, width=1, color='0.2', which='major',
                       pad=0.5, labelsize=self.fontsize)

        if 'CR' in r_score:
            ax.set_xlabel(r"$\bar{z}_{\Delta \tau \: \bar{se}} \: Cue$", fontsize=self.fontsize, labelpad=0)
        elif 'Co' in r_score:
            ax.set_xlabel(r"$\bar{z}_{\Delta \tau \: \bar{se}} \: Rw$", fontsize=self.fontsize, labelpad=0)
        else:
            ax.set_xlabel(r"$\bar{z}_{\Delta \tau-se}$", fontsize=self.fontsize, labelpad=0)
        sns.despine(ax=ax)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_linewidth(1)
            ax.spines[sp].set_color('k')

        xt = 0.1
        r = np.around(session_means[['x', 'y']].corr(method=corr_method).iloc[0, 1], 2)
        if corr_method == 'kendall':
            ax.text(xt, 0.05, r"$\tau$={}".format(r), fontsize=self.legend_fontsize, transform=ax.transAxes)
        else:
            ax.text(xt, 0.05, r"$\rho$={}".format(r), fontsize=self.legend_fontsize, transform=ax.transAxes)

        legend_params = dict(handlelength=0.25, handletextpad=0.2, bbox_to_anchor=[1.0, 1], loc='upper left',
                             frameon=False, labelspacing=0.02,
                             fontsize=self.legend_fontsize - 1)

        legend_elements = []
        pal = params['color_pal']
        for ii, ss in enumerate(self.summary_info.subjects[:-1]):
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', alpha=params['dot_alpha'], markersize=params['legend_markersize'],
                           lw=0, mew=0, color=pal[ii], label=f"s$_{{{ii + 1}}}$"))
        l1 = ax.legend(handles=legend_elements, **legend_params)

        legend_params = dict(handlelength=0.25, handletextpad=0.3, bbox_to_anchor=[1.0, 0], loc='lower left',
                             frameon=False, fontsize=self.legend_fontsize - 1,
                             labelspacing=0.1)

        l2 = ax.add_artist(plt.legend(handles=legend_size_marker_handles, **legend_params))
        ax.add_artist(l1)

    def plot_subject_remap_beh_slope(self, ax, r_score, b_score, **params):

        corr_method = params['corr_method']
        table = self.zrc_b.copy()
        table['r_score'] = table[r_score]
        table['b_score'] = table[b_score]

        table = table[['subject', 'task', 'session', 'unit_type', 'r_score', 'b_score']].copy()
        table['x'] = table['r_score']
        table['y'] = table['b_score']
        table = table.dropna()

        if params['rescale_behav']:
            table['y'] = self._logit(table['y'])

        table_m = table.groupby(['subject', 'task', 'session']).mean()
        table_m = table_m.reset_index()
        subjects = self.summary_info.subjects
        n_subjects = len(subjects)
        n_boot = 500
        rb = np.zeros((n_subjects, n_boot)) * np.nan
        for ii, ss in enumerate(subjects):
            sub_table = table_m.loc[table_m.subject == ss, ['x', 'y']]
            rb[ii] = rs.bootstrap_corr(sub_table.x.values, sub_table.y.values, n_boot, corr_method=corr_method)
        boot_behav_remap = pd.DataFrame(rb.T, columns=subjects).melt(value_name='r', var_name='subject')

        subset = boot_behav_remap.dropna()
        pal = params['color_pal']
        sns.pointplot(ax=ax[0], x='subject', y='r', hue='subject', data=subset, palette=pal,
                      **params['point_plot_params'])
        ax[0].get_legend().remove()

        ylabel = r'$\tau_{(\bar z_{\Delta \tau}, p_{se})}$'
        if 'CR' in r_score:
            ylabel +='-Cue'
        elif 'Co' in r_score:
            ylabel +='-Rw'

        ax[0].set_ylabel(ylabel, fontsize=self.fontsize)
        ax[0].set_xticklabels([f"s$_{{{ii}}}$" for ii in range(1, len(subjects))], fontsize=self.fontsize)
        ax[0].set_xlabel('Subjects', fontsize=self.fontsize)
        ax[0].xaxis.set_label_coords(0.5+1/6, -0.2, transform=ax[0].transAxes)

        for spine in ['top', 'right']:
            ax[0].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax[0].spines[spine].set_linewidth(1)
            ax[0].spines[spine].set_color('k')

        # summary
        scale = params['summary_xrange_scale']
        subset2 = boot_behav_remap.groupby("subject").mean().loc[subjects[:-1]]
        subset2['color'] = pal[:len(subset2)]
        x_locs = np.random.rand(len(subset2)) * scale
        x_locs = x_locs - x_locs.mean()
        ax[1].scatter(x_locs, subset2.r, c=subset2.color, **params['scatter_summary_params'])

        ax[1].plot(np.array([-1, 1]) * scale, [subset2.r.mean()] * 2, lw=params['summary_lw'],
                   color=params['summary_c'])

        ax[1].set_xlim(np.array([-1, 1]) * scale * 2)
        ax[1].set_xticks([0])
        ax[1].set_xticklabels([r" $\bar s$ "], fontsize=self.fontsize)
        for spine in ['top', 'right', 'left']:
            ax[1].spines[spine].set_visible(False)
        for spine in ['bottom']:
            ax[1].spines[spine].set_linewidth(1)
            ax[1].spines[spine].set_color('k')

        for ii in range(2):
            ax[ii].set_ylim(-1.01, 1.01)
            ax[ii].set_yticks([-1, 0, 1])
            ax[ii].set_yticklabels([-1, 0, 1], fontsize=self.fontsize)
            ax[ii].tick_params(axis="both", direction="out", length=3, width=1, color='0.2', which='major', pad=0.5,
                              labelsize=self.fontsize)
            ax[ii].grid(linewidth=0.5)
        ax[1].tick_params(axis='y', left=False)
        ax[1].set_yticklabels(['']*3)

    def _combine_tables(self, zrc, b_table):
        zrc_b = zrc.copy()
        b_table = b_table.copy()
        b_table.set_index('session', inplace=True)

        b_cols = ['pct_correct', 'pct_sw_correct', 'pct_vsw_correct', 'pct_L_correct', 'pct_R_correct']
        for session in b_table.index:
            z_index = zrc_b.session == session
            zrc_b.loc[z_index, b_cols] = b_table.loc[session, b_cols].values

        zrc_b['task'] = zrc_b.session.apply(self._get_task_from_session)

        if self.unit_type != 'all':
            zrc_b = zrc_b[zrc_b.unit_type == self.unit_type]
        return zrc_b

    @staticmethod
    def _logit(p):
        return np.log(p / (1 - p))

    @staticmethod
    def _get_task_from_session(session):
        return session.split("_")[1]


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


def plot_kde_dist(data, color='k', lw=1.0, v_lines=None, v_ls=':', v_lw=0.5, ax=None, label=None, smoothing=1,
                  **kde_params):
    if v_lines is None:
        v_lines = []

    if ax is None:
        _, ax = plt.subplots()

    sns.kdeplot(data=data, fill=False, color=color, linewidth=lw, ax=ax, label=label, bw_adjust=smoothing, **kde_params)

    if 'alpha' in kde_params.keys():
        alpha = kde_params['alpha']
    else:
        alpha = 1

    if v_lines is not None:
        xt, yt = ax.lines[0].get_data()
        try:
            for x_loc in v_lines:
                y_height = yt[np.argmin(abs(xt - x_loc))]
                ax.plot([x_loc] * 2, [0, y_height], color=color, linestyle=v_ls, linewidth=v_lw,
                        zorder=-1, alpha=alpha)
        except TypeError:
            x_loc = v_lines
            y_height = yt[np.argmin(abs(xt - x_loc))]
            ax.plot([x_loc] * 2, [0, y_height], color=color, linestyle=v_ls, linewidth=v_lw,
                    zorder=-1, alpha=alpha)
        finally:
            return ax

    return ax


# noinspection PyTypeChecker
def get_colors_from_data(data, **args):
    """
    provides an ordered colored array for the data.
    returns an array with colors codes of the same length as data.
    :param data: array of data
    :param n_color_bins: number of color bins to discretize the data
    :param color_map: colormap
    :param nans_2_zeros: if true, converts nan values to zero
    :param div: if true, colors go from -max to max
    :param max_value:
    :param min_value:
    :param color_values_array:
    :return:
    """
    data = np.copy(data)

    nan_idx = np.isnan(data)
    data[nan_idx] = 0

    params = dict(color_map='RdBu_r',
                  n_color_bins=25, nans_2_zeros=True, div=False,
                  max_value=None, min_value=None, color_values_array=None)

    params.update(args)
    if params['color_values_array'] is not None:
        color_values_array = params['color_values_array']
        params['n_color_bins'] = len(color_values_array)
    else:
        if params['div']:
            if params['max_value'] is None:
                max_value = np.ceil(np.max(np.abs(data)) * 100) / 100
                min_value = -max_value
            else:
                max_value = params['max_value']
                min_value = params['min_value']
        else:
            if params['max_value'] is None:
                max_value = np.ceil(np.max(data) * 100) / 100
            else:
                max_value = params['max_value']

            if params['min_value'] is None:
                min_value = np.ceil(np.min(data) * 100) / 100
            else:
                min_value = params['min_value']
        color_values_array = np.linspace(min_value, max_value, params['n_color_bins'] - 1)

    color_val_idx = np.digitize(data, color_values_array).astype(int)
    color_map = np.array(sns.color_palette(params['color_map'], params['n_color_bins']))

    data_colors = color_map[color_val_idx]

    if not params['nans_2_zeros']:
        data_colors[nan_idx] = np.ones(3) * np.nan

    return data_colors, color_values_array


def get_reg_ci(x, y, reg_type='siegel', nboot=100, alpha=0.05, eval_x=None):
    n = len(x)
    assert n == len(y)

    boot_mb = np.zeros((nboot, 2))

    if reg_type == 'siegel':
        xp = y.copy()
        yp = x.copy()
        reg_func = stats.siegelslopes
    else:
        xp = x
        yp = y

        def reg_func(_x, _y):
            return np.polyfit(_x, _y, 1)

    for boot in range(nboot):
        samps = np.random.choice(n, n)
        boot_mb[boot, :] = reg_func(xp[samps], yp[samps])

    if eval_x is None:
        xx = np.linspace(x.min(), x.max(), 100)
    else:
        xx = eval_x
    y_mb = boot_mb[:, 1][:, np.newaxis] + np.outer(boot_mb[:, 0], xx)
    y_bot, y_top = np.percentile(y_mb, np.array([alpha / 2, 1 - alpha / 2]) * 100, axis=0)

    return y_bot, y_top, xx


def get_color_bar_axis(cax, color_array, color_map='cividis', **args):
    params = dict(tick_fontsize=7,
                  label_fontsize=7)

    params.update(args)

    color_norm = mpl.colors.Normalize(vmin=color_array[0], vmax=color_array[-1])  # , clip=True)
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=color_norm)
    sm.set_array([])

    color_bar = plt.colorbar(sm, cax=cax)
    # color_bar.set_ticks([0, color_array[-1]])
    # cax.set_yticklabels(['', int(color_array[-1])], fontsize=params['tick_fontsize'], ha='center')
    # cax.set_yticklabels([''], fontsize=params['tick_fontsize'], ha='center')
    color_bar.set_ticks([])
    # cax.set_yticklabels([''])
    cax.tick_params(pad=4, length=0, grid_linewidth=0)

    cax.text(1.05, 1, int(color_array[-1]), fontsize=params['tick_fontsize'], ha='left', va='center',
             transform=cax.transAxes)
    if 'label' in params:
        # cax.set_ylabel(params['label'], rotation='horizontal', va='top', ha='right',
        #                fontsize=params['label_fontsize'], labelpad=0)

        cax.text(1.05, 0, params['label'], fontsize=params['label_fontsize'], ha='left', va='center',
                 transform=cax.transAxes)

    for pos in ['right', 'top', 'bottom', 'left']:
        cax.spines[pos].set_visible(False)
    color_bar.outline.set_color("None")

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
