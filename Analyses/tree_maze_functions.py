import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal

import Analyses.spatial_functions as spatial_funcs
import Utils.filter_functions as filt_funcs
from scipy.signal import filtfilt

import Pre_Processing.pre_process_functions as pp_funcs

import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import Analyses.plot_functions as pf
import Analyses.subject_info as si
import Utils.filter_functions as filter_funcs

class Points2D:
    def __init__(self, x, y, polar=False):

        if not isinstance(x, np.ndarray):
            x = np.array([x]).flatten()
        if not isinstance(y, np.ndarray):
            y = np.array([y]).flatten()

        assert len(x) == len(y), 'different lengths'

        self.n = len(x)
        if not polar:
            self.x = np.array(x)
            self.y = np.array(y)
            self.xy = np.column_stack((self.x, self.y))
            self.r, self.ang = self.polar()
        else:
            self.r = x
            self.ang = np.mod(y, 2 * np.pi)
            self.x, self.y = self.eu()
            self.xy = np.column_stack((self.x, self.y))

    def polar(self):
        r = np.sqrt(self.x ** 2 + self.y ** 2)
        ang = np.zeros(self.n)

        for ii in range(self.n):
            ang[ii] = np.math.atan2(self.y[ii], self.x[ii])
        ang = np.mod(ang, 2 * np.pi)
        return r, ang

    def eu(self):
        x = self.r * np.cos(self.ang)
        y = self.r * np.sin(self.ang)
        return x, y

    def half_ang(self):
        ang2 = np.array(self.ang)
        ang2[ang2 > np.pi] = np.mod(ang2[ang2 > np.pi], -np.pi)
        return ang2

    def __add__(self, b):
        if isinstance(b, (int, float)):
            return Points2D(self.x + b, self.y + b)

        if isinstance(b, Points2D):
            return Points2D(self.x + b.x, self.y + b.y)
        else:
            raise NotImplementedError

    def __sub__(self, b):
        if isinstance(b, (int, float)):
            return Points2D(self.x - b, self.y - b)

        if isinstance(b, Points2D):
            return Points2D(self.x - b.x, self.y - b.y)
        else:
            raise NotImplementedError

    def __rsub__(self, b):
        if isinstance(b, (int, float)):
            return Points2D(b - self.x, b - self.y)

        if isinstance(b, Points2D):
            return Points2D(b.x - self.x, b.y - self.y)
        else:
            raise NotImplementedError

    def __mul__(self, b):
        if isinstance(b, (int, float, np.float, np.int)):
            return Points2D(b * self.x, b * self.y)

        if isinstance(b, Points2D):
            return b.x @ self.x + b.y @ self.y
        else:
            raise NotImplementedError

    def __rmul__(self, b):
        if isinstance(b, (int, float, np.float, np.int)):
            return Points2D(b * self.x, b * self.y)
        elif isinstance(b, Points2D):
            if self.n == b.n:
                return Points2D(b.x * self.x, b.y @ self.y)
            if self.n == 1 or b.n == 1:
                return
        else:
            raise NotImplementedError

    def __truediv__(self, b):
        return Points2D(self.r / b, self.ang, polar=True)

    def __rdiv__(self, b):
        return Points2D(self.r / b, self.ang, polar=True)

    def __getitem__(self, i):
        if isinstance(i, (int, np.int, np.ndarray)):
            return Points2D(self.x[i], self.y[i])
        else:
            raise NotImplementedError

    def __len__(self):
        return self.n

    def __str__(self):
        print((self.x, self.y))
        return ''

    def _repr_html_(self):
        print(self.xy)
        return ''


class TreeMazeZones:
    # zones_coords = {'Home': [(-300, -80), (-300, 80), (300, 80), (300, -80)],
    #                 'Desc': [(-80, 500), (-95, 400), (-150, 400), (-150, 652),
    #                          (-75, 550), (0, 607), (75, 550), (150, 652), (150, 400),
    #                          (95, 400), (80, 500)],
    #                 'A': [(-150, 80), (-80, 500), (80, 500), (150, 80)],
    #                 'B': [(0, 607), (0, 700), (200, 1000), (329, 900), (75, 550)],
    #                 'C': [(610, 1180), (610, 800), (329, 900), (450, 1180)],
    #                 'D': [(200, 1000), (50, 1230), (450, 1230), (450, 1180)],
    #
    #                 'E': [(0, 607), (0, 700), (-200, 1000), (-329, 900), (-75, 550)],
    #                 'F': [(-200, 1000), (-50, 1230), (-450, 1230), (-450, 1180)],
    #                 'G': [(-610, 1180), (-610, 800), (-329, 900), (-450, 1180)],
    #
    #                 'G1': [(610, 1180.5), (800, 1180.5), (800, 800), (610, 800)],
    #                 'G2': [(50, 1230), (50, 1450), (450, 1450), (450, 1230)],
    #                 'G3': [(-50, 1230), (-50, 1450), (-450, 1450), (-450, 1230)],
    #                 'G4': [(-610, 1180.5), (-800, 1180.5), (-800, 800), (-610, 800)],
    #
    #                 'i1': [(200, 1000), (450, 1180), (329, 900)],
    #                 'i2': [(-329, 900), (-450, 1180), (-200, 1000)],
    #                 }

    zones_coords = {'Home': [(-300, -80), (-300, 80), (300, 80), (300, -80)],
                    'Desc': [(-80, 500), (-88, 450), (-150, 450), (-150, 600),
                             (0, 700), (150, 600), (150, 450),
                             (88, 450), (80, 500)],

                    'A': [(-150, 80), (-80, 500), (80, 500), (150, 80)],
                    'B': [(0, 700), (220, 990), (329, 900), (150, 600)],
                    'C': [(570, 1180), (620, 800), (329, 900), (450, 1180)],
                    'D': [(220, 990), (50, 1180), (450, 1300), (450, 1180)],

                    'E': [(0, 700), (-220, 990), (-329, 900), (-150, 600)],
                    'F': [(-220, 990), (-50, 1250), (-450, 1250), (-450, 1180)],
                    'G': [(-600, 1180), (-600, 800), (-329, 900), (-450, 1180)],

                    'G1': [(570, 1180.5), (800, 1180.5), (800, 800), (620, 800)],
                    'G2': [(50, 1180), (50, 1450), (450, 1450), (450, 1300)],
                    'G3': [(-50, 1250), (-50, 1450), (-450, 1450), (-450, 1250)],
                    'G4': [(-600, 1180.5), (-800, 1180.5), (-800, 800), (-600, 800)],

                    'i1': [(220, 990), (450, 1180), (329, 900)],
                    'i2': [(-329, 900), (-450, 1180), (-220, 990)],
                    }


    zone_names = list(zones_coords.keys())
    zone_label_coords = {'Home': (0, 0),
                         'Desc': (155, 500),
                         'A': (0, 250),
                         'B': (170, 800),
                         'C': (500, 1000),
                         'D': (250, 1140),
                         'E': (-170, 800),
                         'G': (-500, 1000),
                         'F': (-250, 1140),
                         'G1': (730, 1000),
                         'G2': (250, 1300),
                         'G3': (-250, 1300),
                         'G4': (-730, 1000),
                         'i1': (300, 1000),
                         'i2': (-300, 1000),
                         }

    zones_geom = {}
    for zo in zone_names:
        zones_geom[zo] = Polygon(zones_coords[zo])

    dirs = {'out': {'A': 'N',
                    'B': 'N', 'C': 'E', 'D': 'N',
                    'E': 'N', 'F': 'N', 'G': 'W'},
            'in': {'A': 'S',
                   'B': 'S', 'C': 'W', 'D': 'S',
                   'E': 'S', 'F': 'S', 'G': 'E'}
            }

    linear_segs = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    stem_seg = ['A']

    split_segs = {'left': {'goals': ['G1', 'G2'],
                           'segs': ['B', 'C', 'D'],
                           'intersection': ['i1']},
                  'right': {'goals': ['G3', 'G4'],
                            'segs': ['E', 'F', 'G'],
                            'intersection': ['i2']},
                  'stem': {'goals': ['Home', 'Desc'],
                           'segs': ['SegA'],
                           'intersection': ['i2']}
                  }

    seg_splits = {'Home': 'stem',
                  'Desc': 'stem',
                  'A': 'stem',
                  'B': 'right',
                  'C': 'right',
                  'D': 'right',
                  'E': 'left',
                  'G': 'left',
                  'F': 'left',
                  'G1': 'right',
                  'G2': 'right',
                  'G3': 'left',
                  'G4': 'left',
                  'i1': 'right',
                  'i2': 'left',
                  }

    split_colors = {'left': 'green', 'right': 'purple', 'stem': 'grey'}

    def __init__(self, sub_seg_length=60):
        self._dir_num_dict = {a: b for a, b in zip(['E', 'N', 'W', 'S'], range(1, 5))}

        self.sub_segs = {seg: {} for seg in self.linear_segs}
        for seg in self.linear_segs:
            for lin_dir in ['in', 'out']:
                self.sub_segs[seg][lin_dir] = self.divide_seg(seg, sub_seg_length, lin_dir)

    def divide_seg(self, seg_name, sub_seg_length, direction='out'):

        seg_dir = self.dirs[direction][seg_name]
        seg_dir_num = self._dir_num_dict[seg_dir]

        # bounding rectangle coords
        seg_geom = self.zones_geom[seg_name]
        xx, yy = seg_geom.minimum_rotated_rectangle.exterior.xy
        pp = Points2D(xx, yy)

        # get ccw line vectors for bounding rect
        l_segs = np.diff(pp.xy, axis=0)
        l_segs_v = Points2D(l_segs[:, 0], l_segs[:, 1])

        # get direction
        l_segs_dirs = np.digitize(l_segs_v.ang, np.arange(-np.pi / 4, 2 * np.pi + np.pi / 4 + 0.001, np.pi / 2))
        l_segs_dirs[l_segs_dirs == 5] = 1

        l_dir_id = np.where(l_segs_dirs == seg_dir_num)[0][0]

        L = l_segs[l_dir_id]
        L = Points2D(L[0], L[1])

        a = pp.xy[:4][l_dir_id - 1]
        a = Points2D(a[0], a[1])
        b = pp.xy[:4][l_dir_id]
        b = Points2D(b[0], b[1])

        n_subsegs = int(L.r // sub_seg_length)

        delta = L.r / n_subsegs
        sub_segs = np.zeros(n_subsegs, dtype=object)

        for ii in range(n_subsegs):
            p0 = Points2D(ii * delta, L.ang, polar=True) + a
            p1 = Points2D((ii + 1) * delta, L.ang, polar=True) + a
            p2 = Points2D((ii + 1) * delta, L.ang, polar=True) + b
            p3 = Points2D(ii * delta, L.ang, polar=True) + b

            sub_seg = Polygon([p0.xy[0],
                               p1.xy[0],
                               p2.xy[0],
                               p3.xy[0]])

            sub_segs[ii] = seg_geom.intersection(sub_seg)
        return sub_segs

    def plot_maze(self, axis=None, sub_segs=None, seg_dir=None, zone_labels=False,
                  seg_color='powderblue', seg_alpha=0.3, lw=1.5, line_alpha=1,
                  sub_seg_color=None, sub_seg_lw=0, font_size=10):
        """
        method to plot the maze with various aesthetic options
        :param axis: axis of the figure, if None, creates figure
        :param sub_segs: None, list, or str
            None -> no subsegments are plotted
            list -> subsegments to plot,
            'all' -> to indicate all subsegments
        :param seg_dir: None or str, ignored if sub_segs==None
            None -> no gradient of directionality applied; and 'out'
            'in', 'out' -> colors the segments in the direction of a inward or outward trajectory
        :param zone_labels: bool, if true, labels the displayed
        :param seg_color: shade color of the segments
        :param seg_alpha: alpha level of segment
        :param lw: float, line width
        :param line_alpha: float [0-1], alpha level of segment lines
        :param sub_seg_color: str, or dict for colors by goal
            str -> a valid color string
            'cue'-> uses default class colors for segments
            dict -> if a dict, mapping of colors to subsegments
        :param sub_seg_lw: float[0-1]
        :return:
        """

        plt.rcParams['lines.dash_capstyle'] = 'round'
        plt.rcParams['lines.solid_capstyle'] = 'round'

        if axis is None:
            f, axis = plt.subplots(figsize=(10, 10))

        seg_color_plot = {}
        if seg_color == 'cue':
            for zone in self.zone_names:
                seg_color_plot[zone] = self.split_colors[self.seg_splits[zone]]
        elif type(seg_color) == str:
            for zone in self.zone_names:
                seg_color_plot[zone] = seg_color
        elif type(seg_color) == dict:
            seg_color_plot = seg_color
        else:
            print("Error. invalid seg_color entry.")

        for zone, geom in self.zones_geom.items():
            pf.plotPoly(geom, axis, alpha=seg_alpha,
                        color=seg_color_plot[zone], lw=lw, line_alpha=line_alpha)

        if zone_labels:
            for zone, coords in self.zone_label_coords.items():
                if zone == 'Desc':
                    axis.text(coords[0], coords[1], zone, fontsize=font_size,
                              horizontalalignment='left', verticalalignment='center')
                else:
                    axis.text(coords[0], coords[1], zone, fontsize=font_size,
                              horizontalalignment='center', verticalalignment='center')

        if sub_segs is not None:
            if sub_segs == 'all':
                sub_segs_to_plot = self.linear_segs
            else:
                sub_segs_to_plot = sub_segs

            if seg_dir is None:
                inout = 'out'
            else:
                inout = seg_dir

            for zone in sub_segs_to_plot:
                if sub_seg_color is None:
                    col = None
                elif sub_seg_color == 'cue':
                    col = self.split_colors[self.seg_splits[zone]]
                elif type(sub_seg_color) == str:
                    col = sub_seg_color
                else:
                    col = sub_seg_color[zone]

                sub_seg = self.sub_segs[zone][inout]
                n_segs = len(sub_seg)
                if seg_dir is not None:
                    alphas = 0.5 / n_segs * np.arange(n_segs)
                else:
                    alphas = np.ones(n_segs) * seg_alpha
                for ii in range(n_segs):
                    pf.plotPoly(sub_seg[ii], axis, alpha=alphas[ii], color=col, lw=sub_seg_lw)

        axis.axis('off')
        axis.axis('equal')
        return axis


# class BehaviorData:
#     def __init__(self, session_info: si.SubjectSessionInfo):
#         self.t_rs = session_info.get_time()
#         # get raw behavior
#         t_vt, x_vt, y_vt, ha_vt = session_info.get_raw_track_data()
#


def get_tree_maze_track_data(session_info: si.SubjectSessionInfo):

    # get resampled time
    t_rs = session_info.get_time()

    # get raw behavior
    t_vt, x_vt, y_vt, ha_vt = session_info.get_raw_track_data()

    # convert head angle into radians
    ha_vt = np.mod(np.deg2rad(ha_vt), 2 * np.pi)  # convert to radians.

    # pre process track data
    t1 = time.time()
    x_s, y_s , ha_s, nan_vals_idx =  pre_process_track_data(x_vt, y_vt, ha_vt, t_vt,
                                                            t_rs, session_info.task_params)
    t2 = time.time()
    print('track data pre-processing completed: {0:0.2f} s '.format(t2 - t1))
    

def pre_process_track_data(x, y, ha, t, t_rs, track_params):
    """
    This function performs the following tasks in order:
        1) masks xy pixel data for out track bound spaces. values set to np.nan
        2) centers the xy pixel data
        3) rotates xy to the experimenters perspective
        4) rescales data from pixels to mm
        5) masks velocity to create a masks for a velocity threshold
        6) fill in nan values with last value
        7) apply median filters
        8) final smoothing using filtfilt
    :param x: tracking data x position
    :param y: tracking data y position
    :param ha: tracking data head angle
    :param track_params: parameters of the track
    :return: processed x,y, ha, and nan_idx
    """

    p = track_params

    # make deep copies
    x = x.copy()
    y = y.copy()

    # 1. mask pixels that are out of bounds
    mask_x = np.logical_or(x < p.x_pix_lims[0], x > p.x_pix_lims[1])
    mask_y = np.logical_or(y < p.y_pix_lims[0], y > p.y_pix_lims[1])
    mask = np.logical_or(mask_x, mask_y)

    x[mask] = np.nan
    y[mask] = np.nan

    # 2. centering / pixel translation
    x2 = x + p.x_pix_bias
    y2 = y + p.y_pix_bias

    # 3. rotate to experimenter's pov
    x3, y3 = spatial_funcs.rotate_xy(x2, y2, p.xy_pix_rot_rad)

    # 4. convert to mm / re-scales; bias term re-frames the image
    x4 = x3 * p.x_pix_mm + p.x_mm_bias
    y4 = y3 * p.y_pix_mm + p.y_mm_bias

    # 5. filter by valid speed values
    with np.errstate(invalid='ignore'):  # avoids warnings about comparing nan values
        # 5a. compute velocity to create speed threshold
        dx = np.append(0, np.diff(x4))
        dy = np.append(0, np.diff(y4))
        dr = np.sqrt(dx ** 2 + dy ** 2)
        mask_r = np.abs(dr) > p.max_speed_thr

        # 5b. mask creating out of bound zones in mm space
        mask_x = np.logical_or(x4 < p.x_mm_lims[0], x4 > p.x_mm_lims[1])
        mask_y = np.logical_or(y4 < p.y_mm_lims[0], y4 > p.y_mm_lims[1])
        mask = np.logical_or(mask_x, mask_y)
        mask = np.logical_or(mask, mask_r)

    # 5c. apply masks
    x5 = x4.copy()
    y5 = y4.copy()
    x5[mask] = np.nan
    y5[mask] = np.nan
    ha5 = ha.copy()
    ha5[mask] = np.nan

    # get nan idx for future use.
    nan_idx = np.where(np.logical_or(np.isnan(x5), np.isnan(y5)))[0]

    # 6. fill in nans
    x6 = filter_funcs.fill_nan_vals(x5)
    y6 = filter_funcs.fill_nan_vals(y5)
    ha6 = filter_funcs.fill_nan_vals(ha5)

    # 7. median filter
    x7 = filter_funcs.median_window_filter_causal(x6, p.temporal_window_size)
    y7 = filter_funcs.median_window_filter_causal(y6, p.temporal_window_size)
    ha7 = filter_funcs.median_window_filter_causal(ha6, p.temporal_angle_window_size)

    # 8. final filter / smoothing
    x8 = filtfilt(p.filter_coef_, 1, x7)
    y8 = filtfilt(p.filter_coef_, 1, y7)
    ha8 = filter_funcs.angle_filtfilt(ha7, p.filter_coef_angle_)

    # 9. resample
    # resampling the data
    x9 = filt_funcs.resample_signal(t, t_rs, x8)
    y9 = filt_funcs.resample_signal(t, t_rs, y8)
    ha9 = filt_funcs.resample_signal(t, t_rs, ha8)

    return x9, y9, ha9, nan_idx


def correctXY(EventDat, x, y):
    xd = [0, 0, 650, 250, -250, -650]
    yd = [45, 560, 1000, 1280, 1280, 1000]

    x2 = np.array(x)
    y2 = np.array(y)

    for z1 in ['D', 'R']:
        cnt = 0
        for z2 in ['H', 'C', '1', '2', '3', '4']:
            z = z1 + z2
            ids = EventDat[z] == 1
            x2[ids] = xd[cnt]
            y2[ids] = yd[cnt]
            cnt += 1

    x2 = filt_funcs.median_window_filtfilt(x2, 5)
    y2 = filt_funcs.median_window_filtfilt(y2, 5)
    for z1 in ['D', 'R']:
        cnt = 0
        for z2 in ['H', 'C', '1', '2', '3', '4']:
            ids = EventDat[z] == 1
            x2[ids] = xd[cnt]
            y2[ids] = yd[cnt]
            cnt += 1

    return x2, y2
