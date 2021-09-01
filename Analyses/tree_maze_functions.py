import os
from pathlib import Path

import numpy as np
import pandas as pd
from types import SimpleNamespace

import Analyses.spatial_functions as spatial_funcs
import Utils.filter_functions as filt_funcs

import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import Analyses.plot_functions as pf
import Analyses.experiment_info as si
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


class Lines2Polygons:

    def extend_line(self, line_points, buffer):

        idx = self.sort_points_idx(line_points)
        sorted_points = line_points[idx]

        p0 = sorted_points[0]
        p1 = sorted_points[1]

        th = self.get_line_angle(p0, p1)
        wh = buffer * np.array((np.cos(th), np.sin(th)))

        out = np.array((p0 - wh, p1 + wh))

        if idx[0] == 0:
            return out
        else:
            return out[::-1]

    def project_line(self, line_points, l):
        p0 = line_points[0]
        p1 = line_points[1]

        th = self.get_line_angle(p0, p1)

        th_d = np.rad2deg(th)
        pi2 = np.pi / 2

        if (th_d >= 0) & (th_d <= 180):
            wh = l * np.array((np.cos(pi2 - th), -np.sin(pi2 - th)))
        else:
            wh = l * np.array((np.cos(5 * pi2 - th), -np.sin(5 * pi2 - th)))

        out = np.array((p0 + wh, p1 + wh))

        return out

    def project_line_repeats(self, line_points, l, n_lines):
        l_s = np.arange(-n_lines * l, n_lines * l + 1, l)

        n_l_s = len(l_s)
        lines = np.zeros((n_l_s, 2, 2))

        for ii in range(n_l_s):
            lines[ii] = self.project_line(line_points, l_s[ii])
        return lines

    def make_poly_sequence_from_parallel_lines(self, lines, order='cw'):
        n_lines = lines.shape[0]
        n_polygons = n_lines - 1
        p = np.zeros(n_polygons, dtype=object)

        for ii in range(n_polygons):
            p[ii] = self.make_polygon_from_2_lines(lines[ii], lines[ii + 1])

        if order == 'cw':
            return p
        else:
            return p[::-1]

    @staticmethod
    def get_line_angle(p0, p1):
        return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

    @staticmethod
    def sort_points_idx(points):
        points = np.asarray(points)
        idx = np.argsort(points[:, 1])
        return idx

    @staticmethod
    def make_polygon_from_2_lines(line1, line2):

        coords = np.zeros((4, 2), dtype=object)
        coords[0, :] = line1[0]
        coords[1, :] = line1[1]
        coords[2, :] = line2[1]
        coords[3, :] = line2[0]

        return Polygon(coords)

    @staticmethod
    def plot_lines(lines, ax=None):
        if ax is None:
            f, ax = plt.subplots()

        lines = np.asarray(lines)
        if lines.ndim <= 1:
            print("invalid input")
            return None
        elif lines.ndim == 2:
            lines = lines[np.newaxis,]

        n_lines = lines.shape[0]

        for ii in range(n_lines):
            line = lines[ii]
            ax.plot(line[:, 0], line[:, 1])
            ax.scatter(line[0, 0], line[0, 1], color='k')
            ax.scatter(line[1, 0], line[1, 1], marker='d', color='k')

        return ax

    @staticmethod
    def plot_polygon_set(polygon_set, ax=None):
        if ax is None:
            f, ax = plt.subplots()
        for ii, pp in enumerate(polygon_set):
            pf.plot_poly(pp, ax, color=f"{ii / 10}")
        return ax


class TreeMazeZones:
    zones_coords = {'H': [(-250, -80), (-250, 80), (250, 80), (250, -80)],
                    'D': [(-80, 500), (-88, 450), (-150, 450), (-150, 600),
                          (0, 700), (150, 600), (150, 450),
                          (88, 450), (80, 500)],

                    'a': [(150, 80), (-150, 80), (-80, 500), (80, 500)],

                    'b': [(150, 600), (0, 700), (220, 990), (329, 900)],
                    'e': [(0, 700), (-150, 600), (-329, 900), (-220, 990)],

                    'c': [(560, 800), (520, 1180), (450, 1180), (329, 900)],
                    'd': [(450, 1260), (50, 1190), (220, 990), (450, 1180)],
                    'f': [(-50, 1190), (-450, 1260), (-450, 1180), (-220, 990)],
                    'g': [(-520, 1180), (-560, 800), (-329, 900), (-450, 1180)],

                    'G1': [(520, 1180.5), (800, 1180.5), (800, 800), (560, 800)],
                    'G2': [(50, 1190), (50, 1450), (450, 1450), (450, 1260)],
                    'G3': [(-50, 1190), (-50, 1450), (-450, 1450), (-450, 1260)],
                    'G4': [(-520, 1180.5), (-800, 1180.5), (-800, 800), (-560, 800)],

                    'i1': [(220, 990), (450, 1180), (329, 900)],
                    'i2': [(-329, 900), (-450, 1180), (-220, 990)],
                    }

    n_zones = len(zones_coords)
    zone_names = ['H', 'a', 'D', 'b', 'i1', 'c', 'G1', 'd', 'G2', 'e', 'i2', 'f', 'G3', 'g', 'G4']
    zone_label_coords = {'H': (0, -140),
                         'D': (155, 500),
                         'a': (180, 250),
                         'b': (290, 710),
                         'c': (500, 780),
                         'd': (80, 1100),
                         'e': (-290, 710),
                         'g': (-500, 780),
                         'f': (-80, 1100),
                         'G1': (880, 1000),
                         'G2': (530, 1350),
                         'G3': (-530, 1350),
                         'G4': (-880, 1000),
                         'i1': (305, 992),
                         'i2': (-305, 992),
                         }

    zones_geom = {}
    for zo in zone_names:
        zones_geom[zo] = Polygon(zones_coords[zo])

    cue_coords = [(-150, 1600), (-150, 1900), (150, 1900), (150, 1600)]
    cue_geom = Polygon(cue_coords)
    cue_label_coords = [0, 1750]

    dirs = {'out': {'a': 'N',
                    'b': 'N', 'c': 'E', 'd': 'N',
                    'e': 'N', 'f': 'N', 'g': 'W'},
            'in': {'a': 'S',
                   'b': 'S', 'c': 'W', 'd': 'S',
                   'e': 'S', 'f': 'S', 'g': 'E'}
            }

    linear_segs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    stem_seg = ['a']

    split_segs = {'right': {'goals': ['G1', 'G2'],
                            'segs': ['b', 'c', 'd'],
                            'intersection': ['i1']},
                  'left': {'goals': ['G3', 'G4'],
                           'segs': ['e', 'f', 'g'],
                           'intersection': ['i2']},
                  'stem': {'goals': ['H', 'D'],
                           'segs': ['a'],
                           'intersection': ['i2']}
                  }

    seg_splits = {'H': 'stem',
                  'D': 'stem',
                  'a': 'stem',
                  'b': 'right',
                  'c': 'right',
                  'd': 'right',
                  'e': 'left',
                  'g': 'left',
                  'f': 'left',
                  'G1': 'right',
                  'G2': 'right',
                  'G3': 'left',
                  'G4': 'left',
                  'i1': 'right',
                  'i2': 'left',
                  }

    split_colors = {'left': 'green', 'right': 'purple', 'stem': 'None',
                    'L': 'green', 'R': 'purple'}

    well_coords = {"H": (0, 20),
                   'D': (0, 560),
                   'G1': (650, 1000),
                   'G2': (240, 1300),
                   'G3': (-240, 1300),
                   'G4': (-650, 1000)}

    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    layout = plt.imread(dir_path.parent / "TreeMazeLayout_plain.jpg")
    maze_lims = {'x': (-1295, 1295), 'y': (-156, 1526)}

    ext_length = 200
    sub_seg_length = 60
    n_repeats = 10
    min_sub_seg_area = 6000

    pos_dist_thr = 100

    _poly_funcs = Lines2Polygons()

    def __init__(self):
        self.seg_orig_line = {k: np.asarray(self.zones_coords[k][slice(2)]) for k in self.linear_segs}
        self.sub_segs = self.divide_segs()

    # def divide_seg_orig(self, seg_name, direction='out'):
    #
    #     seg_dir = self.dirs[direction][seg_name]
    #     seg_dir_num = self._dir_num_dict[seg_dir]
    #
    #     # bounding rectangle coords
    #     seg_geom = self.zones_geom[seg_name]
    #     xx, yy = seg_geom.minimum_rotated_rectangle.exterior.xy
    #     pp = Points2D(xx, yy)
    #
    #     # get ccw line vectors for bounding rect
    #     l_segs = np.diff(pp.xy, axis=0)
    #     l_segs_v = Points2D(l_segs[:, 0], l_segs[:, 1])
    #
    #     # get direction
    #     l_segs_dirs = np.digitize(l_segs_v.ang, np.arange(-np.pi / 4, 2 * np.pi + np.pi / 4 + 0.001, np.pi / 2))
    #     l_segs_dirs[l_segs_dirs == 5] = 1
    #
    #     l_dir_id = np.where(l_segs_dirs == seg_dir_num)[0][0]
    #
    #     L = l_segs[l_dir_id]
    #     L = Points2D(L[0], L[1])
    #
    #     a = pp.xy[:4][l_dir_id - 1]
    #     a = Points2D(a[0], a[1])
    #     b = pp.xy[:4][l_dir_id]
    #     b = Points2D(b[0], b[1])
    #
    #     n_subsegs = int(L.r // self.sub_seg_length)
    #
    #     delta = L.r / n_subsegs
    #     sub_segs = np.zeros(n_subsegs, dtype=object)
    #
    #     for ii in range(n_subsegs):
    #         p0 = Points2D(ii * delta, L.ang, polar=True) + a
    #         p1 = Points2D((ii + 1) * delta, L.ang, polar=True) + a
    #         p2 = Points2D((ii + 1) * delta, L.ang, polar=True) + b
    #         p3 = Points2D(ii * delta, L.ang, polar=True) + b
    #
    #         sub_seg = Polygon([p0.xy[0],
    #                            p1.xy[0],
    #                            p2.xy[0],
    #                            p3.xy[0]])
    #
    #         sub_segs[ii] = seg_geom.intersection(sub_seg)
    #     return sub_segs

    def divide_segs(self):

        sub_segs = {}
        for seg_name in self.linear_segs:
            seg_geom = self.zones_geom[seg_name]

            # get origin line from the geometry
            line = self.seg_orig_line[seg_name]

            # extend line
            e_line = self._poly_funcs.extend_line(line, self.ext_length)

            # get multiple parallel lines at a given distance and creaty sub polygons from them
            lines = self._poly_funcs.project_line_repeats(e_line, self.sub_seg_length, self.n_repeats)
            sub_polys = self._poly_funcs.make_poly_sequence_from_parallel_lines(lines)

            # loop through polygons and intersect segment geometry to obtain subsegments
            sub_segs[seg_name] = []
            union_sub_segs = Polygon()
            small_seg = Polygon()
            cnt = -1
            for p in sub_polys:
                # get intesection between generated subsegments and segment geometry
                inter = seg_geom.intersection(p)
                # print(inter.area)
                if inter.area >= self.min_sub_seg_area:
                    sub_segs[seg_name].append(inter)
                    cnt += 1

                    # if there is a small segment in the cache, add it to the current subsegment
                    if small_seg.area > 1:
                        sub_segs[seg_name][cnt] = sub_segs[seg_name][cnt].union(small_seg)
                        small_seg = Polygon()

                elif inter.area > 1:
                    if cnt > 0:
                        # if current subsegment is too small, add to the previous subsegment
                        sub_segs[seg_name][cnt] = sub_segs[seg_name][cnt].union(inter)
                    else:
                        small_seg = inter
                else:
                    continue

                # if conjuctive polygon is the same as the segment, break loop
                if seg_geom.difference(union_sub_segs).area < 1:
                    print(sub_segs[seg_name])
                    print(seg_name)
                    break

        return sub_segs

    def get_pos_zone_ts(self, x, y):
        """ 
        for a given x,y returns the zone at which that points corresponds to.
        :param x: 
        :param y: 
        :return:
        z: np.array of zones
        """
        n_samps = len(x)
        assert n_samps == len(y)

        # Get zones that contains each x,y point
        pos_zones = np.zeros(n_samps, dtype=int)
        p_cnt = -1
        for xp, yp in zip(x, y):
            p_cnt += 1
            if not np.isnan(xp):
                p_zone_dist = np.zeros(self.n_zones)
                p = Point(xp, yp)
                for z_cnt, zo in enumerate(self.zone_names):
                    # get point distance to zone
                    p_zone_dist[z_cnt] = self.zones_geom[zo].distance(p)

                    # check if point is in zone
                    if self.zones_geom[zo].contains(p):
                        pos_zones[p_cnt] = z_cnt
                        break
                else:  # didn't find a match
                    # option1. assign to closest zone
                    if np.min(p_zone_dist) < self.pos_dist_thr:
                        pos_zones[p_cnt] = np.argmin(p_zone_dist)
                    # option 2. assign to previous zone
                    else:
                        pos_zones[p_cnt] = pos_zones[p_cnt - 1]
            else:  # in the case of a nan, assign to previous
                pos_zones[p_cnt] = pos_zones[p_cnt - 1]

        return pos_zones

    def get_pos_zone_mat(self, pos_zone_ts):
        M = np.full((len(pos_zone_ts), self.n_zones), 0)
        for z in np.arange(self.n_zones):
            M[pos_zone_ts == z, z] = 1
        M = pd.DataFrame(M, columns=self.zone_names)
        return M

    def plot_segs(self, segment_polygons, alpha=0.2, color='white', lw=1, axis=None):
        if axis is None:
            f, axis = plt.subplots(figsize=(2, 2), dpi=100)

        for seg in segment_polygons:
            pf.plot_poly(seg, axis, alpha=alpha, color=color, lw=lw)

    def plot_maze(self, axis=None, sub_segs=None, seg_dir=None, zone_labels=False, tm_layout=False, plot_cue=False,
                  seg_color='powderblue', seg_alpha=0.3, lw=1.5, line_alpha=1, line_color='0.5',
                  sub_seg_color=None, sub_seg_lw=0.1, cue_color=None, fontsize=10):
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
        :param tm_layout: TreeMaze layout overlay
        :param sub_seg_lw: float[0-1]
        :return:
        """

        plt.rcParams['lines.dash_capstyle'] = 'round'
        plt.rcParams['lines.solid_capstyle'] = 'round'

        if axis is None:
            f, axis = plt.subplots(figsize=(10, 10), dpi=500)

        seg_color_plot = {}
        if seg_color is None:
            for zone in self.zone_names:
                seg_color_plot[zone] = 'None'
        elif seg_color == 'cue':
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
            if zone[0] == 'i':
                pf.plot_poly(geom, axis, alpha=seg_alpha,
                             color=seg_color_plot[zone],
                             lw=0, line_alpha=line_alpha, line_color=line_color)
            else:
                pf.plot_poly(geom, axis, alpha=seg_alpha,
                             color=seg_color_plot[zone],
                             lw=lw, line_alpha=line_alpha, line_color=line_color)

        if plot_cue:
            if cue_color is None:
                cue_color = 'white'
                cue_lw = lw
                cue_txt = 'Cue'
            elif cue_color in ['L', 'left', 'Left']:
                cue_color = self.split_colors['left']
                cue_lw = 0
                cue_txt = 'L'
            elif cue_color in ['R', 'right', 'Right']:
                cue_lw = 0
                cue_color = self.split_colors['right']
                cue_txt = 'R'
            else:
                cue_lw = 0
                cue_txt = ''
                cue_color = 'white'

            pf.plot_poly(self.cue_geom, axis, alpha=1,
                         color=cue_color, lw=cue_lw, line_color=line_color)

            if zone_labels:
                axis.text(self.cue_label_coords[0], self.cue_label_coords[1], cue_txt, fontsize=fontsize,
                          horizontalalignment='center', verticalalignment='center')

        if zone_labels:
            for zone, coords in self.zone_label_coords.items():
                if zone[0] in ['G', 'i']:
                    zone_txt = f"{zone[0]}$_{{{zone[1]}}}$"
                else:
                    zone_txt = zone

                if (zone in self.linear_segs) or (zone[0] == 'i'):
                    zone_txt_col = '0.3'
                else:
                    zone_txt_col = 'k'
                if zone == 'D':
                    axis.text(coords[0], coords[1], zone_txt, fontsize=fontsize, color=zone_txt_col,
                              horizontalalignment='left', verticalalignment='center')
                else:
                    axis.text(coords[0], coords[1], zone_txt, fontsize=fontsize, color=zone_txt_col,
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
                    col = 'None'
                elif sub_seg_color == 'cue':
                    col = self.split_colors[self.seg_splits[zone]]
                elif type(sub_seg_color) == str:
                    col = sub_seg_color
                else:
                    col = sub_seg_color[zone]

                if inout == 'in':
                    sub_seg = self.sub_segs[zone][::-1]
                else:
                    sub_seg = self.sub_segs[zone]

                n_sub_segs = len(sub_seg)
                if seg_dir is not None:
                    alphas = 0.5 / n_sub_segs * np.arange(n_sub_segs)
                else:
                    alphas = np.ones(n_sub_segs) * seg_alpha
                for ii in range(n_sub_segs):
                    pf.plot_poly(sub_seg[ii], axis, alpha=alphas[ii], color=col, lw=sub_seg_lw)

        axis.axis('off')
        axis.axis('equal')

        if tm_layout:
            self.plot_layout(axis, cue=plot_cue)

        return axis

    def plot_layout(self, ax=None, cue=False):

        if ax is None:
            f, ax = plt.subplots(dpi=500)
            ax.axis('off')
        else:
            f = ax.figure

        # ax modifications
        ax_pos = ax.get_position()
        x0_mod = -0.001
        w_mod = 1
        if cue:
            y0_mod = 0.038
            h_mod = 0.666
        else:
            y0_mod = 0.04
            h_mod = 0.866

        ax_pos = [ax_pos.x0 + x0_mod * ax_pos.width / 0.775,
                  ax_pos.y0 + y0_mod * ax_pos.height / 0.755,
                  ax_pos.width * w_mod,
                  ax_pos.height * h_mod]
        newax = f.add_axes(ax_pos, anchor='C', zorder=-1)
        newax.imshow(self.layout)
        newax.axis('off')

        return ax

    def plot_zone_ts_window(self, pos_zones, samps=None, t=None, trial_table=None, trial_nums=None, ax=None,
                            h_lw=0.2, v_lw=0.7, samp_buffer=20, fontsize=10):

        if ax is None:
            f, ax = plt.subplots(figsize=(2, 1), dpi=500)
        else:
            f = ax.figure

        if (trial_table is not None) and (trial_nums is not None):
            samps = np.arange(trial_table.loc[trial_nums[0], 't0'] - samp_buffer,
                              trial_table.loc[trial_nums[-1], 'tE'] + samp_buffer)
            pf.sns.heatmap(pos_zones.loc[samps].T, ax=ax, yticklabels=self.zone_names, cbar=0, cmap='Greys_r', vmin=-1,
                           vmax=1.1)

            ax.set_xticks([])
            x_ticks = []
            x_ticklabels = []
            for trial in trial_nums:

                t0 = trial_table.loc[trial, 't0'] - samps[0]
                tE = trial_table.loc[trial, 'tE'] - samps[0]

                ax.vlines(t0, *ax.get_ylim(), color='0.3', linestyle='-.', lw=v_lw)
                if trial_table.loc[trial, 'correct']:
                    ax.vlines(tE, *ax.get_ylim(), color='b', linestyle='-.', lw=v_lw)
                else:
                    ax.vlines(tE, *ax.get_ylim(), color='r', linestyle='-.', lw=v_lw)

                x_ticks.append(t0)

                x_ticklabels.append(f"tr$_{{{trial}}}$")

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, fontsize=fontsize)
            ax.xaxis.set_ticks_position("top")

            cols = [self.split_colors[cue] for cue in trial_table.loc[trial_nums, 'cue']]
            for ticklabel, tickcolor in zip(ax.get_xticklabels(), cols):
                ticklabel.set_color(tickcolor)

            for tick in ax.get_xticklabels():
                tick.set_rotation(0)

            ax.tick_params(axis="x", top=True, direction="out", length=1, width=v_lw,
                           color="0.3", which='major', pad=0)

            x = ax.get_xticks().astype(int)
            sec_ax = ax.secondary_xaxis('bottom')
            sec_ax.tick_params(axis='x', which='major', bottom=True, length=1, width=v_lw,
                               color="0.3", pad=0)
            pf.sns.despine(ax=sec_ax, top=True, bottom=True, left=True, right=True)

            sec_ax.set_xticks(x)
            if t is not None:
                _ = sec_ax.set_xticklabels(t[samps][x].astype(int), fontsize=fontsize)
                sec_ax.set_xlabel("Time [s]", fontsize=fontsize, labelpad=0)

        elif samps is not None:
            pf.sns.heatmap(pos_zones.loc[samps].T, ax=ax, yticklabels=self.zone_names, cbar=0, cmap='Greys_r', vmin=-1,
                           vmax=1.1)

            x = ax.get_xticks().astype(int)
            x = np.linspace(x[0], x[-1], 6, endpoint=False).astype(int)
            x = x[1::]
            ax.set_xticks(x)
            ax.vlines(x, *ax.get_ylim(), color='0.3', linestyle='-.', lw=v_lw)
            if t is not None:
                _ = ax.set_xticklabels(np.round(t[samps][x]).astype(int), fontsize=fontsize)

        else:
            print("Missing Samps or Trial info.")
            raise ValueError

        ax.hlines(np.arange(self.n_zones + 1), *ax.get_xlim(), color='0.7', lw=h_lw)

        y_ticks = ax.get_yticks()
        y_ticklabels = self.zone_names

        y_ticklabels_1 = np.asarray(y_ticklabels)
        y_ticklabels_1[1::2] = ''

        y_ticklabels_2 = np.asarray(y_ticklabels)
        y_ticklabels_2[::2] = ''

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels_1, fontsize=fontsize, ha='right')
        ax.tick_params(axis='y', which='major', left=False, pad=-3)

        sec_ax = ax.secondary_yaxis('right')
        pf.sns.despine(ax=sec_ax, right=True, left=True)

        sec_ax.set_yticks(y_ticks)
        sec_ax.set_yticklabels(y_ticklabels_2, fontsize=fontsize, ha='left')
        sec_ax.tick_params(axis='y', which='major', left=False, right=False, pad=-3)

        return f, ax

    # def fig1_layout(self, ax=None, **params):
    #     if ax is None:
    #         f, ax = plt.subplots(figsize=(1, 1), dpi=600)
    #     else:
    #         f = ax.figure
    #
    #     fig_params = dict(seg_color='None', seg_alpha=0.2, zone_labels=True, seg_dir='in', fontsize=3, lw=0.1,
    #                       sub_segs=None, sub_seg_lw=0.01, sub_seg_color='None', tm_layout=True, plot_cue=True)
    #
    #     fig_params.update(params)
    #
    #     ax = self.plot_maze(axis=ax, **fig_params)
    #
    #     leg_ax = f.add_axes(ax.get_position())
    #     leg_ax.axis("off")
    #
    #     cue_w = 0.1
    #     cue_h = 0.1
    #
    #     cues_p0 = dict(right=np.array([0.65, 0.15]),
    #                    left=np.array([0.25, 0.15]))
    #
    #     text_strs = dict(right=r"$H \rightarrow D \rightarrow G_{1,2}$",
    #                      left=r"$G_{3,4} \leftarrow D \leftarrow H$")
    #
    #     txt_ha = dict(right='left', left='right')
    #
    #     txt_hspace = 0.05
    #     txt_pos = dict(right=cues_p0['right'] + np.array((0, -txt_hspace)),
    #                    left=cues_p0['left'] + np.array((cue_w, -txt_hspace)))
    #
    #     for cue in ['right', 'left']:
    #         cue_p0 = cues_p0[cue]
    #         cue_coords = np.array([cue_p0, cue_p0 + np.array((0, cue_h)),
    #                                cue_p0 + np.array((cue_w, cue_h)), cue_p0 + np.array((cue_w, 0)), ])
    #         cue_poly = Polygon(cue_coords)
    #         pf.plotPoly(cue_poly, ax=leg_ax, lw=0, alpha=0.9, color=self.split_colors[cue])
    #
    #         leg_ax.text(txt_pos[cue][0], txt_pos[cue][1], text_strs[cue], fontsize=fig_params['fontsize'],
    #                     horizontalalignment=txt_ha[cue], verticalalignment='center')
    #     leg_ax.set_xlim(0, 1)
    #     leg_ax.set_ylim(0, 1)
    #
    #     leg_ax.text(cues_p0['right'][0] + cue_w, cues_p0['right'][1] + cue_h // 2,
    #                 'Right Cue', fontsize=fig_params['fontsize'],
    #                 horizontalalignment='left', verticalalignment='bottom')
    #     leg_ax.text(cues_p0['left'][0], cues_p0['left'][1] + cue_h // 2,
    #                 'Left Cue', fontsize=fig_params['fontsize'],
    #                 horizontalalignment='right', verticalalignment='bottom')
    #
    #     return f, ax


class BehaviorData:
    ## constants
    n_wells = 6

    # time durations in seconds
    reward_dur = 0.5
    detection_dur = 0.1
    post_trial_dur = 1.0
    post_trial_correct = 0.3
    post_trial_incorrect = 0.1
    led_default_dur = 0.5
    is_near_thr = 0.01
    max_LED_dur = 200
    min_event_dist = 0.01

    # event end criteria
    _t3hg_end_criteria = {'L1': ['RW1'], 'L2': ['RW2'],
                          'L3': ['RW3', 'DE5', 'DE6'], 'L4': ['RW4', 'DE5', 'DE6'],
                          'L5': ['RW5', 'DE3', 'DE4'], 'L6': ['RW6', 'DE3', 'DE4'],
                          'CL': ['RW5', 'RW6', 'DE3', 'DE4'],
                          'CR': ['RW3', 'RW4', 'DE5', 'DE6']}
    _t3ij_end_criteria = {'L1': ['RW1'], 'L2': ['RW2'],
                          'L3': ['RW3', 'DE5', 'DE6'], 'L4': ['RW4', 'DE5', 'DE6'],
                          'L5': ['RW5', 'DE3', 'DE4'], 'L6': ['RW6', 'DE3', 'DE4'],
                          'CL': ['RW2'],
                          'CR': ['RW2']}
    _event_end_criteria = {'T3g': _t3hg_end_criteria, 'T3h': _t3hg_end_criteria,
                           'T3j': _t3ij_end_criteria, 'T3i': _t3ij_end_criteria,
                           'T3gj': _t3hg_end_criteria}

    trial_end_criteria = {'CL': ['RW5', 'RW6', 'DE3', 'DE4'],
                          'CR': ['RW3', 'RW4', 'DE5', 'DE6']}

    trial_valid_detections = {'CL': ['DE5', 'DE6'],
                              'CR': ['DE4', 'DE3']}

    event_names = ['RH', 'RD', 'R1', 'R2', 'R3', 'R4', 'RG', 'R',
                   'DH', 'DD', 'D1', 'D2', 'D3', 'D4',
                   'LH', 'LD', 'L1', 'L2', 'L3', 'L4',
                   'CL', 'CR', 'DL', 'DR',
                   'Tr', 'cTr', 'iTr',
                   'Out']

    n_event_types = len(event_names)

    def __init__(self, session_info, overwrite=False):
        self.task = session_info.task

        # get resampled time
        self.t = session_info.get_time()
        self.tB = self.t[0]
        self.tE = self.t[-1]
        self.time_step = session_info.params['time_step']
        self.n_samples = len(self.t)

        # compute constants in samples based on time_step.
        self.detection_dur_samps = int(self.detection_dur // self.time_step)
        self.reward_dur_samps = int(self.reward_dur // self.time_step)
        self.led_default_dur_samps = int(self.led_default_dur // self.time_step)
        self.post_trial_dur_samps = int(self.post_trial_dur // self.time_step)
        self.post_trial_correct_samps = int(self.post_trial_correct // self.time_step)
        self.post_trial_incorrect_samps = int(self.post_trial_incorrect // self.time_step)
        self.is_near_thr_samps = int(self.is_near_thr // self.time_step)
        self.max_LED_dur_samps = int(self.max_LED_dur // self.time_step)
        self.min_event_dist_samps = int(self.min_event_dist // self.time_step)

        # get events
        self.event_end_criteria = self._event_end_criteria[self.task]
        self.events = session_info.get_raw_events()
        self._get_reward_stamps()

        # get trial table
        if (not session_info.paths['trial_table'].exists()) or overwrite:
            self.trial_table = self.get_trial_table()
            self.trial_table.to_csv(session_info.paths['trial_table'])
        else:
            self.trial_table = pd.read_csv(session_info.paths['trial_table'], index_col=0)
        self.n_trials = self.trial_table.shape[0]
        self.session_perf = self.get_session_perf()

        # get event table
        if (not session_info.paths['event_table'].exists()) or overwrite:
            self.event_table = self.get_event_table()
            self.event_table.to_csv(session_info.paths['event_table'])
        else:
            self.event_table = pd.read_csv(session_info.paths['event_table'], index_col=0)

        # get event time series
        if (not session_info.paths['event_time_series'].exists()) or overwrite:
            self.events_ts = self.get_event_time_series()
            self.events_ts.to_csv(session_info.paths['event_time_series'])
        else:
            self.events_ts = pd.read_csv(session_info.paths['event_time_series'], index_col=0)

    def get_trial_table(self):
        """
        function that returns a data frame of trial information based on events. *** in samples***
        Trial outcome based on termination criteria.
        Criteria is based on reward events or detections at incorrect wells. Exit case also includes
        the start of another trial.

        Inputs:
            ev -> event directory. This must already include Reward events through the
            'getRewardStamps' function
            t -> regularly sampled time vector that covers the length of a recording
        Outputs:
            CueDurSamps -> dict of cue durations in samples
            TrialEvents > dict of trial start times and post correct/incorrect trial times

        """
        events = self.events
        n_L_cues = len(events['CL'])
        n_R_cues = len(events['CR'])
        n_cues = n_L_cues + n_R_cues

        # left -> 1, right -> 2
        all_cues = np.concatenate((np.ones(n_L_cues), 1 + np.ones(n_R_cues)))
        all_cues_times = np.concatenate((events['CL'], events['CR']))

        # sort by time
        sorted_cue_ids = np.argsort(all_cues_times)
        sorted_cue_times = all_cues_times[sorted_cue_ids]
        sorted_cues = all_cues[sorted_cue_ids]

        t_last = self.t[-1]
        # variables to be created / stored
        n_trials = n_cues
        df = pd.DataFrame(index=range(n_trials), columns=['t0', 'tE', 'dur', 'cue', 'dec', 'correct',
                                                          'long', 'goal', 'sw', 'vsw'])
        for ii in np.arange(n_trials):
            # t0 ->  current trial start time stamp
            # t0_next -> next trial start time
            # tE -> end of current trial
            t0 = sorted_cue_times[ii]  #
            if ii == (n_cues - 1):
                t0_next = t_last
            else:
                t0_next = sorted_cue_times[ii + 1]

            corr = 0
            goal = -1
            long = -1
            dec = ''
            cue = ''
            tE = t0_next
            if sorted_cues[ii] == 1:  # left
                cue = 'L'
                for end_ev in self.trial_end_criteria['CL']:  # end events
                    end_ev_id = np.logical_and(events[end_ev] >= t0, events[end_ev] < t0_next)
                    if any(end_ev_id):
                        if end_ev[:2] == 'RW':  # correct end events are rewards
                            dec = 'L'
                            corr = 1
                            goal = int(end_ev[2])
                        else:  # incorrect end event
                            dec = 'R'
                            corr = 0
                            goal = -1
                        tE = events[end_ev][end_ev_id][0]
                        break

            elif sorted_cues[ii] == 2:  # right
                cue = 'R'
                for end_ev in self.trial_end_criteria['CR']:  # end evens
                    end_ev_id = np.logical_and(events[end_ev] >= t0, events[end_ev] < t0_next)
                    if any(end_ev_id):
                        if end_ev[:2] == 'RW':  # correct end events
                            dec = 'R'
                            corr = 1
                            goal = int(end_ev[2])
                        else:  # incorrect end events
                            dec = 'L'
                            corr = 0
                            goal = -1
                        tE = events[end_ev][end_ev_id][0]
                        break
                        # check if there were detections on both target wells:
            if corr:
                long = 0
                for de in self.trial_valid_detections['C' + cue]:
                    if any(np.logical_and(events[de] >= t0, events[de] <= tE)):
                        if de[2] != str(goal):
                            long = 1
                            break
            else:
                long = -1

            if goal > 2:
                goal = goal - 2
            df.loc[ii, ['t0', 'tE', 'dur', 'cue', 'dec', 'correct', 'long', 'goal']] = \
                t0, tE, tE - t0, cue, dec, corr, long, goal
        df['t0'] = ((df['t0'] - self.tB) // self.time_step).astype(int)
        df['tE'] = ((df['tE'] - self.tB) // self.time_step).astype(int)
        df['dur'] = (df['dur'] // self.time_step).astype(int)

        # switch trial
        df['sw'] = 0
        df.loc[1:, 'sw'] = df['cue'][:-1].values != df['cue'][1:].values

        # valid switch trial: switch after a correct trial
        df['vsw'] = 0
        df.loc[1:, 'vsw'] = df['correct'][:-1].values & df['sw'][1:].values

        df = df.astype({'sw': 'int', 'vsw': 'int'})
        return df

    def get_event_table(self):
        """
        Function to obtain durations of the durations of all events based on termination criteria.
        Criteria is based on reward events or detections at incorrect wells. Exit case also includes
        the start of another trial. Note that this code would Only be applicable for T3 sessions.

        Outputs:
            dataframe table of information
        """

        # get and sort all the events
        n_events = 0

        all_events_list = []
        all_events_times = np.zeros(0)
        for k_event, k_event_times in self.events.items():
            n_k_events = len(k_event_times)
            all_events_list += [k_event] * n_k_events
            all_events_times = np.concatenate((all_events_times, k_event_times))
            n_events += n_k_events

        all_events_times = all_events_times - self.tB
        sorted_events_idx = np.argsort(all_events_times)

        # create data frame of events with their starting point
        df = pd.DataFrame(index=range(n_events), columns=['event', 't0', 'tE', 'dur', 'trial_num', 'out_bound'])
        df['event'] = np.array(all_events_list)[sorted_events_idx]
        df['t0'] = (all_events_times[sorted_events_idx] // self.time_step).astype(int)
        df['out_bound'] = False
        for trial_num in range(self.n_trials - 1):
            df.loc[(df.t0 >= self.trial_table.loc[trial_num, 't0']) &
                   (df.t0 <= self.trial_table.loc[trial_num + 1, 't0']), 'trial_num'] = trial_num
            df.loc[(df.trial_num == trial_num) & (df.t0 <= self.trial_table.loc[trial_num, 'tE']), 'out_bound'] = True

        trial_num = self.n_trials - 1
        if trial_num > 0:
            df.loc[(df.t0 >= self.trial_table.loc[trial_num, 't0']), 'trial_num'] = trial_num
            df.loc[(df.trial_num == trial_num) & (df.t0 <= self.trial_table.loc[trial_num, 'tE']), 'out_bound'] = True

        # go through all events and fill in the tE column by case
        for ii in range(n_events):
            ev = df.loc[ii, 'event']
            t0 = df.loc[ii, 't0']
            trial_num = df.loc[ii, 'trial_num']
            if ev[0] == 'D':  # Detection Event
                tE = t0 + self.detection_dur_samps
            elif ev[0] == 'R':  # Reward Event
                tE = t0 + self.reward_dur_samps
            elif ev[0] == 'S':  # start/stop events
                tE = t0
            elif ev[0] == 'L':  # LED Event
                end_criteria_match = df.loc[(df.t0 > t0 + self.min_event_dist_samps) &
                                            (df.t0 <= t0 + self.max_LED_dur_samps) &
                                            (df.event.isin(self.event_end_criteria[ev]))]['t0'].values
                if len(end_criteria_match) > 0:
                    tE = end_criteria_match[0]
                else:
                    tE = np.min((t0 + self.max_LED_dur_samps, self.n_samples))
            elif ev[0] == 'C':  # Cue Event
                if ~np.isnan(trial_num):
                    end_criteria_match = df.loc[(df.t0 > t0 + self.min_event_dist_samps) &
                                                (df.t0 <= self.trial_table.loc[trial_num, 'tE']) &
                                                (df.event.isin(self.event_end_criteria[ev]))]['t0'].values
                    if len(end_criteria_match) > 0:
                        tE = end_criteria_match[0]
                    else:
                        try:  # assign cue end to the start of the next trial
                            tE = self.trial_table.loc[trial_num + 1, 't0']
                        except:
                            pass
                else:  # undefined scenario
                    tE = t0
            else:  # undefined scenario
                tE = t0

            df.loc[ii, 'tE'] = tE

        # add decision events after the end of reward 2 up to end of trial.
        df2 = pd.DataFrame(index=range(self.n_trials), columns=['event', 't0', 'tE', 'dur', 'trial_num', 'out_bound'])
        df2['out_bound'] = True

        rw2 = df.loc[df.event == 'RW2', ['tE', 'trial_num']].copy()
        trials_to_rw2 = rw2.trial_num.unique()
        for dec in ['L', 'R']:
            dec_trial_table = self.trial_table[self.trial_table.dec == dec].copy()
            trials = np.intersect1d(dec_trial_table.index.values, trials_to_rw2)
            df2.loc[trials, 'event'] = 'D' + dec
            df2.loc[trials, 't0'] = rw2.loc[rw2.trial_num.isin(trials), 'tE'].values
            df2.loc[trials, 'tE'] = dec_trial_table.loc[trials, 'tE'].values
            df2.loc[trials, 'trial_num'] = trials

        df = pd.concat((df, df2), ignore_index=True)
        df = df.sort_values('t0', ignore_index=True)
        df['dur'] = df['tE'] - df['t0']
        df.fillna(-1, inplace=True)
        df = df.astype({'t0': int, 'tE': int, 'dur': int, 'trial_num': int, 'out_bound': int})
        return df

    def _get_reward_stamps(self):
        """ finds rewards time stamps for a specific well number"""
        for well in np.arange(1, self.n_wells + 1, dtype=int):
            detection_matches, reward_matches = \
                is_near(self.events['DE' + str(well)], self.events['RD'], self.is_near_thr)
            self.events['RW' + str(well)] = self.events['RD'][reward_matches > 0]

    def get_event_time_series(self):
        """
        Main Wrapper Function to obtain the event matrix Decribing the animals
        behavior during the TreeMaze Task.

        Outputs:
            evMat   -> a tall/skinny binary matrix of events, each column indicates
                    a different event, each row indicates time such that row i occurs
                    'step' seconds after row i-1.
        """

        event_mat = pd.DataFrame(data=np.zeros((self.n_samples, self.n_event_types), dtype=int),
                                 columns=self.event_names)

        # well events: rewards, detections, LEDS
        for well in np.arange(1, self.n_wells + 1):
            if well == 1:
                suf_str = 'H'
            elif well == 2:
                suf_str = 'D'
            else:
                suf_str = str(well - 2)

            for e in ['RW', 'DE', 'L']:
                if e == 'RW':
                    event_mat['R' + suf_str] = self._make_event_vector(e + str(well))
                elif e == 'DE':
                    event_mat['D' + suf_str] = self._make_event_vector(e + str(well))
                elif e == 'L':
                    event_mat['L' + suf_str] = self._make_event_vector(e + str(well))

        event_mat['RG'] = event_mat['R1'] + event_mat['R2'] + event_mat['R3'] + event_mat['R4']
        event_mat['R'] = event_mat['RH'] + event_mat['RD'] + event_mat['RG']

        # cue
        for cue in ['CL', 'CR']:
            event_mat[cue] = self._make_event_vector(cue)

        # decisions
        for dec in ['DL', 'DR']:
            event_mat[dec] = self._make_event_vector(dec)

        # trials
        for trial in range(self.n_trials):
            t0 = self.trial_table.loc[trial, 't0']
            tE = self.trial_table.loc[trial, 'tE']
            event_mat.loc[t0:(tE + 1), 'Tr'] = trial
            event_mat.loc[t0:(tE + 1), 'Out'] = 1

            if self.trial_table.loc[trial, 'correct']:
                event_mat.loc[tE:(tE + self.post_trial_correct_samps), 'cTr'] = 1
            else:
                event_mat.loc[tE:(tE + self.post_trial_incorrect_samps), 'iTr'] = 1

        return event_mat

    def _make_event_vector(self, event):
        """
        Creates binary vector for "event", the event needs to be one in the event_durs table.
        """
        event_table = self.event_table[self.event_table.event == event].copy()
        event_table.reset_index(drop=True, inplace=True)
        n_events = event_table.shape[0]

        event_vector = np.zeros(self.n_samples, dtype=int)
        for ev in range(n_events):
            i0 = event_table.loc[ev, 't0']
            iE = event_table.loc[ev, 'tE']
            event_vector[i0:(iE + 1)] = 1

        return event_vector

        # tt = isClosest(t, evTimes)
        # if type(evDurs) == int:
        #     evVec = signal.lfilter(np.ones(evDurs), 1, tt > 0)
        #     evVec[evVec > 1] = evValue
        #     return evVec
        # elif nEvents == len(evDurs):
        #     locs = np.where(tt > 0)[0]
        #     evVec = np.zeros(N)
        #     for i in np.arange(nEvents):
        #         idx = np.arange(locs[i], locs[i] + evDurs[i])
        #         if type(evValue) == int:
        #             evVec[idx] = evValue
        #         else:
        #             evVec[idx] = evValue[i]
        #     return evVec
        # else:
        #     print('Event and Event Duration mismatch: {} and {}'.format(nEvents, len(evDurs)))
        #     return []

    def get_session_perf(self):

        df = self.trial_table
        perf = pd.DataFrame(index=[0])
        perf['n_trials'] = df.shape[0]
        perf['n_sw_trials'] = df.sw.sum()
        perf['n_vsw_trials'] = df.vsw.sum()
        perf['n_L_trials'] = (df.cue == 'L').sum()
        perf['n_R_trials'] = (df.cue == 'R').sum()
        perf['pct_correct'] = df.correct.mean()
        perf['pct_sw_correct'] = (df.sw & df.correct).sum() / perf.n_sw_trials
        perf['pct_vsw_correct'] = (df.vsw & df.correct).sum() / perf.n_vsw_trials
        perf['pct_L_correct'] = ((df.cue == 'L') & df.correct).sum() / perf.n_L_trials
        perf['pct_R_correct'] = ((df.cue == 'R') & df.correct).sum() / perf.n_R_trials

        return perf


def pre_process_track_data(x, y, ha, t, t_rs, track_params, return_all=False):
    """
    This function performs the following tasks in order:
        1) masks xy pixel data for out track bound spaces. values set to np.nan
        2) centers the xy pixel data
        3) rotates xy to the experimenters perspective
        4) rescales data from pixels to mm
        5) masks velocity to create a masks for a velocity threshold
        6) smooth the data for excluding nan vals (non-causal)
        7) fill in nan values with last value (causal)
        8) apply median filters (causal)
        9) re-sample
    *Note, steps 2-4 are not applied to the head angle entry
    :param x: tracking data x position (expected in pixels)
    :param y: tracking data y position (expected in pixels
    :param ha: tracking data head angle (in degrees ; will returns radians)
    :param t: tracking time in seconds
    :param t_rs: resampled time
    :param track_params: parameters of the track
    :param return_all: bool. if true, returns an object array with all the steps
    :return: processed x,y, ha, and nan_idx
    """

    if type(track_params) == dict:
        p = SimpleNamespace(**track_params)
    elif type(track_params) == SimpleNamespace:
        p = track_params
    else:
        print("Parameters Need to be Dictionary")
        raise ValueError
    n_steps = 9

    xs = np.zeros(n_steps, dtype=object)
    ys = np.zeros(n_steps, dtype=object)
    has = np.zeros(n_steps, dtype=object)

    # convert ha to radians
    ha = np.mod(np.deg2rad(ha), 2 * np.pi)

    # 1. mask pixels that are out of bounds
    step = 0
    xs[step] = x.copy()
    ys[step] = y.copy()
    has[step] = ha.copy()

    mask_x = np.logical_or(x < p.x_pix_lims[0], x > p.x_pix_lims[1])
    mask_y = np.logical_or(y < p.y_pix_lims[0], y > p.y_pix_lims[1])
    mask = np.logical_or(mask_x, mask_y)

    xs[step][mask] = np.nan
    ys[step][mask] = np.nan
    has[step][mask] = np.nan

    # 2. centering / pixel translation
    step += 1
    xs[step] = xs[step - 1] + p.x_pix_bias
    ys[step] = ys[step - 1] + p.y_pix_bias

    # 3. rotate to experimenter's pov
    step += 1
    xs[step], ys[step] = spatial_funcs.rotate_xy(xs[step - 1], ys[step - 1], p.xy_pix_rot_rad)

    # 4. convert to mm / re-scales; bias term re-frames the image
    step += 1
    xs[step] = -(xs[step - 1] * p.x_pix_mm + p.x_mm_bias)
    xs[step][xs[step] < 0] = xs[step][xs[step] < 0] * p.x_neg_warp_factor
    ys[step] = ys[step - 1] * p.y_pix_mm + p.y_mm_bias

    # 5. filter by valid speed values
    with np.errstate(invalid='ignore'):  # avoids warnings about comparing nan values
        # 5a. compute velocity to create speed threshold
        dx = np.append(0, np.diff(xs[step]))
        dy = np.append(0, np.diff(ys[step]))
        dr = np.sqrt(dx ** 2 + dy ** 2)
        mask_r = np.abs(dr) > p.max_speed_thr

        # 5b. mask creating out of bound zones in mm space
        mask_x = np.logical_or(xs[step] < p.x_mm_lims[0], xs[step] > p.x_mm_lims[1])
        mask_y = np.logical_or(ys[step] < p.y_mm_lims[0], ys[step] > p.y_mm_lims[1])
        mask = np.logical_or(mask_x, mask_y)
        mask = np.logical_or(mask, mask_r)

    # 5c. apply masks
    step += 1
    xs[step] = xs[step - 1].copy()
    ys[step] = ys[step - 1].copy()
    has[step] = has[0].copy()

    xs[step][mask] = np.nan
    ys[step][mask] = np.nan
    has[step][mask] = np.nan

    # get nan idx for future use.
    nan_idx = np.where(np.logical_or(np.isnan(xs[step]), np.isnan(ys[step])))[0]

    # 6. smooth non nan-vals
    step += 1
    xs[step] = xs[step - 1].copy()
    ys[step] = ys[step - 1].copy()
    has[step] = has[step - 1].copy()

    xs[step][~nan_idx] = filter_funcs.filtfilt(p.filter_coef_, 1, xs[step - 1][~nan_idx])
    ys[step][~nan_idx] = filter_funcs.filtfilt(p.filter_coef_, 1, ys[step - 1][~nan_idx])
    has[step][~nan_idx] = filter_funcs.angle_filtfilt(has[step - 1][~nan_idx], p.filter_coef_angle_)

    # 7. fill in nan vals
    step += 1
    xs[step] = filter_funcs.fill_nan_vals(xs[step - 1])
    ys[step] = filter_funcs.fill_nan_vals(ys[step - 1])
    has[step] = filter_funcs.fill_nan_vals(has[step - 1])

    # 8. median filter
    step += 1
    xs[step] = filter_funcs.median_window_filter_causal(xs[step - 1], p.temporal_window_size)
    ys[step] = filter_funcs.median_window_filter_causal(ys[step - 1], p.temporal_window_size)
    has[step] = filter_funcs.median_window_filter_causal(has[step - 1], p.temporal_angle_window_size)

    # 9. resample
    step += 1
    xs[step] = filt_funcs.resample_signal(t, t_rs, xs[step - 1])
    ys[step] = filt_funcs.resample_signal(t, t_rs, ys[step - 1])
    has[step] = filt_funcs.resample_signal(t, t_rs, has[step - 1])

    if return_all:
        return xs, ys, has, nan_idx
    else:
        return xs[step], ys[step], has[step], nan_idx


def correct_xy(x, y, event_locs, goal_locs):
    raise NotImplementedError

    xd = [0, 0, 650, 250, -250, -650]
    yd = [45, 560, 1000, 1280, 1280, 1000]

    x2 = np.array(x)
    y2 = np.array(y)

    for z1 in ['D', 'R']:
        cnt = 0
        for z2 in ['H', 'C', '1', '2', '3', '4']:
            z = z1 + z2
            ids = event_locs[z] == 1
            x2[ids] = xd[cnt]
            y2[ids] = yd[cnt]
            cnt += 1

    x2 = filt_funcs.median_window_filtfilt(x2, 5)
    y2 = filt_funcs.median_window_filtfilt(y2, 5)
    for z1 in ['D', 'R']:
        cnt = 0
        for z2 in ['H', 'C', '1', '2', '3', '4']:
            ids = event_locs[z] == 1
            x2[ids] = xd[cnt]
            y2[ids] = yd[cnt]
            cnt += 1

    return x2, y2


def is_near(x, y, thr):
    """Find x,y points within the thr"""
    x_out = np.full_like(x, -1)
    y_out = np.full_like(y, -1)
    match_cnt = 1
    for ii, xx in enumerate(x):
        for jj, yy in enumerate(y):
            if abs(xx - yy) <= thr:
                x_out[ii] = match_cnt
                y_out[jj] = match_cnt
                match_cnt += 1
    return x_out, y_out


def isClosest(t, X):
    '''Find closest sample in t that matches X'''
    t_out = np.full_like(t, -1)
    cnt1 = 1
    for x in X:
        idx = np.argmin(np.abs(t - x))
        if x - t[idx] >= 0:
            t_out[idx] = cnt1
        else:  # always assign to the earliest sample
            t_out[idx - 1] = cnt1
        cnt1 += 1
    return t_out


def isbefore(X, Y, thr, minTime=0):
    '''Find x,y points within the thr and such that x happens before y'''
    x_out = np.full_like(X, -1)
    y_out = np.full_like(Y, -1)
    match_cnt = 0
    cnt1 = 0
    for x in X:
        cnt2 = 0
        for y in Y:
            if y - x <= thr and y - x >= minTime:
                x_out[cnt1] = match_cnt
                y_out[cnt2] = match_cnt
                match_cnt += 1
                break
            cnt2 += 1

        cnt1 += 1
    return x_out, y_out
