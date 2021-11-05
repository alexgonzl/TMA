import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from types import SimpleNamespace

import Analyses.spatial_functions as spatial_funcs
import Utils.filter_functions as filt_funcs
import Utils.robust_stats as rs

import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import Analyses.plot_functions as pf
import Utils.filter_functions as filter_funcs
from scipy.stats import ttest_ind, ttest_1samp
from joblib import delayed, Parallel


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

                    'G1': [(520, 1180), (800, 1180.5), (800, 800), (560, 800)],
                    'G2': [(50, 1190), (50, 1450), (450, 1450), (450, 1260)],
                    'G3': [(-50, 1190), (-50, 1450), (-450, 1450), (-450, 1260)],
                    'G4': [(-520, 1180), (-800, 1180.5), (-800, 800), (-560, 800)],

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
    zone_numbers = {zone: ii for ii, zone in enumerate(zone_names)}
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

    wells = ['H', 'D', 'G1', 'G2', 'G3', 'G4']

    bigseg_names = ['left', 'stem', 'right']
    split_segs = {'right': ['b', 'c', 'd', 'i1', 'G1', 'G2'],
                  'left': ['e', 'f', 'g', 'i2', 'G3', 'G4'],
                  'stem': ['H', 'a', 'D']}

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

    maze_union = Polygon()
    for z, geom in zones_geom.items():
        maze_union = maze_union.union(geom)

    def __init__(self):
        self.seg_orig_line = {k: np.asarray(self.zones_coords[k][slice(2)]) for k in self.linear_segs}
        self.sub_segs = self.divide_segs()

        self.sub_segs_names = {}
        for seg in self.linear_segs:
            self.sub_segs_names[seg] = list(self.sub_segs[seg].keys())

        self.all_segs_names = []
        for zo in self.zone_names:
            if zo in self.linear_segs:
                self.all_segs_names += self.sub_segs_names[zo]
            else:
                self.all_segs_names.append(zo)
        self.n_all_segs = len(self.all_segs_names)

        self.split_zones_all = {}
        for split, zones in self.split_segs.items():
            split_zones = []
            for zone in zones:
                if zone in self.linear_segs:
                    subsegs = list(self.sub_segs[zone].keys())
                    split_zones += subsegs
                else:
                    split_zones.append(zone)
            self.split_zones_all[split] = split_zones

        self.valid_transition_mat = self.get_valid_transition_mat()

        self.subseg2seg = self._subseg2seg()
        self.seg2bigseg = self._seg2bigseg()
        self.subseg2bigseg = self.subseg2seg @ self.seg2bigseg
        self.subseg2wells = self._subseg2wells()
        self.subseg2bigseg_nowells = self._subseg2bigseg_nowells()

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

        # make it a dictionary
        sub_segs_dict = {}
        for zo, subsegs_geom in sub_segs.items():
            sub_segs_dict[zo] = {}
            for ii, subseg_geom in enumerate(subsegs_geom):
                sub_segs_dict[zo][f"{zo}_{ii}"] = subseg_geom

        return sub_segs_dict

    def _subseg2seg(self):
        subsegs = self.all_segs_names
        segs = self.zone_names

        df = pd.DataFrame(0, index=subsegs, columns=segs)

        for seg in segs:
            if seg in self.linear_segs:
                df.loc[self.sub_segs_names[seg], seg] = 1
            else:
                df.loc[seg, seg] = 1
        return df

    def _seg2bigseg(self):
        segs = self.zone_names
        bigsegs = self.bigseg_names
        df = pd.DataFrame(0, index=segs, columns=bigsegs)
        for bigseg in bigsegs:
            df.loc[self.split_segs[bigseg], bigseg] = 1
        return df

    def _subseg2wells(self):
        subsegs = self.all_segs_names
        wells = self.wells
        df = pd.DataFrame(0, index=subsegs, columns=wells)

        for seg in subsegs:
            if seg in wells:
                df.loc[seg, seg] = 1

        return df

    def _subseg2bigseg_nowells(self):

        bigsegs = self.bigseg_names
        df = pd.DataFrame(0, index=self.all_segs_names, columns=bigsegs)
        for bigseg in bigsegs:
            idx = np.setdiff1d(self.split_segs[bigseg], self.wells)
            for seg in idx:
                if seg in self.linear_segs:
                    df.loc[self.sub_segs_names[seg], bigseg] = 1
                else:  # intersections
                    df.loc[seg, bigseg] = 1
        return df

    def get_segment_type_names(self, segment_type):

        if segment_type == 'bigseg':
            return self.bigseg_names
        elif segment_type == 'subseg':
            return self.all_segs_names
        elif segment_type == 'seg':
            return self.zone_names
        elif segment_type == 'bigseg_nowells':
            return self.bigseg_names
        elif segment_type == 'wells':
            return self.wells

    def subseg_pz_mat_transform(self, subseg_zone_mat, segment_type):
        """
        converts input matrix subseg_zone_mat [n x n_subsegs] to [n x n_segments], where n_segments is the number
        of segments for segment_type
        """

        if segment_type == 'subseg':
            return subseg_zone_mat
        elif segment_type == 'bigseg':
            return subseg_zone_mat @ self.subseg2bigseg
        elif segment_type == 'seg':
            return subseg_zone_mat @ self.subseg2seg
        elif segment_type == 'wells':
            return subseg_zone_mat @ self.subseg2wells
        elif segment_type == 'bigseg_nowells':
            return subseg_zone_mat @ self.subseg2bigseg_nowells
        else:
            raise NotImplementedError

    def subseg_pz_mat_segment_norm(self, subseg_zone_mat, segment_type, seg_mat=None):
        """
        primarely used to appropriate scale firing rates on single trials.
        normalizes subseg_zone_mat by the number of samples in a region to be transformed.
        :param subseg_zone_mat: [n x n_subsegs]
        :param seg_mat: [n x n_segs], where the segments contain subsegs
        :param segment_type:
        :return:
            [n x n_subsegs] matrix re-scaled by the number of samples in the semgent that contain the subsegs
        """

        if seg_mat is None:
            seg_mat = self.subseg_pz_mat_transform(subseg_zone_mat, segment_type)
        if segment_type == 'bigseg':
            transform_mat = self.subseg2bigseg
        elif segment_type == 'bigseg_nowells':
            transform_mat = self.subseg2bigseg_nowells
        elif segment_type == 'seg':
            transform_mat = self.subseg2wells
        else:
            raise NotImplementedError

        return subseg_zone_mat / (seg_mat @ transform_mat.T)

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
        pos_zones = np.zeros(n_samps, dtype=object)
        invalid_pz = []
        p_cnt = -1
        for xp, yp in zip(x, y):
            p_cnt += 1
            if not np.isnan(xp):
                p_zone_dist = {z: 0 for z in self.all_segs_names}

                p = Point(xp, yp)
                for zo, z_cnt in self.zone_numbers.items():
                    # get point distance to zone
                    p_zone_dist[zo] = self.zones_geom[zo].distance(p)

                    # check if point is in zone
                    if self.zones_geom[zo].contains(p):
                        if zo in self.linear_segs:
                            zone_subsegs = self.sub_segs[zo]
                            for subseg_name, subseg_geom in zone_subsegs.items():
                                p_zone_dist[zo] = subseg_geom.distance(p)
                                if subseg_geom.contains(p):
                                    pos_zones[p_cnt] = subseg_name
                                    break  # break the subseg loop
                        else:
                            pos_zones[p_cnt] = zo
                        break  # break the zone loop
                else:  # didn't find a match
                    invalid_pz.append(p_cnt)
                    # option1. assign to closest zone
                    min_dist_zone = min(p_zone_dist, key=p_zone_dist.get)
                    if p_zone_dist[min_dist_zone] < self.pos_dist_thr:
                        pos_zones[p_cnt] = min_dist_zone
                    # option 2. assign to previous zone
                    else:
                        pos_zones[p_cnt] = pos_zones[p_cnt - 1]
            else:  # in the case of a nan, assign to previous
                pos_zones[p_cnt] = pos_zones[p_cnt - 1]
                invalid_pz.append(p_cnt)
        return pos_zones, np.array(invalid_pz)

    def get_inmaze_samps(self, x, y):

        cnt = 0
        in_maze_samps = np.ones(len(x), dtype=bool)
        for xi, yi in zip(x, y):
            p = Point(xi, yi)
            in_maze_samps[cnt] = self.maze_union.contains(p)
            cnt += 1
        return in_maze_samps

    def get_pos_zone_mat(self, pos_zones, segment_type='subseg'):

        M = pd.DataFrame(np.full((len(pos_zones), self.n_all_segs), 0, dtype=int),
                         columns=self.all_segs_names)
        for z in self.all_segs_names:
            M.loc[pos_zones == z, z] = 1
        M = self.subseg_pz_mat_transform(M, segment_type=segment_type)

        return M

    def pz_subsegs_to_pz(self, pz):
        pz2 = pz.copy()
        for z in self.all_segs_names:
            if z.split("_")[0] in self.linear_segs:
                pz2[pz == z] = z.split("_")[0]

        return pz2

    def get_valid_transition_mat(self, subsegs=True):
        """
        :param subsegs, bool, if true uses and expects subsegs in pz
        :return:
            pd.dataframe of transition bools by zones/segs. rows sample n, columns sample n+1.
            entry at i,j, if True, indicates zone j coming from zone i is valid.
        """

        if subsegs:
            valid_transition_mat = pd.DataFrame(np.eye(self.n_all_segs, dtype=bool),
                                                index=self.all_segs_names, columns=self.all_segs_names)

            ssn = self.sub_segs_names
            # within linear seg
            for seg, subsegs in ssn.items():
                valid_transition_mat.loc[subsegs, subsegs] = True

            # outward trajectories
            valid_transition_mat.loc['H', ssn['a']] = True
            valid_transition_mat.loc[ssn['a'], 'D'] = True
            valid_transition_mat.loc['D', ssn['b']] = True

            valid_transition_mat.loc[ssn['b'], 'i1'] = True
            valid_transition_mat.loc['i1', ssn['c']] = True
            valid_transition_mat.loc[ssn['c'], 'G1'] = True
            valid_transition_mat.loc['i1', ssn['d']] = True
            valid_transition_mat.loc[ssn['d'], 'G2'] = True

            valid_transition_mat.loc['D', ssn['e']] = True
            valid_transition_mat.loc[ssn['e'], 'i2'] = True
            valid_transition_mat.loc['i2', ssn['f']] = True
            valid_transition_mat.loc[ssn['f'], 'G3'] = True
            valid_transition_mat.loc['i2', ssn['g']] = True
            valid_transition_mat.loc[ssn['g'], 'G4'] = True

            valid_transition_mat.loc[ssn['f'], ssn['g']] = True
            valid_transition_mat.loc[ssn['c'], ssn['d']] = True
            valid_transition_mat.loc[ssn['a'], ssn['b']] = True
            valid_transition_mat.loc[ssn['a'], ssn['e']] = True
        else:
            valid_transition_mat = pd.DataFrame(np.eye(self.n_zones, dtype=bool), index=self.zone_names,
                                                columns=self.zone_names)

            # outward trajectories
            valid_transition_mat.loc['H', 'a'] = True
            valid_transition_mat.loc['a', 'D'] = True
            valid_transition_mat.loc['D', 'b'] = True

            valid_transition_mat.loc['b', 'i1'] = True
            valid_transition_mat.loc['i1', 'c'] = True
            valid_transition_mat.loc['c', 'G1'] = True
            valid_transition_mat.loc['i1', 'd'] = True
            valid_transition_mat.loc['d', 'G2'] = True

            valid_transition_mat.loc['D', 'e'] = True
            valid_transition_mat.loc['e', 'i2'] = True
            valid_transition_mat.loc['i2', 'f'] = True
            valid_transition_mat.loc['f', 'G3'] = True
            valid_transition_mat.loc['i2', 'g'] = True
            valid_transition_mat.loc['g', 'G4'] = True

            valid_transition_mat.loc['f', 'g'] = True
            valid_transition_mat.loc['c', 'd'] = True

        sym_mat = np.triu(valid_transition_mat.values) + np.triu(valid_transition_mat.values, 1).T
        for ii, col in enumerate(valid_transition_mat.columns):
            valid_transition_mat[col].values[:] = sym_mat[:, ii]

        return valid_transition_mat

    def get_transition_counts(self, pz, subsegs=True):
        """

        :param pz: array, position zones
        :param subsegs, bool, if true uses and expects subsegs in pz
        :return:
            pd.dataframe of transition counts by zones/segs. rows sample n, columns sample n+1.
            entry at i,j, indicates the # of times zone j came from zone i.
        """
        if subsegs:
            zone_transition_counts = pd.DataFrame(np.zeros((self.n_zones, self.n_zones), dtype=int),
                                                  index=self.all_segs_names, columns=self.all_segs_names)
            zn = self.all_segs_names
        else:
            zone_transition_counts = pd.DataFrame(np.zeros((self.n_zones, self.n_zones), dtype=int),
                                                  index=self.zone_names, columns=self.zone_names)

            zn = self.zone_names

        for z in zn:
            for z1 in zn:
                zone_transition_counts.loc[z, z1] = ((pz[:-1] == z) & (pz[1:] == z1)).sum()

        return zone_transition_counts

    def check_valid_pos_zones_transitions(self, pz, subsegs=True):
        """
        returns a time series of booleans indicating if the transition of the position
        from the previous sample was valid
        :param pz: array, position zones
        :param subsegs, bool, if true uses and expects subsegs in pz
        :return:
            array of boolean indicating if the each sample comes from a valid transition
        """
        if subsegs:
            valid_transition_mat = self.valid_transition_mat

        else:
            valid_transition_mat = self.get_valid_transition_mat(subsegs=False)

        zones = valid_transition_mat.columns
        valid_transitions = np.ones(len(pz), dtype=bool)
        cnt = 1
        for z, z1 in zip(pz[:-1], pz[1:]):
            if (z in zones) and (z1 in zones):
                if not valid_transition_mat.loc[z, z1]:
                    valid_transitions[cnt] = False
            cnt += 1

        return valid_transitions

    def plot_segs(self, segment_polygons, alpha=0.2, color='white', lw=1, axis=None):
        if axis is None:
            f, axis = plt.subplots(figsize=(2, 2), dpi=100)

        for seg in segment_polygons:
            pf.plot_poly(seg, axis, alpha=alpha, color=color, lw=lw)

    def plot_maze(self, axis=None, sub_segs=None, seg_dir=None, zone_labels=False, tm_layout=False, plot_cue=False,
                  seg_color='powderblue', seg_alpha=0.3, lw=1, line_alpha=1, line_color='0.5',
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
            f, axis = plt.subplots(figsize=(5, 5), dpi=500)

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
                    sub_seg = list(self.sub_segs[zone].values()[::-1])
                else:
                    sub_seg = list(self.sub_segs[zone].values())

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

    def plot_zone_activity(self, zone_activity, ax=None, plot_cue=False, cue_color=None,
                           lw=0.2, line_alpha=1, line_color='0.5', **cm_args):
        if ax is None:
            f, ax = plt.subplots(figsize=(5, 5), dpi=500)
        else:
            f = ax.figure

        cm_params = dict(color_map='YlOrBr_r',
                         n_color_bins=25, nans_2_zeros=True, div=False,
                         max_value=None, min_value=None,
                         label='FR', tick_fontsize=7, label_fontsize=7)
        cm_params.update(cm_args)
        data_colors, color_array = pf.get_colors_from_data(zone_activity.values, **cm_params)

        cnt = 0
        for zone, value in zone_activity.items():
            if zone[0] in self.linear_segs:
                if zone in self.sub_segs[zone[0]]:
                    zone_geom = self.sub_segs[zone[0]][zone]
                else:
                    zone_geom = self.zones_geom[zone]
            else:
                zone_geom = self.zones_geom[zone]
            pf.plot_poly(zone_geom, ax=ax, alpha=1, color=data_colors[cnt],
                         lw=lw, line_alpha=line_alpha, line_color=line_color)
            cnt += 1

        if plot_cue:
            if cue_color is None:
                cue_color = 'white'
                cue_lw = lw
            elif cue_color in ['L', 'left', 'Left']:
                cue_color = self.split_colors['left']
                cue_lw = 0
            elif cue_color in ['R', 'right', 'Right']:
                cue_color = self.split_colors['right']
                cue_lw = 0
            else:
                cue_color = 'white'
                cue_lw = 0

            pf.plot_poly(self.cue_geom, ax, alpha=1,
                         color=cue_color, lw=cue_lw, line_color=line_color)

        ax.axis('off')
        # ax.axis('equal')

        ax_p = ax.get_position()
        w, h = ax_p.width, ax_p.height
        x0, y0 = ax_p.x0, ax_p.y0

        cax_p = [x0 + w * 0.7, y0 + h * 0.1, w * 0.05, h * 0.15]
        cax = f.add_axes(cax_p)

        pf.get_color_bar_axis(cax, color_array, **cm_params)


class BehaviorData:
    ## constants
    n_wells = 6

    # time durations in seconds
    reward_dur = 0.5  # post reward duration
    reward_null = np.array([-0.04, 0.04])  # blank time around reward for elimination of artifacts
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
        self.reward_null_samps = (self.reward_null // self.time_step).astype(int)
        self.led_default_dur_samps = int(self.led_default_dur // self.time_step)
        self.post_trial_dur_samps = int(self.post_trial_dur // self.time_step)
        self.post_trial_correct_samps = int(self.post_trial_correct // self.time_step)
        self.post_trial_incorrect_samps = int(self.post_trial_incorrect // self.time_step)
        self.is_near_thr_samps = int(self.is_near_thr // self.time_step)
        self.max_LED_dur_samps = int(self.max_LED_dur // self.time_step)
        self.min_event_dist_samps = int(self.min_event_dist // self.time_step)
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
            if ev == 'R_blank':
                tE = t0 + self.reward_null_samps[1]
                t0 = t0 + self.reward_null_samps[0]
                df.loc[ii, 't0'] = t0
            elif ev[0] == 'D':  # Detection Event
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
            # df2.loc[trials, 't0'] = rw2.loc[rw2.trial_num.isin(trials), 'tE'].values
            # bug fix. this takes care of multiple instances of rw2 in a trial # 9/24/20 ag
            df2.loc[trials, 't0'] = rw2.loc[rw2.trial_num.isin(trials)].groupby('trial_num').head(1).tE.values
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

        self.events['R_blank'] = self.events['RD']

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


class TrialAnalyses:
    trial_conditions = {'CL': {'cue': 'L'}, 'CR': {'cue': 'R'}, 'DL': {'dec': 'L'}, 'DR': {'dec': 'R'},
                        'Co': {'correct': 1}, 'Inco': {'correct': 0},
                        'CoCL': {'correct': 1, 'cue': 'L'}, 'CoCR': {'correct': 1, 'cue': 'R'},
                        'IncoCL': {'correct': 0, 'cue': 'L'}, 'IncoCR': {'correct': 0, 'cue': 'R'},
                        'Sw': {'sw': 1}, 'CoSw': {'correct': 1, 'sw': 1}, 'IncoSw': {'correct': 0, 'sw': 1},
                        'Even': '', 'Odd': '', 'Out': '', 'In': '', 'All': ''}

    cond_pairs = ['CR-CL',
                  'Co-Inco',
                  'Co-Inco',
                  'Even-Odd',
                  'Even-Odd']

    # balanced approached for groups
    bal_cond_pairs = ['CR_bo-CL_bo',
                      'Co_bo-Inco_bo',
                      'Co_bi-Inco_bi',
                      'Even_bo-Odd_bo',
                      'Even_bi-Odd_bi']

    # bal_conds_sets = { 'CR_bo': {'cond': 'CR', 'sub_conds': ['Co', 'Inco'], 'trial_seg': 'out'},
    #                 'CL_bo': {'cond':'CL', 'sub_conds': ['Co', 'Inco'], 'trial_seg': 'out'},
    #                 'Co_bo': {'cond':'Co', 'sub_conds': ['CL','CR'], 'trial_seg': 'out'},
    #                 'Co_bi': {'cond': 'Co', 'sub_conds': ['CL', 'CR'], 'trial_seg': 'in'},
    #                 'Inco_bo': {'cond':'Inco', 'sub_conds': ['CL','CR'], 'trial_seg': 'out'},
    #                 'Inco_bi': {'cond': 'Inco', 'sub_conds': ['CL', 'CR'], 'trial_seg': 'in'},
    #                 'Even_bo': {'cond':'Even', 'sub_conds': ['CL','CR'], 'trial_seg': 'out'},
    #                 'Even_bi': {'cond': 'Even', 'sub_conds': ['CL', 'CR'], 'trial_seg': 'in'},
    #                 'Odd_bo': {'cond': 'Odd', 'sub_conds': ['CL', 'CR'], 'trial_seg': 'out'},
    #                 'Odd_bi': {'cond': 'Odd', 'sub_conds': ['CL', 'CR'], 'trial_seg': 'in'},
    #                 }

    test_null_bal_cond_pairs = {'CR_bo-CL_bo': 'Even_bo-Odd_bo',
                                'Co_bo-Inco_bo': 'Even_bo-Odd_bo',
                                'Co_bi-Inco_bi': 'Even_bi-Odd_bi'}

    occupation_thrs = {'bigseg': 5, 'seg': 2, 'subseg': 1, 'bigseg_nowells': 5, 'wells': 2}

    def __init__(self, session_info, reward_blank=False, not_inzone_blank=True, valid_transitions_blank=True):
        """
        class for trialwise analyses on session data
        :param session_info:
        :param reward_blank: blank samples near reward period
        :param not_inzone_blank: blank samples that are not in the defined zones
        """

        self.si = session_info
        self.tmz = TreeMazeZones()

        temp = session_info.get_event_behavior()
        self.trial_table = temp.trial_table
        self.event_table = temp.event_table

        self.n_trials = len(self.trial_table)
        self.all_trials = np.arange(self.n_trials)

        self.n_units = self.si.n_units
        self.n_total_samps = self.si.n_samps

        self.trial_times, self.outbound_samps, self.inbound_samps = self.get_trial_times()

        self.track_data = self.si.get_track_data()
        self.pz, self.pz_invalid_samps = self.si.get_pos_zones(return_invalid_pz=True)

        self.fr = self.si.get_fr()
        self.spikes = self.si.get_binned_spikes()

        if not_inzone_blank:
            self._blank_data(self.pz_invalid_samps)

        if valid_transitions_blank:
            self.valid_transitions_samps = self.tmz.check_valid_pos_zones_transitions(self.pz)
            self._blank_data(~self.valid_transitions_samps)

        self.x_edges = self.si.task_params['x_bin_edges_'] * 10  # edges are in cm
        self.y_edges = self.si.task_params['y_bin_edges_'] * 10  # edges are in cm

        self.trial_condition_table = self.generate_trial_condition_table()

        self.bal_cond_sets = {}
        for bal_cond_pair in self.bal_cond_pairs:
            for bc in bal_cond_pair.split('-'):
                self.bal_cond_sets[bc] = self._decode_cond(bc)

        self.trial_zones = {k: self.get_trial_zones(trial_seg=k) for k in ['out', 'in']}
        self.trial_zone_samps_counts_mat = {k: self.get_trial_zones_samps_counts_mat(trial_seg=k) for k in
                                            ['out', 'in']}
        self.trial_zones_rates = {k: self.get_all_trial_zone_rates(trial_seg=k) for k in ['out', 'in']}

        self.zones_by_trial = {k: self.trial_zone_samps_counts_mat[k] > 0 for k in ['out', 'in']}

    def _decode_cond(self, cond_code):

        if cond_code in self.bal_cond_sets.keys():
            return self.bal_cond_sets[cond_code]

        split = cond_code.split('_')

        cond = split[0]
        sub_conds = []
        trial_seg = 'out'

        if len(split) > 1:
            b_seg = split[1]
            if 'b' in b_seg:
                if cond in ['CL', 'CR']:
                    sub_conds = ['Co', 'Inco']
                else:
                    sub_conds = ['CL', 'CR']

            if 'o' in b_seg:
                trial_seg = 'out'
            elif 'i' in b_seg:
                trial_seg = 'in'

        return {'cond': cond, 'sub_conds': sub_conds, 'trial_seg': trial_seg, 'cond_set': {cond: sub_conds}}

    def _blank_data(self, samps):
        if len(samps) > 0:
            self.track_data.loc[samps, :] = np.nan
            self.pz[samps] = np.nan
            self.fr[:, samps] = np.nan
            self.spikes[:, samps] = np.nan

    def generate_trial_condition_table(self):

        n_trials = len(self.trial_table)
        n_conditions = len(self.trial_conditions)

        temp = np.zeros((n_trials, n_conditions), dtype=bool)

        trial_condition_table = pd.DataFrame(temp, columns=list(self.trial_conditions.keys()))

        for condition in self.trial_conditions:
            cond_bool = np.ones(n_trials, dtype=bool)

            if condition in ['All']:
                pass
            elif condition in ['Out', 'In']:
                pass
            elif condition == 'Even':
                cond_bool[::2] = False
            elif condition == 'Odd':
                cond_bool[1::2] = False
            else:
                sub_condition = self.trial_conditions[condition]
                cond_bool = np.ones(n_trials, dtype=bool)

                for sc, val in sub_condition.items():
                    cond_bool = (cond_bool) & (self.trial_table[sc] == val)

            trial_condition_table.loc[:, condition] = cond_bool

        return trial_condition_table

    def get_condition_trials(self, condition, n_sel_trials=None, seed=None):

        if seed is not None:
            np.random.seed(seed)

        if condition in self.trial_condition_table.columns:
            pass
        else:
            raise ValueError

        trials = self.trial_condition_table.index[self.trial_condition_table[condition]].values

        if n_sel_trials is not None:
            trials = np.random.choice(trials, n_sel_trials, replace=False)

        return trials

    def get_trial_times(self, trials=None):
        if trials is None:
            n_trials = self.n_trials
            trials = self.all_trials
        else:
            n_trials = len(trials)

        df = pd.DataFrame(np.zeros((n_trials, 3), dtype=int), columns=['t0', 'tE', 'tR'])
        outbound_samps = np.empty(0, dtype=int)
        inbound_samps = np.empty(0, dtype=int)
        for ii, tr in enumerate(trials):
            df.loc[ii, 't0'] = self.trial_table.loc[tr, 't0']
            df.loc[ii, 'tE'] = self.trial_table.loc[tr, 'tE']
            if tr < (self.n_trials - 1):

                return_home_detections = self.event_table.loc[(self.event_table.out_bound == 0) &
                                                              (self.event_table.trial_num == tr) &
                                                              (self.event_table.event == 'DE1'), 't0'].values
                if len(return_home_detections) > 0:
                    df.loc[ii, 'tR'] = return_home_detections[0]
                else:
                    df.loc[ii, 'tR'] = self.trial_table.loc[tr + 1, 't0']

            else:
                df.loc[ii, 'tR'] = self.n_total_samps - 1

            outbound_samps = np.append(outbound_samps, np.arange(df.t0[ii], df.tE[ii] + 1, dtype=int))
            inbound_samps = np.append(inbound_samps, np.arange(df.tE[ii], df.tR[ii] + 1, dtype=int))

        return df, outbound_samps, inbound_samps

    def get_trial_track_pos(self, trials=None, trial_seg='out'):
        if trials is None:
            n_trials = self.n_trials
            trials = self.all_trials
        else:
            n_trials = len(trials)

        assert trial_seg in ['out', 'in', 'all']

        x = np.zeros(n_trials, dtype=object)
        y = np.zeros(n_trials, dtype=object)

        track_data = self.track_data
        for ii, tr in enumerate(trials):
            if trial_seg == 'out':
                t0 = self.trial_times.t0[tr]
                tE = self.trial_times.tE[tr]
            elif trial_seg == 'in':
                t0 = self.trial_times.tE[tr]
                tE = self.trial_times.tR[tr]
            else:
                t0 = self.trial_times.t0[tr]
                tE = self.trial_times.tR[tr]

            x[ii] = track_data.loc[t0:tE, 'x']
            y[ii] = track_data.loc[t0:tE, 'y']

        return x, y

    def get_trial_zones(self, trials=None, trial_seg='out'):
        if trials is None:
            n_trials = self.n_trials
            trials = self.all_trials
        else:
            n_trials = len(trials)

        assert trial_seg in ['out', 'in', 'all']

        trial_zones = np.zeros(n_trials, dtype=object)
        pz = self.pz
        for ii, tr in enumerate(trials):
            if trial_seg == 'out':
                t0 = self.trial_times.t0[tr]
                tE = self.trial_times.tE[tr]
            elif trial_seg == 'in':
                t0 = self.trial_times.tE[tr]
                tE = self.trial_times.tR[tr]
            elif trial_seg == 'all':
                t0 = self.trial_times.t0[tr]
                tE = self.trial_times.tR[tr]

            trial_zones[ii] = pz[t0:(tE + 1)]

        return trial_zones

    def get_trial_zones_samps_counts_mat(self, trial_seg='out'):

        trial_zones = self.trial_zones[trial_seg]

        df = pd.DataFrame(np.zeros((self.n_trials, self.tmz.n_all_segs)), columns=self.tmz.all_segs_names)
        for tr in range(self.n_trials):
            df.loc[tr] = pd.value_counts(trial_zones[tr])
        df = df.fillna(0)
        return df

    def get_trial_neural_data(self, trials=None, data_type='fr', trial_seg='out'):

        assert trial_seg in ['out', 'in', 'all']

        if trials is None:
            n_trials = self.n_trials
            trials = self.all_trials
        else:
            n_trials = len(trials)

        if data_type == 'fr':
            neural_data = self.fr
        elif data_type == 'spikes':
            neural_data = self.spikes
        else:
            print("Invalid data type.")
            return

        trial_data = np.zeros((self.n_units, n_trials), dtype=object)

        for ii, tr in enumerate(trials):
            if trial_seg == 'out':
                t0 = self.trial_times.t0[tr]
                tE = self.trial_times.tE[tr]
            elif trial_seg == 'in':
                t0 = self.trial_times.tE[tr]
                tE = self.trial_times.tR[tr]
            elif trial_seg == 'all':
                t0 = self.trial_times.t0[tr]
                tE = self.trial_times.tR[tr]

            for unit in range(self.n_units):
                trial_data[unit, ii] = neural_data[unit, t0:(tE + 1)]

        return trial_data

    def get_trial_rate_maps(self, trials=None, data_type='fr', occupation_thr=1, trial_seg='out', occ_rate_mask=False):

        if trials is None:
            trials = self.all_trials

        if data_type in ['fr', 'spikes']:
            neural_data = self.get_trial_neural_data(data_type=data_type, trial_seg=trial_seg)
            if data_type == 'fr':
                rate_map_function = spatial_funcs.firing_rate_2_rate_map
            else:
                rate_map_function = spatial_funcs.spikes_2_rate_map
        else:
            print("Invalid data type.")
            return

        x, y = self.get_trial_track_pos(trials)
        x = np.hstack(x)
        y = np.hstack(y)

        pos_count_map = self.get_trial_pos_counts_map(trials)

        mask = pos_count_map >= occupation_thr

        args = dict(x=x, y=y, x_bin_edges=self.x_edges, y_bin_edges=self.y_edges,
                    pos_count_map=pos_count_map, mask=mask,
                    spatial_window_size=self.si.task_params['spatial_window_size'],
                    spatial_sigma=self.si.task_params['spatial_sigma'],
                    time_step=self.si.params['time_step'])

        # pre-allocate and set up the map function to be looped
        rate_maps = np.zeros((self.n_units, len(self.y_edges) - 1, len(self.x_edges) - 1))

        if occ_rate_mask:
            sm_mask = spatial_funcs.smooth_2d_map(pos_count_map, n_bins=args['spatial_window_size'],
                                                  sigma=args['spatial_sigma'])
            sm_nan_mask = sm_mask < occupation_thr
        for unit in range(self.n_units):
            rate_maps[unit] = rate_map_function(np.hstack(neural_data[unit]), **args)
            if occ_rate_mask:
                rate_maps[unit][sm_nan_mask] = np.nan

        return rate_maps

    def get_trial_pos_counts_map(self, trials=None, trial_seg='out'):

        if trials is None:
            trials = self.all_trials

        x, y = self.get_trial_track_pos(trials, trial_seg=trial_seg)
        x = np.hstack(x)
        y = np.hstack(y)

        # get occupancy map
        pos_count_map = spatial_funcs.histogram_2d(x, y, self.x_edges, self.y_edges)

        return pos_count_map

    def get_all_trial_zone_rates(self, data_type='fr',
                                 occupation_trial_samp_thr=1, trial_seg='out'):

        if data_type == 'fr':
            neural_data = self.get_trial_neural_data(trial_seg=trial_seg)
        elif data_type == 'spikes':
            raise NotImplementedError
        else:
            print("Invalid data type.")
            return

        n_segs = self.tmz.n_all_segs
        seg_names = self.tmz.all_segs_names

        zones_by_trial = self.trial_zones[trial_seg]
        trial_zone_rates = np.zeros(self.n_units, dtype=object)

        dummy_df = pd.DataFrame(np.zeros((self.n_trials, n_segs)) * np.nan, columns=seg_names)
        for unit in range(self.n_units):
            trial_zone_rates[unit] = dummy_df.copy()

        for ii in range(self.n_trials):

            pzm = self.tmz.get_pos_zone_mat(zones_by_trial[ii], segment_type='subseg')
            pz_counts = pzm.sum()
            pzmn = (pzm / pz_counts).fillna(0)  # pozitions zones normalized by occupancy

            trial_data = np.nan_to_num(np.array(list(neural_data[:, ii]), dtype=np.float))
            zone_rates = trial_data @ pzmn
            zone_rates.loc[:, pz_counts < occupation_trial_samp_thr] = np.nan

            for unit in range(self.n_units):
                trial_zone_rates[unit].loc[ii] = zone_rates.loc[unit]

        return trial_zone_rates

    def get_trial_segment_rates(self, trials=None, segment_type='subseg', trial_seg='out', occ_thr=None):
        if trials is None:
            trials = np.arange(self.n_trials, dtype=int)

        if occ_thr is None:
            occ_thr = self.occupation_thrs[segment_type]

        trial_zone_rates = self.trial_zones_rates[trial_seg]

        trial_segment_rates = np.zeros(self.n_units, dtype=object)
        for unit in range(self.n_units):
            trial_segment_rates[unit] = trial_zone_rates[unit].loc[trials].copy().reset_index(drop=True)

        if segment_type != 'subseg':
            trial_zone_samp_counts = self.trial_zone_samps_counts_mat[trial_seg].loc[trials].fillna(0).reset_index(
                drop=True)
            segment_counts = self.tmz.subseg_pz_mat_transform(trial_zone_samp_counts, segment_type)

            trial_zone_samp_seg_norm = self.tmz.subseg_pz_mat_segment_norm(trial_zone_samp_counts,
                                                                           segment_type, segment_counts)

            for unit in range(self.n_units):
                norm_zone_rates = (trial_segment_rates[unit] * trial_zone_samp_seg_norm).fillna(0)
                trial_segment_rates[unit] = self.tmz.subseg_pz_mat_transform(norm_zone_rates, segment_type)
                trial_segment_rates[unit][segment_counts < occ_thr] = np.nan

        return trial_segment_rates

    def get_unit_trial_zone_rates(self, unit=0, trials=None, trial_seg='out'):
        if trials is None:
            trials = self.all_trials

        return self.trial_zones_rates[trial_seg][unit].loc[trials, :]

    def get_avg_zone_rates(self, trials=None, samps=None, data_type='fr',
                           segment_type='subseg', occupation_samp_thr=5, trial_seg='out'):
        """
        returns the average zone firing rates for the given trials or samps
        :param trials: array of ints, trials to generate zone maps(ignored if samps are provided)
        :param samps: array of ints, samples to generate zone maps
        :param data_type: str ['fr', 'spikes'], firing rate or spikes
        :param segment_type: string, ['seg', 'subseg', 'bigseg']
        :param occupation_samp_thr: int, minimun number of samples in a zone, returns nan otherwise for that zone
        :param trial_seg: str ['out', 'in', 'all'], segment of the trials to use, ignored if samps are provided
        :return:
        data frame of n_units x n_zones of rates
        """

        assert trial_seg in ['out', 'in', 'all']

        if samps is None:
            if trials is None:
                if trial_seg == 'out':
                    samps = self.outbound_samps
                elif trial_seg == 'in':
                    samps = self.inbound_samps
                elif trial_seg == 'all':
                    out_samps = self.outbound_samps
                    in_samps = self.inbound_samps
                    samps = np.concatenate((out_samps, in_samps))

            else:
                _, out_samps, in_samps = self.get_trial_times(trials)
                if trial_seg == 'out':
                    samps = out_samps
                elif trial_seg == 'in':
                    samps = in_samps
                elif trial_seg == 'all':
                    samps = np.concatenate((out_samps, in_samps))

        if data_type == 'fr':
            neural_data = self.fr[:, samps]
        elif data_type == 'spikes':
            neural_data = self.spikes[:, samps]
        else:
            print("Invalid data type.")
            return

        pz = self.pz[samps]  # position zones
        pzm = self.tmz.get_pos_zone_mat(pz, segment_type=segment_type)  # position zones matrix

        pz_counts = pzm.sum()
        pzmn = pzm / pz_counts  # pozitions zones normalized by occupancy

        nan_locs = np.isnan(neural_data)
        neural_data[nan_locs] = 0
        zone_rates = neural_data @ pzmn  # matrix multiply
        zone_rates.loc[:, pz_counts < occupation_samp_thr] = np.nan

        return zone_rates

    def get_avg_trial_zone_rates(self, trials=None, segment_type='subseg', occupation_trial_samp_thr=5,
                                 trial_seg='out', reweight_by_trial_zone_counts=False, return_z=False):
        """
        returns the trial average for the given trial.
        :param trials: list of trials
        :param segment_type: 'str', if not subseg, the function will scale the fr data according to the number of
            sampples for that zone in that trial, according to how many samples were in the segment
        :param occupation_trial_samp_thr: minimum number of samples to be included in analyses
        :param trial_seg: what part of the trial.
        :param reweight_by_trial_zone_counts: if True, this is equivalent to pooling all the samples to avg.
                if false, the return value will be the mean across trials.
        :return:
            n_units x n_segs dataframe
        """

        trial_segment_rates = self.get_trial_segment_rates(trials=trials,
                                                           segment_type=segment_type,
                                                           trial_seg=trial_seg,
                                                           occ_thr=occupation_trial_samp_thr)

        # average zone rate
        segs = list(trial_segment_rates[0].columns)
        azr = pd.DataFrame(0, index=range(self.n_units), columns=segs)

        if reweight_by_trial_zone_counts:
            tzc = self.trial_zone_samps_counts_mat['out'].loc[trials].copy().reset_index(drop=True)
            tsc = self.tmz.subseg_pz_mat_transform(tzc, segment_type=segment_type)
            tscn = tsc / tsc.sum()
            for unit in range(self.n_units):
                azr.loc[unit] = (trial_segment_rates[unit] * tscn).sum()
        else:
            for unit in range(self.n_units):
                m = trial_segment_rates[unit].mean()
                if return_z:
                    azr.loc[unit] = m / trial_segment_rates[unit].std()
                else:
                    azr.loc[unit] = m

        return azr

    def get_trials_boot_cond_set(self, cond_set, n_sel_trials=None, n_boot=100, seed=0):
        """
        for a given condition and subconditions (see format below), balances the number of trials across subconditions
        returns a dictionary for each condition with n_trials x n_bootstaps
        :param cond_set: {cond: [sub_cond1, sub_cond2,..]}
        :param n_sel_trials: [None, 'max', 'min', int]
            None -> balances the max number of sub condition trials and min number
            max -> takes the max number of trials from a subcondition, and resamples the other subconditions
                with replacement to match
            min -> like max, but with min
        :param n_boot: number of bootstaps
        :param seed: random seed
        :return:
        dictionary by condition, with a matrix of trials x bootstaps, each entry would be a trial id
        """

        np.random.seed(seed)
        trial_sets = {}
        out_trial_sets = {}
        for cond, sub_conds in cond_set.items():

            cond_trials = self.get_condition_trials(condition=cond)
            trial_sets[cond] = {}
            min_set_length = 1000
            max_set_lenth = 0
            for sc in sub_conds:
                sc_trials = self.get_condition_trials(condition=sc)
                trial_sets[cond][sc] = np.intersect1d(cond_trials, sc_trials)
                min_set_length = min(len(trial_sets[cond][sc]), min_set_length)
                max_set_lenth = max(len(trial_sets[cond][sc]), max_set_lenth)

            if n_sel_trials is None:
                n_sub_cond_trials = int((min_set_length + max_set_lenth) / 2)
            elif n_sel_trials == 'max':
                n_sub_cond_trials = int(max_set_lenth)
            elif n_sel_trials == 'min':
                n_sub_cond_trials = int(min_set_length)
            elif type(n_sel_trials) == int:
                n_sub_cond_trials = n_sel_trials
            else:
                raise ValueError

            n_cond_balanced_trials = n_sub_cond_trials * len(sub_conds)
            out_trial_sets[cond] = np.zeros((n_cond_balanced_trials, n_boot), dtype=int)
            for boot in range(n_boot):
                sub_cond_trials = np.zeros(len(sub_conds), dtype=object)
                for jj, sc in enumerate(sub_conds):
                    sub_cond_trials[jj] = np.random.choice(trial_sets[cond][sc], n_sub_cond_trials, replace=True)
                out_trial_sets[cond][:, boot] = np.hstack(sub_cond_trials)

        if len(out_trial_sets) == 1:
            return out_trial_sets[cond]
        else:
            return out_trial_sets

    def get_boot_zone_rates(self, cond_sets=None, trial_segs=None, segment_type='subseg', n_boot=100, seed=0):

        np.random.seed(seed)

        if cond_sets is None:
            cond_sets = {'All': ['CL', 'CR']}
        n_conds = len(cond_sets)

        if trial_segs is None:
            trial_segs = 'out'

        if trial_segs in ['out', 'in', 'all']:
            trial_cond_segs = {}
            for cond, sub_conds in cond_sets.items():
                trial_cond_segs[cond] = trial_segs
        else:
            trial_cond_segs = trial_segs
            assert len(trial_segs) == n_conds

        trial_sets = self.get_trials_boot_cond_set(cond_sets, n_boot=n_boot)

        if segment_type == 'subseg':
            n_zones = self.tmz.n_all_segs
        elif segment_type == 'bigseg':
            n_zones = len(self.tmz.bigseg_names)
        else:
            n_zones = self.tmz.n_zones

        cond_zr = {cond: pd.DataFrame(np.zeros((n_boot, n_zones * self.n_units)))
                   for cond in cond_sets.keys()}

        for cond in cond_sets.keys():
            for boot in range(n_boot):
                cond_zr[cond].loc[boot] = self.get_avg_zone_rates(trials=trial_sets[cond][:, boot],
                                                                  trial_seg=trial_cond_segs[cond],
                                                                  segment_type=segment_type).values.flatten()

        return cond_zr

    def zone_rate_trial_quantification(self, cond=None, trials=None, trial_seg='out'):
        """
        simple quantification of zone rate maps for a given set of trials or conditions.
        gets spatial information, mean rate on left, right and stem zones,
        mean rates on the reward wells. saves standard deviation for each of the mean rates as well.
        :return:
        data frame with results, index by unit
        """
        if cond is not None:
            trials = self.get_condition_trials(condition=cond)
        elif trials is not None:
            pass
        else:
            trials = np.arange(self.n_trials)

        zones_by_trial = self.zones_by_trial[trial_seg].loc[trials]
        trial_zone_counts = zones_by_trial.sum()
        trial_zone_prob = trial_zone_counts / trial_zone_counts.sum()

        col_names_root = ['all', 'left', 'right', 'stem', 'H', 'D', 'G1', 'G2', 'G3', 'G4']
        col_names = ['si']

        for cn in col_names_root:
            for post_str in ['m', 's', 'n', 'z']:
                col_names.append(f"{cn}_{post_str}")

        df = pd.DataFrame(np.zeros((self.n_units, len(col_names))), columns=col_names)
        df[f"all_n"] = trial_zone_counts.max()

        for well in self.tmz.wells:
            df[f"{well}_n"] = trial_zone_counts[well]

        for split, zones in self.tmz.split_zones_all.items():
            df[f"{split}_n"] = trial_zone_counts[zones].max()

        for unit in range(self.n_units):
            zr = self.get_unit_trial_zone_rates(unit, trials, trial_seg=trial_seg)
            zrm = zr.mean()
            nzr = zrm / zrm.sum()
            zrs = zr.std()

            df.loc[unit, f"all_m"] = (zrm * trial_zone_prob).sum()
            df.loc[unit, f"all_s"] = (zrs * trial_zone_prob).sum()
            df.loc[unit, 'si'] = spatial_funcs.spatial_information(trial_zone_prob, nzr)

            for well in self.tmz.wells:
                df.loc[unit, f"{well}_m"] = zrm[well]
                df.loc[unit, f"{well}_s"] = zrs[well]

            for split, zones in self.tmz.split_zones_all.items():
                z_p = trial_zone_prob[zones]
                df.loc[unit, f"{split}_m"] = (zr[zones].mean() * z_p).sum()
                df.loc[unit, f"{split}_s"] = (zr[zones].std() * z_p).sum()

        for split in col_names_root:
            df[f"{split}_z"] = df[f"{split}_m"] / df[f"{split}_s"]

        return df

    def zone_rate_maps_corr(self, cond1=None, cond2=None, trials1=None, trials2=None, zr_method='trial',
                            samps1=None, samps2=None, trial_seg1='out', trial_seg2='out',
                            corr_method='kendall'):
        """
        method for comparing zone rate maps across conditions or time_samps.
        conditions must be in the list. generates a zone rate maps from the trials, conditions or samples given
        and correlates the maps across conditions for all units.
        :param cond1: string trial condition 1
        :param cond2: string trial condition 2
        :param trials1:  list of trials
        :param trials2: list of trials
        :param zr_method: str, method of obtaining the zone rates (ignored if samps are provided):
            1. 'pooled' - all samples from all trials pooled
            2. 'trial' - trial average of the zones
            3. 'trial_z' - trial average divided by the means
            4. 'old' - numerically equivalent to pooled, uses get_avg_zone_rates function (slower than pooled).
                -> this method is utilized if samps are provided.
        :param samps1: 1st set of samples to create zone rate map
        :param samps2: 2nd set of samples to create zone rate map
        :param trial_seg1: str ['out', 'in', 'all'], segment of the trials to use, ignored if samps1 are provided
        :param trial_seg2: str ['out', 'in', 'all'], segment of the trials to use, ignored if samps2 are provided
        :param corr_method: method for correlation, valid values ['kendall', 'pearson', 'spearman']
        :return: array of correlation, with each entry corresponding to a unit
        """

        if samps1 is not None:
            zr1 = self.get_avg_zone_rates(samps=samps1)
        else:
            # get trials
            if cond1 is not None:
                trials1 = self.get_condition_trials(condition=cond1)
            elif trials1 is not None:
                pass
            else:
                raise ValueError

            if zr_method == 'pooled':
                zr1 = self.get_avg_trial_zone_rates(trials=trials1, trial_seg=trial_seg1,
                                                    reweight_by_trial_zone_counts=True)
            elif zr_method == 'trial':
                zr1 = self.get_avg_trial_zone_rates(trials=trials1, trial_seg=trial_seg1)
            elif zr_method == 'trial_z':
                zr1 = self.get_avg_trial_zone_rates(trials=trials1, trial_seg=trial_seg1, return_z=True)
            elif zr_method == 'old':
                zr1 = self.get_avg_zone_rates(trials=trials1, trial_seg=trial_seg1)
            else:
                raise ValueError

        if samps2 is not None:
            zr2 = self.get_avg_zone_rates(samps=samps2)
        else:
            # get trials
            if cond2 is not None:
                trials2 = self.get_condition_trials(condition=cond2)
            elif trials2 is not None:
                pass
            else:
                raise ValueError

            if zr_method == 'pooled':
                zr2 = self.get_avg_trial_zone_rates(trials=trials2, trial_seg=trial_seg2,
                                                    reweight_by_trial_zone_counts=True)
            elif zr_method == 'trial':
                zr2 = self.get_avg_trial_zone_rates(trials=trials2, trial_seg=trial_seg2)
            elif zr_method == 'trial_z':
                zr2 = self.get_avg_trial_zone_rates(trials=trials2, trial_seg=trial_seg2, return_z=True)
            elif zr_method == 'old':
                zr2 = self.get_avg_zone_rates(trials=trials2, trial_seg=trial_seg1)
            else:
                raise ValueError

        return zr1.corrwith(zr2, axis=1, method=corr_method)

    def zone_rate_maps_t(self, cond1=None, cond2=None, trials1=None, trials2=None,
                         trial_seg1='out', trial_seg2='out', trial_occupation_thr=2):
        """
        method for comparing zone rate maps across trials.
        conditions must be in the list. generates a zone rate maps from the trials, and compares them to the other
        condition using a t statistic by zone.
        :param cond1: string trial condition 1
        :param cond2: string trial condition 2
        :param trials1:  list of trials
        :param trials2: list of trials
        :param trial_seg1: str ['out', 'in'], segment of the trials to use, ignored if samps1 are provided
        :param trial_seg2: str ['out', 'in'], segment of the trials to use, ignored if samps2 are provided
        :param trial_occupation_thr: int,  minimun number of samples in a zone in a trial
        :return:
        data frame of n_units x n_zones of t values across the conditions/trial sets

        """

        if cond1 is not None:
            trials1 = self.get_condition_trials(condition=cond1)
        else:
            assert trials1 is not None

        if cond2 is not None:
            trials2 = self.get_condition_trials(condition=cond2)
        else:
            assert trials2 is not None

        out = pd.DataFrame(np.zeros((self.n_units, self.tmz.n_all_segs)),
                           columns=self.tmz.all_segs_names)

        for unit in range(self.n_units):
            tzr1 = self.get_unit_trial_zone_rates(unit=unit, trials=trials1, trial_seg=trial_seg1)
            tzr2 = self.get_unit_trial_zone_rates(unit=unit, trials=trials2, trial_seg=trial_seg2)
            out.loc[unit] = ttest_ind(tzr1, tzr2, nan_policy='omit')[0].data

        mask1 = self.zones_by_trial[trial_seg1].loc[trials1].sum() < trial_occupation_thr
        mask2 = self.zones_by_trial[trial_seg2].loc[trials2].sum() < trial_occupation_thr
        out.loc[:, (mask1 | mask2)] = np.nan

        return out

    def _zone_rate_maps_permute_corr(self, cond1=None, cond2=None, trials1=None, trials2=None, n_sel_trials=None,
                                     trial_seg1='out', trial_seg2='out', n_perm=100, corr_method='kendall',
                                     min_valid_trials=10, n_jobs=5):
        """
        similar to zone_rate_maps corr but resamples each trial set to have balanced trials sets, then repeats for
        n_perm.
        :param cond1: str, condition 1
        :param cond2 str, condition 2
        :param trials1: array of trials
        :param trials2: array of trials
        :param n_sel_trials: int: defaults simply selects the minimum length of trials1 and trials2
        :param n_perm: int, number of permutations
        :param corr_method: str, correlation method  ['kendall', 'pearson', 'spearman']
        :param min_valid_trials: int, minimum valid number of trials to allow comparison
        :param trial_seg1: str ['out', 'in'], segment of the trials to use
        :param trial_seg2: str ['out', 'in'], segment of the trials to use
        :return:
        pandas data frame with dimensions: n_units x n_perm
        """

        if cond1 is not None:
            trials1 = self.get_condition_trials(condition=cond1)
        else:
            assert trials1 is not None

        if cond2 is not None:
            trials2 = self.get_condition_trials(condition=cond2)
        else:
            assert trials2 is not None

        if n_sel_trials is None:
            n_sel_trials = min(len(trials1), len(trials2))
        else:
            assert type(n_sel_trials) == int

        if (len(trials1) < min_valid_trials) or (len(trials2) < min_valid_trials):
            return pd.DataFrame(np.zeros((self.n_units, n_perm)) * np.nan)

        def _worker():
            p_trials1 = np.random.choice(trials1, n_sel_trials, replace=False)
            p_trials2 = np.random.choice(trials2, n_sel_trials, replace=False)

            p_zr1 = self.get_avg_zone_rates(trials=p_trials1, trial_seg=trial_seg1)
            p_zr2 = self.get_avg_zone_rates(trials=p_trials2, trial_seg=trial_seg2)

            return p_zr1.corrwith(p_zr2, axis=1, method=corr_method)

        with Parallel(n_jobs=n_jobs) as parallel:
            corr = parallel(delayed(_worker)() for _ in range(n_perm))

        return pd.DataFrame(np.array(corr).T)

    def zone_rate_maps_comparison_analyses(self):
        """
        :return:
        """

        group_conds = {'CR-CL': ['CR', 'CL'], 'Even-Odd': ['Even', 'Odd'],
                       'Co-Inco': ['Co', 'Inco'], 'Out-In': ['Out', 'In'],
                       'Left': ['CoCL', 'IncoCR'], 'Right': ['CoCR', 'IncoCL'],
                       'CoSw-IncoSw': ['CoSw', 'IncoSw']}
        analyses = ['corr', 't_m', 't_var']

        col_names = []
        for ii in group_conds.keys():
            for jj in analyses:
                col_names.append(f"{ii}_{jj}")

        df = pd.DataFrame(np.zeros((self.n_units, len(group_conds) * 3)) * np.nan, columns=col_names)

        for group, conds in group_conds.items():

            if group == 'Out-In':
                trial_seg1 = 'out'
                trial_seg2 = 'in'
            else:
                trial_seg1 = 'out'
                trial_seg2 = 'out'

            trials1 = self.get_condition_trials(condition=conds[0])
            trials2 = self.get_condition_trials(condition=conds[1])

            zc1 = self.zones_by_trial[trial_seg1].loc[trials1].sum()
            zc2 = self.zones_by_trial[trial_seg2].loc[trials2].sum()
            zc = (zc1 + zc2) / 2
            zp = zc / zc.sum()

            df[f"{group}_corr"] = self.zone_rate_maps_corr(trials1=trials1, trials2=trials2,
                                                           trial_seg1=trial_seg1, trial_seg2=trial_seg2)

            try:
                ts = self.zone_rate_maps_t(trials1=trials1, trials2=trials2,
                                           trial_seg1=trial_seg1, trial_seg2=trial_seg2)

                # trial counts by zone weighted mean and variance
                average = (ts * zp).sum(axis=1).values
                variance = ((ts - average[:, np.newaxis]) ** 2 * zp).sum(axis=1)
                df[f"{group}_t_m"] = average
                df[f"{group}_t_var"] = variance
            except:
                pass

        return df

    def _zone_rate_maps_group_trials_perm_bal_corr(self, n_perm=100, n_sel_trials=None,
                                                   corr_method='kendall', min_valid_trials=10, n_jobs=5,
                                                   group_cond_sets=None):
        """
        this function computes balanced zone rate maps correlations between sets of trials. the main component is the
        group_cond_sets parameter. it is a nested dictionary (3 levels):
            group names -> major condition -> balancing conditions
        See below for example.
        Returns
        :param n_perm: number of permutations
        :param n_sel_trials: int, number of trials to use by condition,
        :param corr_method: str, correlation method
        :param min_valid_trials: int, won't perform computation if a trial combinations is less than this #
        :param n_jobs: int, number of parallel workers for permutations
        :param group_cond_sets: dict, group name -> major condition - balancing conditions
                {'Even-Odd': {'Even':['CL','CR'],'Odd':['CL','CR']},
                 'CR-CL': {'CR':['Co', 'InCo'], 'CL':['Co', 'InCo']} }
        :return:
        dict with group names as keys, and pandas data frame with n_units x n_perm as values
        """

        if group_cond_sets is None:
            group_cond_sets = {'CR-CL': {'CR': ['Co', 'Inco'], 'CL': ['Co', 'Inco']},
                               'Even-Odd': {'Even': ['Co', 'Inco'], 'Odd': ['Co', 'Inco']}}

        out_dict = {}
        n_trials = {}
        for group, cond_sets in group_cond_sets.items():

            conds = list(cond_sets.keys())
            trial_sets = {}
            min_set_length = 1000
            for cond, sub_conds in cond_sets.items():
                cond_trials = self.get_condition_trials(condition=cond)
                trial_sets[cond] = {}
                for sc in sub_conds:
                    sc_trials = self.get_condition_trials(condition=sc)
                    trial_sets[cond][sc] = np.intersect1d(cond_trials, sc_trials)
                    min_set_length = min(len(trial_sets[cond][sc]), min_set_length)

            if n_sel_trials is None:
                n_sel_g_trials = min_set_length
            else:
                n_sel_g_trials = n_sel_trials

            n_trials[group] = n_sel_g_trials

            if n_sel_g_trials < min_valid_trials:
                # invalid set partition, skip
                out_dict[group] = pd.DataFrame(np.zeros((self.n_units, n_perm)) * np.nan)
                continue

            def _worker():
                _cond_trial_sets = {}

                for _cond, _sub_conds in cond_sets.items():
                    _cond_trials = np.zeros(len(_sub_conds), dtype=object)

                    for jj, _sc in enumerate(_sub_conds):
                        _cond_trials[jj] = np.random.choice(trial_sets[_cond][_sc], n_sel_g_trials, replace=False)

                    _cond_trial_sets[_cond] = np.hstack(_cond_trials)

                return self.zone_rate_maps_corr(trials1=_cond_trial_sets[conds[0]], trials2=_cond_trial_sets[conds[1]],
                                                corr_method=corr_method)

            with Parallel(n_jobs=n_jobs) as parallel:
                corr = parallel(delayed(_worker)() for _ in range(n_perm))

            out_dict[group] = pd.DataFrame(np.array(corr).T)
        return out_dict, n_trials

    def zone_rate_maps_bal_conds_boot_corr(self, n_boot=100,
                                           corr_method='kendall', min_valid_trials=5, n_jobs=5,
                                           bal_cond_pair=None, zr_method='trial',
                                           parallel=None):
        """
        this function computes balanced zone rate maps correlations between sets of trials.
        using bootstrap to equate the samples sizes between groups, size is the mean between the set sizes.
        the main component is the bal_cond_pair parameter, a pair of conditions that must be in bal_cond_pairs.
        bal_cond_pairs is a class parameter that indicates valid comparisons.
        See below for example.
        Returns
        :param n_boot: number of permutations
        :param corr_method: str, correlation method
        :param min_valid_trials: int, won't perform computation if a trial combinations is less than this #
        :param n_jobs: int, number of parallel workers for permutations
        :param bal_cond_pair: list of the balanced pair to use, entry must be in bal_cond_pairs
        :param zr_method: str, see zone_rate_maps_corr for details
        :param parallel: parallel object to avoid multiple instantiations of workers.

        :return:
        dict with group names as keys, and pandas data frame with n_units x n_perm as values
        """

        if bal_cond_pair is None:
            bal_conds = self.bal_cond_pairs[0].split('-')
        else:
            bal_conds = bal_cond_pair.split('-')

        df = pd.DataFrame(np.nan, index=range(self.n_units), columns=range(n_boot))

        trial_sets = {}
        trial_segs = {}
        for bal_cond in bal_conds:
            if bal_cond in self.bal_cond_sets:
                bal_cond_set = self.bal_cond_sets[bal_cond]
            else:
                bal_cond_set = self._decode_cond(bal_cond)

            cond = bal_cond_set['cond']
            sub_conds = bal_cond_set['sub_conds']

            cond_set = {cond: sub_conds}
            trial_segs[bal_cond] = bal_cond_set['trial_seg']

            try:
                trial_sets[bal_cond] = self.get_trials_boot_cond_set(cond_set, n_boot=n_boot)
            except ValueError:
                print("error getting the trial sets", bal_cond)
                #traceback.print_exc(file=sys.stdout)
                #return df

        ok_cond_trials_flag = True
        for bal_cond in bal_conds:
            if (trial_sets[bal_cond].shape[0]) < min_valid_trials:
                ok_cond_trials_flag = False
                break

        if not ok_cond_trials_flag:
            return df

        def _worker(boot):
            return self.zone_rate_maps_corr(trials1=trial_sets[bal_conds[0]][:, boot],
                                            trials2=trial_sets[bal_conds[1]][:, boot],
                                            trial_seg1=trial_segs[bal_conds[0]], trial_seg2=trial_segs[bal_conds[1]],
                                            corr_method=corr_method, zr_method=zr_method)

        try:
            if isinstance(parallel, Parallel):
                corr = parallel(delayed(_worker)(boot) for boot in range(n_boot))
                df = pd.DataFrame(np.array(corr).T)

            elif parallel == True:
                with Parallel(n_jobs=n_jobs) as parallel:
                    corr = parallel(delayed(_worker)(boot) for boot in range(n_boot))
                    df = pd.DataFrame(np.array(corr).T)
            else:
                for boot in range(n_boot):
                    df[boot] = _worker(boot)
        except:
            print("Error on parallel step.")
            #traceback.print_exc(file=sys.stdout)
        return df

    def all_zone_remapping_analyses(self, corr_method='kendall', n_boot=100, n_jobs=5, zr_method='trial'):
        """
        Runs zone_rate_maps_comparison_analyses and zone_rate_maps_bal_conds_boot_corr and puts them into a single
        data frame.
        :param zr_method: str, method to compute zone rates
        :param n_jobs: number of workers to initiate and use to bootstrap
        :param n_boot: number of bootstraps for the trial sets.
        :param corr_method: string, correlation method to use
        :return:
            data frame: n_units x n_analyses
        """
        with np.errstate(divide='ignore'):
            # main analyses
            df = self.zone_rate_maps_comparison_analyses()

            n_zones = self.tmz.n_all_segs
            with Parallel(n_jobs=n_jobs) as parallel:
                bcorrs = {}
                for cond_pair in self.bal_cond_pairs:
                    bcorrs[cond_pair] = self.zone_rate_maps_bal_conds_boot_corr(bal_cond_pair=cond_pair,
                                                                                corr_method=corr_method,
                                                                                n_boot=n_boot,
                                                                                zr_method=zr_method,
                                                                                parallel=parallel)

            if corr_method == 'kendall':
                def _transform_corr(_c):
                    return rs.fisher_r2z(rs.kendall2pearson(_c))
            else:
                def _transform_corr(_c):
                    return rs.fisher_r2z(_c)

            for group, corrs in bcorrs.items():
                try:
                    c = corrs.copy()
                    c.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df[f"{group}-corr_m"] = c.mean(axis=1)

                    z = _transform_corr(corrs)
                    z.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df[f"{group}-corr_z"] = z.mean(axis=1)
                except:
                    df[f"{group}-corr_m"]= np.nan
                    df[f"{group}-corr_z"]= np.nan


            for test, null in self.test_null_bal_cond_pairs.items():
                try:
                    zc, pc = rs.compare_corrs(bcorrs[test], bcorrs[null],
                                              n_zones, n_zones, corr_method=corr_method)

                    zc.replace([np.inf, -np.inf], np.nan, inplace=True)
                    pc = pd.DataFrame(pc).replace([np.inf, -np.inf], np.nan)

                    df[f"{test}-{null}-corr_zm"] = zc.mean(axis=1)
                    df[f"{test}-{null}-corr_zp"] = rs.combine_pvals(pc, axis=1)

                    temp = ttest_1samp(zc, 0, nan_policy='omit', axis=1)
                    df[f"{test}-{null}-corr_zt"] = temp[0]
                    df[f"{test}-{null}-corr_ztp"] = temp[1]
                except:
                    df[f"{test}-{null}-corr_zm"] = np.nan
                    df[f"{test}-{null}-corr_zp"] = np.nan
                    df[f"{test}-{null}-corr_zt"] = np.nan
                    df[f"{test}-{null}-corr_ztp"] = np.nan
                    #traceback.print_exc(file=sys.stdout)
            return df

    def bal_conds_segment_rate_analyses(self, segment_type, n_boot=100, n_jobs=5):

        m, n, t, p = self.bal_conds_segment_rate_boot(segment_type=segment_type, n_boot=n_boot, n_jobs=n_jobs)

        conds = list(self.bal_cond_sets.keys())
        cond_pairs = self.bal_cond_pairs

        seg_names = self.tmz.get_segment_type_names(segment_type)

        n_units = self.n_units

        cols1 = []
        cols2 = []
        for seg in seg_names:
            for cond in conds:
                cols1.append(f"{cond}-{seg}-m")
                cols1.append(f"{cond}-{seg}-n")

            for cond_pair in cond_pairs:
                cols2.append(f"{cond_pair}-{seg}-t")
                cols2.append(f"{cond_pair}-{seg}-p")

        cols = cols1 + cols2
        df = pd.DataFrame(index=range(n_units), columns=cols)

        for cond in conds:
            cond_col = [f"{cond}-{seg}-m" for seg in seg_names]
            df[cond_col] = m[cond].mean(axis=0)

            cond_col = [f"{cond}-{seg}-n" for seg in seg_names]
            df[cond_col] = n[cond].mean(axis=0)

        for cond_pair in cond_pairs:
            cond_col = [f"{cond_pair}-{seg}-t" for seg in seg_names]
            df[cond_col] = t[cond_pair].mean(axis=0)

            cond_col = [f"{cond_pair}-{seg}-p" for seg in seg_names]
            df[cond_col] = rs.combine_pvals(p[cond_pair], axis=0)

        return df

    def bal_conds_segment_rate_boot(self, segment_type='bigseg', n_boot=100, n_jobs=5):
        """
        performs bootstrap analyses on segments of the maze and their comparisons.
        :param segment_type:
        :param zr_method: str, method to compute zone rates
        :param n_jobs: number of workers to initiate and use to bootstrap
        :param n_boot: number of bootstraps for the trial sets.
        :return:
            -> m: dict of mean activity by condition. array of n_units x n_boot x n_segments
            -> t: dict of stat by condition pairs. array of n_units x n_boot x n_segments
            -> p: dict of pvals for condition pairs. array of n_units x n_boot x n_segments
        """

        bal_cond_sets = self.bal_cond_sets
        conds = list(bal_cond_sets.keys())
        cond_pairs = self.bal_cond_pairs

        seg_names = self.tmz.get_segment_type_names(segment_type)

        n_units = self.n_units
        n_segs = len(seg_names)
        n_conds = len(conds)

        trial_sets = {}
        for cond in conds:
            trial_sets[cond] = self.get_trials_boot_cond_set(bal_cond_sets[cond]['cond_set'])

        def _worker(boot):
            _m = {_cond: np.zeros((n_units, n_segs)) for _cond in conds}
            _n = {_cond: np.zeros((n_units, n_segs)) for _cond in conds}
            _t = {cond_pair: np.zeros((n_units, n_segs)) for cond_pair in cond_pairs}
            _p = {cond_pair: np.zeros((n_units, n_segs)) for cond_pair in cond_pairs}

            trial_segment_rates = {}

            for _cond in conds:
                bal_cond_set = bal_cond_sets[_cond]
                trial_segment_rates[_cond] = self.get_trial_segment_rates(trial_sets[_cond][:, boot],
                                                                          segment_type=segment_type,
                                                                          trial_seg=bal_cond_set['trial_seg'])
                for _unit in range(n_units):
                    _m[_cond][_unit] = trial_segment_rates[_cond][_unit].mean()
                    _n[_cond][_unit] = trial_segment_rates[_cond][_unit].count()

            for cond_pair in cond_pairs:
                cond1, cond2 = cond_pair.split('-')
                for _unit in range(n_units):
                    temp = ttest_ind(trial_segment_rates[cond1][_unit],
                                     trial_segment_rates[cond2][_unit],
                                     nan_policy='omit')
                    _t[cond_pair][_unit], _p[cond_pair][_unit] = temp[0], temp[1]

            return _m, _n, _t, _p

        with Parallel(n_jobs=n_jobs) as parallel:
            out = parallel(delayed(_worker)(boot) for boot in range(n_boot))

        # reformat output
        m = {cond: np.zeros((n_boot, n_units, n_segs)) for cond in conds}
        n = {cond: np.zeros((n_boot, n_units, n_segs)) for cond in conds}
        t = {cond_pair: np.zeros((n_boot, n_units, n_segs)) for cond_pair in cond_pairs}
        p = {cond_pair: np.zeros((n_boot, n_units, n_segs)) for cond_pair in cond_pairs}

        for boot in range(n_boot):
            _m, _n, _t, _p = out[boot]

            for cond in _m.keys():
                m[cond][boot] = _m[cond]
                n[cond][boot] = _n[cond]
            for cond_pair in _t.keys():
                t[cond_pair][boot] = _t[cond_pair]
                p[cond_pair][boot] = _p[cond_pair]

        return m, n, t, p

    def get_avg_seg_rates_boot(self, segment_type='bigseg', n_boot=100, occ_thr=None, n_jobs=5):

        if occ_thr is None:
            occ_thr = self.occupation_thrs[segment_type]

        conds = list(self.bal_cond_sets.keys())

        seg_names = self.tmz.get_segment_type_names(segment_type)
        n_segs = len(seg_names)
        n_units = self.n_units
        n_conds = len(conds)

        n_rows = n_boot * n_units * n_segs * n_conds
        df = pd.DataFrame(index=range(n_rows), columns=['boot', 'cond', 'unit', 'seg', 'm'])
        cnt = 0
        block_idx_len = n_units * n_segs

        def _worker(_boot):
            return self.get_avg_trial_zone_rates(trials=trial_set[:, _boot], segment_type=segment_type,
                                                          trial_seg=trial_seg, occupation_trial_samp_thr=occ_thr)

        with Parallel(n_jobs=n_jobs) as parallel:
            for cond in conds:
                bal_cond_set = self.bal_cond_sets[cond]
                cond_set = bal_cond_set['cond_set']
                trial_seg = bal_cond_set['trial_seg']

                trial_set = self.get_trials_boot_cond_set(cond_set)

                temp = parallel(delayed(_worker)(boot) for boot in range(n_boot))
                for boot in range(n_boot):
                    idx = np.arange(block_idx_len) + cnt * block_idx_len
                    b_temp = temp[boot]
                    b_temp['unit'] = b_temp.index
                    b_temp['cond'] = cond
                    b_temp['boot'] = boot

                    b_temp = b_temp.melt(id_vars=['boot', 'cond', 'unit'], value_name='seg', var_name='m')
                    df.loc[idx] = b_temp.set_index(idx)

                    cnt += 1
        # for cond in conds:
        #     bal_cond_set = self.bal_cond_sets[cond]
        #     cond_set = bal_cond_set['cond_set']
        #     trial_seg = bal_cond_set['trial_seg']
        #
        #     trial_set = self.get_trials_boot_cond_set(cond_set)
        #
        #     for boot in range(n_boot):
        #         idx = np.arange(block_idx_len) + cnt * block_idx_len
        #
        #         trials = trial_set[:, boot]
        #         temp = self.get_avg_trial_zone_rates(trials=trials, segment_type=segment_type,
        #                                              trial_seg=trial_seg, occupation_trial_samp_thr=occ_thr)
        #         temp['unit'] = temp.index
        #         temp['cond'] = cond
        #         temp['boot'] = boot
        #
        #         temp = temp.melt(id_vars=['boot', 'cond', 'unit'], value_name='seg', var_name='m')
        #         df.loc[idx] = temp.set_index(idx)
        #
        #         cnt += 1

        df = df.astype({'m': float})

        return df

    def get_seg_rate_boot(self, bal_cond, segment_type='subseg', n_boot=100, occ_thr=1):

        """failed implementation. too slow to have everything on a pandas array :-/"""
        if not (bal_cond in self.bal_cond_sets.keys()):
            raise ValueError
        else:
            bal_cond_set = self.bal_cond_sets[bal_cond]

        seg_names = self.tmz.get_segment_type_names(segment_type)
        trial_seg = bal_cond_set['trial_seg']
        cond = bal_cond_set['cond']
        sub_conds = bal_cond_set['sub_conds']
        cond_set = {cond: sub_conds}

        trial_sets = self.get_trials_boot_cond_set(cond_set, n_boot=n_boot)[cond]
        all_zr = self.get_trial_segment_rates(segment_type=segment_type, trial_seg=trial_seg,
                                              occ_thr=occ_thr)

        n_trials = len(trial_sets)
        n_segs = len(seg_names)
        n_units = self.n_units
        n_rows = n_units * n_boot * n_trials * n_segs

        df = pd.DataFrame(np.nan, index=range(n_rows), columns=['cond', 'unit', 'boot', 'trial', 'seg', 'activity'])
        df['cond'] = cond

        units = np.arange(n_units)
        boot_block_len = n_units * n_trials * n_segs
        unit_block_len = n_trials * n_segs

        for boot in range(n_boot):
            boot_idx_start = boot_block_len * boot
            boot_idx = np.arange(boot_block_len) + boot_idx_start

            df.loc[boot_idx, 'boot'] = boot

            trials = trial_sets[:, boot]
            for unit in units:
                unit_block_idx = np.arange(unit_block_len) + unit * unit_block_len + boot_idx_start
                df.loc[unit_block_idx, 'unit'] = unit

                temp = all_zr[unit].loc[trials].copy()
                temp['trial'] = temp.index
                temp = temp.melt(id_vars='trial', value_name='activity', var_name='seg')
                temp = temp.set_index(unit_block_idx)

                print(temp.shape)
                df.loc[unit_block_idx, ['trial', 'seg', 'activity']] = temp

        return df

    # def zone_rate_maps_group_trials_boot_bal_corr(self, n_boot=100,
    #                                           corr_method='kendall', min_valid_trials=5, n_jobs=5,
    #                                           group_cond_sets=None, group_trial_segs=None, zr_method='trial',
    #                                           parallel=None):
    #     """
    #     this function computes balanced zone rate maps correlations between sets of trials.
    #     using bootstrap to equate the samples sizes between groups, size is the mean between the set sizes.
    #     the main component is the group_cond_sets parameter.
    #      it is a nested dictionary (3 levels):
    #         group names -> major condition -> balancing conditions
    #     See below for example.
    #     Returns
    #     :param n_boot: number of permutations
    #     :param corr_method: str, correlation method
    #     :param min_valid_trials: int, won't perform computation if a trial combinations is less than this #
    #     :param n_jobs: int, number of parallel workers for permutations
    #     :param group_cond_sets: dict, group name -> major condition - balancing conditions
    #             example:
    #             {'Even-Odd': {'Even':['CL','CR'],'Odd':['CL','CR']},
    #              'CR-CL': {'CR':['Co', 'InCo'], 'CL':['Co', 'InCo']} }
    #     :param group_trial_segs: dict indicating the trial segment for each subcondition in a group:
    #             example:
    #             {'CR-CL': ['out', 'out'], 'Even-Odd': ['out', 'out']}
    #     :param zr_method: str, see zone_rate_maps_corr for details
    #     :param parallel: parallel object to avoid multiple instantiations of workers.
    #
    #     :return:
    #     dict with group names as keys, and pandas data frame with n_units x n_perm as values
    #     """
    #
    #     if group_cond_sets is None:
    #         group_cond_sets = {'CR-CL': {'CR': ['Co', 'Inco'], 'CL': ['Co', 'Inco']},
    #                            'Even-Odd': {'Even': ['Co', 'Inco'], 'Odd': ['Co', 'Inco']}}
    #         group_cond_trial_segs = {'CR-CL': ['out', 'out'],
    #                             'Even-Odd': ['out', 'out']}
    #
    #     out_dict = {}
    #     for group, group_set in group_cond_sets.items():
    #         trial_segs = group_cond_trial_segs[group]
    #         try:
    #             trial_sets = self.get_trials_boot_cond_set(group_set, n_boot=n_boot)
    #         except ValueError:
    #             out_dict[group] = pd.DataFrame(np.zeros((self.n_units, n_boot)) * np.nan)
    #             continue
    #
    #         conds = list(group_set.keys())
    #
    #         ok_cond_trials_flag = True
    #         for cond in conds:
    #             if (trial_sets[cond].shape[0]) < min_valid_trials:
    #                 ok_cond_trials_flag = False
    #                 break
    #         if not ok_cond_trials_flag:
    #             # invalid set partition, skip
    #             out_dict[group] = pd.DataFrame(np.zeros((self.n_units, n_boot)) * np.nan)
    #             continue
    #
    #         def _worker(boot):
    #             return self.zone_rate_maps_corr(trials1=trial_sets[conds[0]][:, boot],
    #                                             trials2=trial_sets[conds[1]][:, boot],
    #                                             trial_seg1=trial_segs[0], trial_seg2=trial_segs[1],
    #                                             corr_method=corr_method, zr_method=zr_method)
    #
    #         try:
    #             if parallel is None:
    #                 with Parallel(n_jobs=n_jobs) as parallel:
    #                     corr = parallel(delayed(_worker)(boot) for boot in range(n_boot))
    #             else:
    #                 corr = parallel(delayed(_worker)(boot) for boot in range(n_boot))
    #
    #             out_dict[group] = pd.DataFrame(np.array(corr).T)
    #         except:
    #             out_dict[group] = pd.DataFrame(np.zeros((self.n_units, n_boot)) * np.nan)
    #
    #     return out_dict

    # def all_zone_rate_cond_diffs(self, n_boot=100, n_jobs=5):
    #     out_dict = {}
    #     for group, group_set in self.group_cond_sets.items():
    #         trial_segs = self.group_cond_trial_segs[group]
    #         try:
    #             trial_sets = self.get_trials_boot_cond_set(group_set, n_boot=n_boot)
    #         except ValueError:
    #             out_dict[group] = pd.DataFrame
    #             continue


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
