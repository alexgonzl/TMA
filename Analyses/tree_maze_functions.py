import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal

import Analyses.spatial_functions as spatial_funcs
import Utils.filter_functions as filt_funcs
import Pre_Processing.pre_process_functions as pp_funcs

import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


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
    zones_coords = {'Home': [(-300, -80), (-300, 80), (300, 80), (300, -80)],
                    'Center': [(-80, 500), (-95, 400), (-150, 400), (-150, 652),
                               (-75, 550), (0, 607), (75, 550), (150, 652), (150, 400),
                               (95, 400), (80, 500)],
                    'A': [(-150, 80), (-80, 500), (80, 500), (150, 80)],
                    'B': [(0, 607), (0, 700), (200, 1000), (329, 900), (75, 550)],
                    'C': [(610, 1180), (610, 800), (329, 900), (450, 1180)],
                    'D': [(200, 1000), (50, 1230), (450, 1230), (450, 1180)],
                    'E': [(0, 607), (0, 700), (-200, 1000), (-329, 900), (-75, 550)],
                    'F': [(-200, 1000), (-50, 1230), (-450, 1230), (-450, 1180)],
                    'G': [(-610, 1180), (-610, 800), (-329, 900), (-450, 1180)],

                    'G1': [(610, 1180), (800, 1180), (800, 800), (610, 800)],
                    'G2': [(50, 1230), (50, 1450), (450, 1450), (450, 1230)],
                    'G3': [(-50, 1230), (-50, 1450), (-450, 1450), (-450, 1230)],
                    'G4': [(-610, 1180), (-800, 1180), (-800, 800), (-610, 800)],

                    'I1': [(200, 1000), (450, 1180), (329, 900)],
                    'I2': [(-329, 900), (-450, 1180), (-200, 1000)],
                    }

    zone_names = list(zones_coords.keys())
    zone_label_coords = {'Home': (0, 0),
                         'Center': (155, 500),
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
                         'I1': (300, 1000),
                         'I2': (-300, 1000),
                         }

    zones_geom = {}
    for zo in zone_names:
        zones_geom[zo] = Polygon(zones_geom[zo])

    out_dirs = {'Home': (0, 0),
                'Center': (155, 500),
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
                'I1': (300, 1000),
                'I2': (-300, 1000),
                }
    def __init__(self):
        self._dir_num_dict = {a: b for a, b in zip(['E', 'N', 'W', 'S'], range(1, 5))}

    def divide_seg(self, seg_name, subseg_length, direction='N'):
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

        l_dir_id = np.where(l_segs_dirs == self._dir_num_dict[direction])[0][0]

        L = l_segs[l_dir_id]
        L = Points2D(L[0], L[1])

        a = pp.xy[:4][l_dir_id - 1]
        a = Points2D(a[0], a[1])
        b = pp.xy[:4][l_dir_id]
        b = Points2D(b[0], b[1])

        n_subsegs = int(L.r // subseg_length)

        delta = L.r / n_subsegs
        subsegs = np.zeros(n_subsegs, dtype=object)

        for ii in range(n_subsegs):
            p0 = Points2D(ii * delta, L.ang, polar=True) + a
            p1 = Points2D((ii + 1) * delta, L.ang, polar=True) + a
            p2 = Points2D((ii + 1) * delta, L.ang, polar=True) + b
            p3 = Points2D(ii * delta, L.ang, polar=True) + b

            subseg = Polygon([p0.xy[0],
                              p1.xy[0],
                              p2.xy[0],
                              p3.xy[0]])

            subsegs[ii] = seg_geom.intersection(subseg)
        return subsegs
