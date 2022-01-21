#!/usr/bin/env python
"""Spectral libraries for PROSPECT + SAIL
"""
import pkgutil
from io import StringIO
from collections import namedtuple

import csv
import os
from os import path
import numpy as np
import time

Spectra = namedtuple('Spectra', 'prospect5 prospectd soil light')
Prospect5Spectra = namedtuple('Prospect5Spectra',
                                'nr kab kcar kbrown kw km')
ProspectDSpectra = namedtuple('ProspectDSpectra',
                                'nr kab kcar kbrown kw km kant')
SoilSpectra = namedtuple("SoilSpectra", "rsoil1 rsoil2")
LightSpectra = namedtuple("LightSpectra", "es ed")


def get_spectra():
    """Reads the spectral information and stores is for future use."""

    # PROSPECT-D
    prospect_d_spectraf = pkgutil.get_data('src', 'prospect_d_spectra.txt')

    start = time.time()

    rows = []
    with open(path.abspath('./prospect_d_spectra.txt')) as file:
        r = csv.reader(file, delimiter='\t')
        for i,row in enumerate(r):
            if i < 20:
                continue
            # if i > 40:
            #     break
            # print(f'{i}: {row}')
            rows.append(row)

    # A = np.array(rows)
    wls =    np.array([int(x[0]) for x in rows])
    nr =     np.array([float(x[1]) for x in rows])
    kab =    np.array([float(x[2]) for x in rows])
    kcar =   np.array([float(x[3]) for x in rows])
    kant =   np.array([float(x[4]) for x in rows])
    kbrown = np.array([float(x[5]) for x in rows])
    kw =     np.array([float(x[6]) for x in rows])
    km =     np.array([float(x[7]) for x in rows])
    # r_list = np.array([r for _, r in sorted(zip(wls, result_dict[C.key_sample_result_r]))])

    # _, nr, kab, kcar, kant, kbrown, kw, km = np.loadtxt(StringIO(prospect_d_spectraf), unpack=True)
    prospect_d_spectra = ProspectDSpectra(nr, kab, kcar, kbrown, kw, km, kant)

    end = time.time()
    print(f'Reading stuff took: {end-start} s')

    # # PROSPECT 5
    # prospect_5_spectraf = pkgutil.get_data('prosail', 'prospect5_spectra.txt')
    # nr, kab, kcar, kbrown, kw, km =  np.loadtxt(StringIO(prospect_5_spectraf),
    #                                             unpack=True)
    # prospect_5_spectra = Prospect5Spectra(nr, kab, kcar, kbrown, kw, km)
    # # SOIL
    # soil_spectraf = pkgutil.get_data('prosail', 'soil_reflectance.txt')
    # rsoil1, rsoil2 =  np.loadtxt(StringIO(soil_spectraf),
    #                                             unpack=True)
    # soil_spectra = SoilSpectra(rsoil1, rsoil2)
    # # LIGHT
    # light_spectraf = pkgutil.get_data('prosail', 'light_spectra.txt')
    # es, ed =  np.loadtxt(StringIO(light_spectraf),
    #                                             unpack=True)
    # light_spectra = LightSpectra(es, ed)
    # spectra = Spectra(prospect_5_spectra, prospect_d_spectra,
    #                   soil_spectra, light_spectra)
    # spectra = Spectra(prospect_d_spectra)
    return prospect_d_spectra
