#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides the predicted boresight gains of the relevant horn antennas.

Created on Sat Aug 20 04:10:41 2022

@author: zackashm
"""

# Path set up
from pathlib import Path
HERE = Path(__file__).parent.absolute()

# go to data dir
import os
os.chdir(HERE)
DATA_DIR = HERE / "Data"

import numpy as np
from scipy.interpolate import interp1d


# Boresight gains of UCLA, Toyon, and RFSpin, used in gain calculation
__UCLA = np.genfromtxt(DATA_DIR / "uclahorn_gain_10m.csv", delimiter=',', unpack=True, usecols=[0,1]) # f [MHz], gain [dB]
# __UCLA = np.genfromtxt(DATA_DIR / "RGainvFreq-UCLAHorn.txt", delimiter='\t', unpack=True) # f [GHz], gain [dB]
__TOYON = np.genfromtxt(DATA_DIR / "Toyon_digitized.txt", delimiter=',', unpack=True)   # f [GHz], gain [dB]
__RFSPIN = np.genfromtxt(DATA_DIR / "RFSpin_digitized.txt", delimiter=',', unpack=True) # f [GHz], gain [dB]

# waveform filenames will somewhat follow these shorthands
catalog = {
    'U':'UCLA',
    'T':'TOYON',
    'R':'RFSPIN',
    }

# allows access to the desired file via the filename labeling / catalog
gain = {
    'U':interp1d(__UCLA[0] * 1e6, __UCLA[1], assume_sorted=True, bounds_error=True),
    # 'U':interp1d(__UCLA[0] * 1e9, __UCLA[1], assume_sorted=True, bounds_error=True),
    'T':interp1d(__TOYON[0] * 1e9, __TOYON[1], assume_sorted=True, bounds_error=True),
    'R':interp1d(__RFSPIN[0] * 1e9, __RFSPIN[1], assume_sorted=True, bounds_error=True),
    }


#EXAMPLE
# foo = waveform("UCLA_TO_R2B_...")
# Tx_label = foo.Tx[0]
# ucla_gain = antennas.gain[ Tx_label ]
# Rx_label = foo.Rx[0]
# rfspin_gain = antennas.gain[ Rx_label ]
# ucla_plot_label = antennas.catalog[ Tx_label ]
# rfspin_plot_label = antennas.catalog[ Rx_label ]