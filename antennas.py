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


# Transmitters
__UCLA10m = np.genfromtxt(DATA_DIR / "uclahorn_gain_10m.csv", delimiter=',', unpack=True, usecols=[0,1]) # f [MHz], gain [dB]
# __UCLA1m = np.genfromtxt(DATA_DIR / "uclahorn_gain_1m.csv", delimiter=',', unpack=True, usecols=[0,1]) # f [MHz], gain [dB]
# __UCLA = np.genfromtxt(DATA_DIR / "RGainvFreq-UCLAHorn.txt", delimiter='\t', unpack=True) # f [GHz], gain [dB]

# __CHIC1mH = np.genfromtxt(DATA_DIR / "UChicagoHorn/UChicagoHorn_gain_1m_H.csv", delimiter=',', unpack=True, usecols=[0,1]) # f [GHz], gain [dBi]
__CHIC10mH = np.genfromtxt(DATA_DIR / "UChicagoHorn/UChicagoHorn_gain_10m_H.csv", delimiter=',', unpack=True, usecols=[0,1]) # f [GHz], gain [dBi]
# __CHIC1mV = np.genfromtxt(DATA_DIR / "UChicagoHorn/UChicagoHorn_gain_1m_V.csv", delimiter=',', unpack=True, usecols=[0,1]) # f [GHz], gain [dBi]
# __CHIC10mV = np.genfromtxt(DATA_DIR / "UChicagoHorn/UChicagoHorn_gain_10m_V.csv", delimiter=',', unpack=True, usecols=[0,1]) # f [GHz], gain [dBi]


# Prototypes
__TOYON = np.genfromtxt(DATA_DIR / "Toyon_digitized.txt", delimiter=',', unpack=True)   # f [GHz], gain [dB]
# __RFSPIN = np.genfromtxt(DATA_DIR / "RFSpin_digitized.txt", delimiter=',', unpack=True) # f [GHz], gain [dB]

# Production (f [GHz], gain [dBi])
__SN001A = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN001_A_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN001B = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN001_B_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN002A = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN002_A_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN002B = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN002_B_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN003A = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN003_A_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN003B = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN003_B_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN004A = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN004_A_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN004B = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN004_B_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN005A = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN005_A_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN005B = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN005_B_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN006A = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN006_A_gain.csv", delimiter=',', unpack=True, usecols=[0,1])
__SN006B = np.genfromtxt(DATA_DIR / "ProductionCompanyMeasurements/SN006_B_gain.csv", delimiter=',', unpack=True, usecols=[0,1])

# allows access to the desired file via the filename labeling / catalog
gain = {
    'UCLA':interp1d(__UCLA10m[0] * 1e6, __UCLA10m[1], assume_sorted=True, bounds_error=True),
    'UCHI':interp1d(__CHIC10mH[0] * 1e9, __CHIC10mH[1], assume_sorted=True, bounds_error=True),
    'PSN001A':interp1d(__TOYON[0] * 1e9, __TOYON[1], assume_sorted=True, bounds_error=True),
    # 'R':interp1d(__RFSPIN[0] * 1e9, __RFSPIN[1], assume_sorted=True, bounds_error=True),
    'SN001A':interp1d(__SN001A[0] * 1e9, __SN001A[1], assume_sorted=True, bounds_error=True),
    'SN001B':interp1d(__SN001B[0] * 1e9, __SN001B[1], assume_sorted=True, bounds_error=True),
    'SN002A':interp1d(__SN002A[0] * 1e9, __SN002A[1], assume_sorted=True, bounds_error=True),
    'SN002B':interp1d(__SN002B[0] * 1e9, __SN002B[1], assume_sorted=True, bounds_error=True),
    'SN003A':interp1d(__SN003A[0] * 1e9, __SN003A[1], assume_sorted=True, bounds_error=True),
    'SN003B':interp1d(__SN003B[0] * 1e9, __SN003B[1], assume_sorted=True, bounds_error=True),
    'SN004A':interp1d(__SN004A[0] * 1e9, __SN004A[1], assume_sorted=True, bounds_error=True),
    'SN004B':interp1d(__SN004B[0] * 1e9, __SN004B[1], assume_sorted=True, bounds_error=True),
    'SN005A':interp1d(__SN005A[0] * 1e9, __SN005A[1], assume_sorted=True, bounds_error=True),
    'SN005B':interp1d(__SN005B[0] * 1e9, __SN005B[1], assume_sorted=True, bounds_error=True),
    'SN006A':interp1d(__SN006A[0] * 1e9, __SN006A[1], assume_sorted=True, bounds_error=True),
    'SN006B':interp1d(__SN006B[0] * 1e9, __SN006B[1], assume_sorted=True, bounds_error=True),
    }

#EXAMPLE
# foo = waveform("UCLA_TO_R2B_...")
# Tx_label = foo.Tx[0]
# ucla_gain = antennas.gain[ Tx_label ]
# Rx_label = foo.Rx[0]
# rfspin_gain = antennas.gain[ Rx_label ]
# ucla_plot_label = antennas.catalog[ Tx_label ]
# rfspin_plot_label = antennas.catalog[ Rx_label ]