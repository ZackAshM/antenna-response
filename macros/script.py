#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses waveform.py, antennaData.py and antenna_response.py to plot horn measurement data.

Created on Fri Aug 19 07:51:41 2022

@author: zackashm
"""

# Path set up
from pathlib import Path
HERE = Path(__file__).parent.absolute()

# for quick importing reasons
import os
os.chdir(HERE)

# now import safely
import some_functions as sf
from waveform import waveform
from antennaData import antennaData
from antenna_response import antenna_response as aresp
# import antennas

import numpy as np
# from scipy.interpolate import interp1d
# from scipy.constants import c
# import pandas as pd

# plotting imports
import matplotlib.pyplot as plt
# from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from cycler import cycler
sns.set_theme(font_scale=5, rc={'font.family':'Helvetica', 'axes.facecolor':'#e5ebf6',
                                'legend.facecolor':'#dce3f4', 'legend.edgecolor':'#000000',
                                'axes.prop_cycle':cycler('color', 
                                                         ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', 
                                                          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', 
                                                          '#FF97FF', '#FECB52']),
                                'axes.xmargin':0, 'axes.ymargin':0.05})
COLORS=['#636EFA', '#EF553B', '#00CC96', '#AB63FA',
        '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', 
        '#FF97FF', '#FECB52']
# import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='png'     # for displaying on noninteractive programs



# from numpy.fft import fft, ifft, rfft, irfft, ifftshift, rfftfreq

# ---------------------------------
# -- FILE PARSING PARAMETERS
# ---------------------------------

# data path set up
DATA_DATE = "20230712"
PULSER_DATE = DATA_DATE #"20220816"
DATA_DIR = HERE.parent / 'Data/{}/'.format(DATA_DATE)
PULSER_DIR = HERE.parent / 'Data/PULSER/{}/'.format(PULSER_DATE)
ALLDATA = np.array([file for file in DATA_DIR.glob('*.csv')] + [file for file in PULSER_DIR.glob('*.csv')])  # list of every data file name
# print("All data paths found: \n", *[f.name for f in ALLDATA], sep='\n')

# -- Now parse through the files to get the ones you want.
# Here's one way
# regex parsers (.* just means anything)
PULSER = "avtech.*Ch1.*"
Tx = "UCHI.*"
Rx = "SN004.*"
PORT = "A"
PLANE = "H"
ANGLE = "([A-Z]*)[0-9]+"#"([A-Z]*)[30]"#",30]" #"0" 
TYPE = ".*Ch1"

# put it all together
PARSER = "_".join([Tx,Rx,PORT,PLANE,ANGLE,TYPE])

# use some_functions.re_search to regex find the files
# these are both an array of Paths, which you input into antenna_response(), or feed each one into waveform()
PULSE_PATH = sf.re_search(PULSER, ALLDATA)
SIGNAL_PATHS = sf.re_search(PARSER, ALLDATA)
DISTANCE = 5.03

# make sure you know what files are going to be used, or if any were found at all
print("Pulser path(s) found: \n", PULSE_PATH)
print("Signal path(s) found: \n", SIGNAL_PATHS)

# ---------------------------------------------------------

# set to true whatever you want to get
TESTING = 0
WAVEFORMS = 0
BORESIGHT = 0
BEAMPATTERN = 1
IMPULSE_RESP = 0

# TESTING
# this is messy, not meant to be best practice. Use this like a scratch pad to test the script
if TESTING:
    
    print("TESTING") 

# ------------------------------------
# ---- MAIN --------------------------
# ------------------------------------


# plot waveforms
# --------------
if WAVEFORMS:
    
    print("WAVEFORMS")
    
    # call waveforms
    pul_wf = antennaData(PULSE_PATH[0])
    sig_wfs = [antennaData(sp) for sp in SIGNAL_PATHS]
    
    # truncate, using best estimated window here
    pul_wf.truncate(pul_wf.estimate_window())
    for wf in sig_wfs:
        wf.truncate(wf.estimate_window())

    # plot
    pul_wf.plot(tscale=1e9, vscale=1e3, title=pul_wf.label)
    
    # separately plot signals or altogether in common Axes
    ax1 = plt.subplots(figsize=[30,20])[1]   # together
    for wf in sig_wfs:
        # wf.plot(tscale=1e9, vscale=1e3, title=wf.label)   # separate
        wf.plot(ax=ax1, tscale=1e9, vscale=1e3,label=wf.label)#, title="All Signals")    # together
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set(title="Port "+wf.port+" Plane "+wf.plane)
    # plt.savefig("_".join(["waveform",wf.port,wf.plane]),bbox_inches='tight')
    
    
    # similar for plotting FFTs
    
    pul_wf.plot_fft(fscale=1e-9, flim=(0,2), dBlim=(-15,10), title=pul_wf.label, legend=True) # just the pulse
    
    ax2 = plt.subplots(figsize=[30,20])[1]   # together
    pul_wf.plot_fft(ax=ax2, fscale=1e-9, flim=(0,2), label=pul_wf.label, legend=True)       # for together plot
    for wf in sig_wfs:
        wf.plot_fft(ax=ax2, fscale=1e-9, flim=(0,2), dBlim=(-75,15), title="FFT Comparison", show_PUEO_band=False, label=wf.label)  # together
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set(title="Port "+wf.port+" Plane "+wf.plane)
    # plt.savefig("_".join(["fft",wf.port,wf.plane]),bbox_inches='tight')
    
    # or separately
    # ax3 = plt.subplots(figsize=[30,15])[1]
    # pul_wf.plot_fft(ax=ax3, fscale=1e-9, flim=(0,2), label=pul_wf.label, legend=True)
    # wf.plot_fft(ax=ax3, fscale=1e-9, flim=(0,2), dBlim=(-75,15), title=wf.label)

# ---------------------------------------------------------


# plot boresight gain
# -------------------
if BORESIGHT:
    
    print("BORESIGHT")
    
    # call antenna_response
    horn_data = aresp(PULSE_PATH, SIGNAL_PATHS, DISTANCE)
    
    # plot
    horn_data.plot_gain()

# ---------------------------------------------------------


# plot beampattern
# ----------------
if BEAMPATTERN:
    
    print("BEAMPATTERN")
    
    # call antenna_response
    horn_data = aresp(PULSE_PATH, SIGNAL_PATHS, DISTANCE)
    
    # plot
    fGHz = np.append(np.arange(0.3,1.20,0.1), 1.2)
    horn_data.plot_beampattern(fGHz=fGHz, polar=True,dBlim=[-20,20])
    

# ---------------------------------------------------------


# get impulse response NOT WORKING
# ---------------------
if IMPULSE_RESP:
    
    print("IMPULSE_RESP")
    
    print("Not working yet...")

# ---------------------------------------------------------
