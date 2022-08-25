#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses waveform.py and antenna_response.py to plot horn measurement data.

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
from antenna_response import antenna_response as aresp

import numpy as np

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
sns.set_theme(font_scale=5, rc={'font.family':'Helvetica', 'axes.facecolor':'#e5ebf6',
                                'legend.facecolor':'#dce3f4', 'legend.edgecolor':'#000000',
                                'axes.prop_cycle':cycler('color', 
                                                         ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', 
                                                          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', 
                                                          '#FF97FF', '#FECB52']),
                                'axes.xmargin':0, 'axes.ymargin':0.05})
import plotly.io as pio
pio.renderers.default='png'     # for displaying on noninteractive programs


# ---------------------------------
# -- FILE PARSING PARAMETERS
# ---------------------------------

# data path set up
DATA_DATE = "20220819"
PULSER_DATE = DATA_DATE #"20220816"
DATA_DIR = HERE / 'Data/{}/'.format(DATA_DATE)
PULSER_DIR = HERE / 'Data/PULSER/{}/'.format(PULSER_DATE)
ALLDATA = np.array([file for file in DATA_DIR.glob('*.csv')] + [file for file in PULSER_DIR.glob('*.csv')])  # list of every data file name
# print("All data paths found: \n", ALLDATA)

# -- Now parse through the files to get the ones you want.
# Here's one way

# regex parsers (.* just means anything)
PULSER = "PULSE.*R2B.*Ch1"
Tx = "U.*"
Rx = "R2B.*"
POL = "VPOL"
DESC = "E"
ANGLE = "([A-Z]*)[0-9]+" #"([A-Z]*)[0,30,60,90]"
TRIAL = "01"
TYPE = ".*Ch1"

# put it all together
if DESC == "":
    PARSER = "_".join([Tx,Rx,POL,ANGLE,TRIAL,TYPE])
else:
    PARSER = "_".join([Tx,Rx,POL,DESC,ANGLE,TRIAL,TYPE])

# use some_functions.re_search to regex find the files
# these are both an array of Paths, which you input into antenna_response(), or feed each one into waveform()
PULSE_PATH = sf.re_search(PULSER, ALLDATA)
SIGNAL_PATHS = sf.re_search(PARSER, ALLDATA)

# make sure you know what files are going to be used, or if any were found at all
# print("Pulser path(s) found: \n", PULSE_PATH)
# print("Signal path(s) found: \n", SIGNAL_PATHS)

# ---------------------------------------------------------

# set to true whatever you want to get
TESTING = 0
WAVEFORMS = 1
BORESIGHT = 1
BEAMPATTERN = 1
IMPULSE_RESP = 0

# TESTING
# this is messy, not meant to be best practice
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
    pul_wf = waveform(PULSE_PATH[0])
    sig_wfs = [waveform(sp) for sp in SIGNAL_PATHS]
    
    # truncate, using best estimated window here
    pul_wf.truncate(pul_wf.estimate_window())
    for wf in sig_wfs:
        wf.truncate(wf.estimate_window())

    # plot
    pul_wf.plot(tscale=1e9, vscale=1e3, title=pul_wf.label)
    
    # separately plot signals or altogether in common Axes
    ax1 = plt.subplots(figsize=[30,15])[1]   # together
    for wf in sig_wfs:
        # wf.plot(tscale=1e9, vscale=1e3, title=wf.label)   # separate
        wf.plot(ax=ax1, tscale=1e9, vscale=1e3, title="All Signals")    # together
    
    
    # similar for plotting FFTs
    
    # pul_wf.plot_fft(fscale=1e-9, flim=(0,2), dBlim=(-15,10), title=pul_wf.label, legend=True) # just the pulse
    
    ax2 = plt.subplots(figsize=[30,15])[1]   # together
    pul_wf.plot_fft(ax=ax2, fscale=1e-9, flim=(0,2), label=pul_wf.label, legend=True)       # for together plot
    for wf in sig_wfs:
        wf.plot_fft(ax=ax2, fscale=1e-9, flim=(0,2), dBlim=(-75,15), title="FFT Comparison", show_PUEO_band=False)  # together
        
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
    horn_data = aresp(PULSE_PATH, SIGNAL_PATHS)
    
    # give the correct distances
    horn_data.front_dist_m = 8
    horn_data.back_dist_m = 10
    
    # plot
    horn_data.plot_boresight()

# ---------------------------------------------------------


# plot beampattern
# ----------------
if BEAMPATTERN:
    
    print("BEAMPATTERN")
    
    # call antenna_response
    horn_data = aresp(PULSE_PATH, SIGNAL_PATHS)
    
    # plot
    fGHz = np.arange(0.3,1.21,0.2)
    horn_data.plot_beampattern(fGHz=fGHz, polar=True)
    

# ---------------------------------------------------------


# get impulse response NOT WORKING
# ---------------------
if IMPULSE_RESP:
    
    print("IMPULSE_RESP")
    
    print("Not working yet...")

# ---------------------------------------------------------
