#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Just some useful functions that don't really belong to a class. 
Usually imported with alias 'sf'.

Created on Tue Jul  5 10:38:45 2022
Rev: 2022 08 20

@author: zackashm
"""

# Path set up
from pathlib import Path
HERE = Path(__file__).parent.absolute()

# for quick importing reasons
import os
os.chdir(HERE)

# normal imports
import numpy as np
import re

# -- Table of Contents --
# - re_search()
# - restict_to_band()
# - make_plotly_polar_look_nice()


# -- Functions --

# since i like glob but it doesn't use regex by itself
def re_search(regex, allfiles):
    '''
    Search with regex through a list, allfiles, and return a list of elements
    matching the regex pattern.
    
    Parameters
    ----------
    regex : string, regular expression
        Regex pattern to be matched in the search.
    allfiles : list
        A list of filenames to search through.

    Returns
    -------
    Generator
        Generator of filenames which match the regex pattern.

    '''

    # generate
    new_list = np.asarray([file for file in allfiles if re.search(regex, str(file))])
    new_list.sort()
        
    #return
    return new_list

# will restrict everything to the PUEO band to prevent interpolating outside of range
def restrict_to_band(bandGHz, fHz, *args):
    '''
    Truncate data arrays to a given frequency band. This function assumes that
    the arrays in args are sorted by the given fHz frequency array.
    
    Parameters
    ----------
    bandGHz : tuple(2)
        The band (min,max) in GHz to restrict the given arrays to.
    fHz : np.ndarray
        Frequency array in Hz used to determine the truncating mask with bandGHz.
    *args : np.ndarray
        Arrays to truncate matching fHz.

    Returns
    -------
    return_arrays
        The given fHz and arrays in args truncated to the bandGHz limits.
    '''
    
    # extract the min and max
    fGHzmin, fGHzmax = bandGHz
    
    # set the mask
    within_band = np.logical_and(fGHzmin*1e9 <= fHz,fHz <= fGHzmax*1e9)
    
    # make the list of truncated arrays
    return_arrays = [fHz[within_band]] + [arr[within_band] for arr in args]
    
    return return_arrays

# Since there's so much just to make the plot look nice, I separated it into this function
# If you haven't seen plotly before, prepare yourself for many dictionaries (though imo it's
# easier to customize plot settings this way)
def make_plotly_polar_look_nice(fig, rlim=(None,None), rlabel='', alltitle='', kwargs_for_trace={}, **kwargs):
    '''
    Makes plotly's polar plot look nice for -90 to 90 deg antenna data.

    Parameters
    ----------
    fig : plotly.Figure
        The figure to make look nice.
    rlim : tuple(2)
        Radial axis (min,max) limits.
    rlabel : str
        The radial axis label. The default is ''.
    alltitle : str
        The overall title. The default is ''.
    kwargs_for_trace : dict
        Kwargs specifically for plotly.Figure.update_traces(). I planned to use this
        if I wanted multiple data set ups plotted and wanted to distinguish them by
        changing the traces to a certain style.
    **kwargs
        plotly.Figure.update_layout() kwargs. 

    Returns
    -------
    None.

    '''
    
    # kwarg defaults
    polar = kwargs.pop('polar', 
                       dict(                                # polar setting
                           sector = [0,180],                # set chart shape (half)
                           angularaxis = dict(              # angle axis settings
                               dtick = 10,                  # angle tick increment
                               rotation = 90,               # rotates data to be on upper half
                               direction = "clockwise",     # -90 to 90 left to right
                               tickfont=dict(size=20),      # angle tick size
                               ),
                           radialaxis=dict(                 # radial axis settings
                               exponentformat="E",          # tick label in exponential form
                               title=dict(                  # radial axis label settings
                                   text=rlabel,             # the label text
                                   font=dict(               # label font settings
                                       size = 23,           # label text size
                                       ),
                                   ),
                               tickfont=dict(size=20),      # radial tick size
                               range=rlim,                  # radial limits
                                )
                           ),
                       )
    
    autosize = kwargs.pop('autosize', False)                # we'll custom size the figure
    
    width = kwargs.pop('width', 1100)                       # canvas width
    
    height = kwargs.pop('height', 620)                      # canvas height
    
    legend = kwargs.pop('legend',                           # legend settings
                        dict(                        
                             bgcolor="#d8e4f5",             # legend bg color
                             xanchor="right",               # reference for x pos
                             x=1.05,                        # legend x pos percentage
                             yanchor="top",                 # reference for x pos
                             y=0.95,                        # legend y pos percentage
                             font=dict(size=20),            # font size
                             itemsizing='constant',         # symbol size to not depend on data traces
                             itemwidth=30,                  # symbol size
                             )
                        )
    
    margin = kwargs.pop('margin',                           # margins
                        dict(                               
                            b=0,
                            t=0,
                            l=70,
                            r=60,
                            ),
                        )
    
    title = kwargs.pop('title',                             # title settings
                       dict(                                
                           text=alltitle,                   # title text
                           x = 0.03,                        # title position
                           y = 0.98,
                           font=dict(
                               size = 28,                   # title font size
                               ),
                           ),
                       )
    
    # update plotly figure
    fig.update_layout(
            polar=polar,
            autosize=autosize,
            width=width,
            height=height,
            legend=legend,
            margin=margin,
            title=title,
            **kwargs,       # other user specific kwargs
            )
    
    # for the traces themselves
    fig.update_traces(**kwargs_for_trace)


# plot multiple antenna beampattern

# plot impulse response (TODO: Implement this into horn_response class)
# def impulse_response(pulse_dict, signal_dict, antenna_dist_ft=34, ax=None, labels=None, **kwargs):
    
#     # get the FFT of the pulse
#     for p_file in pulse_dict:
#         pulse_df = pulse_dict[p_file].copy()
#         # pdB, p_fftfreq, p_fft = calc_fft(pulse_df)
#         pvdata = pulse_df[pulse_df.columns[1]].values
#         p_fft = fft(pvdata)[1:]
        
#         # get fft freqs
#         ptdata = pulse_df[pulse_df.columns[0]].values
#         ptsize = ptdata.size
#         ptstep = ptdata[1] - ptdata[0]
#         p_fftfreq = fftfreq(ptsize, ptstep)[1:]*1e-9
        
#         # time domain for future use
#         tdata = np.array([i*ptstep for i in range(0,ptsize)])*1e9

#     # get signal FFTs
#     s_ffts = []
#     for s_file in signal_dict:
#         sig_df = signal_dict[s_file].copy()
#         # sdB, s_fftfreq, s_fft_ = calc_fft(sig_df)
#         sig_vdata = sig_df[sig_df.columns[1]].values
#         s_fft_ = fft(sig_vdata)[1:]
#         s_ffts.append(s_fft_)
        
#         # s_fftfreq and p_fftfreq should be the same if the datalength and samplerate are the same
#         stdata = sig_df[sig_df.columns[0]].values
#         stsize = stdata.size
#         ststep = stdata[1] - stdata[0]
#         s_fftfreq = fftfreq(stsize, ststep)[1:]*1e-9
#         dfftfreq = np.abs(p_fftfreq - s_fftfreq)
#         if np.any(dfftfreq > 0.01):
#             warn("WARNING: in impulse_response(), corresponding FFT frequencies between signal and pulse do not match. Are the lengths and samplerate different?")
    
#     # calc consts
#     fGHz = p_fftfreq
#     antenna_dist_m = 0.3048 * antenna_dist_ft
#     c_mns = c * 1e-9
    
#     # get impulse response for each signal
#     IRffts = []
#     IRs = []
#     for s_fft in s_ffts:
        
#         # calculate squared IR
#         # IRfft2 = (antenna_dist_m * c_mns * s_fft) / (1j * fGHz * p_fft)
#         # use wiener deconvolution
#         snr = 1e-3
#         # PROBLEM: snr arbitrarily chosen changes the amplitude, so the amplitude
#         # is not correctly determined
        
#         # snr as function of f (lower snr at out of band higher freq)
#         # snr = np.ones(len(fGHz))
#         # outofband = np.logical_or(fGHz < 0.2, fGHz > 1.3)
#         # inband = np.logical_and(0.2 <= fGHz, fGHz <= 1.3)
#         # snr[outofband] = 1e-3
#         # snr[inband] = 1e-3
#         # print(snr)
        
#         p_fft_eff = (1j * fGHz * p_fft) / (antenna_dist_m * c_mns)
#         g_fft = (1 / p_fft_eff) * (1 / (1 + 1 / (np.abs(p_fft_eff)**2 * snr)))
        
#         IRfft2 = g_fft * s_fft
        
#         # separate mag and phase to explicitly unwrap phase
#         fft2mag = np.abs(IRfft2)
#         fft2phi = np.angle(IRfft2)
#         fft2phiunwrapped = np.unwrap(fft2phi)
        
#         # take sqrt
#         IRfft = np.sqrt(fft2mag)*np.exp(0.5*fft2phiunwrapped*1j)
#         IRffts.append(IRfft)
        
#         # get impulse response in time domain
#         IR_ = np.real(ifft(IRfft, n=tdata.size))
#         IRs.append(np.roll(IR_, int(0.2*len(IR_))))
    
#     # plot
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(30,15))
    
#     title = lambda x : ' '.join(re.split(r'[_.]',x.name)[:4])
    
#     i = 0
#     for IR in IRs:
#         if labels is None:
#             label = title(list(signal_dict.keys())[i])
#         else:
#             label = labels[i]
#         ax.plot(tdata, IR, label=label, **kwargs)
#         i+=1
    
#     ax.legend(loc='best')
#     ax.set(xlabel='time [ns]', ylabel='response [m/ns]')
#     for line in ax.legend().get_lines(): 
#         line.set_linewidth(15)
    
#     return (IRs, tdata, IRffts, p_fftfreq)
        