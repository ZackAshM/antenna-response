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
    
        