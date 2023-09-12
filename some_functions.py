#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Just some useful functions that don't really belong to a class. 
Usually imported with alias 'sf'.

Created on Tue Jul  5 10:38:45 2022
Rev: 2022 08 20

Rev: 2023 07 26
Removed class specific functions, and placed them into the classes they were used
for. Eg. tukey_filter() is now in waveform(), and restrict_band() and 
make_plotly_polar_look_nice() are in antenna_response().
This file is still included in case non-class specific functions in the future are
needed.

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
        