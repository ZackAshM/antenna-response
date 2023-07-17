#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class inherits the waveform class to be more specific to the antenna
response data, including filename parsing for specifying the measurement set ups.

Created on Sun May 14 20:42:22 2023

@author: zackashm
"""

# Path set up
from pathlib import Path
HERE = Path(__file__).parent.absolute()

# for quick importing reasons
import os
os.chdir(HERE)

from waveform import waveform

class antennaData(waveform):
    '''
    This class handles the output data from swept gain antenna measurements. Its
    main purpose is to use parse the data file names to identify the set up for
    the given measurements and organize them in plots or elsewhere.
    
    Parameters
    ----------
    waveform.path : str or pathlib.Path
        The file path containing the csv data.
    parse_name : bool [default=True]
        Set attributes based on the filename convention:
        {transmitter}_{receiver}_{port}_{plane}_{angle}_{scope channel}.csv
    date : str [default=None]
        The date of the data. If None, it is assumed to be the parent directory
        of the data directory.
    
    Attribute
    ---------
    date : str
        The str of the data date (data parent directory is assumed to be the 
        date for default).
    pulse : bool
        True if the data is pulser data, with "PULSE(R)" in the file name.
    Tx : str
        File name part referring to the transmitting antenna.
    Rx : str
        File name part referring to the receiving antenna.
    port : str
        File name part referring to the Rx port.
    plane : str
        File name part referring to the swept plane (E or H).
    angle_str : str
        File name part referring to the angle in string format.
    angle : int
        File name part referring to the angle in int format.
    channel : str
        File name part referring to the associated channel on the oscilloscope.
    label : str
        An appropriate label for the data based on the parsed attributes.
        
    Inherited Attributes
    --------------------
    path : pathlib.Path
        The full path of the file.
    name : str
        The data file name.
    vcol : str
        The voltage column label.
    tcol : str
        The time column label.
    data : pandas.Dataframe
        A Dataframe of the data with column names matching tcol and vcol.
    filter : numpy.ndarray
        If filtered using tukey_filter, this is the array of the filter array
        in the data time domain.
    vdata : numpy.ndarray
        The voltage data array in volts.
    tdata : numpy.ndarray
        The time data array in seconds.
    datasize : np.int64
        The size of the data, assumed to be based on vdata.
    samplerate : np.float64
        The samplerate of the data, determined by the mean of the stepsizes in tdata.
        
    Inherited Methods
    -----------------
    truncate
        Truncates the time and voltage data to a given time window.
    estimate_window
        Returns a time domain window based on the voltage peak, with the goal
        to encapsulate the whole pulse. This will usually be fed into self.truncate.
    tukey_filter
        Filter the voltage data based on a Tukey filter in a given time window.
    calc_fft -> tuple[np.ndarray,np.ndarray,np.ndarray]
        Calculates and returns the FFT (frequency [Hz], voltage [V], power [dB]).
    plot
        Plot the voltage vs time waveform.
    plot_fft
        Plot the voltage vs frequency FFT.
        
    '''
    
    def __init__(self, path, parse_name=True, date=None):
        
        # inherit from waveform class
        super().__init__(path)
        self.date = str(self.path).split("/")[-2] if date is None else date
        
        if parse_name:
            self._parse_filename()
            # --> self.pulse, self.Tx, self.Rx, self.port, self.plane, 
            #     self.angle_str, self.angle, self.channel
        

    @property
    def label(self) -> str:
        if self.pulse:
            parts = self.name.split("_")
            return " ".join(parts[:2])
        else:
            return " ".join([self.Tx,"to",self.Rx,"Port",self.port,self.plane,"Plane",str(self.angle)+r'$\degree$'])
    
    # Private Methods
    # ---------------
    
    # attribute setter for self.pulse, self.Tx, self.Rx, self.port, self.plane, self.angle_str, self.angle
    def _parse_filename(self):
        '''
        Based on the filename convention
        {transmitter}_{receiver}_{port}_{plane}_{angle}_{scope channel}.csv
        set each parameter as an attribute to the waveform.
        If the data is a pulser data, then assume format
        {DEVICE}_pulser_{Description}.
        '''
        
        parts = self.name.split("_")
        
        self.pulse = ('pulse' in parts) or ('pulser' in parts)   # this is pulser data
        if self.pulse:
            self.Tx = ""
            self.Rx = ""
            self.port = ""
            self.plane = ""
            self.angle_str = ""
            self.angle = ""
            self.channel = ""
        else:                                                    # this is signal data
            self.Tx = parts[0]
            self.Rx = parts[1]
            self.port = parts[2]
            self.plane = parts[3]
            self.angle_str = parts[4]
            self.angle = -1*int(self.angle_str[3:]) if "NEG" in self.angle_str else int(self.angle_str)
            self.channel = parts[5]
    
