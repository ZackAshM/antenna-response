#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to handle all relevant things waveform related (V vs s output from 
oscilloscope data). 

Created on Fri Aug 19 03:18:41 2022

@author: zackashm
"""

# i like pathlib's path handling
from pathlib import Path, PosixPath

import some_functions as sf

# functional imports
import numpy as np
import pandas as pd
from numpy.fft import fft as FFT, fftfreq as FFTfreq, rfft as rFFT, rfftfreq as rFFTfreq

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
sns.set_theme(font_scale=5, rc={'font.family':'Helvetica', 'axes.facecolor':'#e5ebf6', 
                                'axes.prop_cycle':cycler('color', 
                                                         ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', 
                                                          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', 
                                                          '#FF97FF', '#FECB52']),
                                'axes.xmargin':0, 'axes.ymargin':0.05})

# helpful, but not necessary imports
from warnings import warn


class waveform:
    '''
    Handles oscilloscope output data (assuming volts in V and time in seconds) for
    a single waveform.
    
    Parameters
    ----------
    path : str or pathlib.Path
        The file path containing the csv data.
        
    Attributes
    ----------
    path : pathlib.Path
        The full path of the file.
    date : str
        The str of the data date (assuming the data parent directory is the date).
    name : str
        The data file name.
    pulse : bool
        True if the data is pulser data, with "PULSE" in the file name.
    Tx : str
        File name part referring to the transmitting antenna.
    Rx : str
        File name part referring to the receiving antenna.
    pol : str
        File name part referring to the polarization.
    descriptor : str
        File name part referring to an optional descriptor.
    angle_str : str
        File name part referring to the angle.
    angle : int
        The angle in the file name converted from a string to a number.
    trial : str
        File name part referring to the trial number.
    ch : str
        The scope channel of the data.
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
    label : str
        Return a label for the waveform based on the file name. Format:
        "Tx to Rx pol (descriptor) angle"
        If self.pulse is True, then this is just "{DEVICE} PULSE"
    
    Methods
    -------
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
    
    def __init__(self, path, parse_name=True):
        
        # parse filename info
        self.path = Path(path).resolve() if type(path) != PosixPath else path
        self.date = str(self.path).split("/")[-2]
        self.name = self.path.name
        if parse_name:
            self._parse_filename() # --> self.pulse, self.Tx, self.Rx, self.pol, self.descriptor, self.angle_str, self.angle, self.trial, self.ch
        
        # assumes oscilloscope data, [V,t] on columns [3,4] with units [volts, seconds]
        self.vcol = "V [V]"
        self.tcol = "time [s]"
        self._rawdata = pd.read_csv(self.path, names=[self.tcol, self.vcol], 
                                    usecols=[3,4], dtype={self.tcol:np.float64,self.vcol:np.float64})
        self.data = self._rawdata.copy()  # in cases of data manipulation to keep rawdata untouched
        self.filter = None
        
        # -properties-
        # self.vdata
        # self.tdata
        # self.datasize
        # self.samplerate
        # self.label
        

        
    @property
    def vdata(self) -> np.ndarray:
        return self.data[self.vcol].values
    
    @vdata.setter
    def vdata(self, new_v: np.ndarray):
        self.data[self.vcol] = new_v
    
    @property
    def tdata(self) -> np.ndarray:
        return self.data[self.tcol].values
    
    @property
    def datasize(self) -> np.int64:
        return np.int64(self.data.size / 2)
    
    @property
    def samplerate(self) -> np.float64:
        mean_stepsize = np.mean([np.abs(self.tdata[i+1] - self.tdata[i]) for i in range(self.datasize-1)])
        return float('{:0.5e}'.format(mean_stepsize))  # round to a reasonable number
    
    @property
    def label(self) -> str:
        if self.pulse:
            return self.descriptor
        if self.descriptor:
            return " ".join([self.Tx,"to",self.Rx,self.pol,self.descriptor,str(self.angle)+r'$\degree$'])
        else:
            return " ".join([self.Tx,"to",self.Rx,self.pol,str(self.angle)+r'$\degree$'])
    
    # attribute setter for self.pulse, self.Tx, self.Rx, self.pol, self.descriptor, self.angle_str, self.angle, self.trial
    def _parse_filename(self):
        '''
        Based on the filename convention
        {transmitter}_to_{receiver}_{polarization}_{optional descriptor}_{angle}_{trial}_{scope channel}.csv
        set each parameter as an attribute to the waveform.
        If the data is a pulser data, then assume format
        {DEVICE}_PULSE_{optional descriptor}_{trial}
        and only set the waveform class' self.descriptor to be {DEVICE}_PULSE and trial.
        '''
        
        parts = self.name.split("_")
        
        self.pulse = ('PULSE' in parts) or ('PULSER' in parts)   # this is pulser data
        if self.pulse:
            self.Tx = ""
            self.Rx = ""
            self.pol = ""
            self.descriptor = " ".join(parts[:2])
            self.angle_str = ""
            self.angle = ""
            self.trial = parts[-2]
            self.ch = ""
            
        else:                           # this is signal data
            self.Tx = parts[0]
            self.Rx = parts[2]
            self.pol = parts[3]
            self.descriptor = parts[4] if len(parts) == 8 else ""
            self.angle_str = parts[-3]
            self.angle = -1*int(self.angle_str[3:]) if "NEG" in self.angle_str else int(self.angle_str)
            self.trial = parts[-2]
            self.ch = parts[-1][:3]
    
    # - Methods -
    def truncate(self, window: tuple, zeropad: int = 0) -> None:
        '''
        Truncate the waveform in time domain.

        Parameters
        ----------
        window : tuple
            The time domain window (tmin, tmax) in seconds to truncate the waveform
            instance's self.data.
        zeropad : int
            Zeropad the voltage data with the given length, appended at the end of 
            the data.
        
        Returns
        -------
        None
        
        '''
        
        # don't mess up the original data
        rawdata = self._rawdata.copy()
        
        # set the time limits
        t_min, t_max = window
        bounds = np.logical_and(t_min < rawdata[self.tcol], rawdata[self.tcol] < t_max)
        
        # overwrite instance self.data
        self.data = rawdata[bounds].reset_index(drop=True)
        
        # zeropad the end
        if zeropad > 0:
            self._zeropad(size=zeropad)
        elif zeropad < 0:
            raise ValueError("Error [waveform.truncate()]: zeropad cannot be negative.")
            
        # for easy FFT'ing, just make it odd length (drop last row if even size)
        if self.vdata.size % 2 == 0: # even
            self.data.drop(self.data.tail(1).index, inplace=True)
        
    # We can use this instead of having to find the window manually every time.
    # The default window length and offset we chosen based on the worst case
    # scenarios (largest angles waveforms) without jeopardizing 'cleaner' waveforms.
    def estimate_window(self, dt_ns: int = 70, t_ns_offset: float = 20) -> tuple:
        '''
        Give a best estimate for a time domain window containing the pulse of the waveform.

        Parameters
        ----------
        dt_ns : int
            The window time length in nanoseconds. The default is 70.
        t_ns_offset : float
            The window time offset (to the left) in nanoseonds from the time of the voltage peak.

        Returns
        -------
        (tmin,tmax) : tuple(2)
            The time min and max of the window.
        '''
        
        # get the index of the voltage peak
        v = self._rawdata[self.vcol].values
        t = self._rawdata[self.tcol].values
        peak_ind = (np.abs(v - np.max(np.abs(v)))).argmin()
        
        # determine the window min and max
        tmin = t[peak_ind] - t_ns_offset*1e-9
        tmax = tmin + dt_ns*1e-9
        
        return (tmin,tmax)
    
    def tukey_filter(self, time_window=None, r=0.5):
        """
        Filter vdata using a Tukey filter w/ parameter r at a given time window.

        Parameters
        ----------
        time_window : tuple(2), 'peak', optional
            The (t_initial, t_final) window location in seconds. By default (None), 
            the full time range is used (although this is usually not particularly a 
            useful choice other than zeroing the ends).
        r : TYPE, optional
            A parameter defining the edge of filter (see Tukey window). The default is 0.5.

        Returns
        -------
        None

        """
        
        # handle the filter window
        if time_window is None:     # full time range by default
            window_size = self.datasize
            window_t0 = self.tdata[0]
        else:
            window_size = np.int64( (time_window[1] - time_window[0]) / self.samplerate )
            window_t0 = time_window[0]
        
        # get the filter values in (0,1) domain
        tukey = sf.tukey_window(window_size, r=r)
        
        # define the array holding the filter values in the time domain
        filtering = np.zeros(self.datasize)
        
        
        # TO-DO: make so that windowing can start filling in tukey before actual tdata is reached
        # place tukey filter values in the corresponding time bins
        window_iter = 0
        for sample in range(self.datasize):
            if self.tdata[sample] < window_t0:
                continue
            if window_iter < window_size:
                filtering[sample] = tukey[window_iter]
                window_iter += 1
            else:
                break
        
        # filter the vdata
        self.vdata *= filtering
        
        # return the filtering in the time domain in case its needed (i.e. plotting)
        self.filter = filtering
        
    
    def calc_fft(self, rfft: bool = False, ignore_DC: bool = True) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        '''
        Calculate the waveform's discrete Fourier transformation.

        Parameters
        ----------
        rfft : bool, optional
            If True, the FFT is calculated using numpy.rfft(). Otherwise, numpy.fft() is used.
            The default is False.
        ignore_DC : bool, optional
            If True, the 0 Hz component is dropped from the data. The default is True.

        Returns
        -------
        fHz : numpy.ndarray
            The FFT frequency array in Hz.
        fftdata : numpy.ndarray
            The FFT voltage values.
        fftdB : numpy.ndarray
            The FFT in power dB, assuming 50 Ohm impedance.
            dB = 10 * np.log10( np.abs(fftdata)**2 / 50)
            
        '''
        
        # frequency slice if we're ignoring 0 Hz or not
        freq_slice = slice(1,None) if ignore_DC else slice(None,None)
        
        # get the voltage data and zero mean it
        vdata = self.vdata.copy()
        vdata += -1 * np.mean(vdata)  # zeromean
        
        # use the corresponding fft function and slice according to freq slice
        fftdata = rFFT(vdata)[freq_slice] if rfft else FFT(vdata)[freq_slice]
        
        # convert dB
        fftdB = 10 * np.log10( np.abs(fftdata)**2 / 50)
        
        # get the corresponding frequency array using the corresponding fft function
        fHz = rFFTfreq(self.datasize, self.samplerate)[freq_slice] if rfft else FFTfreq(self.datasize, self.samplerate)[freq_slice] #ignore DC bin
        
        return (fHz, fftdata, fftdB)
        
    def plot(self, ax=None, tscale=1, vscale=1, 
             tlabel=None, vlabel=None, title="", **kwargs) -> None:
        '''
        Plots the voltage waveform.

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            The matploblib Axes to use. If None, one is created internally.
        tscale : float, optional
            Multiply time data by this scale factor. If set to one of {1,1e3,1e6,1e9},
            then tlabel will change to reflect the correct unit prefix. The default is 1.
        vscale : float, optional
            Multiply voltage data by this scale factor. If set to one of {1,1e3,1e6},
            then vlabel will change to reflect the correct unit prefix. The default is 1.
        tlabel : str, optional
            The time axis (x axis) label. Specify the label to include the correct units
            if tscale is not one of {1,1e3,1e6,1e9}.
        vlabel : str, optional
            The voltage axis (y axis) label. Specify the label to include the correct units
            if vscale is not one of {1,1e3,1e6}.
        title : str, optional
            The plot title. The default is "".
        **kwargs
            matplotlib.Axes.plot() kwargs.

        '''
        
        # get plot Axes
        ax = plt.subplots(figsize=[30,20])[1] if ax is None else ax
        
        # unit handling for the labels
        tunits = {1:'[s]', 1e3:'[ms]', 1e6:'[us]', 1e9:'[ns]'}
        known_tunit = tscale in list(tunits.keys())
        if known_tunit:
            tunit = tunits[tscale]
        else:
            warn("WARNING [waveform.plot()]: tunit is unknown, provide tlabel to specify tunit")
            tunit = ""
        tlabel = r'time {}'.format(tunit) if tlabel is None else tlabel
        
        vunits = {1:'[V]', 1e3:'[mV]', 1e6:'[uV]'}
        known_vunit = vscale in list(vunits.keys())
        if known_vunit:
            vunit = vunits[vscale]
        else:
            warn("WARNING [waveform.plot()]: vunit is unknown, provide vlabel to specify vunit")
            vunit = ""
        vlabel = r'V {}'.format(vunit) if vlabel is None else vlabel
        
        # plot settings
        ax.set(xlabel=tlabel, ylabel=vlabel)
        ax.set_title(title)
        
        # some default plot params
        lw = kwargs.pop('lw', 6)
        
        # plot
        ax.plot(self.tdata*tscale, self.vdata*vscale, lw=lw, **kwargs)
        
    def plot_fft(self, ax=None, fscale=1, flabel=None, flim=None, dBlim=None,
                 title="", legend=False, show_PUEO_band=True, **kwargs) -> None:
        '''
        Plots the waveform FFT in dB.

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            The matploblib Axes to use. If None, one is created internally.
        fscale : float, optional
            Multiply freq data by this scale factor. If set to one of {1,1e-3,1e-6,1e-9},
            then flabel will change to reflect the correct unit prefix. The default is 1.
            then vlabel will change to reflect the correct unit prefix. The default is 1.
        flabel : str, optional
            The time axis (x axis) label. Specify the label to include the correct units
            if fscale is not one of {1,1e-3,1e-6,1e-9}.
        flim : tuple(2), optional
            Set (min, max) limits on the plotted frequency domain.
        dBlim : tuple(2), optional
            Set (min, max) limits on the plotted dB range.
        title : str, optional
            The plot title. The default is "".
        legend : bool, optional
            If True, then legend will be shown.
        show_PEUO_band : bool, optional
            If True, the frequencies outside of the PUEO band, (0.3, 1.2) GHz,
            will be greyed out.
        **kwargs
            matplotlib.Axes.plot() kwargs.

        '''
        
        # get plot Axes
        ax = plt.subplots(figsize=[30,20])[1] if ax is None else ax
        
        # unit handling for the labels
        funits = {1:'[Hz]', 1e-3:'[kHz]', 1e-6:'[MHz]', 1e-9:'[GHz]'}
        known_funit = fscale in list(funits.keys())
        if known_funit:
            funit = funits[fscale]
        else:
            warn("WARNING [waveform.plot_fft()]: funit is unknown, provide flabel to specify funit")
            funit = ""
        flabel = r'Freq {}'.format(funit) if flabel is None else flabel
        
        dBlabel = r"$10\log_{10}$(|rfft($V)|^{2} / 50)$ [dB]"
        
        # plot settings
        ax.set(xlabel=flabel, ylabel=dBlabel)
        if flim is not None: ax.set(xlim=flim)
        if dBlim is not None: ax.set(ylim=dBlim)
        ax.set_title(title)
        
        # some default plot params
        lw = kwargs.pop('lw', 3)
        
        # get the plot data and plot
        fHz, _, fftdB = self.calc_fft(rfft=True)
        ax.plot(fHz*fscale, fftdB, lw=lw, **kwargs)
        
        # show the PUEO band
        if show_PUEO_band:
            dBmin, dBmax = ax.get_ylim()
            fmin, fmax = ax.get_xlim()
            ax.axvline(x=0.3, c='gray', ls='--', lw=4, label="PUEO Band")
            ax.axvline(x=1.2, c='gray', ls='--', lw=4)
            ax.axvspan(0, 0.3, facecolor='grey', alpha=0.2)
            ax.axvspan(1.2, fmax, facecolor='grey', alpha=0.2)
        
        if legend:
            ax.legend(loc='upper right')
            
    # zero pad the waveform (to be used when truncating)
    def _zeropad(self, size=4):
        '''
        Zero pad the voltage data at the end.
        
        Parameters
        ----------
        size : int, optional
            The size of the padding.
        
        Returns
        -------
        None
        '''
        
        # get time info
        dt = self.samplerate
        tf = self.tdata[-1]
        
        # create padded data, time continuation, 0 for voltage
        t_padded = [tf + dt*i for i in range(size)]
        v_padded = np.zeros(size)
        
        # the padded dataframe
        padding = pd.DataFrame({self.tcol: t_padded, self.vcol: v_padded})
        
        # set this object's data to the padded dataframe
        self.data = pd.concat((self.data,padding), ignore_index=True)
        
    
    # check power is conserved after FFT
    def _check_parseval(self, rfft: bool = True) -> None:
        '''
        Check that Parseval's theorem for power conservation during an FFT is satisfied.
        sum(abs(V(t))**2) = sum(abs(V(f))**2) / N
        If fails, a warning message appears.
        
        Parameters
        ----------
        rfft : bool, optional
            If True, the FFT is calculated using numpy.rfft(). Otherwise, numpy.fft() is used.
            The default is False.
            
        Returns
        -------
        None
        
        '''
        
        # don't mess up the original data
        vdata = self.vdata.copy()
        vdata += -1 * np.mean(vdata)      # zeromean
        
        # get the FFT
        fftdata = self.calc_fft(rfft, ignore_DC = False)[1]
        
        # account for rfft being only half of most frequency bins
        if rfft:
            fftdata[1:] *= np.sqrt(2)     # DC is not binned twice
            if self.datasize % 2 == 0:    # even, nyquist bin last element
                fftdata[-1] /= np.sqrt(2) # nyquist is not binned twice
        
        # calculate the power in time and in freq
        tpower = np.sum(np.abs(vdata)**2)
        fpower = np.sum(np.abs(fftdata)**2) / self.datasize
        
        # determine if the values are close enough
        close = np.abs(tpower - fpower) < 1e-3
        result = 'succeeds' if close else 'fails'
        
        # warn if failure
        if not close:
            warn("Parseval's Theorem {} with time domain power calculated".format(result) +
                 " to be {} and frequency domain power calculated to be {}".format(tpower, fpower))
       
        
# EXAMPLE
# foo = waveform("Data/20220818/R2A_TO_R3A_VPOL_NEG10_01_Ch1.csv")
# print(foo.date)
# print(foo.path)
# print(foo.vcol)
# print(foo.tcol)
# print(foo.data)
# print(foo.name)
# print(foo.Tx)
# print(foo.Rx)
# print(foo.pol)
# print(foo.descriptor)
# print(foo.angle_str)
# print(foo.angle)
# print(foo.trial)
# print(foo.ch)
# print(foo.vdata)
# print(foo.tdata)
# print(foo.datasize)
# print(foo.samplerate)
# print(foo.label)
# foo.truncate(foo.estimate_window())
# foo.plot(tscale=1e9, vscale=1e3)
# foo.plot_fft(fscale=1e-9, flim=(0.1,2), dBlim=(-80,-10))