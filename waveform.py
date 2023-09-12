#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to handle all relevant things waveform related (V vs s output from 
oscilloscope data).

Created on Fri Aug 19 03:18:41 2022

Updated 2023 05 14
This class is now filename independent. All measurement set up and file name
parsing should be handled by the antennaData class.

@author: zackashm
"""

# i like pathlib's path handling
from pathlib import Path, PosixPath

# functional imports
import numpy as np
import pandas as pd
from numpy.fft import fft as FFT, fftfreq as FFTfreq, rfft as rFFT, rfftfreq as rFFTfreq
from scipy.signal import resample

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
    a single waveform. Oscilloscope output format is assumed to have columns
    3 and 4 be the x and y axes respectively.
    To avoid different FFT effects on data length, the waveform is kept at an odd 
    length.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Provide data via file path containing the csv data.
    data : pandas.DataFrame
        Provide data via pandas DataFrame.
        
    Attributes
    ----------
    path : pathlib.Path
        The full path of the file if path given.
    name : str
        The data file name if path given.
    vcol : str
        The voltage column label.
    tcol : str
        The time column label.
    data : pandas.Dataframe
        A DataFrame of the data with column names matching tcol and vcol.
    filter : numpy.ndarray
        If filtered using tukey_filter, this is the filter array in the time domain.
    vdata : numpy.ndarray
        The voltage data array. Volts is assumed.
    tdata : numpy.ndarray
        The time data array. Seconds is assumed.
    datasize : np.int64
        The size of the data. vdata and tdata sizes are assumed equal.
    samplerate : np.float64
        The sample rate of the data in samples per second. Mean sample spacing
        is used.
    
    Methods
    -------
    reset
        Restore the waveform back to its original initialization, before any 
        manipulations.
    flip
        Flip the vdata sign.
    zeromean
        Zero mean the data by bringing the average noise level offset to 0.
    resample
        Resample the waveform to match a given constant sample rate.
    truncate
        Truncates the time and voltage data to a given time window.
    tukey_filter
        Filter the voltage data based on a Tukey filter in a given time window.
    zeropad
        Zeropad the vdata (extends the tdata).
    estimate_window
        Returns a time domain window based on the voltage peak, with the goal
        to encapsulate the whole pulse. Typically fed into self.truncate.
    clean
        A quick data clean. Zeromean, truncate, filter, and zeropad the waveform.
    plot
        Plot the waveform in the time domain.
    calc_fft -> tuple[np.ndarray,np.ndarray,np.ndarray]
        Calculates and returns the FFT (frequency [Hz], voltage [V], power [dB]).
    plot_fft
        Plot the voltage vs frequency FFT.
    
    '''
    
    def __init__(self, path=None, data=None):
        
        # set data columns
        self.vcol = "V [V]"
        self.tcol = "time [s]"
        
        # default data
        self._RAWDATA = pd.DataFrame({self.tcol:[0], self.vcol:[0]}, dtype=np.float64)
        
        # first try path
        if path is not None:
            # parse filename info
            self.path = Path(path).resolve() if type(path) != PosixPath else path
            self.name = self.path.name
            
            # assumes oscilloscope data, [V,t] on columns [3,4] with units [volts, seconds]
            self._RAWDATA = pd.read_csv(self.path, names=[self.tcol, self.vcol], 
                                        usecols=[3,4], dtype={self.tcol:np.float64,self.vcol:np.float64})
        else:
            self.path = ""
            self.name = ""
        
        # then try input data
        if data is not None:
            self._RAWDATA = data    # overwrites if both data and path given
            self.tcol = data.columns[0]
            self.vcol = data.columns[1]
        
        # set data
        self.data = self._RAWDATA.copy()  # in cases of data manipulation to keep rawdata untouched
        self.filter = None
        self._keep_odd()
        
        # -properties-
        # self.vdata
        # self.tdata
        # self.datasize
        # self.samplerate

        
    @property
    def vdata(self) -> np.ndarray:
        return self.data[self.vcol].values
    
    @property
    def tdata(self) -> np.ndarray:
        return self.data[self.tcol].values
    
    @property
    def datasize(self) -> np.int64:
        return np.int64(self.data.size / 2)
    
    @property
    def samplerate(self) -> np.float64:
        mean_stepsize = np.diff(self.tdata).mean()
        return np.float64('{:0.5e}'.format(1. / mean_stepsize))  # round to a reasonable number
    
    # Public Methods
    # --------------
    
    def reset(self) -> None:
        '''
        Restore the waveform back to its original initialization, before any manipulations.

        Returns
        -------
        None
        
        '''
        del self.data
        del self.filter 
        
        self.data = self._RAWDATA.copy()
        self.filter = None
        self._keep_odd()
        
    def flip(self):
        '''
        Flip the vdata sign. Paritcularly useful for polarity concerns.

        '''
        self.data[self.vcol] *= -1
        
    def zeromean(self, noise_window_ns='best'):
        '''
        Zero mean the data by bringing the average noise level offset to 0.
        
        Parameters
        ----------
        noise_window_ns : 'best', tuple(2), optional
            The time window in ns containing just noise which to calculate the mean offset.
            If 'best', the window will be estimated based on the longest tail end
            of the data relative to the peak.
            
        Returns
        -------
        None
        
        '''
        
        v = self.vdata
        t = self.tdata
        
        if noise_window_ns == 'best':
            
            # get the index of the voltage peak
            peak_ind = (v**2).argmax()
            
            # noise sampling window 5% of total length
            window_size = int(0.05 * self.datasize)
            
            # get noise offset using window at longest tail end
            tailsize = int(self.datasize - peak_ind)
            noise_offset = v[:window_size].mean() if peak_ind > tailsize else v[-1*window_size:].mean()
            
        else:
            
            # set the time limits
            t_min, t_max = noise_window_ns
            bounds = np.logical_and(t_min < t, t < t_max)
            
            noise_offset = v[bounds].mean()
            
        # zeromean
        self.data[self.vcol] += -1*noise_offset
            
    # NOTE: currently, resampling will change the amplitude of the FFT. Not sure why. I don't know if the
    # change is consistent for all signals, so I don't know if it's fine to use when calculating antenna responses.
    def resample(self, new_samplerate=5e9) -> None:
        '''
        Resample the vdata. The tdata will retain the range, and thus the length
        of the data may change from resampling.
        
        Parameters
        ----------
        new_samplerate : float
            The samplerate to force the data to follow in samples per second.
        
        Returns
        -------
        None
        
        '''
        
        v = self.vdata 
        t = self.tdata
        new_sampling = float('{:0.5e}'.format(new_samplerate))
        
        # Calculate the resampling ratio
        resampling_ratio = new_sampling / self.samplerate
        
        # resample
        resampled_vdata, resampled_tdata = resample(v, int(len(self.vdata) * resampling_ratio), t)
        
        # set to resampled data
        del self.data
        self.data = pd.DataFrame({self.tcol : resampled_tdata, self.vcol : resampled_vdata}, 
                                 dtype=np.float64)
        
        self._keep_odd()
    
    def truncate(self, t_ns_window = 'best') -> None:
        '''
        Truncate the waveform in time domain.

        Parameters
        ----------
        t_ns_window : 'best', tuple
            The time domain window (tmin, tmax) in seconds to truncate the waveform
            instance's self.data. If 'best', waveform.estimate_window will be used.
        
        Returns
        -------
        None
        
        '''
    
        # set the time limits
        if t_ns_window == 'best':
            t_ns_window = self.estimate_window()
        t = self.tdata
        t_min, t_max = t_ns_window
        bounds = np.logical_and(t_min < t, t < t_max)
        
        # truncate filter if there is one
        if self.filter is not None:
            self.filter = self.filter[bounds]
        
        # overwrite instance self.data
        self.data = self.data[bounds].reset_index(drop=True)
        
        self._keep_odd()
        
    def tukey_filter(self, t_ns_window=None, peak_width_ns=50, r=0.5) -> None:
        """
        Filter vdata using a Tukey filter w/ parameter r at a given time window.

        Parameters
        ----------
        t_ns_window : tuple(2), 'peak', optional
            The (t_initial, t_final) window location in nanoseconds. If 'peak', a window
            centered at the signal peak is used, the width is given by peak_width_ns. By 
            default (None), the full time range is used (although this is usually not 
            particularly a useful choice other than zeroing the ends).
        peak_width_ns : float
            If t_ns_window is 'peak' this is the width of the window in nanoseconds centered 
            at the peak.
        r : TYPE, optional
            A parameter defining the edge of filter (see Tukey window). The default is 0.5.

        Returns
        -------
        None

        """
        
        dt = 1. / self.samplerate
        t = self.tdata
        
        # handle the filter window
        if t_ns_window is None:     # full time range by default
            window_size = self.datasize
            window_t0 = t[0]
            window_tf = t[-1]
        elif t_ns_window == 'peak':
            vraw = self._RAWDATA[self.vcol].values
            traw = self._RAWDATA[self.tcol].values
            peak_ind = (vraw**2).argmax()
            window_t0 = traw[peak_ind] - peak_width_ns*1e-9/2
            window_tf = traw[peak_ind] + peak_width_ns*1e-9/2
            window_size = np.int64( (window_tf - window_t0) / dt )
        else:
            window_size = np.int64( (t_ns_window[1] - t_ns_window[0]) * 1e-9 / dt )
            window_t0 = t_ns_window[0]
            window_tf = t_ns_window[1]
        
        # get the filter values in (0,1) domain
        tukey = self._tukey_window(window_size, r=r)
        
        # extend time domain to fit window
        min_t = min(window_t0, t[0])
        max_t = max(window_tf, t[-1])
        filt_t = np.arange(min_t, max_t+dt, dt)
            
        # define the array holding the filter values in the time domain
        filtering = np.zeros(filt_t.size)
        
        # place tukey filter values in the corresponding time bins
        window_iter = 0
        for sample in range(filt_t.size):
            if filt_t[sample] < window_t0:
                continue
            if window_iter < window_size:
                filtering[sample] = tukey[window_iter]
                window_iter += 1
            else:
                break
        
        # truncate filter to match data size
        filtering = filtering[filt_t >= t[0]-dt][:self.datasize]
        
        # filter the vdata
        filt_vdata = self.vdata*filtering
        self.data[self.vcol] = filt_vdata
        
        # return the filtering in the time domain in case it's needed (i.e. plotting)
        self.filter = filtering
        
        self._keep_odd()
        
    def zeropad(self, length=6, where='both'):
        '''
        Zeropad the vdata (extends the tdata).

        Parameters
        ----------
        length : int, optional
            The number of zeros to append. The default is 6.
        where : 'before', 'after', 'both', optional
            If 'before', the zeros are appended at the beginning. If 'after', append at the end.
            If 'both' append to both sides (total padding is length*2). The default is 'both'.

        Returns
        -------
        None

        '''
        
        v = self.vdata 
        t = self.tdata
        dt = 1. / self.samplerate
        
        # determine where to pad
        if where == 'before':
            before = length
            after = 0
        elif where == 'after':
            before = 0
            after = length 
        elif where == 'both':
            before = length
            after = length
        else:
            raise ValueError("Error: Argument 'where' must be one of 'before', 'after' or 'both'")
            
        # pad
        padded_vdata = np.pad(v, (before,after))
        padded_tdata_before = np.linspace(t[0] - before*dt, t[0] - dt, before)
        padded_tdata_after = np.linspace(t[-1] + dt, t[-1] + after*dt, after)
        padded_tdata = np.append(np.append(padded_tdata_before, t), padded_tdata_after)
        
        # make filter same length
        if self.filter is not None:
            self.filter = np.pad(self.filter, (before,after))
        
        # set padded data
        self.data = pd.DataFrame({self.tcol : padded_tdata, self.vcol : padded_vdata}, 
                                 dtype=np.float64)
        
        self._keep_odd()
        
    # We can use this instead of having to find the window manually every time.
    # The default window length and offset we chose based on the worst case
    # scenarios (largest angles waveforms) without jeopardizing 'cleaner' waveforms.
    def estimate_window(self, dt_ns: int = 80, t_ns_offset: float = 30) -> tuple:
        '''
        Give a best estimate for a time domain window containing the pulse of the waveform.

        Parameters
        ----------
        dt_ns : int
            The window time length in nanoseconds. The default is 70.
        t_ns_offset : float
            The window time offset (to the left) in nanoseonds from the time of the voltage peak.
            Default is 20.

        Returns
        -------
        (tmin,tmax) : tuple(2)
            The time min and max of the window.
        '''
        
        # get the index of the voltage peak
        v = self._RAWDATA[self.vcol].values
        t = self._RAWDATA[self.tcol].values
        peak_ind = (v**2).argmax()
        
        # determine the window min and max
        tmin = t[peak_ind] - t_ns_offset*1e-9
        tmax = tmin + dt_ns*1e-9
        
        return (tmin,tmax)
    
    def clean(self, reset = True,
              zeromean_kwargs = {'noise_window_ns':'best'},
              truncate_kwargs = {'t_ns_window':'best'}, 
              tukey_filter_kwargs = {'t_ns_window':'peak', 'peak_width_ns':50},
              # resample_kwargs = {'new_samplerate':5e9},
              zeropad_kwargs = {'length':6, 'where':'both'}) -> None:
        '''
        Clean the data: Zeromean, truncate, filter, and zeropad the waveform.
        
        Parameters
        ----------
        reset : Bool
            If True, manipulations are applied to the initialized version of the
            waveform. Otherwise, further calls of this method can stack.
        zeromean_kwargs : False, Dict
            If False, do not zeromean. Otherwise, provide Dict of waveform.zeromean
            keyword arguments.
        truncate_kwargs : False, Dict
            If False, do not truncate. Otherwise, provide Dict of waveform.truncate
            keyword arguments.
        tukey_filter_kwargs : False, Dict
            If False, do not filter. Otherwise, provide Dict of waveform.tukey_filter
            keyword arguments.
        zeropad_kwargs : False, Dict
            If False, do not zeropad. Otherwise, provide Dict of waveform.zeropad
            keyword arguments.

        Returns
        -------
        None

        '''
        if reset:
            self.reset()
        
        if zeromean_kwargs:
            self.zeromean(**zeromean_kwargs)
            
        if truncate_kwargs:
            self.truncate(**truncate_kwargs)
        
        if tukey_filter_kwargs:
            self.tukey_filter(**tukey_filter_kwargs)
        
        # if resample_kwargs:
        #     self.resample(**resample_kwargs)
        
        if zeropad_kwargs:
            self.zeropad(**zeropad_kwargs)
        
        self._keep_odd()
        
    def plot(self, ax=None, tscale=1, vscale=1, tlabel=None, vlabel=None, 
             show_filter=False, title="", **kwargs) -> None:
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
        show_filter : bool, optional
            If True and a filter has been applied, the filter will be plotted with the
            waveform, normalized to the peak of the waveform.
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
        
        # show filter
        if show_filter and (self.filter is not None):
            ax.plot(self.tdata*tscale, self.filter*self.vdata.max()*vscale, c='black', lw=lw, ls='--', alpha=0.5)
    
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
        
        # get the voltage data
        vdata = self.vdata.copy()
        
        # use the corresponding fft function and slice according to freq slice
        fftdata = rFFT(vdata)[freq_slice] if rfft else FFT(vdata)[freq_slice]
        
        # convert dB
        fftdB = 10 * np.log10( np.abs(fftdata)**2 / 50)
        
        # get the corresponding frequency array using the corresponding fft function
        fHz = rFFTfreq(self.datasize, 1. / self.samplerate)[freq_slice] if rfft else FFTfreq(self.datasize, 1. / self.samplerate)[freq_slice] #ignore DC bin
        
        return (fHz, fftdata, fftdB)
        
    def plot_fft(self, ax=None, fscale=1, flabel=None, flim=None, dBlim=None,
                 title="", legend=False, GHz_band=None, **kwargs) -> None:
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
        GHz_band : tuple(2), None, optional
            If True, grey out areas outside the given band.
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
        if GHz_band is not None:
            bandmin, bandmax = GHz_band
            dBmin, dBmax = ax.get_ylim()
            fmin, fmax = ax.get_xlim()
            ax.axvline(x=bandmin*1e9*fscale, c='gray', ls='--', lw=4, label="Band")
            ax.axvline(x=bandmax*1e9*fscale, c='gray', ls='--', lw=4)
            ax.axvspan(0, bandmin*1e9*fscale, facecolor='grey', alpha=0.2)
            ax.axvspan(bandmax*1e9*fscale, fmax, facecolor='grey', alpha=0.2)
        
        if legend:
            ax.legend(loc='upper right')
            
            
    # Private Methods
    # ---------------
        
    def _keep_odd(self):
        '''
        Check the data length, and if even, remove the last point to maintain
        an odd data length.
        '''
        if (self.vdata.size % 2 == 0) or (self.tdata.size % 2 == 0): # even
            self.data.drop(self.data.tail(1).index, inplace=True)
            
            # same for filter if there is one
            if self.filter is not None:
                self.filter = self.filter[:-1]
        
    # for filtering using a tukey window
    def _tukey_window(self, size, r=0.5):
        '''
        Defines the Tukey window filter.
        '''

        # define the window and domain
        w = np.zeros(size)
        x = np.linspace(0, 1, size, endpoint=True)
        
        # define the piecewise function
        left = np.logical_and(0 <= x, x < r/2)
        middle = np.logical_and(r/2 <= x, x < 1 - r/2)
        right = np.logical_and(1 - r/2 <= x, x <= 1)
        
        w[left] = 0.5 * ( 1 + np.cos( (2*np.pi / r) * (x[left] - r/2) ) )
        w[middle] = 1
        w[right] = 0.5 * ( 1 + np.cos( (2*np.pi / r) * (x[right] - 1 + r/2) ) )

        # return the window
        return w
    
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
       