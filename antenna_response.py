#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses waveform class to manipulate and compare multiple waveform data to pulse data.

Created on Fri Aug 19 07:38:58 2022

@author: zackashm
"""

# Path set up
from pathlib import Path
HERE = Path(__file__).parent.absolute()

# for quick importing reasons
import os
os.chdir(HERE)

# functional imports
from waveform import waveform
import some_functions as sf
import antennas

import numpy as np
from scipy.constants import c
from scipy.interpolate import interp1d

# plotting imports
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from cycler import cycler
sns.set_theme(font_scale=5, rc={'font.family':'Helvetica', 'axes.facecolor':'#e5ebf6',
                                'legend.facecolor':'#dce3f4', 'legend.edgecolor':'#000000',
                                'axes.prop_cycle':cycler('color', 
                                                         ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', 
                                                          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', 
                                                          '#FF97FF', '#FECB52']),
                                'axes.xmargin':0, 'axes.ymargin':0.05})
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='png'     # for displaying on noninteractive programs

# helpful, but not necessary imports
from warnings import warn


# the class to handle the gain tasks
class antenna_response:
    '''
    This class handles many gain-related calculations for a single horn to horn setup
    at multiple angles. It requires data for both a pulse and signal(s).
    
    Parameters
    ----------
    pulse_path : iterable[str or pathlib.Path]
        The full path of the pulser data. If the iterable size is greater than 1,
        only the first file is chosen as the pulse data.
    signal_paths : iterable[str or pathlib.Path]
        The full paths of the signal data.
    
    Attributes
    ----------
    dt.ns : float
        The time domain window size in nanoseconds of the data truncation.
    pulse : waveform
        The pulser waveform.
    angle_lim : tuple(2)
        The (min,max) angle limit in degrees for inclusion. The rest are ignored.
    angles : ndarray[int]
        The array of sorted angles.
    signals : ndarray[waveform]
        The array containing signal waveforms, with sorting matched with angles.
    labels : list[str]
        The labels of every file, obtained by waveform.label.
    front_dist_m : float
        The distance in meters between the front faces of the antennas.
    back_dist_m : float
        The distance in meters between the backsides of the antennas.
    bandGHz : tuple(2)
        The (min,max) frequency limits in GHz for which all calculations are truncated by.
        This is by default set to the PUEO band (0.3,1.2) GHz in order to prevent
        outside interpolations in scipy.interpolate.interp1d.
    boresight : ndarray[waveform]
        The signal waveform corresponding to 0 degrees.
    gains : ndarray[scipy.interpolate.interp1d]
        An array of gain interpolations as a function of frequency in Hz. These are
        sorted by angle, just like the signals.
    boresight_gain : scipy.interpolate.interp1d
        Like self.gains, but only containing the gain interpolation for angle 0 degrees.
    impulse_response : scipy.interpolate.interp1d
        The calculated impulse response of the boresight signal. THIS IS NOT YET COMPLETED.
    
    Methods
    -------
    set_angles
        Set the instance signals and angles according to self.angle_lim.
    plot_boresight
        Plot the gain response [dB] vs frequency [GHz] of the boresight signal.
    plot_beampattern
        Plot the beampattern of the signals, the gain response [dB] vs angle [degree]
        at select frequencies of interest.
        
    '''
    
    def __init__(self, pulse_path, signal_paths):
        
        self.dt_ns = 70  # window dt
        self.pulse = self._set_data(pulse_path)[0]
        self._signals = self._set_data(signal_paths)
        self.angle_lim = (-180,180)
        self.set_angles(self.angle_lim)  # --> self.angles, self.signals
        self.labels = [self.pulse.label] + [sig.label for sig in self.signals]
        self.front_dist_m = 9.845
        self.back_dist_m = 8.382
        self.bandGHz = (0.3,1.2)
        
        # -properties-
        # self.boresight
        # self.gains
        # self.boresight_gain
        # self.impulse_response
    
    @property
    def boresight(self):# -> ndarray[waveform]:
        
        # to restore angles later
        original_angle_lim = self.angle_lim
        
        # set boresight only
        self.set_angles((0,0))
        
        # there should only be 1 boresight signal
        if len(self.signals) > 1:
            warn("WARNING [antenna_response.boresight()]: More than 1 boresight file " +
                 "was found. The antenna_response class should only be used for a " +
                 "single horn to horn setup.")
        
        # get current (boresight) files
        boresight_signals = self.signals
        
        # reset angles
        self.set_angles(original_angle_lim)
        
        return boresight_signals
    
    @property
    def gains(self):# -> np.ndarray[interp1d]:
        
        # get pulse FFT, restrict it to PUEO band (with 5% margin)
        pul_fHz, pul_rfft, pul_dB = self.pulse.calc_fft(rfft=True, ignore_DC=True)
        # band = (self.bandGHz[0]*0.95, self.bandGHz[1]*1.05) # band with 5% margins, because it yells if we want .3 GHz when it restricts to 0.3000001 GHz or whatever
        # pul_fHz, pul_rfft, pul_dB = sf.restrict_to_band(band, pul_fHz_, pul_rfft_, pul_dB_)
        
        # hard to trust freq arrays will always match, so let's interpolate
        # to prevent edge extrapolations,  we've ensured the arrays are restricted to the band of interest
        pul_gain = interp1d(pul_fHz, pul_dB, assume_sorted=True, bounds_error=True)
        
        # use these later to check consistency in all of the files (but does not fix inconsistencies)
        datalength = self.pulse.datasize
        datasamplerate = self.pulse.samplerate
        
        # check if power is conserved for the pulser fft
        # self.pulse._check_parseval()
        
        # check that the freq range covers full band
        self._check_pueoband(pul_fHz, self.pulse.label, 'antenna_response.gains()')
        
        # get signal FFT
        sig_gains = []
        for sig_waveform in self.signals:
            
            # get signal FFT, restrict it to PUEO band (with 5% margin)
            sig_fHz, sig_rfft, sig_dB = sig_waveform.calc_fft(rfft=True, ignore_DC=True)
            # sig_fHz, sig_rfft, sig_dB = sf.restrict_to_band(band, sig_fHz_, sig_rfft_, sig_dB_)
            
            # hard to trust freq arrays will always match, so let's interpolate
            # to prevent edge extrapolations,  we've ensured the arrays are restricted to the band of interest
            _sig_gain = interp1d(sig_fHz, sig_dB, assume_sorted=True, bounds_error=True)
            
            # add to the list
            sig_gains.append(_sig_gain)
            
            # check if power is conserved for the signal fft
            # sig_waveform._check_parseval()
            
            # check length and samplerate are consistent
            self._check_datarecord(sig_waveform, datalength, datasamplerate, 'antenna_response.gains()')
                
            # check that the freq range covers full band
            self._check_pueoband(sig_fHz, sig_waveform.label, 'antenna_response.gains()')
            
        # gain calculation time
        
        # set our domain, this should be sorted already, postive freqs in Hz
        fHz_ = pul_fHz.copy()
        band = (self.bandGHz[0]*0.95, self.bandGHz[1]*1.05) # band with 5% margins, because it yells if we want .3 GHz when it restricts to 0.3000001 GHz or whatever
        fHz = sf.restrict_to_band(band, fHz_)[0]
        
        # get distance linear interpolation
        fftsize = fHz.size
        ant_dist = np.linspace(self.front_dist_m, self.back_dist_m, fftsize)
        
        # the friis term containing the info about the physical environment
        # https://www.antenna-theory.com/basics/friis.php
        friis_term = 20 * np.log10( np.abs((4 * np.pi * ant_dist * fHz) / c ) )
        
        # assuming all of the signals given are the same set up
        Tx = self.signals[0].Tx
        Rx = self.signals[0].Rx
        same_antenna = Tx[0] == Rx[0]
        
        # gain calc depending on set up
        if same_antenna:
            _gains = [(sig_gain(fHz) - pul_gain(fHz) + friis_term) / 2 for sig_gain in sig_gains]
        else:
            _gains = [sig_gain(fHz) - pul_gain(fHz) + friis_term - antennas.gain[Tx[0]](fHz) for sig_gain in sig_gains]
        
        # return list of gain interpolations sorted by file
        return np.asarray([interp1d(fHz, gain, assume_sorted=True, bounds_error=True) for gain in _gains])
    
    @property
    def boresight_gain(self):# -> interp1d:
        
        # to restore angles later
        original_angle_lim = self.angle_lim
        
        # get gain for boresight only
        self.set_angles((0,0))
        gain = self.gains[0]
        
        # reset angles
        self.set_angles(original_angle_lim)
        
        return gain
    
    @property
    def impulse_response(self) -> interp1d:
        
        warn("WARNING [antenna_response.impulse_response()]: THIS METHOD IS NOT YET COMPLETE.")
        
        # to restore angles later
        original_angle_lim = self.angle_lim
        
        # only want boresight
        self.set_angles((0,0))
        
        # calculate impulse response (TO DO)
        imp_resp = None
        
        # reset angles
        self.set_angles(original_angle_lim)
        
        return imp_resp
        
    
    def _set_data(self, filepaths: list):
        """
        Sets the instance's waveforms, truncated according to self.dt_ns and using
        waveform.estimate_window().

        Parameters
        ----------
        filepaths : iterable[str or pathlib.Path]
            The full paths of the data.

        Returns
        -------
        The array of truncated waveforms.

        """
        
        # get list of waveforms from filepaths
        data_waveforms = [waveform(path) for path in filepaths]
        
        # truncate all of them using waveform.estimate_window()
        for raw_data in data_waveforms:
            raw_data.truncate(raw_data.estimate_window(self.dt_ns))
        
        # return the array containing the truncated waveforms
        return np.array(data_waveforms)
    
    # attribute setter for self.angles, self.signals
    def set_angles(self, angle_lim: tuple = None) -> None:
        """
        Set the instance signals and angles according to self.angle_lim.

        Parameters
        ----------
        angle_lim : tuple(2), optional
            The (min,max) angles in degree for determining which signal waveforms
            to use. If None, this defaults to the currently set self.angle_lim.
            Otherwise, self.angle_lim will be set to the new angle_lim given.

        """
        
        # redefine instance's angle lim if given a new one
        if angle_lim is not None:
            self.angle_lim = angle_lim

        # parse the bounds
        angle_min, angle_max = self.angle_lim
            
        # get everything within the angle limits
        within_lim = lambda theta: np.logical_and(angle_min<=theta,theta<=angle_max)
        angles = [sig.angle for sig in self._signals if within_lim(sig.angle)]
        signals = [sig for sig in self._signals if within_lim(sig.angle)]
        
        # index sorter will be used for signals too
        angle_sorter = np.argsort(angles)
        
        # sort based on angle sorter
        self.angles = np.array(angles)[angle_sorter]
        self.signals = np.array(signals)[angle_sorter]
            
    def plot_boresight(self, fGHz=None, ax=None, **kwargs) -> None:
        """
        Plot the gain response [dB] vs frequency [GHz] of the boresight signal.
        
        Parameters
        ----------
        fGHz : ndarray, optional
            The frequency domain array in GHz. If None, this defaults to the
            PUEO band in steps of 0.1 GHz.
        ax : matplotlib.Axes, optional
            The matploblib Axes to use. If None, one is created internally.
        **kwargs
            matplotlib.Axes.plot() kwargs.
        """
        
        # set the frequency domain to be within the PUEO band
        _fGHz = self._force_band(fGHz, 'antenna_response.plot_boresight()')
        
        # get the boresight waveform and gain
        bs_waveform = self.boresight[0]
        bs_gain = self.boresight_gain
        
        # handle plot instance
        ax = plt.subplots(figsize=[30,20])[1] if ax is None else ax
        
        # get the boresight gain prediction
        predicted_gain = antennas.gain[bs_waveform.Rx[0]](_fGHz*1e9)
        
        # for labeling purposes
        antenna_name = antennas.catalog[bs_waveform.Rx[0]]
        
        # plot predicted and calculated
        ax.plot(_fGHz, predicted_gain, lw=6, label='Predicted {}'.format(antenna_name))
        ax.plot(_fGHz, bs_gain(_fGHz*1e9), c='black', label='Measured {}'.format(antenna_name),
                ls='',marker='o',lw=5,ms=20)
        
        # plot settings
        ax.set(xlim=(0.3 - 0.05, 1.2 + 0.05), ylim=(0, 16.5),
               xlabel="freq [GHz]", ylabel='Gain [dB]')
        title = " ".join([bs_waveform.date, bs_waveform.label])
        ax.set_title(title)
        legend = ax.legend(loc='lower right')
        for line in legend.get_lines(): 
            line.set_linewidth(15)
    
        # find and plot the mean of the residuals
        delta = np.mean(bs_gain(_fGHz*1e9)-predicted_gain)
        ax.add_artist(AnchoredText("Mean residual: {:0.3} dB".format(delta), loc="lower left", frameon=True))
            
    
    def plot_beampattern(self, fGHz=None, polar=True, plot_object=None, dBlim=(None,None),
                         label_add='', title_add='', **kwargs):
        """
        Plot the beampattern of the signals, the gain response [dB] vs angle [degree]
        at select frequencies of interest.
        
        Parameters
        ----------
        fGHz : ndarray, optional
            The frequency domain array in GHz. If None, this defaults to the
            PUEO band in steps of 0.1 GHz.
        polar : bool, optional
            If True, the plot will be in polar coordinates and use plotly.graph_objects.Figure.
            Otherwise, the plot will use rectangular coordinates and use matplotlib.Axes.
            If plot_object is given, this should match the corresponding object with polar.
        plot_object : matplotlib.Axes or plotly.graph_objects.Figure
            The plot object to use. This should match the corresponding object
            that polar would use. True: plotly.graph_objects.Figure, False: matplotlib.Axes.
            If not given, the plot object is created internally.
        dBlim : tuple(2)
            The (min,max) dB limits. The default is (None,None).
        label_add : str
            This is added to the frequency labels in the legend if needed.
        title_add : str
            This is added to the plot title if needed.
        **kwargs
            If polar is True, this is plotly.Figure.update_layout() kwargs. 
            Otherwise, this is matplotlib.Axes.plot() kwargs.
            
        """
        
        # set the frequency domain to be within the PUEO band
        _fGHz = self._force_band(fGHz, 'antenna_response.plot_beampattern()')
        
        # get a good title
        wf = self.signals[0]  # one of the signal waveforms, assuming they all share base names
        title = " ".join([wf.date,wf.Tx,"to",wf.Rx,wf.pol,title_add])
        
        # the plot object
        if polar:  # plotly much nicer for polar plots
        
            # polar should use go.Figure()
            if isinstance(plot_object, plt.Axes):
                warn("WARNING [antenna_response.plot_beampattern()]: Polar is set to True, but " +
                      "the plot object is a matplotlib Axes instead of plotly graph object. " +
                      "Defaulting to plot_object to None.")
                plot_object = None
                
            fig = plot_object if plot_object is not None else go.Figure()
            
        else:      # standard rectangular coords
        
            # not polar should use plt.Axes
            if isinstance(plot_object, go.Figure):
                warn("WARNING [antenna_response.plot_beampattern()]: Polar is set to False, but " +
                      "the plot object is a plotly graph object instead of matplotlib Axes. " +
                      "Defaulting to plot_object to None.")
                plot_object = None
            
            ax = plot_object if plot_object is not None else plt.subplots(figsize=[30,20])[1]
            ax.set(xlabel=r"theta [$\degree$]", ylabel="Gain [dB]", title=title,ylim=dBlim)
        
        # now plot the gain response vs angle at each frequency of interest
        for fGHz in _fGHz:
            
            # get the gains
            gains = np.asarray([gain(fGHz*1e9) for gain in self.gains])
            
            # and plot
            if polar:
                fig.add_trace(
                    go.Scatterpolar(
                        r = gains,
                        theta = self.angles,
                        mode = 'lines+markers',
                        name = '{:.1f} GHz {}'.format(fGHz,label_add),
                        marker=dict(symbol='0',size=8,opacity=1)
                        )
                    )
            else:
                ax.plot(self.angles, gains, ls='-', marker="o", lw=4, ms=13, 
                        label='{:.1f} GHz {}'.format(fGHz,label_add), **kwargs)
        
        # plot settings
        if polar:
            sf.make_plotly_polar_look_nice(fig, rlim=dBlim, rlabel='Gain [dB]', alltitle=title, **kwargs)
            fig.show()
        else:
            legend = ax.legend(loc='upper right')
            for line in legend.get_lines(): 
                line.set_linewidth(15)
    
    def _force_band(self, fGHz=None, location="antenna_response"):
        """
        Truncate the given frequency array to self.bandGHz. This method also
        gives a warning if this has been done.

        Parameters
        ----------
        fGHz : ndarray, optional
            The frequency array in GHz to truncate. If None, an array of the full
            band at 0.1 GHz steps is returned.
        location : str, optional
            The function name to indicate where the warning is being raised.
            The default is the class name.

        Returns
        -------
        fGHz : ndarray
            A frequency array in GHz that is within the self.bandGHz.

        """
        
        if fGHz is None:   # then return full band
            fGHz = np.arange(self.bandGHz[0], self.bandGHz[1]+0.01, 0.1)
        else:      # then truncate to band if necessary
            if (fGHz.min() < self.bandGHz[0]) or (fGHz.max() > self.bandGHz[1]):
                warn('WARNING [{}: Chosen frequencies of interest '.format(location) +
                     'are outside of antenna_response band, and thus outside of interpolation range. Frequencies ' +
                     'outside of {}-{} GHz have been truncated.'.format(*self.bandGHz))
                
                fGHz = sf.restrict_to_band(self.bandGHz, fGHz)[0]
        
        # return band suitable frequency array
        return fGHz
    
    def _check_pueoband(self, data_fHz, data_label, location='antenna_response'):
        """
        Warn if the frequency array doesn't cover the entire PUEO band.

        Parameters
        ----------
        data_fHz : ndarray, optional
            The frequency array in Hz to check.
        data_label : str
            The label of the data whose frequency is put into question.
        location : str, optional
            The function name to indicate where the warning is being raised.
            The default is the class name.

        """
        
        fGHzmin = data_fHz.min() * 1e-9
        fGHzmax = data_fHz.max() * 1e-9
        
        if fGHzmin < 0:
            warn('WARNING [{}]: Data "{}" frequency minimum is negative at {} GHz'.format(
                location, data_label, fGHzmin) )
        if fGHzmin > 0.3:
            warn('WARNING [{}]: Data "{}" frequency minimum of {} GHz is greater than PUEOs 0.3 GHz'.format(
                location, data_label, fGHzmin) )
        if fGHzmax < 1.2:
            warn('WARNING [{}]: Data "{}" frequency maximum of {} is less than PUEOs 1.2 GHz'.format(
                location, data_label, fGHzmax) )
        
    def _check_datarecord(self, data_waveform, test_size, test_samplerate, location="antenna_response"):
        """
        Warn if a waveform's length or sample rate does not match with a test
        length or test sample rate.

        Parameters
        ----------
        data_waveform : waveform
            The waveform whose length and sample rate to check.
        test_size : int
            The length to test the waveform data length against.
        test_samplerate : float
            The sample rate to test the waveform data sample rate against.
        location : str, optional
            The function name to indicate where the warning is being raised.
            The default is the class name.

        """
        
        data_size = data_waveform.datasize
        data_samplerate = data_waveform.samplerate
        label = data_waveform.label
        
        if data_size != test_size:
            warn('WARNING [{}]: Data "{}" mismatching common size of {} instead of {}'.format(
                location, label, data_size, test_size) )
        if test_samplerate != data_samplerate:
            warn('WARNING [{}]: Data "{}" mismatching common sample rate of {} instead of {}'.format(
                location, label, data_samplerate, test_samplerate) )




