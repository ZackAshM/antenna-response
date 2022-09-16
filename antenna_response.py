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
import copy

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
    pulse_path : iterable[str or pathlib.Path]
        The full path of the pulser data. If the iterable size is greater than 1,
        only the first file is chosen as the pulse data.
    signal_paths : iterable[str or pathlib.Path]
        The full paths of the signal data.
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
    boresight_window : tuple(2)
        The best truncation window for boresight data. This is recommended to be
        applied to all of the signal waveforms.
    tukey_window : tuple(2)
        The best Tukey window for filtering, based on the boresight signal. It is
        a narrower window than the boresight window, 4ns before the voltage peak
        and 15ns wide (values based on observation of previous data, but is not 
        necessarily the best for all data). This is recommended to be applied
        to all of the signal waveforms.
    
    Methods
    -------
    set_data
        Set the pulse or signal waveforms according to a given truncate window and/or
        Tukey filter window.
    set_angles
        Set the instance signals and angles according to self.angle_lim.
    plot_boresight
        Plot the gain response [dB] vs frequency [GHz] of the boresight signal.
    plot_beampattern
        Plot the beampattern of the signals, the gain response [dB] vs angle [degree]
        at select frequencies of interest.
        
    '''
    
    def __init__(self, pulse_path, signal_paths):
        
        self.pulse_path = pulse_path
        self.signal_paths = signal_paths
        self.angle_lim = (-180,180)
        self.set_data(data='pulse')# --> self.pulse
        self.set_data(data='signal', set_angles=self.angle_lim)# --> self.boresight, self.angles, self.signals
        self.labels = [self.pulse.label] + [sig.label for sig in self.signals]
        self.front_dist_m = 8.382
        self.back_dist_m = 9.845
        self.bandGHz = (0.3,1.2)
        
        # -properties-
        # self.boresight
        # self.gains
        # self.boresight_gain
        # self.impulse_response
    
    @property
    def boresight(self):
        return [sig for sig in self.signals if sig.angle == 0][0]
    
    @property
    def boresight_window(self):# -> tuple(2)
        return self.boresight.estimate_window()
    
    @property
    def tukey_window(self):# -> tuple(2)
        return self.boresight.estimate_window(dt_ns=16, t_ns_offset=5)
    
    @property
    def gains(self):# -> np.ndarray[interp1d]:
        
        # get pulse FFT, restrict it to PUEO band (with 5% margin)
        pul_fHz, pul_rfft, pul_dB = self.pulse.calc_fft(rfft=True, ignore_DC=True)
        
        # hard to trust freq arrays will always match, so let's interpolate
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
            
            # hard to trust freq arrays will always match, so let's interpolate
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
        
        # set our domain, postive freqs in Hz
        fHz = np.linspace(self.bandGHz[0]*1e9, self.bandGHz[1]*1e9, pul_fHz.size, endpoint=True)
        
        # get distance linear interpolation
        # NOTE: this is not accurate at different angles. The distance should be set
        # for every angle. But this approximation for all angles is good enough for now.
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
        
        # get pulser and signal, fHz should be the same for both (if same samplerate)
        bs = self.boresight[0]
        print('bs size: ', bs.vdata.size)
        print('pulse size: ', self.pulse.vdata.size)
        # self.pulse.truncate((98e-9,113e-9))
        # bs.truncate((525e-9,540e-9))
        
        fHz, pfft, _ = self.pulse.calc_fft()#rfft=True)
        _ , sfft, _ = bs.calc_fft()#rfft=True)    # only want boresight signal
        
        print('bs fft: ', sfft.size)
        print('pfft: ', pfft.size)
        
        # noise = self.pulse#boresight[0]
        # noise.truncate(noise.estimate_window(t_ns_offset = 170))
        # _, nfft, _ = noise.calc_fft()
        # noise.plot()
        
        # self.pulse.truncate((4.89e-7,5.01e-7))
        # bs.truncate((5.23e-7,5.35e-7))
        
        # self.pulse.plot()
        # bs.plot()
        
        # get distance linear interpolation
        fftsize = fHz.size
        print('freq: ', fftsize)
        pos_size = int(fftsize/2) if fftsize % 2 == 0 else int(fftsize/2) + 1  # size for positive freq side
        pos_dist = np.linspace(self.front_dist_m, self.back_dist_m, pos_size)
        neg_dist = np.linspace(self.back_dist_m, self.front_dist_m, int(fftsize/2))
        dist = np.append(neg_dist, pos_dist)
        # dist = np.linspace(self.front_dist_m, self.back_dist_m, fftsize)
        
        # the physical factor
        phys = (dist * c) / (1j * fHz)   # [m^2]
        
        # calculate impulse response (TO DO)
        
        # Simply division: IR = sqrt((r*c*V_r(f)) / (i*f*V_src(f)))
        IRfft2 = phys * ( sfft / pfft )  # the fft of imp resp squared (radicand)
        
        # -- wiener deconvolution --
        
        # snr determination
        # snr = 100            # constant all freq
        
        fGHz_ordered = np.fft.fftshift(fHz)*1e-9
        fGHz_pos = fGHz_ordered[fGHz_ordered > 0]
        
        lowband1 = np.logical_and( 0<=fGHz_pos, fGHz_pos<0.15 )
        lowband2 = np.logical_and( 0.15<=fGHz_pos, fGHz_pos<0.3 )
        withinband = np.logical_and( 0.3<=fGHz_pos, fGHz_pos<=1.2 )
        aboveband1 = np.logical_and( 1.2<fGHz_pos, fGHz_pos<1.35 )
        aboveband2 = np.logical_and( 1.35<=fGHz_pos, fGHz_pos<np.inf )
        
        below1 = np.ones(len(fGHz_pos[lowband1]))
        below2 = np.linspace(1,1000,len(fGHz_pos[lowband2]),endpoint=True)
        within = 1000*np.ones(len(fGHz_pos[withinband]))
        above1 = np.linspace(1000,1,len(fGHz_pos[aboveband1]),endpoint=True)
        above2 = np.ones(len(fGHz_pos[aboveband2]))
        
        snr_pos = np.hstack((below1, below2, within, above1, above2))
        snr = np.hstack((snr_pos,np.flip(snr_pos)))
        
        print('snr: ', snr.size)
        
        # snr = np.array([1 if np.abs(f)<0.3e9 else 1000 if np.abs(f)<=1.2e9 else 1 for f in fHz])    # constant inband snr vs constant outofband snr
        # snr = sfft / nfft
        
        # print(fHz*1e-9)
        # print(snr)
        
        # wiener filtering
        pfft_eff = pfft / phys
        gfft = (1 / pfft_eff) * (1 / (1 + 1 / (np.abs(pfft_eff)**2 * snr)))
        
        # impulse response calc
        IRfft2w = gfft * sfft
        
        # --------------------------
        
        # other methods...
        
        # unwrap phase to take the sqrt
        IRfft2mag = np.abs(IRfft2)
        IRfft2phi = np.angle(IRfft2)
        IRfft2phiunwrapped = np.unwrap(IRfft2phi)
        
        IRfft2magw = np.abs(IRfft2w)
        IRfft2phiw = np.angle(IRfft2w)
        IRfft2phiunwrappedw = np.unwrap(IRfft2phiw)
        
        # take the sqrt (half of the phase)
        phases = 0.5 * IRfft2phiunwrapped
        phases = (phases + np.pi) % (2 * np.pi) - np.pi  # wrap again (do we need to do this?)
        IRfft = np.sqrt(IRfft2mag) * np.exp(1j * phases)
        
        phasesw = 0.5 * IRfft2phiunwrappedw
        phasesw = (phasesw + np.pi) % (2 * np.pi) - np.pi  # wrap again (do we need to do this?)
        IRfftw = np.sqrt(IRfft2magw) * np.exp(1j * phasesw)
        
        # ------
        
        ax = plt.subplots(figsize=(30,15))[1]
        ax.plot(np.fft.fftshift(fHz*1e-9), np.fft.fftshift(10*np.log10(np.abs(IRfft)**2/50)), ls="--", lw=6, c='black',label="Simple Division")
        ax.plot(np.fft.fftshift(fHz*1e-9), np.fft.fftshift(10*np.log10(np.abs(IRfftw)**2/50)), lw=6, c='black', label="Wiener Deconvolution")
        ax.set(xlim=(0,None), xlabel="f [GHz]", ylabel=r"$10\log{$|fft|$^2/50}$ [dB]")
        ax.set_title(antennas.catalog[bs.Rx[0]] + " Impulse Response FFT")
        
        pueoband = np.logical_and(0.3e9 < fHz, fHz < 1.2e9)
        int_power = 2*np.sum(np.abs(IRfftw[pueoband])**2/50)#, dx=fHz[1]-fHz[0])#, x=fHz[pueoband])
        
        dBmin, dBmax = ax.get_ylim()
        fmin, fmax = ax.get_xlim()
        ax.axvline(x=0.3, c='gray', ls='--', lw=4, label="PUEO Band")
        ax.axvline(x=1.2, c='gray', ls='--', lw=4)
        ax.axvspan(0, 0.3, facecolor='grey', alpha=0.2)
        ax.axvspan(1.2, fmax, facecolor='grey', alpha=0.2)
        
        ax.add_artist(AnchoredText("Integrated Power In Band: {:0.3} W".format(int_power), loc="lower right", frameon=True))
        
        leg = ax.legend(loc="lower left")
        for line in leg.get_lines(): 
            line.set_linewidth(15)
        
        # ------
        
        # inverse fft
        IRfftw = np.insert(IRfftw, 0, 0)
        # print(IRfft)
        imp_resp = np.fft.fftshift(np.fft.ifft(IRfftw))
        
        # -- Checking --
        
        # stest = np.convolve(imp_resp, np.convolve(imp_resp, np.gradient(self.pulse.vdata), mode="same"), mode='same')[:-1] / phys
        # ax = plt.subplots(figsize=(30,15))[1]
        # ax.plot(bs.tdata[:-1], stest, label="Convolved")
        # self.boresight[0].plot(ax=ax, label="Original")
        # ax.legend(loc="upper right")
        
        # --------------
        
        # imp_resp = None
        
        return imp_resp
    
    def plot_impulse_response(self, ax=None, **kwargs):
        
        # bs = self.boresight[0]
        
        # default plot kwargs
        title = kwargs.pop("title",self.signals[0].Rx + " Impulse Response")
        
        imp_resp = self.impulse_response
        
        if ax is None:
            ax = plt.subplots(figsize=(30,15))[1]
        
        dt = self.signals[0].samplerate
        
        imp_resp_t = [dt*i*1e9-20 for i in range(len(imp_resp))]#np.linspace(0, 70, len(imp_resp))
        ax.plot(imp_resp_t, imp_resp, lw=6, **kwargs)#bs.tdata[:-1], imp_resp, lw=6)
        ax.set(ylabel="Response [m/ns]", xlabel='t [ns]')
        ax.set_title(title)
        # bs.plot(ax=ax)
        # self.boresight[0].plot(ax=ax)
        # self.pulse.plot(ax=ax)
    
    def set_data(self, data, truncate='best', tukey_window='best', set_angles=None) -> None:
        """
        Set and clean the data waveforms. Truncate and/or filter (Tukey).

        Parameters
        ----------
        data : {'pulse', 'signal'}
            The type of the data.
        truncate : tuple(2), 'best', None, optional
            The window in seconds which to truncate the data. If 'best' then each waveform will
            be truncated by the boresight_window if data is 'signal', or waveform.estimate_window()
            if data is 'pulse'. If None, the data will not be truncated. The default is 'best'.
        tukey_window : tuple(2), 'best', 'full', None, optional
            The window in seconds which to apply a Tukey filter to the data. If 'best' then each
            waveform will be filtered with the window given by the tukey_window attribute
            if data is 'signal', or waveform.estimate_window(dt_ns=15, t_ns_offset=4) if data
            is 'pulse'. If 'full', the filter will apply to the whole data range. If None, 
            the data will not be filtered. The default is 'best'.
        set_angles : tuple(2), optional
            Set the angle limit of the data by (angle_min, angle_max).

        Returns
        -------
        None

        """
        
        # settle data type
        pulse = data == 'pulse'
        signal = data == 'signal'
        if not pulse and not signal:
            raise ValueError("Error [antenna_response.set_data]: Parameter 'data' " +
                             "must be one of 'pulse' or 'signal'.")
        
        # set waveforms
        if pulse:
            # data_waveforms = np.array([self._rawpulse])
            data_waveforms = np.array([waveform(path) for path in self.pulse_path])
            self.pulse = data_waveforms[0]
        elif signal:
            # data_waveforms = self._rawsignals
            data_waveforms = np.array([waveform(path) for path in self.signal_paths])
            self.signals = data_waveforms
            
            # set angle limits if given
            if set_angles is None:
                pass
            else:
                # redefine instance's angle lim if given a new one
                self.angle_lim = set_angles

                # parse the bounds
                angle_min, angle_max = self.angle_lim
                    
                # get everything within the angle limits
                within_lim = lambda theta: np.logical_and(angle_min<=theta,theta<=angle_max)
                angles = [sig.angle for sig in data_waveforms if within_lim(sig.angle)]
                signals = [sig for sig in data_waveforms if within_lim(sig.angle)]
                
                # check if there is a boresight signal
                boresight_exists = 0 in angles
                if not boresight_exists:
                    raise RuntimeError("Error [antenna_response.set_data]: A boresight (angle 0) " +
                                       "signal data was not found, but is necessary.")
                
                # index sorter will be used for signals too
                angle_sorter = np.argsort(angles)
                
                # sort based on angle sorter
                self.angles = np.array(angles)[angle_sorter]
                data_waveforms = np.array(signals)[angle_sorter]
                
        
        # truncate
        if truncate == 'best':
            for raw_data in data_waveforms:
                if pulse:
                    raw_data.truncate(raw_data.estimate_window())
                elif signal:
                    raw_data.truncate(self.boresight_window)
                    
        elif truncate is None: # do nothing
            pass
        
        else:
            for raw_data in data_waveforms:
                raw_data.truncate(truncate)
        
        # tukey filtering
        if tukey_window == 'best':
            for raw_data in data_waveforms:
                if pulse:
                    raw_data.tukey_filter(time_window=raw_data.estimate_window(dt_ns=25, t_ns_offset=5))
                elif signal:
                    raw_data.tukey_filter(time_window=self.tukey_window)
                    
        elif tukey_window == 'full':
            if pulse:
                raw_data.tukey_filter()
            elif signal:
                raw_data.tukey_filter()
                
        elif tukey_window is None: # do nothing
            pass
        
        else:
            for raw_data in data_waveforms:
                raw_data.tukey_filter(time_window=tukey_window)
        
        # set the data to the array containing the (cleaned) waveforms
        if pulse:
            self.pulse = data_waveforms[0]
        elif signal:
            self.signals = data_waveforms

            
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
        
        # default plotting kwargs
        color = kwargs.pop("color","black")
        color = kwargs.pop("c","black")
        ls = kwargs.pop("ls","-")
        lw = kwargs.pop("lw",6)
        
        # plot predicted and calculated
        ax.plot(_fGHz, predicted_gain, lw=6, label='Predicted {}'.format(antenna_name))
        ax.plot(_fGHz, bs_gain(_fGHz*1e9), label='Measured {}'.format(antenna_name),
                c=color, ls=ls, lw=lw, **kwargs)#marker='o',lw=6,ms=15)
        
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
        _fGHz = fGHz#self._force_band(fGHz, 'antenna_response.plot_beampattern()')
        
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
                
                
    def save_gains(self, fGHz, savepath: Path = HERE):
        
        # directory existence check
        if not os.path.exists(savepath):
            warn("WARNING [antenna_response.save_gains]: Given save directory does not exist. " +
                 "Creating directory {} ...".format(savepath))
            savepath.mkdir(parents=True)
        
        # get the gains
        gains = [gain(fGHz*1e9) for gain in self.gains]
        
        # set filename parts (assuming all the same set up)
        wf = self.signals[0]
        
        plane = "E" if wf.pol[0] == "H" else "H" if wf.pol[0] == "V" else "ERROR"
        if wf.ch == "Ch2":
            plane += "X"
        
        antenna = antennas.catalog[wf.Rx[0]]
        
        # save to file
        for theta, gain in zip(self.angles, gains):
            
            filename = str(savepath) + '/' + "_".join([antenna, str(theta), plane]) + '.txt'
            
            # filename existence checks
            if os.path.exists(filename):
                proceed = input("WARNING [antenna_response.save_gains]: Filename " +
                                "{} exists. Proceed to overwrite? [y/n]\n".format(filename))
                while proceed != 'y':
                    if proceed == "n":
                        print("Ending program...")
                        return 0
                    proceed = input('Please use y or n\n')
            
            output = np.array([fGHz*1e9, gain]).T
            
            np.savetxt(filename, output, fmt=['%.2e','%.2f'])
            
    
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
        
        if fGHz is None:   # then return full band at 100 MHz intervals
            fGHz = np.linspace(self.bandGHz[0], self.bandGHz[1], 10, endpoint=True)
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




