#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses antennaData objects from pulser data and receiving data in order to determine 
properties such as gain and impulse response of the receiving antenna.

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
from antennaData import antennaData
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
import plotly.express as px
from itertools import cycle
pio.renderers.default='png'     # for displaying on noninteractive programs

# helpful, but not necessary imports
from warnings import warn


# the class to handle the gain tasks
class antenna_response:
    '''
    DOCUMENTATION CURRENTLY OUT OF DATE
    This class handles many gain-related calculations for a single horn to horn setup
    at multiple angles. It requires both pulse and signal data.
    
    Parameters
    ----------
    pulse_path : str or pathlib.Path
        The full path of the pulse waveform data.
    signal_paths : iterable[str or pathlib.Path]
        The full paths of the signal waveform data.
    distance : float or iterable(2)
        The distance in meters between Receiver and Transmitter. If an iterable of 
        size 2 is given, the effective distance is a linear interpolation from the first 
        number to the second (This is usually to assign distances between the front faces 
        and the back panels as an effective frequency dependency).
    bandGHz : optional, tuple(2)
        The (min, max) frequency limits in GHz. Default is the PUEO band (0.3, 1.2).
    
    Attributes
    ----------
    pulse_path : str or pathlib.Path
        The full path of the pulser waveform data.
    signal_paths : iterable[str or pathlib.Path]
        The full paths of the signal waveform data.
    pulse : antennaData
        The pulser waveform data.
    angles : ndarray[int]
        The array of angles in ascending order.
    signals : ndarray[antennaData]
        The array containing signal waveforms, with sorting matched with self.angles.
    labels : list[str]
        The labels of every file, obtained by waveform.label.
    bandGHz : tuple(2)
        The (min,max) frequency limits in GHz.
    front_dist_m : float
        The distance in meters between the front faces of the antennas.
    back_dist_m : float
        The distance in meters between the backsides of the antennas.
    gains : ndarray[scipy.interpolate.interp1d]
        An array of gain interpolations as a function of frequency in Hz. These are
        sorting matched with self.angles.
    impulse_response : scipy.interpolate.interp1d
        The calculated impulse response of the boresight signal.
    
    Methods
    -------
    set_data
        Set the pulse or signal waveforms according to a given truncate window and/or
        Tukey filter window. Can also be used to set limits on angles.
    plot_boresight
        Plot the gain response [dB] vs frequency [GHz] of the boresight signal.
    plot_beampattern
        Plot the beampattern of the signals, the gain response [dB] vs angle [degree]
        at select frequencies of interest.
        
    '''
    
    def __init__(self, pulse_path, signal_paths, distance, bandGHz=(0.3,1.2)):
        
        self.pulse_path = pulse_path
        self.signal_paths = signal_paths
        self._initialize_data()# --> self.pulse, self.angles, self.signals
        self.labels = [self.pulse.label] + [sig.label for sig in self.signals]
        self.bandGHz = bandGHz
        
        _distance = np.array(distance, ndmin=1, dtype=float)
        if len(_distance) > 1:
            self.front_dist_m = _distance[0]
            self.back_dist_m = _distance[1]
        else:
            self.front_dist_m = _distance[0]
            self.back_dist_m = _distance[0]
        
        # -properties-
        # self.gains
        # self.impulse_response
        
        # -private attributes-
        # self._ANGLES
        # self._PULSE
        # self._SIGNALS
        
    
    @property
    def gains(self):# -> np.ndarray[interp1d]:
        '''
        Calculates the gain using the pulse and signals data. Each element of the returned
        array is a scipy.interpolate.interp1d of the gain values as a function of
        frequency in Hz. The returned array is sorting matched with self.angles (and self.signals).
        
        The gain is calculated by the following steps:
            1. Use waveform.calc_fft to get the relative gain of the pulse and signals waveform data:
            dB = 10 * log10( abs(rfft)**2 / 50)
            2. Create scipy.interpolate.interp1d of each relative gain with the corresponding
            frequency data given by waveform.calc_fft.
            3. Calculate the gain via https://www.antenna-theory.com/basics/friis.php :
            gain = sig_dB - pul_dB + friis_term - transmitted_dB
            where friis_term is the physical propagation term given by
            friis_term = 20 * log10( abs( (4 * pi * dist * fHz) / c ) )

        '''
        
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
            
        # gain calculation
        
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
        port = self.signals[0].port
        same_antenna = Tx[0] == Rx[0]
        if same_antenna:
            Tx = Tx+port
        
        # gain calculation
        _gains = [sig_gain(fHz) - pul_gain(fHz) + friis_term - antennas.gain[Tx](fHz) for sig_gain in sig_gains]
        
        # return list of gain interpolations sorted by file
        return np.asarray([interp1d(fHz, gain, assume_sorted=True, bounds_error=True) for gain in _gains])
    
    @property
    def impulse_response(self):
        '''
        Return the (time [ns], response [m/ns]) impulse response of the boresight signal.
        
        Calculation is done using Weiner deconvolution method to suppress irrelevant high frequencies.
        SNR characteristics are currently fixed across frequencies for all signals, being as low as
        1 outside of the frequency band and 1000 within the frequency band, with a linear slope
        connecting outside and inside levels and spanning 150 MHz.

        Returns
        -------
        (impulse response time, impulse response)
            The impulse response time domain in ns and amplitude in m/ns (complex). 
            Time domain is determined using signal samplerate and impulse response data length.
        '''
        
        warn("WARNING [antenna_response.impulse_response()]: THIS METHOD IS NOT YET FULLY DEBUGGED.")
        
        # boresight
        bs = self.signals[self.angles==0][0]
        
        # get pulser and signal FFTs, fHz should be the same for both
        self._check_datarecord(self.pulse, bs.datasize, bs.samplerate, location='antenna_response.impulse_response', 
                               add='Impulse Response results may misbehave due to uncommon frequency values in FFT')
        fHz, pfft, _ = self.pulse.calc_fft()
        _ , sfft, _ = bs.calc_fft()
        
        # ignore very high frequencies
        fCut = np.logical_and(-2e9 < fHz, fHz < 2e9)
        fHz = fHz[fCut]
        pfft = pfft[fCut]
        sfft = sfft[fCut]
        
        # print('bs fft: ', sfft.size)
        # print('pfft: ', pfft.size)
        
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
        # print('freq: ', fftsize)
        pos_size = int(fftsize/2) if fftsize % 2 == 0 else int(fftsize/2) + 1  # size for positive freq side
        pos_dist = np.linspace(self.front_dist_m, self.back_dist_m, pos_size)
        neg_dist = np.linspace(self.back_dist_m, self.front_dist_m, int(fftsize/2))
        dist = np.append(neg_dist, pos_dist)
        dist = np.linspace(self.front_dist_m, self.back_dist_m, fftsize)
        
        # the physical factor
        phys = (dist * c) / (1j * fHz)   # [m^2]
        
        # calculate impulse response (TO DO)
        
        # Simple division: IR = sqrt((r*c*V_r(f)) / (i*f*V_src(f)))
        # IRfft2 = phys * ( sfft / pfft )  # the fft of imp resp squared (radicand)
        
        # -- wiener deconvolution --
        
        # snr determination
        # snr = 100            # constant all freq
        
        fGHz_ordered = np.fft.fftshift(fHz)*1e-9
        fGHz_pos = fGHz_ordered[fGHz_ordered > 0]
        # fGHz_pos = fHz * 1e-9
        
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
        # snr = np.hstack((below1, below2, within, above1, above2))
        snr = np.hstack((snr_pos,np.flip(snr_pos)))
        
        # print('snr: ', snr.size)
        
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
        # IRfft2mag = np.abs(IRfft2)
        # IRfft2phi = np.angle(IRfft2)
        # IRfft2phiunwrapped = np.unwrap(IRfft2phi)
        
        IRfft2magw = np.abs(IRfft2w)
        IRfft2phiw = np.angle(IRfft2w)
        IRfft2phiunwrappedw = np.unwrap(IRfft2phiw)
        
        # take the sqrt (half of the phase)
        # phases = 0.5 * IRfft2phiunwrapped
        # phases = (phases + np.pi) % (2 * np.pi) - np.pi  # wrap again (do we need to do this?)
        # IRfft = np.sqrt(IRfft2mag) * np.exp(1j * phases)
        
        phasesw = 0.5 * IRfft2phiunwrappedw
        phasesw = (phasesw + np.pi) % (2 * np.pi) - np.pi  # wrap again (do we need to do this?)
        IRfftw = np.sqrt(IRfft2magw) * np.exp(1j * phasesw)
        
        # - POWER INTEGRATION -----
        
        # ax = plt.subplots(figsize=(30,15))[1]
        # ax.plot(np.fft.fftshift(fHz*1e-9), np.fft.fftshift(10*np.log10(np.abs(IRfft)**2/50)), ls="--", lw=6, c='black',label="Simple Division")
        # ax.plot(np.fft.fftshift(fHz*1e-9), np.fft.fftshift(10*np.log10(np.abs(IRfftw)**2/50)), lw=6, c='black', label="Wiener Deconvolution")
        # # ax.plot(fHz*1e-9, 10*np.log10(np.abs(IRfft)**2/50), ls="--", lw=6, c='black',label="Simple Division")
        # # ax.plot(fHz*1e-9, 10*np.log10(np.abs(IRfftw)**2/50), lw=6, c='black', label="Wiener Deconvolution")
        # ax.set(xlim=(0,None), xlabel="f [GHz]", ylabel=r"$10\log{$|fft|$^2/50}$ [dB]")
        # ax.set_title(bs.label + " Impulse Response FFT")
        
        # pueoband = np.logical_and(0.3e9 < fHz, fHz < 0.7e9)
        # int_power = 2*np.sum(np.abs(IRfftw[pueoband])**2/50) / IRfftw[pueoband].size #, dx=fHz[1]-fHz[0])#, x=fHz[pueoband])
        
        # dBmin, dBmax = ax.get_ylim()
        # fmin, fmax = ax.get_xlim()
        # ax.axvline(x=0.3, c='gray', ls='--', lw=4, label="PUEO Band")
        # ax.axvline(x=1.2, c='gray', ls='--', lw=4)
        # ax.axvspan(0, 0.3, facecolor='grey', alpha=0.2)
        # ax.axvspan(1.2, fmax, facecolor='grey', alpha=0.2)
        
        # ax.add_artist(AnchoredText("Integrated Power In Band: {:0.3} W".format(int_power), loc="lower right", frameon=True))
        
        # leg = ax.legend(loc="lower left")
        # for line in leg.get_lines(): 
        #     line.set_linewidth(15)
        
        # ---------------------------
        
        # inverse fft
        IRfftw = np.insert(IRfftw, 0, 0)  # add back in 0Hz slot in first position with amplitude 0
        # print(IRfft)
        imp_resp = np.fft.fftshift(np.fft.ifft(IRfftw))
        # imp_resp = np.fft.irfft(IRfftw)
        
        # -- Checking --
        
        # stest = np.convolve(imp_resp, np.convolve(imp_resp, np.gradient(self.pulse.vdata), mode="same"), mode='same')[:-1] / phys
        # ax = plt.subplots(figsize=(30,15))[1]
        # ax.plot(bs.tdata[:-1], stest, label="Convolved")
        # self.signals[self.angles==0][0].plot(ax=ax, label="Original")
        # ax.legend(loc="upper right")
        
        # --------------
        
        # imp_resp = None
        
        dt = 1. / self.signals[0].samplerate
        imp_resp_t = np.array([dt*i*1e9-20 for i in range(len(imp_resp))])
        
        
        return imp_resp_t, imp_resp
    
    
    def reset(self):
        '''
        Set the angles, pulse, and signals back to the initialized versions.
        '''
        del self.angles
        del self.pulse
        del self.signals
        self.angles = self._ANGLES.copy()
        self.pulse = self._PULSE.copy()
        self.signals = self._SIGNALS.copy()
    
    def plot_impulse_response(self, ax=None, **kwargs):
        
        bs = self.signals[self.angles==0][0]
        
        # default plot kwargs
        title = kwargs.pop("title",bs.label + " Impulse Response")
        
        imp_resp_t, imp_resp = self.impulse_response
        
        if ax is None:
            ax = plt.subplots(figsize=(30,15))[1]
        
        # dt = 1. / self.signals[0].samplerate
        
        # imp_resp_t = [dt*i*1e9-20 for i in range(len(imp_resp))]#np.linspace(0, 70, len(imp_resp))
        ax.plot(imp_resp_t, imp_resp, lw=6, **kwargs)#bs.tdata[:-1], imp_resp, lw=6)
        ax.set(ylabel="Response [m/ns]", xlabel='Time [ns]')
        ax.set_title(title)
        # bs.plot(ax=ax)
        # self.pulse.plot(ax=ax)
    
    def clean_data(self, whichdata,
                   zeromean_kwargs = {'noise_window_ns':'best'},
                   truncate_kwargs = {'t_ns_window':'best'},
                   tukey_filter_kwargs = {'t_ns_window':'peak', 'peak_width_ns':50},
                   zeropad_kwargs = {'length':6, 'where':'both'}) -> None:
        """
        Clean the data waveforms. Choose which data to clean (signals or pulse). 
        Choose to zeromean, truncate, filter (via Tukey), and zeropad.
        Note that these effects are applied to the initialized version of the data 
        (further calls of this method will not stack, but instead overwrite).

        Parameters
        ----------
        whichdata : 'signals', 'pulse', 'all'
            Selects whether to clean the signals or the pulse, or all of them.
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

        """
        
        # set data
        if whichdata=='pulse':
            data_wfs = np.array([self.pulse])
        elif whichdata=='signals':
            data_wfs = self.signals
        elif whichdata=='all':
            data_wfs = np.append(np.array([self.pulse]), self.signals)
        else:
            raise ValueError("Error [antenna_response.clean_data()]: Parameter 'whichdata' must be one of 'pulse', 'signals' or 'all'")
        
        for wf in data_wfs:
            wf.clean(reset = True,
                     zeromean_kwargs = zeromean_kwargs,
                     truncate_kwargs = False, 
                     tukey_filter_kwargs = False,
                     zeropad_kwargs = {'length':1000})   # ensure data length contains all reasonable windows
            wf.clean(reset = False,
                     zeromean_kwargs = zeromean_kwargs,
                     truncate_kwargs = truncate_kwargs, 
                     tukey_filter_kwargs = tukey_filter_kwargs,
                     zeropad_kwargs = zeropad_kwargs)
        
            # check data record
            if whichdata=='pulse':
                data_wf = wf
                test_wf = self.signals[0]
                self._check_datarecord(data_wf, test_wf.datasize, test_wf.samplerate,
                                       location='antenna_response_clean_data()', add='Pulse mismatch with signals.')
            elif whichdata=='signals':
                data_wf = wf
                test_wf = self.pulse
                self._check_datarecord(data_wf, test_wf.datasize, test_wf.samplerate,
                                       location='antenna_response_clean_data()', add='Signals mismatch with pulse.')
            elif whichdata=='all':
                data_wf = wf
                test_wf = self.pulse
                self._check_datarecord(data_wf, test_wf.datasize, test_wf.samplerate,
                                       location='antenna_response_clean_data()', add='Signals mismatch with pulse.')
            
    def plot_gain(self, angle=0, fGHz=None, ax=None, **kwargs) -> None:
        """
        Plot the gain response [dB] vs frequency [GHz].
        
        Parameters
        ----------
        angle : float
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
        waveform = self.signals[self.angles==angle][0]
        gain = self.gains[self.angles==angle][0]
        
        # handle plot instance
        ax = plt.subplots(figsize=[30,20])[1] if ax is None else ax
        
        # get the boresight gain prediction
        predicted_gain = antennas.gain[waveform.Rx+waveform.port](_fGHz*1e9)
        
        # for labeling purposes
        antenna_name = waveform.Rx+waveform.port
        
        # default plotting kwargs
        color = kwargs.pop("color",'#636EFA')
        color = kwargs.pop("c",'#636EFA')
        ls = kwargs.pop("ls","-")
        lw = kwargs.pop("lw",6)
        
        if angle == 0:
            # plot predicted and calculated
            ax.plot(_fGHz, predicted_gain, lw=6, ls='--', c='black', 
                    label='Company Measured {}'.format(antenna_name))
            # find and plot the mean of the residuals
            delta = np.mean(gain(_fGHz*1e9)-predicted_gain)
            ax.add_artist(AnchoredText("Mean residual: {:0.3} dB".format(delta), 
                                       loc="lower left", frameon=True))
        ax.plot(_fGHz, gain(_fGHz*1e9), label='PUEO Measured {}'.format(antenna_name),
                c=color, ls=ls, lw=lw, **kwargs)#marker='o',lw=6,ms=15)
        
        # plot settings
        ax.set(xlim=(0.3 - 0.05, 1.2 + 0.05), ylim=(0, 16.5),
               xlabel="freq [GHz]", ylabel='Gain [dB]')
        title = " ".join([waveform.date, waveform.label])
        ax.set_title(title)
        legend = ax.legend(loc='lower right')
        for line in legend.get_lines(): 
            line.set_linewidth(15)
            
    
    def plot_beampattern(self, fGHz=None, polar=True, plot_object=None, dBlim=(None,None),
                         label_add='', title_add='', close_gap=False, **kwargs):
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
        close_gap : bool
            Connect the first and last points. Default is False.
        **kwargs
            If polar is True, this is plotly.Figure.update_layout() kwargs. 
            Otherwise, this is matplotlib.Axes.plot() kwargs.
            
        """
        
        # set the frequency domain to be within the PUEO band
        _fGHz = fGHz#self._force_band(fGHz, 'antenna_response.plot_beampattern()')
        
        # get a good title
        wf = self.signals[0]  # one of the signal waveforms, assuming they all share base names
        title = " ".join([wf.date,wf.Tx,"to",wf.Rx,wf.port,wf.plane,'Gain [dB]',title_add])
        
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
        
        # set colors for plotting
        COLORS = cycle(px.colors.sequential.OrRd[1:] + ['#530000', '#210000'])
        
        # now plot the gain response vs angle at each frequency of interest
        for fGHz in _fGHz:
            
            # get the gains
            gains = np.asarray([gain(fGHz*1e9) for gain in self.gains])
            angles = self.angles
            if close_gap:
                gains = np.append(gains, gains[0])
                angles = np.append(self.angles, self.angles[0])
            
            # and plot
            if polar:
                fig.add_trace(
                    go.Scatterpolar(
                        r = gains,
                        theta = angles,
                        mode = 'lines',
                        name = '{:.1f} GHz {}'.format(fGHz,label_add),
                        line=dict(width=3),
                        marker=dict(symbol='0',size=8,opacity=1),
                        line_color=next(COLORS),
                        ),
                    )
            else:
                ax.plot(angles, gains, ls='-', marker="o", lw=4, ms=13, 
                        label='{:.1f} GHz {}'.format(fGHz,label_add), **kwargs)
        
        # plot settings
        if polar:
            
            # add solid lines for main grid / major ticks
            rmin,rmax = dBlim
            for deg in np.arange(-180,180,30):
                fig.add_trace(
                    go.Scatterpolar(
                        r = (rmin,rmax),
                        theta = (deg,deg),
                        mode = 'lines',
                        name = None,
                        line=dict(color='#cccccc',width=1),
                        showlegend=False,
                        )
                    )
            circ = np.linspace(-180,180,500)
            for dB in range(int(rmin), int(rmax)):
                if dB % 5 == 0:
                    fig.add_trace(
                        go.Scatterpolar(
                            r = dB*np.ones(circ.size),
                            theta = circ,
                            mode = 'lines',
                            name = None,
                            line=dict(color='#cccccc',width=1),
                            showlegend=False,
                            )
                        )
            
            self._make_plotly_polar_look_nice(fig, rlim=dBlim, rlabel='', alltitle=title, **kwargs)
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
        
        antenna = wf.Rx+wf.port
        
        # save to file
        for theta, gain in zip(self.angles, gains):
            
            filename = str(savepath) + '/' + "_".join([antenna, str(theta), wf.plane]) + '.txt'
            
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
            
            
    def _initialize_data(self):
        """
        Set the data waveforms based on the provided filenames in the class.
        Initializes self.angles, self.pulse, and self.signals. Angles and signals
        are commonly sorted in order of increasing angles.
        """
        
        # set waveforms
        self._PULSE = antennaData(self.pulse_path)
        self.pulse = self._PULSE.copy()
        signal_wfs = np.array([antennaData(path=path) for path in self.signal_paths])
            
        # set angles and sort
        angles = [sig.angle for sig in signal_wfs]
        signals = [sig for sig in signal_wfs]
        
        angle_sorter = np.argsort(angles)
        self._ANGLES = np.array(angles)[angle_sorter]
        self.angles = self._ANGLES.copy()
        self._SIGNALS = np.array(signals)[angle_sorter]
        self.signals = self._SIGNALS.copy()

        # check data
        test_wf = self.signals[self.angles==0][0]
        for sample_wf in np.append(self.signals, self.pulse):
            self._check_datarecord(sample_wf, test_wf.datasize, test_wf.samplerate, 
                                       location="antenna_response.set_data()", 
                                       add='Tested against boresight data {}.'.format(test_wf.label))
            
    # will restrict everything to a band to prevent interpolating outside of range
    def _restrict_to_band(self, bandGHz, fHz, *args):
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
        within_band = np.logical_and(fGHzmin*1e9 < fHz,fHz < fGHzmax*1e9)
        
        # set the edges to be one step further to include the bounds within the range
        idx = np.nonzero(within_band)
        within_band[idx[0] - 1] = True
        within_band[idx[-1] + 1] = True
        
        # make the list of truncated arrays
        return_arrays = [fHz[within_band]] + [arr[within_band] for arr in args]
        
        return return_arrays
    
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
                
                fGHz = self.restrict_to_band(self.bandGHz, fGHz)[0]
        
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
        
    def _check_datarecord(self, data_waveform, test_size, test_samplerate, location="antenna_response", add=""):
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
        add : str, optional
            Add a custom string to the end of the warning message.

        """
        
        data_size = data_waveform.datasize
        data_samplerate = data_waveform.samplerate
        label = data_waveform.label
        
        if data_size != test_size:
            warn('WARNING [{}]: "{}" mismatching size of {} instead of {}. '.format(
                location, label, data_size, test_size) + add)
        if test_samplerate != data_samplerate:
            warn('WARNING [{}]: "{}" mismatching sample rate of {} instead of {}. '.format(
                location, label, data_samplerate, test_samplerate) + add)

    # Since there's so much just to make the plot look nice, I separated it into this function
    # If you haven't seen plotly before, prepare yourself for many dictionaries (though imo it's
    # easier to customize plot settings this way)
    def _make_plotly_polar_look_nice(self, fig, rlim=(None,None), rlabel='', sector=(0,180), alltitle='', 
                                    kwargs_for_trace={}, **kwargs):
        '''
        Makes plotly's polar plot look nice, to the extent it can on Python...
    
        Parameters
        ----------
        fig : plotly.Figure
            The figure to make look nice.
        rlim : tuple(2)
            Radial axis (min,max) limits.
        rlabel : str
            The radial axis label. The default is ''.
        sector : tuple(2)
            Set the angular sector of the chart. Default is (0,180), i.e. upper half.
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
                               sector = sector,                 # set chart shape (half)
                               bgcolor='#fcfcfc',
                               angularaxis = dict(              # angle axis settings
                                   dtick = 5,                   # angle tick increment
                                   ticklabelstep=6,             # tick label increment
                                   gridcolor='#d6d6d6',         # tick line color
                                   griddash='dot',              # tick line style
                                   # rotation = 90,               # rotates data to be on upper half
                                   direction = "counterclockwise",     # -90 to 90 right to left
                                   tickfont=dict(size=20),      # angle tick size
                                   ),
                               radialaxis=dict(                 # radial axis settings
                                   angle=0,                     # angle where the tick axis is
                                   tickangle=0,                 # rotation of ticklabels
                                   dtick=1,                     # radial tick increment
                                   ticklabelstep=10,             # tick label increment
                                   gridcolor='#d6d6d6',         # tick line color
                                   griddash='dot',              # tick line style
                                   linecolor='#d6d6d6',         # axis color
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
        
        height = kwargs.pop('height', 820)                      # canvas height
        
        legend = kwargs.pop('legend',                           # legend settings
                            dict(                        
                                 bgcolor="#d8e4f5",             # legend bg color
                                 xanchor="right",               # reference for x pos
                                 x=1.05,                        # legend x pos percentage
                                 yanchor="top",                 # reference for x pos
                                 y=1.05,                        # legend y pos percentage
                                 font=dict(size=20),            # font size
                                 itemsizing='constant',         # symbol size to not depend on data traces
                                 itemwidth=30,                  # symbol size
                                 )
                            )
        
        margin = kwargs.pop('margin',                           # margins
                            dict(                               
                                b=30,
                                t=70,
                                l=40,
                                r=60,
                                ),
                            )
        
        title = kwargs.pop('title',                             # title settings
                           dict(                                
                               text=alltitle,                   # title text
                               x = 0.03,                        # title position
                               y = 0.98,
                               font=dict(
                                   size = 24,                   # title font size
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


