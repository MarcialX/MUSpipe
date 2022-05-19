# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# TimeStreamObs.py
# Functions to read Sweeps and Time Streams files from MUSCAT observations
#
# Marcial Becerril, @ 12 January 2022
# Latest Revision: 12 Jan 2022, 17:15 GMT-6
#
# TODO list:
# Functions missing:
#   + Convert data to FITS files
#   + Get sweep data
#       - As dictionary?
#   + Analyse sweep data: Fit resonator model: Get F0, Qi, Qc, Qr.
#   + Get TimeStream data [Done 17012022]
#   + Plot Sweep data: Sweep, IQ circles, ts_point
#   + Get PSD, noise (NEP)
#   + Fit data curves to PSD
#   + Plot NEP noise and time streams
#   + Filter Cosmic-Rays
#   + Apply PCA
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import os as _os
import numpy as _np

import cmath as _cmath
import pygetdata as _gd

import yaml
import collections

import lmfit
from lmfit.models import ConstantModel

from scipy.signal import savgol_filter
from scipy import signal
from scipy import interpolate
from scipy.signal import find_peaks

from misc.msg_custom import *
from misc.timeout import timeout
from resonator_model import *


class TimeStreamObs(object):
    """
        Time Stream Observation object for a given channel.
        It contains all the methods to read, analyse and plot time stream data.
        Parameters
        ----------
        directory : string
            Directory path of the observation
        ----------
    """
    def __init__(self, directory, *args, **kwargs):

        # Key arguments
        # ----------------------------------------------

        # ----------------------------------------------

        # Loading available fields
        try:
            with open('./var/data_fields.yaml') as file:
                self.FIELDS = yaml.safe_load(file)

        except Exception as e:
            print('Fail loading keys. '+str(e), 'fail')

        # Read time stream dirfile
        self.data_dirfile = _gd.dirfile(directory, _gd.RDONLY)

        # Get some file propierties
        self.dirfile = self.data_dirfile.name

        # Get KID's names
        self.kids_stream = self.data_dirfile.get_sarray(self.FIELDS['gral']['names'])

        # Get tones list
        self.lo = self.data_dirfile.get_constant(self.FIELDS['gral']['lo'])
        self.tones = self.data_dirfile.get_carray(self.FIELDS['gral']['tones'])

        # Get observation time
        self.time = self.data_dirfile.getdata(self.FIELDS['gral']['py_time'])
        self.fine_time = self.data_dirfile.getdata(self.FIELDS['gral']['fine_time'])
        self.pps = self.data_dirfile.getdata(self.FIELDS['gral']['pps'])

        # Get data fields
        raw_fields = self.data_dirfile.field_list()
        self.raw_fields = self._get_fields(raw_fields)

        # Cryostat data
        self.sensors = {
            'MC1_Built_In': self.data_dirfile.getdata(self.FIELDS['gral']['m1']),
            'MC2_Cald': self.data_dirfile.getdata(self.FIELDS['gral']['m2'])
        }

        # Sweep path
        self.sweep_path = self.data_dirfile.get_string('sweep.'+self.FIELDS['sweep']['file'])


    def sweep(self, keys='f-s21', kids='all'):
        """
            Get sweep data
            Parameters
            ----------
            Input:
                keys : string
                    String chain of keys to read.
                    Every key is separated by a '-'
                    Example: 'f-s21-s21_cal' -> Read frequency, sweep and calibration sweep
                    Keys are available in /var/data_fields.yaml
            Output:
                kids : string/list
                    List of KIDs to read
            ----------
        """
        meta = b'sweep'
        meta_str = meta.decode('utf-8')

        # Validate KIDs list
        kids_list = self._validate_kids(kids)

        # Sweeping fields to read
        sweep_keys = self._parse_keys(keys, meta_str)

        if b'f_s21' in sweep_keys:
            # Get frequency parameters
            lo_freqs = self.data_dirfile.get_carray(meta + b'.' + b'lo_freqs')
            bb_freqs = self.data_dirfile.get_carray(meta + b'.' + b'bb_freqs')

        # Sweeping data as required in 'keys'
        sweep_data = {}

        for n, kid in enumerate(kids_list):
            # Check the rest of the required fields
            sweep_data[kid] = {}
            for key in sweep_keys:
                # Frequencies
                if key == b'f_s21':
                    freqs = lo_freqs + bb_freqs[n]
                    sweep_data[kid][b'f'] = freqs
                # Other data
                else:
                    sweep_data[kid][key] = self.data_dirfile.get_carray(meta+b'.'+kid)

        return sweep_data


    def time_stream(self, keys='I-Q-df', kids='all'):
        """
            Get time stream data
            Parameters
            ----------
            Input:
                keys : string
                    String chain of keys to read.
                    Every key is separated by a '-'
                    Example: 't-df' -> Read time and fractional-df
                    Keys are available in /var/data_fields.yaml
            Output:
                kids : string/list
                    List of KIDs to read
            ----------
        """

        # As time stream metadata is empty
        meta = b''
        meta_str = 'time-stream'

        # Validate KIDs list
        kids_list = self._validate_kids(kids)

        # Stream fields to read
        stream_keys = self._parse_keys(keys, meta_str)

        # Stream data as required in 'keys'
        stream_data = {}

        for n, kid in enumerate(kids_list):
            # Check the rest of the required fields
            stream_data[kid] = {}
            for key in stream_keys:
                #print(meta+kid+b'_'+key)
                stream_data[kid][key] = self.data_dirfile.getdata(meta+kid+b'_'+key)

        return stream_data


    def fit_s21(self, s21, method='nonlinear', **kwargs):
        """
            Fit S21 paramter of resonators
            Parameters
            ----------
            s21 : dict
                S21 and frequency for each KID
            method : string
                Fit method: linear/non-linear
            ----------
        """

        # Key arguments
        # ----------------------------------------------
        # Add verbose
        verbose = kwargs.pop('verbose', False)
        # ----------------------------------------------

        # Validate it is a method available
        if not method in ['linear', 'nonlinear']:
            msg('Method not valid', 'error')
            return

        # Create directory
        kid_params = {}

        # Define the constant model one
        one = lmfit.Model(constant_model)

        prefix = "mus0_"

        for kid in s21.keys():

            msg('======================================', 'info')
            msg(str(kid), 'info')
            msg('======================================', 'info')

            if b'f' in list(s21[kid].keys()) and b'sweep' in list(s21[kid].keys()):

                # Extract frequency and S21 params
                freq_kid = s21[kid][b'f']
                s21_kid = s21[kid][b'sweep']

                # Check if data is adjustable
                # 1. Check if the peaks are positive
                # delta_height = np.max(-1*np.abs(s21_kid)) - np.min(-1*np.abs(s21_kid))
                # peaks, _ = find_peaks(-1*np.abs(s21_kid), distance=len(freq_kid)/4, height=delta_height/3+np.min(-1*np.abs(s21_kid)) )

                # if len(peaks) == 1:
                #     if np.mean(np.abs(s21_kid)) >  np.abs(s21_kid)[peaks[0]]:

                try:
                    result = self._perform_fit(method, freq_kid, s21_kid, **kwargs)

                    # # Plot them
                    # subplot(121)
                    # plot(fit_s21.real, fit_s21.imag, 'b')
                    # plot(guess_s21.real, guess_s21.imag, 'r')
                    # plot(s21_kid.real, s21_kid.imag, 'k')
                    # plot(x_center, y_center, 'ko')
                    # plot(x_guess, y_guess, 'ro')
                    # subplot(122)
                    # plot(np.abs(fit_s21), 'b')
                    # plot(np.abs(guess_s21), 'r')

                    # Extract results
                    kid_params[kid] = {}
                    kid_params[kid]['params'] = {}
                    kid_params[kid]['params']['A'] = result.values[prefix+'A']

                    kid_params[kid]['params']['f0'] = result.values[prefix+'f0']
                    kid_params[kid]['params']['alpha'] = result.values[prefix+'alpha']
                    kid_params[kid]['params']['Qr'] = result.values[prefix+'Qr']
                    kid_params[kid]['params']['Qc_real'] = result.values[prefix+'Qc_real']
                    kid_params[kid]['params']['Qc_imag'] = result.values[prefix+'Qc_imag']

                    # Extract error
                    kid_params[kid]['error'] = {}
                    kid_params[kid]['error']['A'] = result.params[prefix+'A'].stderr

                    kid_params[kid]['error']['f0'] = result.params[prefix+'f0'].stderr
                    kid_params[kid]['error']['alpha'] = result.params[prefix+'alpha'].stderr
                    kid_params[kid]['error']['Qr'] = result.params[prefix+'Qr'].stderr
                    kid_params[kid]['error']['Qc_real'] = result.params[prefix+'Qc_real'].stderr
                    kid_params[kid]['error']['Qc_imag'] = result.params[prefix+'Qc_imag'].stderr

                    if method == 'nonlinear':
                        kid_params[kid]['params']['a'] = result.values[prefix+'a']
                        kid_params[kid]['error']['a'] = result.params[prefix+'a'].stderr

                    if verbose:
                        msg("A = {:.2f}".format(kid_params[kid]['params']['A']), 'ok')
                        msg("f0 = {:.2f} Hz".format(kid_params[kid]['params']['f0']), 'ok')
                        msg("alpha = {:.2f}°".format(kid_params[kid]['params']['alpha']*180/_np.pi), 'ok')
                        msg("Qr = {:.0f}".format(kid_params[kid]['params']['Qr']), 'ok')
                        msg("Qc = {:.0f} + {:.0f}j".format(kid_params[kid]['params']['Qc_real'], kid_params[kid]['params']['Qc_imag']), 'ok')

                        if method == 'nonlinear':
                            msg("a = {:.3f}".format(kid_params[kid]['params']['a']), 'ok')

                except Exception as e:
                    msg(str(e)+'\nResonator not fitted. It might be a noisy resonator...', 'warn')

            else:
                msg('There is not enough data to fit', 'fail')

        return kid_params


    def interactive_mode(self, sweep, chn, q=25e3, lw=7, method='qfactor'):

        self._lw = lw
        self._q = q

        self._method = method

        self.flag_inter = False

        self.kid_man = {}
        self.num_man = 0

        self._fig = figure()
        self._ax = self._fig.add_subplot(111)

        self._freqs = sweep[0].real
        self._s21 = sweep[1]

        self._ax.plot(self._freqs, 20*np.log10(np.abs(self._s21)), 'r')

        for kid in chn.keys():
            self._ax.plot(chn[kid]['freq'], 20*np.log10(np.abs(chn[kid]['s21'])), 'k--', lw=1)
            self._ax.text(chn[kid]['freq'][30], np.mean(20*np.log10(np.abs(chn[kid]['s21']))), kid)

        self._ax.set_xlabel(r'Frequency[Hz]')
        self._ax.set_ylabel(r'Amplitude')

        self._onclick_xy = self._fig.canvas.mpl_connect('button_press_event', self._onclick)
        self._keyboard = self._fig.canvas.mpl_connect('key_press_event', self._key_pressed)


    def _key_pressed(self, event):
        """
            Keyboard event to save/discard line fitting changes
        """
        sys.stdout.flush()
        if event.key in ['x', 'q', 'w']:
            if event.key == 'x':

                msg('Changes saved', 'ok')
                self._fig.canvas.mpl_disconnect(self._onclick_xy)
                close(self._fig)
            elif event.key == 'q':

                self.kid_man = {}
                self.num_man = 0
                msg('No changes to the fitting', 'warn')
                self._fig.canvas.mpl_disconnect(self._onclick_xy)
                close(self._fig)

            elif event.key == 'w':
                self.flag_inter = not self.flag_inter
                if self.flag_inter:
                    msg('START', 'ok')
                else:
                    msg('STOP', 'warn')


    def _onclick(self, event):
        """
            On click event to select lines
        """
        if event.inaxes == self._ax:
            # Left-click
            if event.button == 1:
                if self.flag_inter:
                    ix, iy = event.xdata, event.ydata
                    # Add detectors
                    xarg = np.where(self._freqs>ix)[0][0]
                    self._ax.axvline(ix)

                    kid_name = 'M'+str(self.num_man).zfill(4)

                    step = np.mean(np.diff(self._freqs))
                    max_f = np.max(self._freqs)
                    one_lw = ix/self._q

                    width = int(self._lw*one_lw/step)

                    start = xarg-width
                    end = xarg+width

                    if self._method == 'intermediate':
                        # First element. Start span takes the value of the end span
                        if i == 0:
                            span_end = (xarg[i+1] - xarg[i])/2
                            span_start = -span_end
                        # Last element. End span takes the value of the start span
                        elif i == len(xarg)-1:
                            span_start = -(xarg[i] - xarg[i-1])/2
                            span_end = -span_start
                        else:
                            span_start = -(xarg[i] - xarg[i-1])/2
                            span_end = (xarg[i+1] - xarg[i])/2
                    # Selection by q factor
                    elif self._method == 'qfactor':
                        span = self._lw*freq[xarg]/self._q
                        span_start = -int(span/2./step)
                        span_end = int(span/2./step)

                    start = xarg + span_start
                    end = xarg + span_end

                    if start < 0:
                        start = 0
                    if end > len(freq)-1:
                        end = len(freq)-1

                    freq_kid = self._freqs[start:end]
                    s21_kid = self._s21[start:end]

                    self._ax.plot(freq_kid, 20*np.log10(np.abs(s21_kid)), 'k--')
                    self._ax.text(freq_kid[30], np.mean(20*np.log10(np.abs(s21_kid))), kid_name)

                    self.kid_man[kid_name] = {}
                    self.kid_man[kid_name]['freq'] = freq_kid
                    self.kid_man[kid_name]['s21'] = s21_kid

                    self.num_man += 1
                    self._fig.canvas.draw_idle()


    def _validate_kids(self, kids):
        """
            Validate kids list.
            Parameters
            ----------
            Input:
                kids : string/list
                    List of KIDs to get information.
                    As a single string 'K000' or '000'.
                    As an array ['K000', 1, '25']
            Output:
                kids_list : list
                    List of KIDs available
            ----------
        """

        kids_list = []

        if isinstance(kids, list) or isinstance(kids, _np.ndarray):
            for kid in kids:
                kid = bytes(str(kid), 'utf-8')
                if kid.isdigit():
                    kids_list.append(b'K'+kid.zfill(3))
                else:
                    kids_list.append(kid)
        else:
            if isinstance(kids, int) or isinstance(kids, float):
                kids = str(int(kids))

            kids = bytes(kids, 'utf-8')
            if kids == b'all':
                kids_list = self.kids_stream
            elif kids.isdigit():
                kids_list = [b'K'+kids.zfill(3)]
            else:
                kids_list.append(kids)

        # Check if KIDs are available in the data
        for kid in kids_list:
            if not kid in self.kids_stream:
                kids_list.remove(kid)
                msg(str(kid)+' is not available.', 'warn')

        return kids_list


    @timeout(2)
    def _perform_fit(self, method, freq_kid, s21_kid, **kwargs):
        """
            Apply the resonator model to the data
            Parameters
            ----------
            method : string
                Method to implement
            freq_kid : array
                Frequency
            s21_kid : array
                S21 parameter
            ----------
        """

        # Create the model
        model_res = ResonatorModel(method)

        f0_guess = int(len(s21_kid)/2)+1
        pars = model_res.guess(freq_kid, s21_kid, f0_guess, **kwargs)

        # Evaluating the guessing
        guess_s21 = model_res.eval(params=pars, f=freq_kid)

        # Getting the fit
        result = model_res.fit(s21_kid, params=pars, f=freq_kid)

        # Evaluate the results
        fit_s21 = model_res.eval(params=result.params, f=freq_kid)

        return result


    def _parse_keys(self, keys, metadata):
        """
            Parse keys.
            ----------
            Input:
                keys : string
                    String chain with the keys parameters
                metadata : string
                    Data origin: sweep, time_stream or general
            Output:
                keys_parsed : list
                    List of keys available
            ----------
        """

        keys_array = keys.split('-')

        keys_parsed = []
        for key in keys_array:
            # Get keys
            if key in self.FIELDS[metadata].keys():
                key_vars = self.FIELDS[metadata][key]
                key_vars = bytes(key_vars, 'utf-8')
                keys_parsed.append(key_vars)
            else:
                msg(key+' is not available.', 'warn')

        return keys_parsed


    def _get_fields(self, raw_fields):
        """
            Get field names.
            It does not include the fields + KID's names.
            Parameters
            ----------
            Input:
                raw_fields : list
                    Raw field list
            Output:
                fields : list
                    Filter field list
            ----------
        """

        fields = []
        for field in raw_fields:

            filter_field = field

            # Check if the field has a KID number
            if b'K' in field:
                k_idx = field.index(b'K')
                len_num = min(k_idx+4, len(field))
                k_num = field[k_idx+1:len_num]
                if k_num.isdigit():
                    if k_idx > 0 and (field[k_idx-1] == 46 or field[k_idx-1] == 95):
                        filter_field = field[:k_idx-1]+field[len_num:]
                    elif field[len_num] == 46 or field[len_num] == 95:
                        max_char = min(k_idx+5, len(field)-1)
                        filter_field = field[:k_idx]+field[max_char:]
                    else:
                        filter_field = field[:k_idx]+field[len_num:]

            if not filter_field in fields:
                fields.append(filter_field)

        return fields


    # A N A L Y S I S   F U N C T I O N S
    # ----------------------------------------------------
    def get_dxdf(self, freqs, x, **kwargs):
        """
            Get dx/df, where x could be i or q
            Parameters
            ----------
            Input:
                freqs : array
                    Frequencies
                x : array
                    I or Q signals
            Output:
                dx_df : array
                    I or Q Gradient
            ----------
        """

        # Key arguments
        # ----------------------------------------------
        # Smoothing the I/Q signals. Using Savinsky-Golay filter
        smooth = kwargs.pop('smooth', True)
        # Order
        order = kwargs.pop('order', 3)
        # Number of points
        npoints = kwargs.pop('npoints', 15)
        # ----------------------------------------------

        if smooth:
            x = savgol_filter(x, npoints, order)

        dx_df = _np.gradient(x)/_np.gradient(freqs)

        return dx_df


    def get_df(self, kids, **kwargs):
        """
            Get df. Displacemente of the resonance frequency
            Parameters
            ----------
            Input:
                kids : array
                    kids to analise
                x : array
                    I or Q signals
            Output:
                df : dict
                    df for each KID in the input.
            ----------
        """

        kids_avail = self._validate_kids(kids)

        # Get sweep data
        s21 = self.sweep(keys='f-s21', kids=kids)
        # Get IQ data
        IQ = self.time_stream(keys='I-Q', kids=kids)
        # Define df
        dfs = {}

        for kid in s21.keys():

            # Get frequencies
            fs = s21[kid][b'f']

            # Get f0
            f0 = int(len(fs)/2)

            I_sweep = s21[kid][b'sweep'].real
            Q_sweep = s21[kid][b'sweep'].imag

            I = IQ[kid][b'I']
            Q = IQ[kid][b'Q']

            didf = self.get_dxdf(fs, I_sweep, **kwargs)
            dqdf = self.get_dxdf(fs, Q_sweep, **kwargs)

            # First get the speed magnitude
            speed_mag = didf[f0]**2 + dqdf[f0]**2

            # Calculate the df
            df = (((I_sweep[f0]-I)*didf[f0]) + ((Q_sweep[f0]-Q)*dqdf[f0])) / speed_mag

            dfs[kid] = df/fs[f0]

        return dfs


    def get_phase(self, cplx_IQ, deg=False):
        """
            Get the phase from I,Q signals
        """
        phase = _cmath.phase(cplx_IQ)
        if deg:
            return phase*180/_np.pi
        else:
            return phase


    def get_speed(self, freqs, cplx_IQ):
        """
            Get the speed from I,Q signals
        """
        di_df = _np.gradient(cplx_IQ.real)/_np.gradient(freqs)
        dq_df = _np.gradient(cplx_IQ.imag)/_np.gradient(freqs)

        speed = _np.sqrt(di_df**2 + dq_df**2)

        return freqs, speed


    def get_psd(self, stream, kids='all', Fs=488.28125):
        """
            Get Power Spectral Density
            Parameters
            ----------
            Input:
                stream : data
                    Stream data
                keys : string
                    String chain of keys to read.
                    Every key is separated by a '-'
                    Example: 'f-s21' -> Read frequency and sweep
                    Keys are available in /var/data_fields.yaml
                Fs : float
                    Frequency sample
            Output:
                psd : dict
                    Power Spectral Density for each KID
            ----------
        """

        # As time stream metadata is empty
        meta = b''
        meta_str = 'time-stream'

        # Validate KIDs list
        kids_list = self._validate_kids(kids)

        # Stream data as required in 'keys'
        psd = {}

        for kid in kids_list:

            if kid in stream.keys():
                # A new PSD is defined
                psd[kid] = {}
                # Check if the stream data
                if b'df' in stream[kid].keys():
                    # Get f0
                    f0 = self.tones[int(kid[1:])]
                    # Power Spectral Density
                    freq, psd_kid = signal.periodogram(f0*stream[kid][b'df'], Fs)
                    psd_kid = 10*_np.log10(psd_kid)

                    psd[kid]['psd'] = psd_kid
                    psd[kid]['freq'] = freq

        return psd




# Obtener y colocar en algun sitio los datos mas importantes de cada sweep y timestream,
# como f0, step, npoints, fs, etc.
