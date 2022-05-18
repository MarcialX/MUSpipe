# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# MapBuilding.py
# Functions to create and analise maps
#
# Marcial Becerril, @ 17 January 2022
# Latest Revision: 17 Jan 2022, 01:42 GMT-6
#
# TODO list:
# Functions missing:
#   + Create maps
#   + Save maps as a FITS image
#   + Analyse beam mapping
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
import copy

import yaml

from datetime import datetime
from matplotlib.pyplot import *
ion()

from tqdm import tqdm
import data_processing as dp
from scipy.signal import savgol_filter
from scipy import signal
from scipy.optimize import curve_fit

from TimeStreamObs import TimeStreamObs
from TelescopeObs import TelescopeObs
from misc.msg_custom import *


# Loading instrument characteristics
try:
    with open('./var/general_params.yaml') as file:
        GENERAL_PARAMS = yaml.safe_load(file)
        
        TELESCOPE = GENERAL_PARAMS['telescope']
        CAMERA = GENERAL_PARAMS['camera']

        ANALYSIS_PATH = GENERAL_PARAMS['analysis_path']
        LOGS_PATH = GENERAL_PARAMS['logs_path']

except Exception as e:
    print('Fail loading general parameters. '+str(e), 'fail')

import logging
logging.basicConfig(level=logging.INFO, filename=ANALYSIS_PATH+LOGS_PATH,  filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


STATUSMODEL = {
    'despiking': {
        'active': False,
        'win_size': None,
        'sigma_thresh': None,
        'peak_pts': None
    },
    'decorrelation': {
        'active': False,
        'ncomps': None
    }
}



class MuscatChannel(TimeStreamObs):
    """
        MUSCAT Channel object for a given channel.
        It contains the high-level methods to analyse channel data
        Parameters
        ----------
        stream_dir : string
            Time Stream directory path
        channel : string
            MUSCAT channel
        obs : string
            Telescope observation number
        sub_obs : string
            Sub-observation? (ask to Kamal, David S. or Andreas P.)
        scan_rate : float
            Telescope scan rate [rad/s]
        kids : string/list
            Set of kids to map.
            If kids = [], the class is instanced without any kid, useful to load previuos analysis
        bad_kids : list
            Flagged KIDs
        ----------
    """
    def __init__(self, stream_dir, channel, obs, sub_obs='00_0001', scan_rate=0.0005, kids='K000', bad_kids=[], keys='df', *args, **kwargs):

        # Key arguments
        # ----------------------------------------------
        # Processing paths
        self.analysis_path = kwargs.pop('analysis_path', ANALYSIS_PATH)
        # Verbose
        verbose = kwargs.pop('verbose', True)
        # Telescope
        self.tel_id = kwargs.pop('tel', TELESCOPE)
        # Instrument
        self.cam_id = kwargs.pop('cam', CAMERA)
        # ----------------------------------------------

        # Paths
        self.stream_path = stream_dir+'/'+obs+'/'+sub_obs+'/'+channel

        # Inheritance the Time Stream data
        TimeStreamObs.__init__(self, self.stream_path)

        # Loading telescope and camera parameters
        try:
            with open('./var/map_fields.yaml') as file:
                self.HW_KEYS = yaml.safe_load(file)[self.tel_id]

        except Exception as e:
            print('Fail loading keys. '+str(e), 'fail')

        # Channel
        self.channel = channel

        # Status variables
        self.status = {}

        # Processing path
        self.obs_num = obs 
        name_path = obs+'-'+sub_obs

        if not _os.path.isdir(self.analysis_path+'/processing/'):
            _os.mkdir(self.analysis_path+'/processing/')

        if not _os.path.isdir(self.analysis_path + '/processing/'+name_path):
            _os.mkdir(self.analysis_path+'/processing/'+name_path)

        self.process_path = self.analysis_path + '/processing/'+name_path

        # Raw and processed data
        self.raw_data = {}
        self.data = {}
        self.add_kids(kids=kids, keys=keys, mode='w')

        # Name of analysis
        now =  datetime.now()
        self.process_name = now.strftime("%Y%m%d-%H%M%S")

        # Interpolation data
        self.az_inter = _np.array([])
        self.el_inter = _np.array([])

        # Define the beam size in number of stream points
        # Telescope resolution [arcsec]
        self.res = 206265*1.22*self.HW_KEYS['inst'][self.cam_id]['wavelength']/self.HW_KEYS['diameter']
        # Scan rate [arcsec/s]
        self.scan_rate =scan_rate * 3600 * 180 / _np.pi
        self.line_width = self.HW_KEYS['inst'][self.cam_id]['sample'] * self.res / self.scan_rate

        self.bad_kids = bad_kids

        # SNR parameters
        self.SNR = {}
        self.noise = {}



    # F I L E   F U N C T I O N S
    # -------------------------------------------------------------
    def add_kids(self, kids, keys='I-Q-df', mode='a'):
        """
            Add KIDs to the analysis.
            Parameters
            ----------
            kids : string/list
                Set of kids to map
            keys : string
                Keys chain of variables to get
            mode : string
                Adding mode: 
                'w': Writting. If the kid already exists, overwrites
                'a': Appending. Append kid/key
            ----------
        """

        raw_data = self.time_stream(keys=keys, kids=kids)

        for kid in raw_data.keys():
            # Writting mode
            if mode == 'w':
                # Clean KID field
                self.raw_data[kid] = {}
                self.raw_data[kid] = raw_data[kid]
                self.data[kid] = self.raw_data[kid].copy()
                self.status[kid] = copy.deepcopy(STATUSMODEL)
            # Append mode
            elif mode == 'a':
                if not kid in self.raw_data.keys():
                    # Clean KID field
                    self.raw_data[kid] = {}
                    self.raw_data[kid] = raw_data[kid]
                    self.data[kid] = self.raw_data[kid].copy()
                    self.status[kid] = copy.deepcopy(STATUSMODEL)
                else:
                    for key in raw_data[kid].keys():
                        if not key in self.raw_data[kid].keys():
                            self.raw_data[kid][key] = raw_data[kid][key]
                            self.data[kid][key] = self.raw_data[kid][key].copy()        

        # KIDs to analyse
        self.kids = self.data.keys()


    def new_analysis(self, name=None):
        """
            Start a new analysis.
            Parameters
            ----------
            name(opt) : string
                Name for a new processed data.
                If name is None, it generates one based on the date
            ----------
        """

        # Saving the previuos analysis
        self.save_analysis(name=self.process_name)

        self._create_process_name(name)

        # Reset a new analysis
        self.data = self.raw_data
        self.status = {}
        for kid in self.kids():
            self.status[kid] = copy.deepcopy(STATUSMODEL)

        msg('New data analysis '+self.process_name, 'ok')
        logging.info('Processing data path updated: '+self.process_name)


    def reset_analysis(self, kids=None, keys=None):
        """ 
            Reset the processing data.
            Parameters
            ----------
            kids : list of string
                Kids to reset
            keys : list of string
                Keys to reset
            ----------
        """

        msg('Reset data analysis '+self.process_name, 'info')
        logging.info('Reseting data analysis '+self.process_name) 

        if kids is None:
            kids = self.kids 

        for kid in kids:
            if keys is None:
                check_keys = self.data[kid].keys()
            else:
                check_keys = keys

            self.status[kid] = {}
            self.status[kid] = copy.deepcopy(STATUSMODEL)

            # Reset all the kids and keys
            for key in check_keys:
                # Check if kid and key are bytes
                if not isinstance(kid, bytes):
                    kid = bytes(kid, 'utf-8')
                if not isinstance(key, bytes):
                    key = bytes(key, 'utf-8')

                self.data[kid][key] = self.raw_data[kid][key]

                logging.info('Reset '+str(kid) )

        msg('Data reset :)', 'ok')  


    def save_analysis(self, name=None):
        """
            Save processed data.
            Parameters
            ----------
            name(opt) : string
                Name for a new processed data.
                If name is None, it generates one based on the date
            ----------
        """

        # If there is not name create one
        if not name is None:
            self._create_process_name(name)

        elif self.process_name == '':
            self._create_process_name()

        # Processing data name
        process_name = self.process_path+'/'+self.process_name
        
        msg('Saving processing data '+self.process_name, 'info')

        try:
            # Save data
            _np.save(process_name, self.data)
            _np.save(process_name+'-status', self.status)

            msg('Done', 'ok')
            logging.info(process_name+' saved')

        except Exception as e:
            msg('An error ocurred saving data.\n'+str(e), 'fail')
            logging.error(process_name+' not saved')


    def load_analysis(self, name):
        """
            Load proccessed data.
            Parameters
            ----------
            name(opt) : string
                Name of the data to load
            ----------
        """
        msg('Loading processed data', 'info')

        if name is None:
            pro_data = self.process_path+'/'+self.process_name
        else:
            pro_data = self.process_path+'/'+name

        # Adding npy ending
        pro_data = pro_data + '.npy'

        if _os.path.isfile(pro_data):
            self.data = _np.load(pro_data, allow_pickle=True).item()
            self.status = _np.load(pro_data[:-4]+'-status.npy', allow_pickle=True).item()

            msg('Done', 'ok')
            logging.info(pro_data+' loaded')

        else:
            msg('Processed data not loaded', 'warn')
            logging.info(pro_data+' not loaded')

    # -------------------------------------------------------------



    # HIGH-LEVEL FUNCTIONS
    # -------------------------------------------------------------
    def despiking(self, param=[b'df'], *args, **kwargs):
        """
            Filter spikes due to electrical features or cosmic rays
            Parameters
            ----------
            param : list
                Parameters to despike
            ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Window size
        win_size = kwargs.pop('win_size', 150)
        # Sigma Threshold
        sigma_thresh = kwargs.pop('sigma_thresh', 4)
        # Peaks points
        peak_pts = kwargs.pop('peak_pts', 3)
        # Verbose
        verbose = kwargs.pop('verbose', False)
        # ---------------------------------------------

        msg('Spikes filtering...', 'info')

        # Perform cosmic ray filter per each kid detector
        for kid in tqdm(self.kids, desc='Removing Cosmic Rays'):
            if not kid in self.bad_kids:
                for pm in self.data[kid].keys():
                    self.data[kid][pm] = dp.cr_filter(self.data[kid][pm], win_size=win_size, 
                                                  sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)[0]

                    # Update status
                    self.status[kid]['despiking']['active'] = True
                    self.status[kid]['despiking']['win_size'] = win_size
                    self.status[kid]['despiking']['sigma_thresh'] = sigma_thresh
                    self.status[kid]['despiking']['peak_pts'] = peak_pts

                    logging.info(kid.decode('utf-8')+' ['+pm.decode('utf-8')+'] '+' spikes filtered')

        msg('Done', 'ok')
        logging.info('Spikes filtering done')


    def rm_baseline(self, param=[b'df'], *args, **kwargs):
        """
            Remove baseline through Savinsky-Golay filter.
            It also estimates the SNR for all the signals detected
            ----------
            param : list
                Parameters to remove the baseline
            ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Verbose
        verbose = kwargs.pop('verbose', False)
        # ---------------------------------------------

        msg('Removing baseline...', 'info')
        msg('NOTE. This method applies a Savinsky-Golay filter to suppress atmosphere, there is no decorrelation analysis', 'info')

        for kid in tqdm(self.kids, desc='Substracting baseline'):
            for pm in self.data[kid].keys():

                noise, signal_noise = dp.pro_sv_filter(self.data[kid][pm], **kwargs)
                
                self.data[kid][pm] = signal_noise
                
                self.noise[kid] = {}
                self.noise[kid][pm] = noise

                logging.info(kid.decode('utf-8')+ ' baseline substracted')

        # Calculate SNR
        self.get_snr(*args, **kwargs)

        msg('Done', 'ok')
        logging.info('Baseline removed :)')


    def apply_pca(self, param=b'df', *args, **kwargs):
        """
            Clean signal through PCA(Principal Component Analysis) algorithm.
            It also estimates the SNR for all the signals detected
            Parameters
            ----------
            param : string
                Parameter to apply PCA filter
            ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Verbose
        verbose = kwargs.pop('verbose', False)
        # Number of components
        comps = kwargs.pop('comps', 6)
        # Window Savinsky golay filter
        window = kwargs.pop('window', 2500)
        # Size of window to get noise
        noise_win = kwargs.pop('noise_win', 20)
        #   - Signal linewidths
        #   - Noise linewidths
        sg_lw = kwargs.pop('sg_lw', 4)
        ns_lw = kwargs.pop('ns_lw', 2)
        # Noise Method
        noise_method = kwargs.pop('nm', 'psd')
        # Region
        #   - Define a region to analise
        #region = kwargs.pop('region', 220000) # Para clones
        #region = kwargs.pop('region', 0) # Para sith
        region = kwargs.pop('region', 20000) # Para empire
        # ---------------------------------------------

        msg('Applying PCA...', 'info')

        nokid = []
        detector_data = []
        for kid in self.kids:
            if (not np.isnan(np.nanmean(self.data[kid][param]))) and (not kid in self.bad_kids):
                detector_data.append(self.data[kid][param][region:])#[int(self.HW_KEYS['inst'][self.cam_id]['sample']):])                
            else:
                nokid.append(kid)

        detector_data = np.array(detector_data)
        dims = np.shape(detector_data)
        if dims[0] < 2:
            msg('PCA not applied. It needs more than one KID to work', 'fail')
            return

        # Applying PCA cleaning
        signal_clean = dp.cleanPca(detector_data, comps)

        # Second check for peaks detection
        i = 0
        for kid in tqdm(self.kids, desc='Strongest signal selection'):

            # Update status
            self.status[kid]['decorrelation']['active'] = True
            self.status[kid]['decorrelation']['ncomps'] = comps

            self.SNR[kid] = {}
            self.noise[kid] = {}

            if not kid in nokid:
                # Check every single spike
                new_peaks = []
                N_std = _np.nanstd(signal_clean[i])
                mu_std = _np.nanmean(signal_clean[i])
                sig_peaks = dp.find_profile_peaks(signal_clean[i], distance=self.line_width*5, height=6*N_std+mu_std)
                
                for s in sig_peaks:
                    # Get sample
                    sm = signal_clean[i][s-int((noise_win)*self.line_width):s+int((noise_win)*self.line_width)]
                    # Get base level
                    lvl_sm = _np.mean(_np.concatenate((sm[:int(2*self.line_width/3)], sm[-int(2*self.line_width/3):])))

                    # Check if the width correspond to a real line
                    hp = (signal_clean[i][s]-lvl_sm)/2. + lvl_sm

                    high_idx = _np.where(sm>hp)[0]
                    diff_high = _np.diff(high_idx)

                    if np.count_nonzero(diff_high == 1) >= self.line_width/3:
                        new_peaks.append(s)

                if len(new_peaks) > 0:
                    max_peaks = _np.argmax(signal_clean[i][new_peaks])
                    sig_peaks = new_peaks[max_peaks]
                    
                    # Get noise
                    noise = _np.concatenate((signal_clean[i][sig_peaks-noise_win*int(self.line_width):sig_peaks-int((1*noise_win/3)*self.line_width)], 
                            signal_clean[i][(sig_peaks+int((1*noise_win/3)*self.line_width)):sig_peaks+noise_win*int(self.line_width)]))

                    N = _np.nanstd(noise)
                    mu = _np.nanmean(noise)

                    sample = signal_clean[i][sig_peaks-int((1*noise_win/3)*self.line_width):sig_peaks+int((1*noise_win/3)*self.line_width)]
                    time_sample = np.arange(len(sample))

                    # Fit a Gaussian curve on detection
                    popt, pcov = curve_fit(dp.gaussian, time_sample, sample, p0=[signal_clean[i][sig_peaks]-mu,
                                len(time_sample)/2, self.line_width, mu])

                    # --------------------
                    # Extracting the noise
                    noise_data = dp.extract_noise(signal_clean[i], np.array(new_peaks), sg_lw=sg_lw, ns_lw=ns_lw, lw=self.line_width)
                    f0 = self.tones[int(kid[1:])]

                    freq, psd = signal.periodogram(f0*noise_data, self.HW_KEYS['inst'][self.cam_id]['sample'])
                    avg_noise_psd = np.mean(psd[np.where((freq >= 95) & (freq <= 142))[0]])
                    
                    # D E B U G G I N G
                    # -----------------------
                    # if kid == b'K001':
                    #     plot(time_sample, sample, 'c', lw=3, label='Subregion selected')
                    #     plot(time_sample, dp.gaussian(time_sample, *popt), 'k--', lw=3, label='Fit curve')
                    #     legend()
                    #     return

                    # if kid == b'K235':
                    #     figure()
                    #     semilogx(freq[3:], psd[3:])
                    #     semilogx(freq[np.where((freq >= 95) & (freq <= 142))[0]], psd[np.where((freq >= 95) & (freq <= 142))[0]])
                    #     axhline(avg_noise_psd, color='r')
                        
                    #     figure()
                    #     plot(noise_data)
                    #     plot(signal_clean[i])
                    # ----------------------

                    self.data[kid][param] = signal_clean[i]
                    self.noise[kid][param] = noise

                    self.SNR[kid]['peaks'] = [sig_peaks]
                    self.SNR[kid]['std'] = N

                    signal_amp = popt[0]
                    # self.SNR[kid]['snr'] = (signal_clean[i][sig_peaks]-mu)/N
                    self.SNR[kid]['snr'] = signal_amp/N

                    self.SNR[kid]['sig'] = signal_amp
                    self.SNR[kid]['psd'] = avg_noise_psd 

                else:
                    self.SNR[kid]['peaks'] = [np.nan]
                    self.SNR[kid]['std'] = np.nan
                    self.SNR[kid]['snr'] = np.nan

                    self.SNR[kid]['sig'] = np.nan
                    self.SNR[kid]['psd'] = np.nan

                i += 1

            else:
                self.SNR[kid]['peaks'] = [np.nan]
                self.SNR[kid]['std'] = np.nan
                self.SNR[kid]['snr'] = np.nan

                self.SNR[kid]['sig'] = np.nan
                self.SNR[kid]['psd'] = np.nan

        msg('Done', 'ok')
        logging.info('PCA filtering done')


    def get_snr(self, *args, **kwargs):
        """
            Get the signal to noise ratio for the KIDs available.
        """

        for kid in self.kids:
            for pm in self.data[kid].keys():

                # Calculate SNR
                # Noise
                # ---------------------------------------
                noise = self.noise[kid][pm]
                N = _np.nanstd(noise)
                mean_noise = _np.nanmean(noise)

                # Signal
                signal_noise = self.data[kid][pm]
                sig_idx = dp.find_profile_peaks(signal_noise, distance=self.line_width*5, height=6*N+_np.nanmean(noise))

                self.SNR[kid] = {}
                self.SNR[kid]['peaks'] = sig_idx
                self.SNR[kid]['std'] = N 
                self.SNR[kid]['snr'] = signal_noise[sig_idx]/N
                # ---------------------------------------


    def interp_obs_times(self):
        """
            Match telescope and MUSCAT times.
            This function is OBSOLETE as a new and improved method for mapping is about to be released.
        """

        msg('This function is OBSOLETE!', 'warn')
        msg('Matching observation points to time stream data', 'info')

        # Get data limits 
        stx = _np.where(self.time>self.map_time[0])[0][0]
        enx = _np.where(self.time<self.map_time[-1])[0][-1]

        self.stx = stx 
        self.enx = enx

        # Interpolated rray coordinates
        self.az_inter = _np.zeros(enx-stx+1)
        self.el_inter = _np.zeros(enx-stx+1)
        
        for i in tqdm(range(len(self.map_time)-1)):
            # Find the time period to adjust
            start_idx = _np.where(self.time>self.map_time[i])[0][0]
            end_idx = _np.where(self.time<self.map_time[i+1])[0][-1]

            tm_array = self.time[start_idx:end_idx+1]

            # Fractional time points
            tm_array = tm_array - self.map_time[i]
            frac_tm_array = tm_array/(self.map_time[i+1]-self.map_time[i])

            # Get azimuth
            lon_az = self.az_map[i+1] - self.az_map[i]
            fill_az = lon_az*frac_tm_array + self.az_map[i]

            self.az_inter[start_idx-stx:end_idx-stx+1] = fill_az

            # Get elevation
            # Linear inetrpolation
            m_el = (self.el_map[i+1]-self.el_map[i])/(self.az_map[i+1]-self.az_map[i])
            b = self.el_map[i] - m_el*self.az_map[i]
            fill_el = m_el*fill_az + b

            self.el_inter[start_idx-stx:end_idx-stx+1] = fill_el

        msg('Done', 'ok')
        logging.info('Match telescope and stream time points')


    def create_map(self, kids=None, grid_size=(70, 70), save=False):
        """
            Create maps from observations
            This function is OBSOLETE as a new and improved method for mapping is about to be released.
            Parameters
            ----------
            kids : string/list
                Set of kids to map
            grid_size : tuple
                Grid size of the map
            save : boolean
                Save Map as a image
            ----------
        """

        msg('This function is OBSOLETE!', 'warn')
        msg('Creating map for one pixel', 'info')

        if kids == None:
            kid = self.kids

        for kid in kids:

            # Get map points
            x = self.az_inter
            y = self.el_inter
            z = self.data[kid][b'df'][self.stx:self.enx+1]

            # Grid size
            nx = grid_size[0]
            ny = grid_size[1]

            # Define max and min values
            xmin = _np.min(x)
            xmax = _np.max(x)
            ymin = _np.min(y)
            ymax = _np.max(y)

            xx = _np.linspace(xmin,xmax,nx+1)
            yy = _np.linspace(ymin,ymax,ny+1)

            mapa = _np.zeros((nx+1, ny+1)) #creo una matriz con ceros de dimensions nx y ny
            mapa_aux = [[[]for i in range(nx+1)] for j in range(ny+1)] #matriz de listas vacias de dimensiones nx*ny

            # Map dimensions
            wx = xmax-xmin
            wy = ymax-ymin

            # Correction dimensions
            xcorr = x-xmin
            ycorr = y-ymin

            # Sweep the positions along z-axis
            for i in range(len(z)):
                px = int(_np.floor((nx*(xcorr[i]))/wx)) # que valor tendra en grados
                py = int(_np.floor((ny*(ycorr[i]))/wy)) # redondea el valor a un entero mas bajo
                mapa_aux[px][py].append(z[i])           # agregamos al mapa auxiliar el valor de z de acuerdo a i, en la posicion px, py

            # Asignamos un valor promedio de todos los valores guardados en cada cajoncito
            #temp = []
            for i in range(nx):
                for j in range(ny):
                    avg = _np.mean(mapa_aux[i][j])
                    #temp.append(len(mapa_aux[i][j]))
                    mapa[i][j] = avg    #asignando el valor de average al mapa en la posicion i,j

            imshow(np.transpose(mapa))

            return mapa #, temp
    # -------------------------------------------------------------



    # H I D E   F U N C T I O N S
    # -------------------------------------------------------------
    def _create_process_name(self, name=None):
        """
            Create a new processing data path.
            Parameters
            ----------
            name(opt) : string
                Name for the processed data
                If name is None, it generates one based on the date
            ----------
        """
        
        if name is None:
            now =  datetime.now()
            self.process_name = now.strftime("%Y%m%d-%H%M%S")
        else:
            self.process_name = name

    # -------------------------------------------------------------

