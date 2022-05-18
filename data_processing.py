# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# data_processing.py
# Functions to process time streams data
#
# Marcial Becerril, @ 16 January 2022
# Latest Revision: 16 Jan 2022, 21:49 GMT-6
#
# TODO list:
# Functions missing:
#	+ Savgol filter
#	+ PCA filter
#	+ High/low-pass filter
#	+ Stop band filters (60 Hz signal, PTC, etc.)
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import numpy as _np
import time

from tqdm import tqdm
from random import seed
from random import random

from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

from matplotlib.pyplot import *

from misc.msg_custom import *



def cr_filter(stream, win_size=100, sigma_thresh=4, peak_pts=5, verbose=True):
	"""
		Cosmic Ray filter
		Remove glitches and likely cosmic ray events.
		Parameters
        ----------
        Input:
            stream : array
                Time stream array
            win_size : int
            	Window size filter
            sigma_thresh : float
            	Sigma threshold
        Output:
            stream_filter : array
                Time stream data filtered
        ----------
	"""

	cr_events = 0
	data_len = len(stream)
	
	stream_filter = _np.copy(stream)
	
	if verbose:
		msg('Starting Cosmic Ray removal procedure...', 'info')

	prev_cr_pts = _np.array([])

	start_time = time.time()
	check_time = time.time()

	while True:
		# Derivate data
		data_win_diff = _np.diff(stream_filter)
		# Sigma diff
		sigma_diff = _np.std(data_win_diff)
		# Mean diff
		offset_diff = _np.mean(data_win_diff)

		# Cosmic ray events
		cr_idx = _np.where((data_win_diff > offset_diff+sigma_thresh*sigma_diff) | 
						   (data_win_diff < offset_diff-sigma_thresh*sigma_diff) )[0]

		num_cr_pts = len(cr_idx)

		if check_time - start_time > 10:
			break

		if num_cr_pts <= 0 or _np.array_equal(prev_cr_pts, cr_idx):
			break

		if verbose:
			msg('Cosmic ray events: '+str(num_cr_pts), 'info')

		# Get statistics per each point
		for cr in cr_idx:
			data_win = stream_filter[cr-int(win_size/2):cr+int(win_size/2)]
			sort_data = _np.sort(data_win)
			edge_data = sort_data[:int(3*win_size/4)]
			# Sigma window
			sigma = _np.std(edge_data)
			# Data offset
			offset = _np.mean(edge_data)
			# Validate
			cr_rec = _np.where((data_win > offset+sigma_thresh*sigma) | 
						   (data_win < offset-sigma_thresh*sigma) )[0]

			diff_cr_rec = _np.diff(cr_rec)

			if _np.count_nonzero(diff_cr_rec == 1) < peak_pts:
				# Replace point
				# ----------------
				# New random points with normal distribution
				new_sample = _np.random.normal(offset, sigma)
				# Update points
				stream_filter[cr] = new_sample
				# ----------------

		check_time = time.time()

		prev_cr_pts = cr_idx

	if verbose:
		msg('Done', 'info')

	return stream_filter, cr_events


def pro_sv_filter(stream, *args, **kwargs):
	"""
		Through Savinsky-Golay algorithm this function substrate the baseline.
	"""

	# Key arguments
	# ----------------------------------------------
	# Distance between lines
	window = kwargs.pop('window', 501)
	# Minimum height for the lines
	order = kwargs.pop('order', 3)
	# Linewidth definition
	# 	- # of points for one linewdith
	lw = kwargs.pop('lw', 50)
	#	- Signal linewidths
	#	- Noise linewidths
	sg_lw = kwargs.pop('sg_lw', 4)
	ns_lw = kwargs.pop('ns_lw', 2)
	# ----------------------------------------------

	# 1. Find the peaks associated to a detection
	# Data has to be free os spikes signals!
	height = savgol_filter(stream, 1001, 3)
	scale = 6*_np.nanstd(stream[:100]) + height

	peaks = find_profile_peaks(stream, height=scale, **kwargs)
	print(len(peaks), ' peaks detected')

	# 2. Remove those detected signals
	noise_data = extract_noise(stream, peaks, lw=lw, sg_lw=sg_lw, ns_lw=ns_lw)

	# 3. Get baseline from the noise
	base_noise_signal = savgol_filter(noise_data, window, order)

	noise_data = noise_data - base_noise_signal

	# 4. Remove baseline from the signal
	signal_data = stream - base_noise_signal

	return noise_data, signal_data


def extract_noise(stream, peaks, *args, **kwargs):
	"""
		Extracting noise
	"""

	# Key arguments
	# ----------------------------------------------
	# Linewidth definition
	# 	- # of points for one linewdith
	lw = kwargs.pop('lw', 50)
	#	- Signal linewidths
	#	- Noise linewidths
	sg_lw = kwargs.pop('sg_lw', 4)
	ns_lw = kwargs.pop('ns_lw', 2)
	# ----------------------------------------------

	# Noise processing
	noise_data = _np.copy(stream)

	from_sm = peaks - int(lw*(sg_lw+ns_lw)/2)
	to_sm = peaks + int(lw*(sg_lw+ns_lw)/2)

	from_sg = peaks - int(lw*sg_lw/2)
	to_sg = peaks + int(lw*sg_lw/2)

	for i, peak in enumerate(peaks):
		# a. Get sample
		sample = stream[from_sm[i]:to_sm[i]]
		x = _np.arange(from_sm[i], to_sm[i])

		# b. Remove signal and fit a line with the noise
		no_signal = _np.concatenate(( sample[:from_sg[i]-from_sm[i]], sample[to_sg[i]-from_sm[i]:] ))
		x_no_signal = _np.concatenate(( x[:from_sg[i]-from_sm[i]], x[to_sg[i]-from_sm[i]:] ))

		if len(no_signal) == len(x_no_signal):
			# Get sample baseline
			z = np.polyfit(x_no_signal, no_signal, 1)
			p = np.poly1d(z)

			sample_base = p(x_no_signal)

			# c. Replace signal data with noise
			corr_base = no_signal - sample_base

			mean_ns = _np.nanmean(corr_base)
			std_ns = _np.nanstd(corr_base)

			# Generate the replacement signal
			noise_signal = _np.random.normal(mean_ns, std_ns, to_sg[i]-from_sg[i])
			x_signal = _np.arange(from_sg[i], to_sg[i])

			signal_base = p(x_signal)

			# Replace sample
			noise_data[from_sg[i]:to_sg[i]] = noise_signal + signal_base

	return noise_data


def edge_noise(stream, *args, **kwargs):
	"""
		Get noise from edges of detections.
	"""

	# Key arguments
	# ----------------------------------------------
	# Distance between lines
	window = kwargs.pop('window', 1001)
	# Minimum height for the lines
	order = kwargs.pop('order', 3)
	# Linewidth definition
	# 	- # of points for one linewdith
	lw = kwargs.pop('lw', 50)
	#	- Signal linewidths
	#	- Noise linewidths
	sg_lw = kwargs.pop('sg_lw', 4)
	ns_lw = kwargs.pop('ns_lw', 4)
	# ----------------------------------------------

	# 1. Find the peaks associated to a detection
	# Data has to be free os spikes signals!
	height = savgol_filter(stream, window, order)
	scale = 6*_np.nanstd(stream[:100]) + height

	peaks = find_profile_peaks(stream, height=scale, **kwargs)
	print(len(peaks), ' peaks detected')

	# 2. Remove those detected signals
	noise_data = _np.copy(stream)
	
	from_sm = peaks - int(lw*(sg_lw+ns_lw)/2)
	to_sm = peaks + int(lw*(sg_lw+ns_lw)/2)

	from_sg = peaks - int(lw*sg_lw/2)
	to_sg = peaks + int(lw*sg_lw/2)

	noise = np.array([])
	for i, peak in enumerate(peaks):
		# a. Get sample to get the noise
		sample = np.concatenate((stream[from_sm[i]:from_sg[i]], stream[to_sm[i]:to_sg[i]]))
		noise = np.concatenate((noise, sample))

	return noise


def find_profile_peaks(data_array, **kwargs):
    """
        Find profile peaks
        Parameters
        ----------
            data_array : array
            dist : float
                Minimum distance between peaks
            height_div : float
                Factor which divide the maximum value of the array, to define the
                minimum peak detectable
        ----------
    """

    # Find peaks keyword parameters
    # Distance between lines
    dist = kwargs.pop('dist', 250.0)
    # Minimum height for the lines
    height = kwargs.pop('height', 10.0)

    # Height division
    peaks, _ = find_peaks(data_array, distance=dist, height=height)

    return peaks


def cleanPca(inData, nComps):
	"""
		PCA cleaning. Designed and coded by Andreas Papageorgiou
		Parameters
		----------
		inData: 2d array
			Input data from several KIDs
		nComps: float
			Number of components to substract
	    ----------
	"""
	nDet = inData.shape[0]
	tmpStd = np.nanstd(inData,axis=1)
	tmpStd[tmpStd == 0] = 1
	inData = (inData.T/tmpStd).T

	pca = PCA(n_components=nComps, svd_solver='full')
	
	t0 = time.time()
	Y = pca.fit(inData.T)
	t1 = time.time()
	print('end pca', "%.2fs"%(t1-t0))
	
	loadings = pca.components_.T
	components =  Y.fit_transform(inData.T)
	print('loadings', loadings.shape)
	print('components', components.shape)

	for i in range(nComps):
		print('cleaning comp', i, end='\r')
		for j in range(nDet):
			inData[j, :]-= components[:,i]*loadings[j,i]
	print('cleaning comp', i)
	#inData -= corr.T
	inData = (inData.T*tmpStd).T
	#inData = inData.T
	return inData


def gaussian(x, A, mu, sigma, offset):
    """
        Gaussian function
        Parameters
        ----------
        x : int/float/array
        A : float
            Amplitude
        mu : float
            Mean
        sigma : float
            Dispersion
        offset : float
            Offset
        ----------
    """
    return offset+A*np.exp(-((x-mu)**2)/(2*sigma**2))


def gaussian2d(x, y, A, x0, y0, sigma, offset):
    """
        Gaussian 2d function
        Parameters
        ----------
        x : int/float/array
        y : int/float/array
        A : float
            Amplitude
        x0 : float
            Mean along x-axis
        y0 : float
            Mean along y-axis
        sigma : float
            Dispersion
        offset : float
            Offset
        ----------
    """
    g2d = offset+A*np.exp(-((x - x0)**2 / (2*sigma**2) + (y - y0)**2 / (2*sigma**2)) )
    return g2d



