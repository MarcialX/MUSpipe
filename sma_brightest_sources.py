# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# LMT BRIGTHEST SMA SOURCES
# Script to get the brightest sources from the SMA calibrator webpage:
# http://sma1.sma.hawaii.edu/callist/callist.html
#
# Marcial Becerril, @ 29 March 2022
# Latest Revision: 29 Mar 2022, 19:00 GMT-6
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import os as os
import numpy as np

import requests

from astropy import units as u
from astropy.coordinates import SkyCoord,EarthLocation, AltAz
from astropy.time import Time

from datetime import datetime

import argparse

from tqdm import tqdm
from matplotlib.pyplot import *


# SMA Catalog webpage
SMA_PATH = 'http://sma1.sma.hawaii.edu/callist/callist.html'

# LMT DATA
LMT = {
		"name"	: 'LMT',
		"lon"	: 18.985733,
		"lat"	: -97.314818,
		"alt"	: 4580,
		"utf"	: -5
}

# Telescope location object
location = EarthLocation(lat=LMT['lon']*u.deg, lon=LMT['lat']*u.deg, height=LMT['alt']*u.m)

# Defined colours
HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


# Get SMA sources
def get_sma_src(band='1mm', time=None, num_src=5, alt_range=[30, 85], display=True):
	"""
		Get SMA sources from the SMA calibrators catalog webpage
        Parameters
        ----------
        Input:
			band : string
				Flux density wavelength
			time : string
				Date and time of request.
				Format: y-m-d H:M:S
				If none time is shown, it takes the current time of system
			num_src : int
				Number of sources
			alt_range : array
				Altitude range
			display : bool
				Show results?
        ----------
	"""

	# Get time
	if time is None:
		now = datetime.now()
		current_time = now.strftime('%Y-%m-%d %H:%M:%S')
		obs_time = Time(current_time) - LMT['utf']*u.hour
	else:
		obs_time = Time(time)
	
	# Get SMA objects
	objs = read_webpage()

	objs_name = list(objs.keys())
	flux_source = np.zeros(len(objs_name))
	
	# Get az-alt coordinates of each source
	for i, obj in enumerate(objs_name):

		# Get coordinates of each object
		target = SkyCoord(objs[obj]['ra'], objs[obj]['dec'], unit=(u.hourangle, u.deg), frame='icrs')

		# Convert to horizontal coordinates
		alt_az_frame = AltAz(location=location, obstime=obs_time)
		target_alt_az = target.transform_to(alt_az_frame)

		objs[obj]['alt'] = target_alt_az.alt 
		objs[obj]['az'] = target_alt_az.az

		# Get source bright
		if band in objs[obj]['band'].keys():
			flux_source[i] = objs[obj]['band'][band]['flux']
		else:
			flux_source[i] = np.nan

		#print(target_alt_az.alt, target_alt_az.az)

	# Get the num_src brightest sources available
	i = 0
	cnt = 0
	most_bright = {}
	sort_flux = np.sort(flux_source)[::-1]
	while cnt < num_src:

		sort_flux = sort_flux[i:]

		if not np.isnan(sort_flux[0]):
		
			flux_idx = np.where(sort_flux[0] == flux_source)[0][0]
			obj_sel = objs_name[flux_idx]

			# Check altitude
			if (objs[obj_sel]['alt'].value >= alt_range[0]) and (objs[obj_sel]['alt'].value <= alt_range[1]):  
				most_bright[obj_sel] = objs[obj_sel]
				cnt += 1	

		i += 1

	# Display results
	if display:
		print_res(most_bright)

	return most_bright


# Print results
def print_res(data):
	"""
		Display results in a nice way
	"""
	print(BOLD+'-------------------------------'+ENDC)
	for name in data.keys():
		print('Object: '+UNDERLINE+name+ENDC)
		print('Alias: '+BOLD+data[name]['name']+ENDC)
		print('RA [hrs]: '+BLUE+data[name]['ra']+ENDC)
		print('DEC [deg]: '+BLUE+data[name]['dec']+ENDC)
		band = list(data[name]['band'].keys())[0]
		print('Flux density('+band+') [Jy]: '+RED+str(data[name]['band'][band]['flux'])+ENDC)
		print('AZ [deg]: '+YELLOW+str(data[name]['az'].value)+ENDC)
		print('ALT [deg]: '+YELLOW+str(data[name]['alt'].value)+ENDC)
		print(BOLD+'-------------------------------'+ENDC)


# Read webpage as a dictionary
def read_webpage():
	"""
		Read webpage
	"""

	# Get HTML data
	sma_response = requests.post(SMA_PATH)
	sma_full_text = sma_response.text

	# Get table content
	TABLE_ID = 'class="cals"'

	from_idx = sma_full_text.find(TABLE_ID)
	to_idx = sma_full_text[from_idx:].find('/table')

	sma_table = sma_full_text[from_idx:to_idx]

	# Read row by row
	data = {}
	obj = ['']*9
	start_row = 0
	add_band = 0

	while start_row >= 0:

		start_row = sma_table.find('valign="middle"')
		end_row = sma_table[start_row:].find('/tr')

		obj_info = sma_table[start_row+1:start_row+end_row].split('\n')

		if len(obj_info) <= 6:
			add_band = 4
		else:
			add_band = 0

		c = 0
		for f in obj_info:
			if '<td' in f:
				start_data = f.find('>')
				end_data = f[start_data:].find('<')

				row = f[start_data+1:start_data+end_data]
				obj[c+add_band] = row
				c += 1

		# Source name
		if not (obj[1] in data.keys()):
			data[obj[1]] = {}
			
			idx_name = obj[0].find(';')
			if idx_name > 0:
				obj[0] = obj[0][idx_name+1:]

			data[obj[1]]['name'] = obj[0]
		
		# Coordinates
		data[obj[1]]['ra'] = obj[2].replace(' ', '')
		data[obj[1]]['dec'] = obj[3].replace(' ', '')

		# Band data
		header = ['date', 'obs', 'flux', 'error']

		if not 'band' in data[obj[1]].keys():
			data[obj[1]]['band'] = {}

		band = obj[4].replace('&micro;', 'u')
		data[obj[1]]['band'][band] = {}
		for i, k in enumerate(obj[5:-2]):
			data[obj[1]]['band'][band][header[i]] = k

		# Separate flux and error
		cnt = 0
		for j in obj[-2].split(' '):
			if j.replace('.', '', 1).isdigit():
				data[obj[1]]['band'][band][header[cnt+2]] = float(j)
				cnt += 1

		sma_table = sma_table[end_row:]

	return data


# MAIN PROGRAM

# Get any argument defined
parser = argparse.ArgumentParser()
parser.add_argument('--band', '-b', help="wavelength band", type=str, required=False)
parser.add_argument('--number', '-n', help="number of sources to request", type=int, required=False)
parser.add_argument('--time', '-t', help="time to request", type=str, required=False)
parser.add_argument('--display', '-d', help="display results?", type=bool, required=False)

args = parser.parse_args()

band = args.band
num_src = args.number
time = args.time 
display = args.display

if band is None:
	band = '1mm'

if num_src is None:
	num_src = 5

if display is None:
	display = True 

most_bright = get_sma_src(band=band, num_src=num_src, display=display)

