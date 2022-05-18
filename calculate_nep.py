# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# Scripts
#	Calculate the NEP, from source observations.
# Get the NEP from all the resonators
#
# Marcial Becerril, @ 24 January 2022
# Latest Revision: 24 Jan 2022, 03:03 GMT-6
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

from scipy.optimize import curve_fit
from scipy import interpolate
import scipy.stats as stats
from scipy import signal
from scipy import integrate

from tqdm import tqdm
from matplotlib.pyplot import *

# LaTEX format
rc('text', usetex=True)
rc('font', family='serif', size='20')

from misc.msg_custom import *
import MapBuilding as MB 


# Defining dome constants
h = 6.62607e-34     # Planck constant
c = 299.792458e6    # Light speed [m/s]
K = 1.38064e-23     # Stefan-Boltzmann constant

# Telescope diameter
D = 50
# Telescope area
A = np.pi*(D/2)**2	# Telescope collecting area

# MUSCAT parameters
l = 1.1e-3		# Wavelength [m]
dv = 50e9		# Bandwidth [Hz]
n = 2			# Usually of 2

# Blackbody function
def BB(l, T):
    S = (8*np.pi*h*c/l**5)*(1/(np.exp(h*c/(l*K*T)-1)))
    return S

# Second order equation
def pol2(x, a, b, c):
	Y = a*x**2 + b*x + c
	return Y

# Uranus size
uranus_size = 3.7	# [arcsec]

# Total flux from Uranus
wavelength_uranus = np.array([0.45e-3, 0.85e-3, 1.3e-3])
flux_uranus = np.array([190.24, 70.97, 39.03])

# Get Uranus flux
def uranus_flux(wavelength, size=uranus_size):
	# Fit a curve to Uranus data
	popt, pcov = curve_fit(pol2, wavelength_uranus, flux_uranus)

	# Get flux at wavelength
	S = pol2(wavelength, *popt)

	# Define timeline
	t = np.linspace(-5*np.pi, 5*np.pi, 5000)
	# Beam
	beam = np.sinc(t/(206265*1.22*wavelength/D))**2
	int_beam = integrate.simps(beam, t)
	beam = beam/int_beam

	# Uranus Aperture
	# Diameter 3.7"
	centered = 0
	width = size
	source = stats.uniform(loc=centered-width/2., scale=width)
	delta = t[1]-t[0]
	source = S*source.pdf(t)

	# Get convolution
	conv = signal.convolve(source, beam, mode='same') / sum(beam)

	max_flux = np.max(conv)

	# Print results
	print('Total flux at '+str(wavelength)+': '+str(S))
	print('Max flux after convolution: '+str(max_flux))

	return max_flux


# Get 3C279 flux
def quasar_flux(wavelength, flux_ref=11.8):

	# Define timeline
	t = np.linspace(-5*np.pi, 5*np.pi, 5000)
	# Beam
	beam = np.sinc(t/(206265*1.22*wavelength/D))**2
	int_beam = integrate.simps(beam, t)
	beam = beam/int_beam

	# Quasar aperture
	# Point source
	source = signal.unit_impulse(5000, 'mid')
	int_source = integrate.simps(source, t)
	source = flux_ref*source/int_source

	# Get convolution
	conv = signal.convolve(source, beam, mode='same') / sum(beam)

	max_flux = np.max(conv)

	# Print results
	print('Total flux at '+str(wavelength)+': '+str(flux_ref))
	print('Max flux after convolution: '+str(max_flux))

	return max_flux


# Channels
chns = ['clones', 'sith', 'hope', 'empire', 'jedi']

# Object
obj = 'Uranus'
obs_num = '92937'
root_path = '../data'


# Define channel
chn = 'empire'

print('Channel: ', chn)

# Paths where the files are defined
# Time stream data
stream_num = '20211217_012831'
stream_path = root_path+'/'+obj+'/'+obs_num+'/'+chn+'/'+stream_num

# Observation data
obs_file = 'tel_muscat_2021-12-17_092937_00_0001.nc'
obs_path = root_path+'/pointing/'+obs_file

# KIDs to analyse
kids = 'all'
#kids = 'K005'

# Create observation object
obs = MB.MapBuilding(stream_path, obs_path, kids=kids)

# Spike filter to all the kids
obs.spikes_filter()

# Remove baseline
obs.rm_baseline()

# [Optional] Second spike filtering with a extremely high threshold
for kid in obs.kids:
	# Check every single spike
	new_peaks = []
	new_snr = []
	for i, s in enumerate(obs.SNR[kid]['peaks']):
		# Check if the width correspond to a real line
		hp = obs.pro_data[kid][b'df'][s]/2.
		# Get sample
		sm = obs.pro_data[kid][b'df'][s-int(obs.line_width/2):s+int(obs.line_width/2)]

		high_idx = np.where(sm>hp)[0]
		diff_high = np.diff(high_idx)

		if np.count_nonzero(diff_high == 1) >= obs.line_width/4:
			new_peaks.append(s)
			new_snr.append(obs.SNR[kid]['snr'][i])

	obs.SNR[kid]['peaks'] = new_peaks
	obs.SNR[kid]['snr'] = new_snr

# Get sample frequency
fs = obs.HW_KEYS['inst']['muscat']['sample']

# Collecting the SNR by channel
results = {}

# Uranus flux [Jy]
ura_flux = uranus_flux(l)

# Check KIDs one by one
for kid in obs.kids:
	
	print('------------------------')
	print(kid)

	if len(obs.SNR[kid]['snr'])>0:

		# Get the maximum SNR
		snr = np.max(obs.SNR[kid]['snr'])
		print ('SNR: {0:.2f}'.format(snr))

		if snr > 70:
			msg(str(kid), 'warn')

		# Get frecuency tone
		f0 = obs.tones[int(kid[1:])]
		# Get noise
		noise = f0*obs.SNR[kid]['std']
		print ('Noise[Hz]: {0:.2f}'.format(noise))

		# Get signal
		signal = noise*snr 
		print ('Signal[Hz]: {0:.2f}'.format(signal))

		# Responsivity
		S = signal/ura_flux
		print('Responsivity [Hz/Jy]: {0:.2f}'.format(S))

		# Get NEFD
		NEFD = noise/S/np.sqrt(fs)
		print ('NEFD[mJy/Hz^-1/2]: {0:.2f}'.format(1e3*NEFD))

		# To get NEP
		NEP = 1e-26*A*dv*NEFD/n
		print('NEP[W/Hz^-1/2]: {0:.2e}'.format(NEP))

		# Save results
		results[kid] = {
			'snr': snr,
			'f0': f0,
			'noise': noise,
			'signal': signal,
			'respo': S,
			'nefd': NEFD,
			'nep': NEP
		}
	else:
		msg(str(kid)+' no detection!', 'fail')

# Save file
np.save(obj+'-'+obs_num+'-'+chn, results)
print('File '+obj+'-'+obs_num+'-'+chn+' saved')


# Plotting the results
figure()

nep = []
for kid in results.keys():
	nep.append(results[kid]['nep'])

nbins = 30
logbins = np.logspace(np.log10(np.min(nep)),np.log10(np.max(nep)),30)

xb = hist(nep, bins=logbins, histtype='step', color='magenta', lw=2, label=chn)
max_bin = np.argmax(xb[0])

med = logbins[max_bin]+(logbins[max_bin+1]-logbins[max_bin])/2
axvline(med, color='k')
text(med, xb[0][max_bin]/2, 'NEP: {0:.2e}'.format(med), fontsize=14)


title(r'NEP-'+obj+'-'+obs_num+'-'+chn)
xscale('log')
xlabel(r'NEP [W/Hz$^{-1/2}$]')
ylabel(r'Number of detectors')
grid(True, which="both", ls="-")
legend()