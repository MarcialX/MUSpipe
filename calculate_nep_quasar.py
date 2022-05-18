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
from scipy.signal import savgol_filter
from scipy import integrate

from tqdm import tqdm
from matplotlib.pyplot import *

# LaTEX format
rc('text', usetex=True)
rc('font', family='serif', size='20')

from misc.msg_custom import *
import MuscatChannel as MC


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

# Assuming total flux at 1.1 mm of 11.8 Jy for 3C279
flux_3c279 = 11.8

# Get 3C279 flux
def quasar_flux(wavelength, flux_ref, meas=None):

	# Measurement
	if meas is None:
		ang = 206265*1.22*wavelength/D
	else:
		ang = meas

	# Define timeline
	t = np.linspace(-5*np.pi, 5*np.pi, 5000)
	# Beam
	beam = np.sinc(t/ang)**2
	int_beam = integrate.simps(beam, t)
	beam = beam/int_beam

	# Quasar aperture
	# Point source
	source = signal.unit_impulse(5000, 'mid')
	int_source = integrate.simps(source, t)
	source = flux_ref*source/int_source

	# Get convolution
	conv = signal.convolve(source, beam, mode='same') / sum(beam)

	# plot(t, source, label='3C279. Total flux=11.8Jy')
	# plot(t, beam, label='LMT beam (sinc function)')
	# plot(t, conv, label='Convolutioned source')

	max_flux = np.max(conv)

	# Print results
	print('Total flux at '+str(wavelength)+': '+str(flux_ref))
	print('Max flux after convolution: '+str(max_flux))

	return max_flux


# Channels
chns = ['clones', 'sith', 'hope', 'empire', 'jedi']

# Object
obj = '3C279'
obs_num = '094468'
sub_obs = '00_0001'
root_path = '../data/'

# Define channel
chn = 'clones'

print('Channel: ', chn)

# Paths where the files are defined
# Time stream data
stream_path = root_path+obs_num+'/'+sub_obs+'/'+chn

# Observation data
obs_file = 'tel_muscat_2022-02-25_094468_00_0001.nc'
obs_path = root_path+obs_num+'/'+sub_obs+'/'+obs_file

# KIDs to analyse
kids = 'all'
#kids = 'K005'

# 94468 bad kids
# clones
bad_kids = [b'K002', b'K003', b'K011', b'K020', b'K021', b'K044',
			b'K048', b'K060', b'K063', b'K071', b'K073', b'K075',
			b'K076', b'K077', b'K078', b'K079', b'K084', b'K085',
			b'K091', b'K103', b'K105', b'K106', b'K107', b'K110',
			b'K113', b'K116', b'K117', b'K119', b'K121', b'K122',
			b'K124', b'K125', b'K127', b'K138', b'K144', b'K149',
			b'K179', b'K186', b'K187', b'K219', b'K223', b'K239',
			b'K240', b'K244']

# # sith 
# bad_kids = [b'K004', b'K005', b'K009', b'K010', b'K016', b'K017',
# 			b'K018', b'K019', b'K020', b'K021', b'K027', b'K028',
# 			b'K029', b'K032', b'K033', b'K034', b'K045', b'K046',
# 			b'K047', b'K049', b'K043', b'K080', b'K082', b'K102',
# 			b'K121', b'K168', b'K169', b'K185', b'K186', b'K187',
# 			b'K215', b'K216']

# # empire
# bad_kids = [b'K106', b'K107', b'K112', b'K134', b'K135', b'K153',
# 			b'K172', b'K201', b'K203']


# # jedi
# bad_kids = [b'K057', b'K058', b'K059', b'K062', b'K069', b'K070',
# 			b'K074', b'K075', b'K078', b'K083', b'K124', b'K136',
# 			b'K139', b'K153', b'K158', b'K160', b'K161', b'K162',
# 			b'K163', b'K176'] 

# Create observation object
obs = MB.MapBuilding(stream_path, obs_path, kids=kids, bad_kids=bad_kids)

# Spike filter to all the kids
obs.spikes_filter(win_size=80, peak_pts=4)

# Apply PCA
obs.apply_pca()

# Second spikes filter
obs.spikes_filter(win_size=200)

# Get sample frequency
fs = obs.HW_KEYS['inst']['muscat']['sample']

# Collecting the SNR by channel
results = {}

# Measure FWHM by Andreas
fwhm = 6.5	# arcsec

# Quasar flux [Jy]
qso_flux = quasar_flux(l, flux_3c279, meas=fwhm)
#qso_flux = 11.8

# Check KIDs one by one
for kid in obs.kids:
	if not kid in obs.bad_kids:
		print('------------------------')
		print(kid)

		if not np.isnan(obs.SNR[kid]['snr']):

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
			print ('PSD Noise[Hz] {0:.2f}'.format(np.sqrt(fs*obs.SNR[kid]['psd'])))

			# Get signal
			signal = f0*obs.SNR[kid]['sig']
			print ('Signal[Hz]: {0:.2f}'.format(signal))

			# Responsivity
			S = signal/qso_flux
			print('Responsivity [Hz/Jy]: {0:.2f}'.format(S))

			# Get NEFD
			NEFD = noise/S/np.sqrt(fs)
			#NEFD = np.sqrt(obs.SNR[kid]['psd'])/S
			print ('NEFD[mJy/Hz^-1/2]: {0:.2f}'.format(1e3*NEFD))

			# To get NEP
			NEP = 1e-26*A*dv*NEFD/n
			print('NEP[W/Hz^-1/2]: {0:.2e}'.format(NEP))


			if NEP > 1e-16:
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
				msg('---------->>> '+str(kid), 'fail')

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

xb = hist(nep, bins=logbins, histtype='step', color='red', lw=2, label=chn)
max_bin = np.argmax(xb[0])

med = logbins[max_bin]+(logbins[max_bin+1]-logbins[max_bin])/2
axvline(med, color='k')
text(med, xb[0][max_bin]/2, 'S: {0:.2e}'.format(med), fontsize=14)


title(r'NEP-'+obj+'-'+obs_num+'-'+chn)
xscale('log')
xlabel(r'S [Hz/Jy]')
ylabel(r'Number of detectors')
grid(True, which="both", ls="-")
legend()


# # Plotting the SNR
# figure()

# snr = []
# for kid in results.keys():
# 	snr.append(results[kid]['snr'])

# nbins = 30

# xb = hist(snr, bins=nbins, histtype='step', color='blue', lw=2, label=chn)
# max_bin = np.argmax(xb[0])

# med = logbins[max_bin]+(logbins[max_bin+1]-logbins[max_bin])/2
# #axvline(med, color='k')
# #text(med, xb[0][max_bin]/2, 'SNR: {0:.2f}'.format(med), fontsize=14)


# title(r'SNR-'+obj+'-'+obs_num+'-'+chn)
# xscale('log')
# xlabel(r'SNR')
# ylabel(r'Number of detectors')
# #grid(True, which="both", ls="-")
# grid()
# legend()


# # Plotting the NEFD
# figure()

# nefd = []
# for kid in results.keys():
# 	nefd.append(1e3*results[kid]['nefd'])

# nbins = 30
# logbins = np.logspace(np.log10(np.min(nefd)),np.log10(np.max(nefd)),30)

# xb = hist(nefd, bins=logbins, histtype='step', color='blue', lw=2, label=chn)
# max_bin = np.argmax(xb[0])

# med = logbins[max_bin]+(logbins[max_bin+1]-logbins[max_bin])/2
# #axvline(med, color='k')
# #text(med, xb[0][max_bin]/2, 'SNR: {0:.2f}'.format(med), fontsize=14)


# title(r'NEFD-'+obj+'-'+obs_num+'-'+chn)
# xscale('log')
# xlabel(r'NEFD')
# ylabel(r'Number of detectors')
# grid(True, which="both", ls="-")
# #grid()
# legend()



# # Search for Glitches
# start = 0
# end = 220000

# for kid in obs.kids:
	
# 	df = obs.pro_data[kid][b'df'][start:end]

# 	baseline = savgol_filter(df, 25001, 3)
# 	sig_clean = df - baseline

# 	nstd = np.std(sig_clean[:3000])

# 	#peaks, _ = find_peaks(data_array, distance=dist, height=height)

# 	a = np.where(np.abs(sig_clean) > 7*nstd)[0]

# 	if len(a) > 1000:

# 		print(kid)

# 		#figure()
# 		#plot(df)
# 		#plot(df[a])

# Bad KIDs, or weird response after 220000 points

# b'K055'
# b'K056'
# b'K057'
# b'K058'
# b'K059'
# b'K064'
# b'K065'
# b'K075'
# b'K076'
# b'K080'
# b'K104'
# b'K111'
# b'K116'
# b'K117'
# b'K122'
# b'K126'
# b'K127'
# b'K128'
# b'K133'
# b'K134'
# b'K135'
# b'K136'
# b'K137'
# b'K139'
# b'K140'
# b'K141'
# b'K142'
# b'K192'
# b'K223'
