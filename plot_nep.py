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
ion()

# LaTEX format
rc('text', usetex=True)
rc('font', family='serif', size='20')


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


# Channels
chns = ['clones', 'sith', 'empire', 'jedi']
colors = ['r', 'b', 'c', 'm']

# Object
obj = '3C279'
obs_num = '094468'
root_path = './'


# Read and plot all data
figure()
title(r'Noise-'+obj+'-'+obs_num)
cmap = cm.get_cmap('Spectral')
colors = np.linspace(0, 1, 4)

nbins = 30
logbins = np.logspace(np.log10(2.4),np.log10(100),nbins)

for i, chn in enumerate(chns):
    # Read data
    data = np.load(root_path+obj+'-'+obs_num+'-'+chn+'.npy', allow_pickle=True).item()

    nep = []
    for kid in data.keys():
    	nep.append(data[kid]['noise'])

    # Opción 2
    if i == 0:
        ax = subplot(4,1,i+1)
    else:
        subplot(4,1,i+1, sharex=ax, sharey=ax)

    # Opción 1
    xb = hist(nep, bins=logbins, color=cmap(colors[i]), lw=2, label=chn)
    #max_bin = np.argmax(xb[0])

    #med = logbins[max_bin]+(logbins[max_bin+1]-logbins[max_bin])/2
    #axvline(med, color='k')
    #text(med, xb[0][max_bin]/2, 'NEP: {0:.2e}'.format(med), fontsize=14)

    if i != 3:
        tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    else:
        xlabel(r'Noise [Hz]')

    xscale('log')
    ylabel(r'\# of detectors')
    grid(True, which="both", ls="-")
    legend()
