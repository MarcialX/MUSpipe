# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# Scripts
#	Plot NEP
# Get the NEP from all the resonators
#
# Marcial Becerril, @ 28 February 2022
# Latest Revision: 28 Feb 2022, 23:20 GMT-6
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

from matplotlib.pyplot import *
ion()

# LaTEX format
rc('text', usetex=True)
rc('font', family='serif', size='20')

from misc.msg_custom import *


chns = ['clones', 'sith', 'empire', 'jedi']
colors = ['red', 'blue', 'cyan', 'magenta']

obj = '3C279'
obs_num = '094471'
root_path = './'

# Plotting the results
figure()

for i, chn in enumerate(chns):

	# Read channel
	path = root_path+obj+'-'+obs_num+'-'+chn+'.npy'
	results = np.load(path, allow_pickle=True).item()

	subplot(2,2,i+1)

	nep = []
	for kid in results.keys():
		nep.append(results[kid]['nep'])

	nbins = 30
	logbins = np.logspace(np.log10(np.min(nep)),np.log10(np.max(nep)),30)

	xb = hist(nep, bins=logbins, histtype='step', color=colors[i], lw=2, label=chn)
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


# Plotting One channel, Different observations
figure()

chn = 'empire'
obs_nums = ['094468', '094469', '094471']

nbins = 45
logbins = np.logspace(np.log10(1.75e-15),np.log10(8e-15),nbins)

for i, obs in enumerate(obs_nums):

	# Read channel
	path = root_path+obj+'-'+obs+'-'+chn+'.npy'
	results = np.load(path, allow_pickle=True).item()

	nep = []
	for kid in results.keys():
		nep.append(results[kid]['nep'])

	subplot(4,1,i+2)

	if i != 3:
		tick_params('x', labelbottom=False)
	if i == 0:
		title(r'NEP-'+obj+'-'+chn)
		ax = subplot(4,1,1)
	else:
		subplot(4,1,i+1, sharex=ax)

	xb = hist(nep, bins=logbins, color=colors[i], lw=2, label=obs, alpha=0.75)
	max_bin = np.argmax(xb[0])

	med = logbins[max_bin]+(logbins[max_bin+1]-logbins[max_bin])/2
	axvline(med, color='k', alpha=1)
	text(med, xb[0][max_bin]/2, 'NEP: {0:.2e}'.format(med), fontsize=14)

	axis([1.6e-15, 8e-15, -1, 25])

	xscale('log')
	xlabel(r'NEP [W/Hz$^{-1/2}$]')
	ylabel(r'Number of detectors')
	grid(True, which="both", ls="-")
	legend()
