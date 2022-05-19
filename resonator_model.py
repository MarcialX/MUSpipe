# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Resonator Model
# resonator_model.py
# Fit resonator model using the LM library
#
# Marcial Becerril, @ 16 May 2022
# Latest Revision: 16 May 2022, 12:44 GMT-6
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #

import numpy as np
from matplotlib.pyplot import *
import collections

import lmfit
from lmfit.models import ConstantModel

from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

from misc.msg_custom import *



def linear_resonator(f, A, alpha, f0, Qr, Qc_real, Qc_imag):
    """
        Linear resonator model
        Parameters
        ----------
        f : float/array
            Frequency
        A : float
            Resonator amplitud
        alpha : float
            IQ circle rotation
        f0 : float
            Reosnance frequency
        Qr : float
            Quality factor
        Qc_real : float
            Coupling quality factor Real part
        Qc_imag : float
            Coupling quality factor Imaginary part
        ----------
    """

    # Grouping Qc components in a single complex value
    Qc = Qc_real + 1j*Qc_imag

    # Fractional frequency shift
    x = (f - f0)/f0

    # Resonator model
    S21 = (1 - (Qr/Qc)*(1/(1+2j*Qr*x)))
    S21_rot_x = S21.real*np.cos(alpha) - S21.imag*np.sin(alpha)
    S21_rot_y = S21.real*np.sin(alpha) + S21.imag*np.cos(alpha)

    return A*(S21_rot_x + 1j*S21_rot_y)


def nonlinear_resonator(f, A, alpha, f0, Qr, Qc_real, Qc_imag, a):
    """
        Non-linear resonator model
        Parameters
        ----------
        f : float/array
            Frequency
        A : float
            Resonator amplitud
        alpha : float
            IQ circle rotation
        f0 : float
            Reosnance frequency
        Qr : float
            Quality factor
        Qc_real : float
            Coupling quality factor Real part
        Qc_imag : float
            Coupling quality factor Imaginary part
        a : float
            Nonlinearity coefficient
        ----------
    """

    # Creating the Qc complex value
    Qc = Qc_real + 1j*Qc_imag

    # Fractional frequency shift of non-linear resonator
    y0s = Qr*(f - f0)/f0
    y = np.zeros_like(y0s)

    for i, y0 in enumerate(y0s):
        coeffs = [4.0, -4.0*y0, 1.0, -(y0+a)]
        y_roots = np.roots(coeffs)
        # Extract real roots only. From [ref Pete Code]
        # If all the roots are real, take the maximum
        y_single = y_roots[np.where(abs(y_roots.imag) < 1e-5)].real.max()
        y[i] = y_single

    S21 = (1 - (Qr/Qc)*(1/(1+2j*y)))
    S21_rot_x = S21.real*np.cos(alpha) - S21.imag*np.sin(alpha)
    S21_rot_y = S21.real*np.sin(alpha) + S21.imag*np.cos(alpha)

    S21_rot = S21_rot_x + 1j*S21_rot_y

    return A*S21_rot


def constant_model(f):
    """
        Constant model of ones
        Parameters
        ----------
        f : array
            Matrix shape to mimic with ones
        ----------
    """
    cte = np.ones_like(f)

    return cte


def amp_resonator(f, A):
    """
        Constant model of defined amplitude
        Parameters
        ----------
        f : array
            Matrix shape to mimic with ones
        ----------
    """
    cte = A*constant_model(f)

    return cte



class ResonatorModel(lmfit.model.Model):
    __doc__ = "resonator model" + lmfit.models.COMMON_DOC

    def __init__(self, model, prefix='mus0_', *args, **kwargs):
        """
            Resonator model object: linear or non-linear
            Parameters
            ----------
            model : string
                Resonator model type: linear or non-linear
            prefix : string
                Model prefix. 'mus0' by default
            ----------
        """

        # pass in the defining equation so the user doesn't have to later.
        if model == 'linear':
            super(ResonatorModel, self).__init__(linear_resonator, *args, **kwargs)
        elif model == 'nonlinear':
            super(ResonatorModel, self).__init__(nonlinear_resonator, *args, **kwargs)
        else:
            print('Model not valid')
            return

        self.model = model
        self.prefix = prefix


    def guess(self, f, data, idx_res, **kwargs):
        """
            Guessing resonator parameters
            Parameters
            ----------
            f : array
                Frequency vector
            data : array
                Resonator amplitudes as a funtion of frequency
            idx_res : float
                Index of the resonance frequency
            ----------
        """

        # Key arguments
        # ----------------------------------------------
        # Add verbose
        verbose = kwargs.pop('verbose', None)
        # ----------------------------------------------

        argmin_s21 = int(idx_res)
        fmin = f.min() # Lower frequency limit
        fmax = f.max() # Higher frequency limit

        # Guessing
        # Amplitude
        # Max value of data
        A_guess = np.abs(data).max()

        A_min = np.abs(data).min()
        A_max = 10*np.abs(data).max()       # Por qué 10?

        # Rotation of IQ circle
        y = (data[0].imag + data[-1].imag)/2.
        x = (data[0].real + data[-1].real)/2.
        alpha_guess = np.arctan2(y, x)      # Realmente funciona?

        # Resonance frequency
        # Resonance frequency close to the minimum of |S21|
        f0_guess = f[argmin_s21]

        # Quality factor Qr
        # Qr min es 10 veces menos que el valor que tendría un resonador si ocupara toda la muestra, de extremo a extremo
        Qr_min = 0.1*(f0_guess/(fmax-fmin))
        # Asumiendo que f esta en orden
        delta_f = np.diff(f)
        min_delta_f = delta_f[delta_f > 0].min()    # En caso que delta_f no sea constante
        # Qr max es el valor máximo posible que lograría la muestra, dada la resolución en frecuencia
        Qr_max = f0_guess/min_delta_f

        half_pwr = (np.max(np.abs(data)) - np.min(np.abs(data)))/2.
        down_half = np.where(np.abs(data) < (np.max(np.abs(data)) - half_pwr))[0]
        df = f[down_half[-1]] - f[down_half[0]]
        Qr_guess = 0.75*f0_guess/df

        # Qr es la media geométrica de los limites posibles
        #Qr_guess = np.sqrt(Qr_min*Qr_max)

        # Qc real (Aún no sé de donde proviene)
        Qc_real_guess = Qr_guess/(1-(np.abs(data[argmin_s21])/A_guess))
        # Qc imag assuming an ideal resonator
        Qc_imag_guess = 0.0

        # Nonlinearity parameter
        a_guess = 0.1
        a_max = 10.0    # Podría ser más chico?

        if verbose:
            print ("Model: ", self.prefix)
            print ("fmin=", fmin, " / fmax=", fmax, " / f0_guess=", f0_guess, "\nalpha_guess=", alpha_guess)
            print ("Qr_min=", Qr_min, " / Qr_max=", Qr_max, " / Qr_guess=", Qr_guess, "\nQc_real_guess=", Qc_real_guess)
            if self.model == 'nonlinear':
                print ("\na_guess=", a_guess, "\n")

        # We define the boundaries
        self.set_param_hint('%sA' % self.prefix, value=A_guess, min=A_min, max=A_max)
        self.set_param_hint('%sQr' % self.prefix, value=Qr_guess, min=Qr_min, max=Qr_max)
        self.set_param_hint('%sf0' % self.prefix, value=f0_guess, min=fmin, max=fmax)
        self.set_param_hint('%salpha' % self.prefix, value=alpha_guess, min=-np.pi, max=np.pi)
        self.set_param_hint('%sQc_real' % self.prefix, value=Qc_real_guess)
        self.set_param_hint('%sQc_imag' % self.prefix, value=Qc_imag_guess)

        # If the model is nonlinear, a parameter is added
        if self.model == 'nonlinear':
            self.set_param_hint('%sa' % self.prefix, value=a_guess, min=0, max=a_max)

        # Load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)



class AmplitudeModel(lmfit.model.Model):
    __doc__ = "Amplitude model" + lmfit.models.COMMON_DOC

    def __init__(self, prefix='amp_', *args, **kwargs):
        """
            Resonator model object: linear or non-linear
            Parameters
            ----------
            prefix : string
                Model prefix. 'amp_' by default
            ----------
        """

        # pass in the defining equation so the user doesn't have to later.
        super(AmplitudeModel, self).__init__(amp_resonator, *args, **kwargs)

        self.prefix = prefix


    def guess(self, data, **kwargs):
        """
            Guessing amplitude model
            Parameters
            ----------
            data : array
                Data as function of frequency
            ----------
        """

        # Key arguments
        # ----------------------------------------------
        # Add verbose
        verbose = kwargs.pop('verbose', None)
        # ----------------------------------------------

        # Amplitude
        A_data = np.abs(data).max()
        A_guess = A_data

        A_min = np.abs(data).min()
        A_max = 10*np.abs(data).max()

        if verbose:
            print ("Model: ", self.prefix)
            print ("A=", Amin, " / Amax=", Amax, " / A_guess=", A_guess, "\n")

        # We define the boundaries
        self.set_param_hint('%sA' % self.prefix, value=A_guess, min=A_min, max=A_max)

        # Load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)



# # Q factor limits
# Q = 50000
# Q_max = 2*Q
#
# # Read the VNA sweep file (as numpy array)
# path_sweep = '../Tonelist_MUSCAT_24022020/20200224/ch4_fine_sweep_p-10dB.npy'
# s21_res = SweepResonators(path_sweep, verbose=False)
#
# freq = s21_res.freq
# s21 = s21_res.s21
#
# # Coarse search
# baseline = s21_res.get_baseline(freq, np.abs(s21), Q)
# idx_res = s21_res.find_resonators(freq, np.abs(s21), baseline, Q_max, prom=[2., 2.], distance=50e3)
# msg('No. resonators: '+str(len(idx_res)), 'ok')
#
# lw = 9
# chn = s21_res.get_resonators(freq, s21, idx_res, method='qfactor', lw=lw)
#
# # ==============================================================================
# # INTERACTIVE MODE
# # ==============================================================================
# sweep = np.load('/home/marcial/Tonelist_MUSCAT_24022020/20200224/ch4_fine_sweep_p-10dB.npy')
# ioff()
# s21_res.interactive_mode(sweep, chn, 25e3, lw=lw, method='qfactor')
# ion()
#
# # Caso especial del CANAL 4
# chn['K0208']['s21'] = sweep[1][452500:453600]
# chn['K0208']['freq'] = sweep[0][452500:453600].real
#
# block = ['K0026','K0160','K0162','K0207','K0208','M0003','M0005','M0007']
# delete = ['K0161', 'K0181', 'K0224', 'K0228']
#
# for kid in s21_res.kid_man.keys():
#     chn[kid] = {}
#     chn[kid]['freq'] = s21_res.kid_man[kid]['freq']
#     chn[kid]['s21'] = s21_res.kid_man[kid]['s21']
#     chn[kid]['fine'] = True
#
# for kid in chn.keys():
#     if kid in delete:
#         del(chn[kid])
#     else:
#         if kid in block:
#             chn[kid]['fine'] = False
#         else:
#             chn[kid]['fine'] = True
#
#
# t1 = time.time()
# kid_res = s21_res.fit_sweep(chn, Q_max, distance=10e3)
# print 'Total time CH4: '+ str(time.time() - t1)







# # Fake data
# resonator_nonlinear_model = ResonatorModel('nonlinear')
# true_params = resonator_nonlinear_model.make_params(f0=100, a=0.85, alpha=0, A=0.01, Qr=10000, Qc_real=9000, Qc_imag=-9000)
#
# f = np.linspace(99.95, 100.05, 100)
# true_s21 = resonator_nonlinear_model.eval(params=true_params, f=f)
# noise_scale = 0.0002
# np.random.seed(123)
# measured_s21 = true_s21 + noise_scale*(np.random.randn(100) + 1j*np.random.randn(100))
#
#
# # Fitting process
# # Nonlinear
# print "++++++++++++++++++ N O N L I N E A R     M O D E L +++++++++++++++++++++"
# guess = resonator_nonlinear_model.guess(measured_s21, f=f, verbose=True)
# result = resonator_nonlinear_model.fit(measured_s21, params=guess, f=f, verbose=True)
# result.params.pretty_print()
#
# fit_s21_nonlinear = resonator_nonlinear_model.eval(params=result.params, f=f)
# guess_s21_nonlinear = resonator_nonlinear_model.eval(params=guess, f=f)
#
# # Linear
# print "++++++++++++++++++ L I N E A R     M O D E L +++++++++++++++++++++"
# resonator_linear_model = ResonatorModel('linear')
# guess = resonator_linear_model.guess(measured_s21, f=f, verbose=True)
# result = resonator_linear_model.fit(measured_s21, params=guess, f=f, verbose=True)
# result.params.pretty_print()
#
# fit_s21_linear = resonator_linear_model.eval(params=result.params, f=f)
# guess_s21_linear = resonator_linear_model.eval(params=guess, f=f)
#
# plt.figure()
# plt.plot(f, 20*np.log10(np.abs(measured_s21)), 'o-')
# plt.plot(f, 20*np.log10(np.abs(fit_s21_linear)), 'r.-', label='linear fit')
# plt.plot(f, 20*np.log10(np.abs(fit_s21_nonlinear)), 'k--', label='nonlinear fit')
# plt.legend(loc='best')
# plt.ylabel('|S21| (dB)')
# plt.xlabel('MHz')
#
