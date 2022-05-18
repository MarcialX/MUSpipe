# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Manejo y graficación de parámetros S21 de cada KID
#
# Marcial Becerril, @ 12 February 2022
# Latest Revision: 12 Feb 2022, 21:27 GMT-6
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
ion()

# Leemos el archivo con los parámetros de barrido S21
sweep_file = 'Uranus-92937-sweeps.npy'

chns = np.load(sweep_file, allow_pickle=True).item()

# Seleccionamos el canal y KID a graficar
chn = 'empire'
kid = b'K005'

# Obtenemos el vector de frecuencias en Hz
f = chns[chn][kid][b'f']

# Obtenemos el parámetro S21.
# Este es un número complejo
s21 = chns[chn][kid][b'sweep']

# Calculamos la magnitud de S21
mag_s21 = np.abs(s21)

# Graficamos S21 vs frecuencia
figure()
subplot(121)
plot(f, 20*np.log10(mag_s21))	# S21 se grafica en decibeles [dB]
xlabel('Frecuency [Hz]')
ylabel('|S21| [dB]')
# Plot the tone
axvline(f[int(np.ceil(len(f)/2))], color='red')

# Graficamos S21 en el plano complejo
subplot(122)
plot(s21.real, s21.imag)
xlabel('I')
ylabel('Q')