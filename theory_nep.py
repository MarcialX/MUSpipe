# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# theory_nep.py
# Theoretical NEP
#
# Marcial Becerril, @ 2 February 2022
# Latest Revision: 2 Feb 2022, 11:15 GMT-6
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


from var.constants import *


# MUSCAT propierties
bw = 50e9           # [Hz]
mid_band = 270e9    # [Hz]
mid_lambda = c / mid_band

# Operational temperature
Tamb = 250

# Power
P_noise = 2 * K * Tamb * bw

AOmega = mid_lambda**2

NEP_Shot = (2*h*P_noise*mid_band)**.5
NEP_Wave = (c**2*P_noise**2/mid_band**2/bw/AOmega)**.5

NEP_Phot = (NEP_Wave**2 + NEP_Shot**2)**.5

print (NEP_Phot)