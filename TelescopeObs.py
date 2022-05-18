# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# MapBuilding.py
# Functions to create and analise maps
#
# Marcial Becerril, @ 17 January 2022
# Latest Revision: 17 Jan 2022, 18:17 GMT-6
#
# TODO list:
# Functions missing:
#   + Plot functions
#   + Analysis: stats, etc.
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

import yaml
import netCDF4 as nc

# Loading instrument characteristics
try:
    with open('./var/general_params.yaml') as file:
        GENERAL_PARAMS = yaml.safe_load(file)
        
        ANALYSIS_PATH = GENERAL_PARAMS['analysis_path']
        LOGS_PATH = GENERAL_PARAMS['logs_path']

except Exception as e:
    print('Fail loading general parameters. '+str(e), 'fail')

import logging
logging.basicConfig(level=logging.INFO, filename=ANALYSIS_PATH+LOGS_PATH,  filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

from scipy.signal import savgol_filter

from misc.msg_custom import *


class TelescopeObs(object):
    """
        Telescope observation object.
        It contains all the methods to build and analyse the telescope observation.
        Parameters
        ----------
        obs : string
            Observation path as .nc file
        ----------
    """
    def __init__(self, obs ,*args, **kwargs):

        # Key arguments
        # ----------------------------------------------

        # ----------------------------------------------

        # Paths
        self.obs_path = obs 

        # Get header
        self.map_header, self.map_data = self.read_netcdf(obs)

        # Get observation time
        self.map_time = self.map_data['TelescopeBackend']['TelTime']

        # Pointing Coordinates
        self.az_map, self.el_map = self.map_data['TelescopeBackend']['TelAzMap'],\
                                   self.map_data['TelescopeBackend']['TelElMap'] 


    def read_netcdf(self, path, sep='.'):
        """
            Read nc files as dictionaries
            Parameters
            ----------
            Input:
                path : string
                    File path
                sep(opt) : string
                    Separation key 
            Output:
                header : dict
                    Header info as dictionary
                data : dict
                    Data info as dictionary
            ----------
        """
        ds = nc.Dataset(path)
        keys = ds.variables.keys()

        data_map = {}
        for key in keys:
            split_key = key.split(sep)
            
            title = split_key[0]
            subtitle = split_key[1]
            value = split_key[2] 

            # Header/Data            
            if not title in data_map.keys():
                data_map[title] = {}
            # Field
            if not subtitle in data_map[title].keys():
                data_map[title][subtitle] = {}
            # Value
            if not value in data_map[title][subtitle].keys():
                data = ds[key][:]
                if data.data.size > 1:
                    if isinstance(data.data[0], bytes):
                       word = b''
                       aux = b''
                       for b in data.data:
                            if b == b' ':
                                aux += b
                            else:
                                if len(aux) > 0:
                                    word += aux + b
                                    aux = b''
                                else:
                                    word += b

                       data = word

            data_map[split_key[0]][split_key[1]][split_key[2]] = data

        return data_map['Header'], data_map['Data']


    def obs_summary(self):
        """
            Data observation summary
        """
        msg ('------------------------------------------------', 'info')
        msg ('              Observation summary               ', 'info')
        msg ('------------------------------------------------', 'info')
        print ('Number observation: ', self.map_header['Dcs']['ObsNum'])
        print ('Source: ', self.map_header['Source']['SourceName'])
        print ('RA: ', self.map_header['Source']['Ra'][0]*180/_np.pi, ' deg')
        print ('Dec: ', self.map_header['Source']['Dec'][0]*180/_np.pi, ' deg') 
        print ('Az: ', self.map_header['Source']['Az'][0]*180/_np.pi, ' deg')
        print ('El: ', self.map_header['Source']['El'][0]*180/_np.pi, ' deg') 
        msg ('------------------------------------------------', 'info')
        msg ('              Map characteristics               ', 'info')
        msg ('------------------------------------------------', 'info')
        print ('X length: {0:.2f} arcsec'.format(206265*self.map_header['Map']['XLength']))
        print ('Y length: {0:.2f} arcsec'.format(206265*self.map_header['Map']['YLength']))
        print ('X step: {0:.2f} arcsec'.format(self.map_header['Map']['XStep']))
        print ('Y step: {0:.2f} arcsec'.format(self.map_header['Map']['YStep']))
        print ('Scan rate: {0:.2f} arcsec/s'.format(206265*self.map_header['Map']['ScanRate']))
        print ('Integration time: {0:.2f} s'.format(self.map_header['Dcs']['IntegrationTime']))

