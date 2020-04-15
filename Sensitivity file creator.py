# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:04:50 2020

@author: camilorg
"""
import pandas as pd

FN_GWD = "GroundwaterDepth" # GWD layer field name
FN_TDS = "TDS" # TDS

variables = {'gwd': {'sensitivity_vars': {'low': -10, 
                                          'high': 10}, 
                     'sensitivity_func': 'sum'},
             'tds': {'sensitivity_vars': {'low': 0.5, 
                                          'high': 1.5}, 
                     'sensitivity_func': 'times'}}
names = {'gwd': FN_GWD, 'tds': FN_TDS}

df = pd.read_csv('nwsas_1km_data' + '.gz')
for variable, values in variables.items():
    for level, value in values['sensitivity_vars'].items():
        file_name = f'nwsas_1km_{level}_{variable}'
        dff = df.copy()
        if values['sensitivity_func'] == 'sum':
            dff.loc[dff[names[variable]] > -value, names[variable]] += value
        elif values['sensitivity_func'] == 'times':
            dff[names[variable]] *= value
        dff.to_csv(file_name + ".gz", index = False)    
        