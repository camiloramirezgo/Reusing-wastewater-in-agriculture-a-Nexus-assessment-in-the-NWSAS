#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:06:14 2018

@author: Camilo Ramirez Gomez

Spatial Wastewater Treatment and Allocation Tool
SWaTAT
"""
import glob
import pandas as pd
import numpy as np
import os
import logging
from functools import reduce
from math import pi, exp, sqrt
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from plotnine import *
from pandas.api.types import CategoricalDtype
import pyeto
import math

math.exp = np.exp
math.pow = np.power
math.sqrt = np.sqrt

class DataFrame:
    """
    Processes the dataframe and adds all the columns to determine the cheapest option and the final costs and summaries
    """        
    def __init__(self, create_dataframe = None, lyr_names = None, dir_files: str = None, 
                 input_file = None, file_name = None, save_csv = False, empty = None, 
                 cell_area = None, sensitivity_vars = None, sensitivity_func = None):
        """
        Reads all layer datasets stored under a common directory and merges them into 
        a single dataframe, having the option to save the output into a new csv file.
        """
        if not empty:
            self.lyr_names = lyr_names
    #        self.spe_names = spe_names
            self.cell_area = cell_area # cell area in km2
            
            if create_dataframe:
                try:
                    directory = os.getcwd()
                    
                    os.chdir(dir_files)
                    filenames = glob.glob("*.csv")
                    os.chdir(directory)
                    
                    ldf = []
                    for file in filenames:
                        ldf.append(pd.read_csv(os.path.join(dir_files, file), header = None, names = [lyr_names["X"],lyr_names["Y"],file.split(".")[0]]))
                    
                    self.df = reduce(lambda a,b: pd.merge(a, b, on = [lyr_names["X"], lyr_names["Y"]]), ldf)
                    self.df.loc[self.df[lyr_names["TDS"]] < 0,lyr_names["TDS"]] = 0
    #                self.df.loc[self.df[lyr_names["Landcover"]] != 4,lyr_names["Landcover"]] = 0
    #                self.df.loc[self.df[lyr_names["Landcover"]] == 4,lyr_names["Landcover"]] = 1
                    self.df.loc[self.df[lyr_names["Region"]] == 1,lyr_names["Region"]] = "Algeria"
                    self.df.loc[self.df[lyr_names["Region"]] == 2,lyr_names["Region"]] = "Tunisia"
                    self.df.loc[self.df[lyr_names["Region"]] == 3,lyr_names["Region"]] = "Libya"
                    self.df.loc[self.df[lyr_names["Region"]] == 0,lyr_names["Region"]] = None
                    
                    for variable, value in sensitivity_vars.items():
                        if sensitivity_func == 'sum':
                            self.df.loc[self.df[variable] > -value, variable] += value
                        elif sensitivity_func == 'times':
                            self.df[variable] *= value
                                
                    if save_csv:
                        self.df.to_csv(file_name + ".gz", index = False)
                    
                except FileNotFoundError:
                    print('No .csv files were found in the directory!')
                    
            else:
                try:
                    self.df = pd.read_csv(input_file + '.gz')
                except FileNotFoundError:
                    print('Not such file in directory!')
            
            self.df = self.df.dropna(subset=['Region'])
    
    def copy(self, copy_data):
        """
        Creates a copy of the object
        """
        self.lyr_names = copy_data.lyr_names.copy()
        self.cell_area = copy_data.cell_area
        self.df = copy_data.df.copy()
    
    def is_region(self, value, name = 'Region', over = False):
        '''
        Calculates a boolean vector telling which data cells are from a specific region
        '''
        if over:
            return self.df[name] > value
        else:
            return self.df[self.lyr_names[name]] == value
    
    def is_urban(self):
        '''
        Calculates a boolean vector telling which data cells are urban population
        '''
        return self.df[self.lyr_names['IsUrban']] == 1
    
    def calibrate_pop_and_urban(self, region, pop_actual, pop_future, urban, urban_future, urban_cutoff):
        """
        Calibrate the actual current population, the urban split and forecast the future population
        """
        urban_cutoff = 100
        is_region = self.is_region(region)
        is_urban = self.is_urban()
        max_pop =  self.df.loc[is_region, self.lyr_names["Population"]].sum()
        # Calculate the ratio between the actual population and the total population from the GIS layer
        logging.info('Calibrate current population')
        pop_ratio = pop_actual/self.df.loc[is_region, self.lyr_names["Population"]].sum()

        # And use this ratio to calibrate the population in a new column
        self.df.loc[is_region, self.lyr_names["Population"]] = self.df.loc[is_region, self.lyr_names['Population']] * pop_ratio

        # Calculate the urban split, by calibrating the cutoff until the target ratio is achieved
        # Keep looping until it is satisfied or another break conditions is reached
        logging.info('Calibrate urban split')
        
        if urban == 0:
            urban_cutoff = 'Unknown'
            urban_modelled = 0
            self.df.loc[is_region, self.lyr_names['IsUrban']] = 0
            urban_growth = 0
            rural_growth = ((1 - urban_future) * pop_future) / ((1 - urban) * pop_actual)
        elif urban == 1:
            urban_cutoff = 'Unknown'
            urban_modelled = 1
            self.df.loc[is_region, self.lyr_names['IsUrban']] = 1
            urban_growth = (urban_future * pop_future) / (urban * pop_actual)
            rural_growth = 0
        else:
            count = 0
            prev_vals = []  # Stores cutoff values that have already been tried to prevent getting stuck in a loop
            accuracy = 0.005
            max_iterations = 30
            urban_modelled = 0
            
            while True:
                # Assign the 1 (urban)/0 (rural) values to each cell
                self.df.loc[is_region, self.lyr_names['IsUrban']] = self.df.loc[is_region, self.lyr_names["Population"]] > urban_cutoff
                is_urban = self.is_urban()
                
                # Get the calculated urban ratio, and limit it to within reasonable boundaries
                pop_urb = self.df.loc[(is_region) & (is_urban), self.lyr_names["Population"]].sum()
                urban_modelled = pop_urb / pop_actual
    
                if abs(urban_modelled - urban) < accuracy:
                    break
                else:
                    urban_cutoff = sorted([0.005, urban_cutoff - urban_cutoff * 2 *
                                           (urban - urban_modelled) / urban, max_pop])[1]
                if urban_modelled == 0:
                    urban_modelled = 0.05
                elif urban_modelled == 1:
                    urban_modelled = 0.999
        
                if urban_cutoff in prev_vals:
                    logging.info('NOT SATISFIED: repeating myself')
                    break
                else:
                    prev_vals.append(urban_cutoff)
    
                if count >= max_iterations:
                    logging.info('NOT SATISFIED: got to {}'.format(max_iterations))
                    break
    
                count += 1
    
            # Project future population, with separate growth rates for urban and rural
            logging.info('Project future population')
    
            urban_growth = (urban_future * pop_future) / (urban * pop_actual)
            rural_growth = ((1 - urban_future) * pop_future) / ((1 - urban) * pop_actual)
            
        self.df.loc[(is_urban) & (is_region), self.lyr_names['PopulationFuture']] = self.df.loc[(is_urban) & (is_region), self.lyr_names["Population"]] * urban_growth
        self.df.loc[(1 - is_urban) & (is_region), self.lyr_names['PopulationFuture']] = self.df.loc[(1 - is_urban) & (is_region), self.lyr_names["Population"]] * rural_growth
        
        return urban_cutoff, urban_modelled
    
    def calculate_irrigation_system(self, region, total_irrigated_area, irrigation_per_ha, irrigated_area_growth):
        """
        creates a column with the irrigation water needs per cell area
        """
        is_region = self.is_region(region)
        
        area_ratio = total_irrigated_area/self.df.loc[is_region, self.lyr_names["IrrigatedArea"]].sum()
        
        self.df.loc[is_region, self.lyr_names['IrrigatedArea']] = self.df.loc[is_region, self.lyr_names['IrrigatedArea']] * area_ratio
        self.df.loc[is_region, self.lyr_names['IrrigationWater']] = irrigation_per_ha *  self.df[self.lyr_names['IrrigatedArea']]
        self.df.loc[is_region, 'IrrigatedAreaFuture'] = self.df.loc[is_region, self.lyr_names['IrrigatedArea']] * irrigated_area_growth
        
    def calculate_population_water(self, region, urban_uni_water, rural_uni_water):
        """
        Calculate the population water consumption
        """
        is_region = self.is_region(region)
        is_urban = self.is_urban()
    
        self.df.loc[(is_region) & (is_urban), self.lyr_names['PopulationWater']] = self.df.loc[(is_region) & (is_urban), self.lyr_names["Population"]] * urban_uni_water
        self.df.loc[(is_region) & (1 - is_urban), self.lyr_names['PopulationWater']] = self.df.loc[(is_region) & (1 - is_urban), self.lyr_names["Population"]] * rural_uni_water
    
    def total_withdrawals(self, region = None):
        """
        Calculates the total water withdrawals per cell area
        """
#        is_region = self.is_region(region)
        
        self.df[self.lyr_names['TotalWithdrawals']] = self.df[self.lyr_names['PopulationWater']] + \
                                self.df[self.lyr_names['IrrigationWater']]
    
    def recharge_rate(self, region, recharge_rate, environmental_flow):
        """
        Calculates the recharge rate and environmental flow per cell area
        """
        is_region = self.is_region(region)
        
        self.df.loc[is_region, 'RechargeRate'] = recharge_rate / 1000 * self.cell_area **2 * 1000 ** 2
        self.df.loc[is_region, 'EnvironmentalFlow'] = environmental_flow / 1000 * self.cell_area**2 * 1000 ** 2
    
    def groundwater_stress(self, region, withdrawals, name = 'GroundwaterStress'):
        """
        calculates the groundwater stress of each cell, based on the area annual water withdrawals, 
        the area-average annual recharge rate and the environmental stream flow
        """
        is_region = self.is_region(region)
                
#        recharge_rate = recharge_rate / 1000 * self.cell_area * 1000 ** 2
#        environmental_flow = environmental_flow / 1000 * self.cell_area * 1000 ** 2
        
        self.df.loc[is_region, self.lyr_names[name]] = withdrawals.loc[is_region] \
                                / (self.df.loc[is_region, 'RechargeRate'] -  self.df.loc[is_region, 'EnvironmentalFlow'])
        
    def groundwater_pumping_energy(self, region, hours, density, delivered_head, pump_efficiency = 1, calculate_friction = False, viscosity = None, pipe = None):
        '''
        Calculates the energy requirements for pumping groundwater based on the water table level, 
        and the friction losses (in kWh/m3)
        '''
        is_region = self.is_region(region)
        flow = self.df.loc[is_region, 'IrrigationWater'] / (hours * 60 * 60)
        
        if calculate_friction:
            self.df.loc[is_region, 'GWPumpingEnergy'] = (density * 9.81 * (delivered_head + self.df.loc[is_region, 'GroundwaterDepth']) + \
                   pipe.calculate_pressure_drop(density, flow, viscosity, self.df.loc[is_region,'GroundwaterDepth'])) / 3600000 / pump_efficiency
        else:
            self.df.loc[is_region, 'GWPumpingEnergy'] = (density * 9.81 * (delivered_head + self.df.loc[is_region, 'GroundwaterDepth'])) / 3600000 / pump_efficiency
            
    def reverse_osmosis_energy(self, region, threshold, osmosis):
        """
        Calculates the energy required for desalinisation of groundwater in each cell (kWh/m3)
        """
        is_region = self.is_region(region)
        temperature = self.df.loc[is_region, 'GroundwaterTemperature']
        solutes = self.df.loc[is_region, 'GroundwaterSolutes']
        concentration = self.df.loc[is_region, 'TDS']
        
        self.df.loc[is_region, 'DesalinationEnergy'] = osmosis.minimum_energy(solutes, concentration, temperature)
        self.df.loc[is_region & (self.df['TDS']<=threshold), 'DesalinationEnergy'] = 0
        
    def total_irrigation_energy(self):
        """
        Aggregates groundwater pumping and desalination energy requirements
        """
        self.df['IrrigationPumpingEnergy'] = self.df['GWPumpingEnergy'] * self.df['IrrigationWater']
        self.df['IrrigationDesalinationEnergy'] = self.df['DesalinationEnergy'] * self.df['IrrigationWater']
        self.df['IrrigationEnergyTotal'] = self.df['IrrigationDesalinationEnergy'] + self.df['IrrigationPumpingEnergy']
                
#    def total_energy_irrigation_cost(self, region, electricity_price):
#        """
#        Calculates the total price of the electricity used to pump and desalinate water for irrigation
#        """
#        is_region = self.is_region(region)
#        self.df.loc[is_region, 'IrrigationEnergyCost'] = self.df.loc[is_region,'IrrigationEnergyTotal'] * electricity_price
#        self.df.loc[is_region, 'IrrigationWaterCost'] = self.df.loc[is_region, 'IrrigationEnergyUni'] * electricity_price
        
#    def income_energy_cost_share(self, region, income_per_ha):
#        """
#        Calculates the share of energy costs in the income per hectare for the region
#        """
#        is_region = self.is_region(region)
#        self.df.loc[is_region, 'IncomeEnergyCostShare'] = self.df.loc[is_region,'IrrigationEnergyCost'] / (self.df.loc[is_region,'IrrigatedArea'] * income_per_ha)
#        
    def clustering_algorithm(self, population_min, irrigated_min, cluster_num, clusterize):
        """
        Runs a clustering algorithm that combines and classify the population and irrigated area into clusters
        """
        if clusterize:
            clustering_vector = self.df.loc[(self.df['Population'] > population_min) | (self.df['IrrigatedArea'] > irrigated_min), ['X','Y']]
            
            hc = AgglomerativeClustering(n_clusters=cluster_num, affinity = 'euclidean', linkage = 'ward')
            # save clusters for chart
            y_hc = hc.fit_predict(clustering_vector)
            clustering_vector['Cluster'] = y_hc
            
            self.newdf = self.df.merge(clustering_vector, on = [self.lyr_names["X"], self.lyr_names["Y"]], how='outer')
        else:
            self.df.loc[(self.df['Population'] <= population_min) & (self.df['IrrigatedArea'] <= irrigated_min), 'Cluster'] = None
#        plt.scatter(clustering_vector['X'], clustering_vector['Y'], s=15, c=y_hc, cmap='tab20', marker='o')
    
    def calculate_per_cluster(self, cluster, parameter, variable, min_variable):
        """
        Calculates the sum of a parameter per cluster
        """
        is_region = self.is_region(cluster, 'Cluster')
        is_type = self.is_region(min_variable, variable, over = True)
        self.df.loc[(is_region) & (is_type), parameter + 'PerCluster'] = self.df.loc[(is_region) & (is_type), parameter].sum()
    
    def calculate_reclaimed_water(self, pop_water_fraction, agri_water_fraction):
        """
        Calculates the potential reused water per cluster
        """
        pop_growth = self.df['PopulationFuturePerCluster'].dropna() / \
                   self.df['PopulationPerCluster'].dropna()     
        agri_growth = self.df['IrrigatedAreaFuturePerCluster'].dropna()  / \
                    self.df['IrrigatedAreaPerCluster'].dropna() 
        
        self.df['PopulationReclaimedWater'] = None
        self.df['IrrigationReclaimedWater'] = None
        self.df['PopulationReclaimedWater'] = self.df['PopulationWaterPerCluster'].dropna() * pop_water_fraction
        # self.df['IrrigationReclaimedWater'] = self.df['IrrigationWaterPerCluster'].dropna() * agri_water_fraction
        self.df['PopulationFutureReclaimedWater'] = None
        self.df['IrrigationFutureReclaimedWater'] = None
        self.df['PopulationFutureReclaimedWater'] = self.df['PopulationWaterPerCluster'].dropna() * pop_water_fraction * pop_growth
        self.df['IrrigationFutureReclaimedWater'] = self.df['IrrigationWaterPerCluster'].dropna() * agri_water_fraction * agri_growth
    
    def get_evap_i(self, lat, elev, wind, srad, tmin, tmax, tavg, month):
        J = 15 + (month-1)*30
            
        latitude = pyeto.deg2rad(lat)
        atmosphericVapourPressure = pyeto.avp_from_tmin(tmin)
        saturationVapourPressure = pyeto.svp_from_t(tavg)
        ird = pyeto.inv_rel_dist_earth_sun(J)
        solarDeclination = pyeto.sol_dec(J)
        sha = [pyeto.sunset_hour_angle(l, solarDeclination) for l in latitude]
        extraterrestrialRad = [pyeto.et_rad(x, solarDeclination,y,ird) for 
                               x, y in zip(latitude,sha)]
        clearSkyRad = pyeto.cs_rad(elev,extraterrestrialRad)
        netInSolRadnet = pyeto.net_in_sol_rad(srad*0.001, albedo=0.065)
        netOutSolRadnet = pyeto.net_out_lw_rad(tmin, tmax, srad*0.001, clearSkyRad, 
                                               atmosphericVapourPressure)
        netRadiation = pyeto.net_rad(netInSolRadnet,netOutSolRadnet)
        tempKelvin = pyeto.celsius2kelvin(tavg)
        windSpeed2m = wind
        slopeSvp = pyeto.delta_svp(tavg)
        atmPressure = pyeto.atm_pressure(elev)
        psyConstant = pyeto.psy_const(atmPressure)
        
        return self.fao56_penman_monteith(netRadiation, tempKelvin, windSpeed2m, 
                                          saturationVapourPressure, 
                                          atmosphericVapourPressure,
                                          slopeSvp, psyConstant)

    def get_eto(self, eto, lat, elevation, wind, srad, tmin, tmax, tavg):
        '''
        calculate ETo for each row for each month 
        '''
        for i in range(1,13):
            self.df['{}_{}'.format(eto, i)] = 0
            self.df['{}_{}'.format(eto, i)] = self.get_evap_i(self.df[lat],
                                                              self.df[elevation],
                                                              self.df['{}_{}'.format(wind, i)],
                                                              self.df['{}_{}'.format(srad, i)],
                                                              self.df['{}_{}'.format(tmin, i)],
                                                              self.df['{}_{}'.format(tmax, i)],
                                                              self.df['{}_{}'.format(tavg, i)],
                                                              i) * 30
    
    def fao56_penman_monteith(self, net_rad, t, ws, svp, avp, delta_svp, psy, shf=0.0):
        """
        Estimate reference evapotranspiration (ETo) from a hypothetical
        short grass reference surface using the FAO-56 Penman-Monteith equation.
        Based on equation 6 in Allen et al (1998).
        :param net_rad: Net radiation at crop surface [MJ m-2 day-1]. If
            necessary this can be estimated using ``net_rad()``.
        :param t: Air temperature at 2 m height [deg Kelvin].
        :param ws: Wind speed at 2 m height [m s-1]. If not measured at 2m,
            convert using ``wind_speed_at_2m()``.
        :param svp: Saturation vapour pressure [kPa]. Can be estimated using
            ``svp_from_t()''.
        :param avp: Actual vapour pressure [kPa]. Can be estimated using a range
            of functions with names beginning with 'avp_from'.
        :param delta_svp: Slope of saturation vapour pressure curve [kPa degC-1].
            Can be estimated using ``delta_svp()``.
        :param psy: Psychrometric constant [kPa deg C]. Can be estimatred using
            ``psy_const_of_psychrometer()`` or ``psy_const()``.
        :param shf: Soil heat flux (G) [MJ m-2 day-1] (default is 0.0, which is
            reasonable for a daily or 10-day time steps). For monthly time steps
            *shf* can be estimated using ``monthly_soil_heat_flux()`` or
            ``monthly_soil_heat_flux2()``.
        :return: Reference evapotranspiration (ETo) from a hypothetical
            grass reference surface [mm day-1].
        :rtype: float
        """
        a1 = (0.408 * (net_rad - shf) * delta_svp /
              (delta_svp + (psy * (1 + 0.34 * ws))))
        a2 = (900 * ws / t * (svp - avp) * psy /
              (delta_svp + (psy * (1 + 0.34 * ws))))
        return a1 + a2
    
    def calculate_capex(self, treatment_system_name, treatment_system, values,
                        parameter, variable, limit, limit_func):
        """
        Calculates the CAPEX for each treatment technology in each cluster 
        """
#        if variable == 'Population':
#            growth = self.df['PopulationFuturePerCluster'].dropna() / \
#                   self.df['PopulationPerCluster'].dropna()     
#        elif variable == 'Irrigation':
#            growth = self.df['IrrigatedAreaFuturePerCluster'].dropna()  / \
#                    self.df['IrrigatedAreaPerCluster'].dropna() 
        
        population_total = self.df['PopulationFuturePerCluster'].dropna() 
        water_total = self.df[variable + 'ReclaimedWater'].dropna()
#        water_total = self.df[variable + 'WaterPerCluster'].dropna()  * water_fraction * growth
        limit = np.array(list(limit) * water_total.shape[0])
        
        if 'water' in limit_func:
            water = limit
            limit_multiplier = np.floor(water_total / limit)
            population = np.floor(population_total / limit_multiplier)   
            func = create_function(treatment_system, parameter, values)
            self.df.loc[self.df['Cluster'].notna(), treatment_system_name] = eval(func) * limit_multiplier
            water = water_total % limit
            population = population_total % limit_multiplier
            self.df.loc[self.df['Cluster'].notna(), treatment_system_name] = self.df[treatment_system_name].dropna(subset=['Cluster']) + eval(func)
            
        elif 'population' in limit_func:
            population = limit
            limit_multiplier = np.floor(population_total / limit)
            water = np.floor(water_total / limit_multiplier)   
            func = create_function(treatment_system, parameter, values)
            self.df.loc[self.df['Cluster'].notna() , treatment_system_name] = eval(func) * limit_multiplier
            population = population_total % limit
            water = water_total % limit_multiplier
            self.df.loc[self.df['Cluster'].notna(), treatment_system_name] = self.df[treatment_system_name].dropna(subset=['Cluster']) + eval(func)        
         
        else:
            water = water_total
            population = population_total
            func = create_function(treatment_system, parameter, values)
            self.df.loc[self.df['Cluster'].notna(), treatment_system_name] = eval(func)
         
    
    def calculate_opex(self, treatment_system_name, treatment_system, values,
                        water_fraction, parameter, variable, years):
        """
        Calculates the OPEX for each treatment technology in each cluster 
        """
        if variable == 'Population':
            growth = 'PopulationGrowthPerCluster'
        elif variable == 'Irrigation':
            growth = 'IrrigatedGrowthPerCluster'
        
        year = np.arange(years + 1)
        population = np.array([x * (1 + y)**year for x, y in np.array(self.df[['PopulationPerCluster', growth]].dropna())])
        water = np.array([x  * (1 + y)**year for x, y in np.array(self.df[[variable + 'ReclaimedWater', growth]].dropna())])
        func = create_function(treatment_system, parameter, values)
         
        return eval(func)
    
    def calculate_treatment_energy(self, treatment_system_name, treatment_system, values,
                                   parameter, variable):
        """
        Calculates the energy requirements for the specified treatment system
        """
        population = self.df.groupby('Cluster').agg({'PopulationFuturePerCluster': 'first'})
        water = self.df.groupby('Cluster').agg({variable + 'ReclaimedWater': 'first'})
        func = create_function(treatment_system, parameter, values)
        self.df[treatment_system_name] = self.df['Cluster'].map(eval(func).iloc[:,0])
    
    def calculate_lcow_capex(self, variable, investment_var, water_var,
                             degradation_factor, income_tax_factor, future_var, years, discount_rate):
        """
        Calculates the levelised cost of water for the capex
        """      
#        capacity = self.df[variable + 'ReclaimedWater'].dropna()
#        year = np.arange(years + 1)
#        discount_factor = (1 / (1 + discount_rate))**year
        
        if variable == 'Population':
            growth = 'PopulationGrowthPerCluster'
        elif variable == 'Irrigation':
            growth = 'IrrigatedGrowthPerCluster'       
        
        year = np.arange(years + 1)
        discount_factor = (1 / (1 + discount_rate))**year
        
        water = np.array([x * (1 + y)**year for x, y in np.array(self.df[[variable + 'ReclaimedWater', growth]].dropna())])
        
        self.df[investment_var + '_LCOW'] = None
#        self.df.loc[self.df[variable + 'ReclaimedWater'].notna(), investment_var + '_LCOW'] = income_tax_factor * self.df[investment_var].dropna() / \
#                                            (capacity * sum((1 - degradation_factor)**(year) * discount_factor))
        
        a = self.df.loc[self.df[variable + 'ReclaimedWater'].notna(), investment_var].dropna()
        b = np.array([sum(x * discount_factor) for x in water])
        self.df.loc[self.df[variable + 'ReclaimedWater'].notna(), investment_var + '_LCOW'] = income_tax_factor * np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        
        
    def calculate_lcow_opex(self, variable, op_cost_var, water_var, degradation_factor, 
                            present_var, future_var, years, discount_rate, opex_data):
        """
        Calculates the levelised cost of water for the opex
        """                    
        if variable == 'Population':
            growth = 'PopulationGrowthPerCluster'
        elif variable == 'Irrigation':
            growth = 'IrrigatedGrowthPerCluster'       
        
        year = np.arange(years + 1)
        discount_factor = (1 / (1 + discount_rate))**year
        
        water = np.array([x * (1 + y)**year for x, y in np.array(self.df[[variable + 'ReclaimedWater', growth]].dropna())])
#        opex = opex_data / water
        
        self.df[op_cost_var + '_LCOW'] = None
        self.df.loc[self.df[growth].notna(), op_cost_var + '_LCOW'] = np.array([sum(x * discount_factor) for x in opex_data]) / \
                                            (np.array([sum(x * discount_factor) for x in water]))
        
    
    def calculate_lcow(self, name):
        """
        Calculates the total LCOW for a technology
        """
        self.df[name + 'LCOW'] = self.df[name + 'OPEX_LCOW'] + self.df[name + 'CAPEX_LCOW']
        
    def least_cost_technology(self, systems: set, variable):
        """
        Chooses the least-cost system in the cluster
        """
        systems_list = []
        for system in systems:
            self.df[system] = pd.to_numeric(self.df[system])

        self.df[variable + 'Technology'] = self.df[systems].idxmin(axis=1)
        class_name = {}
        for i, system in enumerate(systems):
            class_name[str(i + 1)] = system.split('LCOW')[0]
            systems_list.append(self.df[system])
            bool_vector = self.df[variable + 'Technology'] == system
            self.df.loc[bool_vector, variable + 'Technology'] = str(i + 1)
            
        self.df[variable] = reduce(lambda a,b: np.minimum(a, b), systems_list)
        self.df.loc[self.df[variable].isna() , variable + 'Technology'] = str(0)
        class_name['0'] = 'Na'
        return class_name
    
    def least_cost_system(self, variables, dic_1, dic_2):
        """
        Gets te least cost system
        """
        class_name = {}
        self.df['LeastCostSystem'] = self.df[variables[0]] + self.df[variables[1]]
        for system in set(self.df['LeastCostSystem'].dropna()):
            class_name[system] = dic_1[system[0]] + ', ' + dic_2[system[1]]
        return class_name
    
    def get_storage(self, leakage, area_percent, storage_depth, agri_water_req, agri_non_recoverable):
        """
        Calculate the losses in the on-farm storage through the year, based on
        a water balance (leakage + evaporation)

        Parameters
        ----------
        leakage : float
            Leakage in mm per day of water percolated in the on-farm storage.
        area_percent : float
            Percentage of area covered by the on-farm storage.
        storage_depth : float
            Depth of the on-farm storage in meters.
        """
        not_na = self.df['IrrigationWaterPerCluster'].notna()
        self.df.loc[not_na, 'AgWaterReq'] = agri_water_req
        self.df.loc[not_na, 'available_storage'] = area_percent  * storage_depth * self.df.loc[not_na, 'IrrigatedArea'] * 10000
        self.df.loc[not_na, 'leakage_month'] = (leakage / 1000) * 30 * area_percent * self.df.loc[not_na, 'IrrigatedArea'] * 10000
        recoverable_water = (self.df.loc[not_na, 'IrrigationWater'] - (self.df.loc[not_na, 'AgWaterReq'] * self.df.loc[not_na, 'IrrigatedArea']/(1-agri_non_recoverable)))
        recoverable_water[recoverable_water<0] = 0
        for i in range(1,13):
            self.df.loc[not_na, f'stored_{i}'] = recoverable_water / 12 - \
                                      (self.df.loc[not_na, 'leakage_month'] + \
                                      ((self.df.loc[not_na, f'eto_{i}'] / 1000) * area_percent * self.df.loc[not_na, 'IrrigatedArea'] * 10000))
            self.df.loc[not_na & (self.df[f'stored_{i}']<0), f'stored_{i}'] = 0
            self.df.loc[not_na, f'stored_percentage_{i}'] = self.df.loc[not_na, f'stored_{i}'] / self.df.loc[not_na, 'available_storage']
            self.df.loc[not_na & (self.df[ f'stored_percentage_{i}'] > 1), f'stored_{i}'] = self.df.loc[not_na & (self.df[ f'stored_percentage_{i}'] > 1), 'available_storage']
    
    def reused_water(self, pop_percentage_of_reuse):
        """
        Calculates the total final amount of water extracted for irrigation after reuse
        """
        self.df['IrrigationReusedWater'] = self.df.filter(regex='stored_[1-9]').sum(axis=1)
        self.df['IrrigationReclaimedWater'] = self.df.set_index('Cluster').index.map(self.df.groupby('Cluster')['IrrigationReusedWater'].sum())
        self.df['FinalIrrigationWater'] = 0
        not_na = self.df['IrrigationWaterPerCluster'].notna()
        self.df.loc[not_na, 'FinalIrrigationWater'] = self.df.loc[not_na, 'IrrigationWater'] - self.df.loc[not_na, 'IrrigationReusedWater']
        self.losses = 0
        self.df['PopulationReusedWater'] = 0
        for cluster in set(self.df['Cluster'].dropna()):
            is_cluster = self.is_region(cluster, 'Cluster')
            count = self.df.loc[is_cluster, 'IrrigationWaterPerCluster'].dropna().count()
                
            pop_water = self.df.loc[is_cluster, 'PopulationReclaimedWater'].dropna().mean() * pop_percentage_of_reuse
            while (count > 0) and (pop_water > 0):
                self.df.loc[(is_cluster) & (not_na) & (self.df['FinalIrrigationWater'] > 0), 'FinalIrrigationWater'] -= (pop_water / count)
                self.df.loc[(is_cluster) & (not_na) & (self.df['FinalIrrigationWater'] > 0), 'PopulationReusedWater'] += (pop_water / count)
                remaining_water = self.df.loc[(is_cluster) & (self.df['FinalIrrigationWater'] < 0), 'FinalIrrigationWater'].dropna().sum()
                self.df.loc[(is_cluster) & (self.df['FinalIrrigationWater'] < 0), 'FinalIrrigationWater'] = 0
                count = self.df.loc[(is_cluster) & (not_na) & (self.df['FinalIrrigationWater'] > 0), 'FinalIrrigationWater'].count()
                pop_water = remaining_water * (-1)
                
        self.df['FinalWaterWithdrawals'] = self.df[['FinalIrrigationWater', 'PopulationWater']].sum(axis=1)

    def get_water_stats(self):
        withdrawals_per_cluster = self.df.groupby('Cluster').agg({
                                    'FinalWaterWithdrawals': 'sum', 
                                    'TotalWithdrawals': 'sum', 
                                    'IrrigationReusedWater': 'sum', 
                                    'PopulationReusedWater': 'sum',
                                    'PopulationWater': 'sum',
                                    'FinalIrrigationWater': 'sum'})
        withdrawals_total = pd.DataFrame({'Irrigation extractions': self.df['FinalIrrigationWater'].sum(),
                                          'Population extractions': self.df['PopulationWater'].sum(),
                                          'Reused water from irrigation': self.df['IrrigationReusedWater'].sum(),
                                          'Reused water from population': self.df['PopulationReusedWater'].sum(),
                                          'Final withdrawals': self.df['FinalWaterWithdrawals'].sum(),
                                          'Baseline withdrawals': self.df['TotalWithdrawals'].sum()}, index=[0])
        withdrawals_baseline = pd.DataFrame({'Irrigation extractions': self.df['IrrigationWater'].sum(),
                                          'Population extractions': self.df['PopulationWater'].sum(),
                                          'Reused water from irrigation': 0,
                                          'Reused water from population': 0,
                                          'Baseline withdrawals': self.df['TotalWithdrawals'].sum()}, index=[0])
#        reused_per_cluster = (withdrawals_per_cluster['TotalWithdrawals'].subtract(withdrawals_per_cluster['FinalWaterWithdrawals'])) / withdrawals_per_cluster['TotalWithdrawals']
#        reused_global = (self.df['TotalWithdrawals'].subtract(self.df['FinalWaterWithdrawals']).sum()) / self.df['TotalWithdrawals'].sum()
        
        return withdrawals_per_cluster, withdrawals_total, withdrawals_baseline
    
    def calculate_final_energy(self, treatment_systems_pop, treatment_systems_agri):
        """
        Calculates the energy requirements for pumping, desalinating and treatment for each cell area
        """     
        
        self.df['FinalPumpingEnergy'] = self.df['GWPumpingEnergy'] * self.df['FinalIrrigationWater']
        self.df['FinalDesalinationEnergy'] = self.df['DesalinationEnergy'] * self.df['FinalIrrigationWater']
        self.df['FinalIrrigationEnergy'] = self.df['FinalPumpingEnergy'] + self.df['FinalDesalinationEnergy']
        
        systems_vector_pop = self.df['PopulationLeastCostTechnology'].apply(lambda row: treatment_systems_pop[row] + 'Energy')
        systems_vector_agri = self.df['IrrigationLeastCostTechnology'].apply(lambda row: treatment_systems_agri[row] + 'Energy')
        self.df['FinalPopTreatmentEnergy'] = None
        self.df['FinalAgriTreatmentEnergy'] = None
        systems_vector_pop.loc[systems_vector_pop == 'NaEnergy'] = None
        systems_vector_agri.loc[systems_vector_agri == 'NaEnergy'] = None
        
        for value in set(systems_vector_pop.dropna()):
            index_vec =  systems_vector_pop == value
            self.df.loc[index_vec, 'FinalPopTreatmentEnergy'] =  self.df.loc[index_vec, value]
            
        for value in set(systems_vector_agri.dropna()):
            index_vec =  systems_vector_agri == value
            self.df.loc[index_vec, 'FinalAgriTreatmentEnergy'] =  self.df.loc[index_vec, value]

        self.df['FinalTreatmentEnergy'] = self.df[['FinalPopTreatmentEnergy', 'FinalAgriTreatmentEnergy']].sum(axis=1)
        
    
    def least_cost_option(self):
        """
        Gets te best option for each cell between the conventional and the 
        least-cost evaluated system
        """
        self.df['IrrigationWaterPerCluster'] = self.df['IrrigationWaterPerCluster'].fillna(0)
        self.df['PopulationWaterPerCluster'] = self.df['PopulationWaterPerCluster'].fillna(0)
        self.df['IrrigationLeastCost'] = self.df['IrrigationLeastCost'].fillna(0)
        self.df['PopulationLeastCost'] = self.df['PopulationLeastCost'].fillna(0)
        self.df['PotentialReusedWater'] = self.df['IrrigationWaterPerCluster'] + self.df['PopulationWaterPerCluster']
        self.df['PotentialTotalCost'] = self.df['IrrigationLeastCost'] * self.df['IrrigationWaterPerCluster'] + \
                                        self.df['PopulationLeastCost'] * self.df['PopulationWaterPerCluster']
        self.df['CombinedLeastCost'] = pd.to_numeric(self.df['PotentialTotalCost'] / self.df['PotentialReusedWater'])
        self.df['LeastCostOption'] = self.df[['CombinedLeastCost', 'IrrigationWaterCost']].idxmin(axis=1)
        self.df.loc[self.df['LeastCostOption'] ==  'CombinedLeastCost','LeastCostOption'] = self.df.loc[self.df['LeastCostOption'] ==  'CombinedLeastCost','LeastCostSystem']
        self.df.loc[self.df['LeastCostOption'] ==  'IrrigationWaterCost','LeastCostOption'] = '-1'

        
class PipeSystem:
    """
    Creates an object for the piping system used in and specific region
    """
    def __init__(self, diameter, roughness):
        """
        Stores the information of the technology into parameters
        """
        self.diameter = diameter
        self.roughness = roughness / 1000
        self.area = pi * (self.diameter/2) ** 2
    
    def calculate_velocity(self, flow):
        """
        Calculates the fluid velocity
        """
        flow = flow
        return flow / self.area
    
    def calculate_reynolds(self, velocity, viscosity):
        """
        Calculates the Reynolds number
        """
        viscosity = viscosity / (1000 ** 2)
        return velocity * self.diameter / viscosity
    
    def calculate_friction_factor(self, velocity, viscosity):
        """
        Calculates the friction factor
        """
        Re = self.calculate_reynolds(velocity, viscosity)
        return 8 * ((8/Re) ** 12 + ((2.457 * (1/((7/Re) ** 0.9) + 0.27 * self.roughness/self.diameter)) ** 16 + (37530/Re) ** 16) ** (-1.5)) ** (1/12)
    
    def calculate_pressure_drop(self, density, flow, viscosity, length):
        """
        Calculates the pressure drop due to friction of the fluid against the walls of the pipe
        """
        velocity = self.calculate_velocity(flow)
        return self.calculate_friction_factor(velocity, viscosity) * (length/self.diameter) * (density * (velocity ** 2)/2)


class ReverseOsmosis:
    """
    Creates an object to model the reverse osmosis energy needs
    """
    def __init__(self, osmotic_coefficient, efficiency):
        """
        Stores the information of the technology into parameters
        """
        self.efficiency = efficiency
        self.osmotic_coefficient = osmotic_coefficient
        self.solutes_dissociation = {'NaCl': 2, 'SrSO4': 2, 'glucose': 1}
        self.solutes_molar_mass = {'NaCl': 58.4, 'SrSO4': 183.6, 'glucose': 180}
        
    def molar_concentration(self, solute, concentration):
        """
        Calculates the molar concentration of ions
        """
        solutes_dissociation = np.array([self.solutes_dissociation[x] for x in solute])
        solutes_molar_mass = np.array([self.solutes_molar_mass[x] for x in solute])
        
        return solutes_dissociation * concentration / (10**3 * solutes_molar_mass)
    
    def osmotic_pressure(self, solutes, concentration, temperature):
        """
        Calculate the osmotic pressure of the feed water
        """
        return self.osmotic_coefficient * self.molar_concentration(solutes, concentration) * 0.083145 * (temperature + 273)
    
    def minimum_energy(self, solutes, concentration, temperature):
        """
        Calculates the minimun energy (in kWh/m3)required for desalination
        """
        return self.osmotic_pressure(solutes, concentration, temperature) / 36

        

def convert_m3_to_mm(cell_area, path, df, *layers):
    """
    Convert water withdrawals from cubic meters to mm, based on the cell area, and saves it
    """
    for layer in layers:
        print('    - Saving {} layer...'.format(layer))
        temp_layer = df.loc[:,['X', 'Y', layer]]
        temp_layer[layer] = temp_layer[layer] / (1000 ** 2) * 100
        temp_layer.to_csv(path + "/CSV/" + layer + ".csv", index = False)
    

def save_layers(path, df, *layers):
    """
    Saves the specified results layers in separate csv files
    """
    try:
        for layer in layers:
            print('    - Saving {} layer...'.format(layer))
            temp_layer = df[['X', 'Y', layer]]
            temp_layer.to_csv(path + "/CSV/" + layer + ".gz", index = False)
    except:
        print(layer + ' layer not found')
        

def delete_files(folder):
    """
    Delete file from folder
    """
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    
def multiple_sheet_excel(xls, input_string):
    """
    Read multiple sheets from excel file into a dictionary
    """
    if input_string != 'all':
        sheets = [xls.sheet_names[int(x.strip()) - 1] for x in input_string.split(',')]
    else:
        sheets = xls.sheet_names
        
    xls_dfs = {}
    for sheet in sheets:
        xls_dfs[sheet] = xls.parse(sheet)
    
    return xls_dfs


def create_function(treatment_system, parameter, values):
    """
    Creates a function based on user input
    """
    func = str(treatment_system.loc[0, parameter])
    func = func.replace(' ','')
#    limit_func = str(treatment_system.loc[0, limit_var])
    for key, value in values.items():
        func = func.replace(key, str(value))
#        limit_func = limit_func.replace(key, str(value))
    return func


def calculate_lcows(data, clusters, variable, water_fraction, degradation_factor, 
                    income_tax_factor, years, discount_rate, treatment_systems, variable_present, opex_data):
    """
    Loops through the given treatment systems and calls for the lcow functions
    """
    for name, system in treatment_systems.items():
                print('\nCalculating {} treatment system LCOW...'.format(name))
                data.df[name + 'LCOW_CAPEX'] = None
                data.df[name + 'LCOW_OPEX'] = None
                data.df[name + 'LCOW'] = None
#                for cluster in set(data.df['Cluster'].dropna()):
#                    print('    - Cluster {} of {}...'.format(int(cluster) + 1, clusters))
                data.calculate_lcow_capex(variable = variable, investment_var = name + 'CAPEX', 
                                         water_var = variable + 'Water',
                                         degradation_factor = degradation_factor, income_tax_factor = income_tax_factor, 
                                         future_var = variable_present + 'Future', years = years, discount_rate = discount_rate)
                data.calculate_lcow_opex(variable = variable, op_cost_var = name + 'OPEX', 
                                         water_var = variable + 'Water',
                                         degradation_factor = degradation_factor, present_var = variable_present, 
                                         future_var = variable_present + 'Future', years = years, discount_rate = discount_rate,
                                         opex_data = opex_data[name])
                data.calculate_lcow(name)



def gws_plot_mathplot(gws_values, file_name):
    """
    Creates the groundwater stress indicator plot
    """
    color_list, color_legend = create_gws_color_list(gws_values)  
    fig = plt.figure()
    text_color = to_rgb(80,80,80)
    p = fig.add_subplot(111)
    box = p.get_position()
    p.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    bar = p.bar(np.arange(len(gws_values)), gws_values, color = color_list,
                edgecolor=[tuple(x) for x in np.array(color_list)*0.8], linewidth=3)
    plt.xticks(np.arange(len(gws_values)), ['Baseline','Reusing Water'], color = text_color)
    plt.ylabel('Groundwater Stress Indicator', color = text_color)
    legend = plt.legend(handles=color_legend, edgecolor=text_color, 
                        facecolor=to_rgb(240,240,240), loc='lower left',
                        bbox_to_anchor=(1,0))  
    plt.setp(legend.get_texts(), color=text_color)
    autolabel(bar, [tuple(x) for x in np.array(color_list)*0.8])
#        p.set_frame_on(False)
    p.spines['right'].set_visible(False)
    p.spines['top'].set_visible(False)
    p.spines['left'].set_color(text_color)
    p.spines['bottom'].set_color(text_color)
    p.tick_params(colors=text_color)
#        plt.setp(p.spines.values(), color=to_rgb(80,80,80))
    plt.show()
    plt.savefig(file_name, format='pdf')

def create_gws_color_list(values):
    """
    Creates te color list for the gws indicatior
    """
    color_values = [to_rgb(255, 254, 187),
                    to_rgb(255, 202, 110),
                    to_rgb(255, 139, 76),
                    to_rgb(245, 55, 43),
                    to_rgb(193, 0, 41)]
    color_list = [] 
    for value in values:
        if value < 1:
            color_list.append(color_values[0])
        elif value < 5:
            color_list.append(color_values[1])
        elif value < 10:
            color_list.append(color_values[2])
        elif value < 20:
            color_list.append(color_values[3])
        else:
            color_list.append(color_values[4])
    
    color_legend = [mpatches.Patch(color=color_values[0], label='Low (<1)'),
                    mpatches.Patch(color=color_values[1], label='Low to medium (1-5)'),
                    mpatches.Patch(color=color_values[2], label='Medium to high (5-10)'),
                    mpatches.Patch(color=color_values[3], label='High (10-20)'),
                    mpatches.Patch(color=color_values[4], label='Extremely high (>20)')]
    return color_list, color_legend

def autolabel(rects, colors):
    """
    Attach a text label above each bar displaying its height
    """
    for i, rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 0.1,
        str(round(height, 2)),
        ha = 'center', va = 'bottom', color = colors[i],
        weight = 'bold')

def to_rgb(*values):
    """
    Creates RGB color from a RGB value (255 scale)
    """
    return tuple(np.array(values) / 255)

def legend_generator(values):
    color_list = [] 
    for value in values:
        if value < 1:
            color_list.append('1')
        elif value < 5:
            color_list.append('2')
        elif value < 10:
            color_list.append('3')
        elif value < 20:
            color_list.append('4')
        else:
            color_list.append('5')
    return color_list

def gws_plot(gws_values, names, order):
    color_values = [to_rgb(255, 254, 187),
                    to_rgb(255, 202, 110),
                    to_rgb(255, 139, 76),
                    to_rgb(245, 55, 43),
                    to_rgb(193, 0, 41)]
    color_list = [colors.to_hex(x) for x in color_values]
    border_color = '#E8E8E8' 
    df = pd.DataFrame(gws_values,columns=['GWS'])
    df['Legend'] = legend_generator(gws_values)
    df['X'] = names
    new_df = df.append(pd.DataFrame({'GWS':[0,0,0,0,0], 'Legend':['1','2','3','4','5'], 'X':'Baseline'}))
    point_df = df.copy()
    point_df['GWS'] += 0.3
    df['GWS'] = round(df['GWS'],2)
    
    new_df['X'] = new_df['X'].astype('category')
    new_df['X_cat'] = new_df['X'].cat.reorder_categories(order, ordered=True)
    
    p = (ggplot() +
         geom_bar(new_df, aes(x='X_cat', y='GWS', fill= 'Legend'), stat='identity', size=0.5, color='gray') +
#         geom_point(point_df.sort_values(['GWS']), aes(x= 'X', y='GWS'), alpha=0) +
         geom_text(df, aes(x='X',y='GWS/2',label='GWS'), color='black', nudge_y=0, size=8) +
         scale_fill_manual(labels=['Low (<1)','Low to medium (1-5)','Medium to high (5-10)','High (10-20)','Extremely high (>20)'],values= color_list) +
         scale_y_continuous(expand = [0, 0]) + 
#         scale_x_continuous(breaks=[0]) +
#         facet_wrap('X', shrink=False) +
#         theme_minimal() +
         coord_flip() +
         theme_classic() + 
         labs(y='Groundwater Stress Indicator', x='Scenario') +
         theme(legend_title=element_blank(),
               axis_title_x=element_text(color='black'),
               axis_title_y=element_text(color='black'))
        )
#    print(p)
    p.save('GWS.pdf', height=5, width=4)
    
def energy_plot(energy_start, energy_end, sensitivity_energy, order):
    energy_start[1].index = ['Desalination energy', 'Pumping energy']
    df_end = pd.DataFrame(columns=list('XYZ'))
    for i, value in energy_end.items():
        energy_end[i].index = ['Desalination energy', 'Pumping energy', 'Treatment energy']
        temp_df = pd.DataFrame({'X': i, 'Y': value.values, 'Z': energy_end[i].index})
        df_end = df_end.append(temp_df)
#    energy_end.index = ['Desalination energy', 'Pumping energy', 'Treatment energy']
    df_start = pd.DataFrame({'X': energy_start[0], 'Y': energy_start[1].values, 'Z': energy_start[1].index})
#    df_end = pd.DataFrame({'X': 'Water reuse', 'Y': energy_end.values, 'Z': energy_end.index})
    df = df_start.append(df_end)
    df['Y'] /= 1000000
    df['label_pos'] = df['Y'] / 2
    df['total'] = 0
    value_groups = df.groupby('X')['Y'].sum()
    for group, value in value_groups.iteritems():   
        df.loc[(df['Z'] != 'Treatment energy') & (df['X'] == group), 'label_pos'] += \
                        float(0 if len(df.loc[(df['Z'] == 'Treatment energy') & (df['X'] == group), 'Y']) == 0 else df.loc[(df['Z'] == 'Treatment energy') & (df['X'] == group), 'Y'])
        df.loc[(df['Z'] == 'Desalination energy') & (df['X'] == group), 'label_pos'] += float(df.loc[(df['Z'] == 'Pumping energy') & (df['X'] == group), 'Y'])
        df.loc[(df['X'] == group), 'total'] = value
    
    sensitivity_df = pd.DataFrame()
    for scenario, item in sensitivity_energy.items():
        if scenario != 'Baseline':
            item['Desalination energy'] += df_end.loc[(df_end['X'] == scenario) & (df_end['Z'] != 'Desalination energy'), 'Y'].sum()
            item['Pumping energy'] += float(df_end.loc[(df_end['X'] == scenario) & (df_end['Z'] == 'Treatment energy'), 'Y'])
            for sensitivity in set(item['SensitivityVar']):
                item = item.append(pd.DataFrame({'SensitivityVar': [sensitivity], 'Scenario': [scenario], 
                                    'Desalination energy': df_end.loc[(df_end['X'] == scenario), 'Y'].sum(),
                                    'Pumping energy': df_end.loc[(df_end['X'] == scenario) & (df_end['Z'] != 'Desalination energy'), 'Y'].sum(),
                                    'Treatment energy': df_end.loc[(df_end['X'] == scenario) & (df_end['Z'] == 'Treatment energy'), 'Y'].sum()}))
        else:
            item['Desalination energy'] += df_start.loc[(df_start['X'] == scenario) & (df_start['Z'] != 'Desalination energy'), 'Y'].sum()
#            item['Pumping energy'] += float(df_start.loc[(df_start['X'] == scenario) & (df_start['Z'] == 'Treatment energy'), 'Y'])
            for sensitivity in set(item['SensitivityVar']):
                item = item.append(pd.DataFrame({'SensitivityVar': [sensitivity], 'Scenario': [scenario], 
                                    'Desalination energy': df_start.loc[(df_start['X'] == scenario), 'Y'].sum(),
                                    'Pumping energy': df_start.loc[(df_start['X'] == scenario) & (df_start['Z'] != 'Desalination energy'), 'Y'].sum(),
                                    'Treatment energy': df_start.loc[(df_start['X'] == scenario) & (df_start['Z'] == 'Treatment energy'), 'Y'].sum()}))
        sensitivity_df = sensitivity_df.append(item.melt(id_vars=['Scenario', 'SensitivityVar']))
#        na_data = sensitivity_df.groupby(['SensitivityVar', 'variable']).mean()
#        na_vector = sensitivity_df.loc[sensitivity_df['variable'] == 'Desalination energy', 'value'].values == float(na_data.loc[na_data.index.get_level_values('variable') == 'Desalination energy', 'value'].values)
#        sensitivity_df.loc[na_vector, 'value'] = None
            
    sensitivity_df = sensitivity_df.dropna(subset=['value'])
    sensitivity_df['value'] /= 1000000 
    sensitivity_df['group'] = sensitivity_df['Scenario'] + sensitivity_df['variable']
    
#    for scenario in set(sensitivity_df['Scenario']):
#        for sensitivity, variable in zip(set(sensitivity_df['SensitivityVar']), set(sensitivity_df['variable'])):
#            sensitivity_df = sensitivity_df.append(pd.DataFrame({'Scenario': scenario, 'SensitivityVar': sensitivity, 'variable': variable, 'value': [0]}))
#    
    df['X'] = df['X'].astype('category')
    df['X_cat'] = df['X'].cat.reorder_categories(order, ordered=True)
    
    p = (ggplot(df) + geom_bar(df,aes(x='X_cat', y='Y', fill='Z'), stat='identity', size=0.3, color='gray', width=0.5) +
        geom_text(df, aes(x='X_cat',y='label_pos',label='[int(round(x,0)) if (x/t)>0.05 else "" for x, t in zip(Y,total)]'), color='#e7e7e7', nudge_y=0, size=8) +
        labs(y='Energy (GWh/yr)', x='Scenario') + 
        scale_fill_brewer(type='qual', palette='Set2') +
        scale_color_brewer(type='qual', palette='Set2') +
#        scale_y_continuous(expand = [0, 0]) +
        coord_flip() + 
        theme_minimal() + geom_vline(xintercept=[1.5 + x for x in range(len(sensitivity_energy.items())-1)], size=0.5, colour="lightgray") +
#        theme_classic() +
        theme(legend_title=element_blank(),
              axis_title_x=element_text(color='black'),
              axis_title_y=element_text(color='black'),
              panel_grid_major_y=element_blank()))
        
    nudge=0.35
    for var in set(sensitivity_df['SensitivityVar']):
        nudge *= -1
        p = (p + geom_line(sensitivity_df.loc[sensitivity_df['SensitivityVar'] == var], aes(x='Scenario', y='value', group='group', color='variable'), position = position_nudge(x = nudge)) +
            geom_point(sensitivity_df.loc[sensitivity_df['SensitivityVar'] == var], aes(x='Scenario', y='value', group='group', color='variable'), shape='|', position = position_nudge(x = nudge), size=2))
#    print(p)
    p.save('Energy.pdf', height=5, width=5)
    
def water_plot(withdrawals_total, order):
    '''
    '''
    p = (ggplot() + coord_flip())
    min_low = 0
    
    for scenario, value in zip(order,[withdrawals_total[key] for key in sorted(withdrawals_total,key=lambda i:order.index(i))]):
        highs = value[['Irrigation extractions',
                    'Population extractions']]
        lows = value[['Reused water from irrigation',
                    'Reused water from population']]
        highs = highs.melt()
        highs['value'] /= 1000000
        lows = lows.melt()
        lows['value'] /= -1000000
        highs['scenario'] = scenario
        lows['scenario'] = scenario
        sum_vals = highs['value'].sum() - lows['value'].sum()
        highs['total'] = sum_vals
        lows['total'] = sum_vals
        highs['label_pos'] = highs['value']/2
        highs.loc[highs['variable'] == 'Irrigation extractions','label_pos'] = highs['value'].sum() - highs['value']/2
        lows['label_pos'] = lows['value']/2
        lows.loc[lows['variable'] == 'Irrigation reused water','label_pos'] = lows['value'].sum() - lows['value']/2
        if min(lows['value']) < min_low:
            min_low = min(lows['value'])
        lows['percentage_pos'] = min_low/10
        lows.loc[lows['scenario'] == 'Baseline','percentage_pos'] = 0
        
        p = (p + geom_bar(highs, aes(x='scenario', y='value', fill='variable'), stat='identity', position='stack', width=0.5, size=0.3, color='gray') 
            + geom_bar(lows, aes(x='scenario', y='value', fill='variable'), stat='identity', position='stack', width=0.5, size=0.3, color='gray')
            + geom_text(highs, aes(x='scenario', y='label_pos', label='[str(int(x)) if (x/t)>0.05 else "" for x, t in zip(value,total)]'), size=7, color='#F7F7F7')
            + geom_text(lows, aes(x='scenario', y='label_pos', label='[str(int(x)) if (x/t)>0.05 else "" for x, t in zip(-value,total)]'), size=7, color='#F7F7F7')
            + geom_line(lows, aes(x='scenario', y='[0,sum(value)/2-np.average(percentage_pos)]'), position = position_nudge(x = 0.4))
            + geom_line(lows, aes(x='scenario', y='[sum(value)/2+np.average(percentage_pos),sum(value)]'), position = position_nudge(x = 0.4))
            + geom_point(lows, aes(x='scenario', y='sum(value)'), shape='|', size= 2, position = position_nudge(x = 0.4))
            + geom_text(lows, aes(x='scenario', y='sum(value)/2', label='str(round(-sum(value)/np.average(total)*100,1)) + "%" if sum(value)<0 else ""'), size= 7, position = position_nudge(x = 0.4)))
    
    text_df_1 = pd.DataFrame({'label': ["Reused water\nfor irrigation"], 
                            'X': [len(list(withdrawals_total.items())) + 1.2], 'Y': [min_low / 4]})
    text_df_2 = pd.DataFrame({'label': ['Water extractions\nfor population and irrigation'], 
                            'X': [len(list(withdrawals_total.items())) + 1.2], 'Y': [- min_low / 4]})
    
    p = (p + theme_minimal() + theme(legend_position = 'top', legend_title=element_blank()) 
        + geom_hline(yintercept = 0, color ="black") + scale_fill_brewer(type='div', palette='PuOr')
        + geom_text(text_df_1, aes(x='X', y='Y', label='label'), ha='right')
        + geom_text(text_df_2, aes(x='X', y='Y', label='label'), ha='left')
        + expand_limits(x=len(list(withdrawals_total.items())) + 2) + labs(x='Scenario', y='Million cubic meters of water per year (Mm3/yr)'))
#    print(p)    
    p.save('Water.pdf', height=5, width=7.5)
    
def water_plot_per_cluster(withdrawals_per_cluster, cluster, admin_names):
    '''
    '''
    for scenario, value in withdrawals_per_cluster.items():
        min_low = 0
        p = (ggplot() + coord_flip())
        order = value[['FinalIrrigationWater',
                       'PopulationWater']].sum(axis=1).sort_values().index.tolist()
        order = [str(x) for x in order]
#        order_cat = CategoricalDtype(categories=order, ordered=True)
        highs = value[['FinalIrrigationWater',
                    'PopulationWater', 'Cluster']]
        lows = value[['IrrigationReusedWater',
                    'PopulationReusedWater', 'Cluster']]
        highs = highs.melt(id_vars=['Cluster'])
        highs['value'] /= 1000000
        highs['Cluster'] = [str(x) for x in highs['Cluster']]
        highs['Cluster'] = highs['Cluster'].astype('category')
#        highs['cluster_cat'] = highs['Cluster'].astype(str).astype(order_cat)
        highs['cluster_cat'] = highs['Cluster'].cat.reorder_categories(order, ordered=True)
        lows = lows.melt(id_vars=['Cluster'])
        lows['value'] /= -1000000
        lows['Cluster'] = [str(x) for x in lows['Cluster']]
        lows['Cluster'] = lows['Cluster'].astype('category')
        lows['cluster_cat'] = lows['Cluster'].cat.reorder_categories(order, ordered=True)
        if min(lows['value']) < min_low:
            min_low = min(lows['value'])
        
        p = (p + geom_bar(highs, aes(x='cluster_cat', y='value', fill='variable'), stat='identity', position='stack', width=0.5)
            + geom_bar(lows, aes(x='cluster_cat', y='value', fill='variable'), stat='identity', position='stack', width=0.5))
    
        text_df_1 = pd.DataFrame({'label': ["Reused water\nfor irrigation"], 
                                'X': [len(set(highs['Cluster'])) + 2], 'Y': [min_low / 4]})
        text_df_2 = pd.DataFrame({'label': ['Water extractions\nfor population and irrigation'], 
                                'X': [len(set(highs['Cluster'])) + 2], 'Y': [- min_low / 4]})
        
        p = (p + theme_minimal() + theme(legend_position = 'top', legend_title=element_blank()) 
            + geom_hline(yintercept = 0, color ="black") + scale_fill_brewer(type='div', palette='PuOr', labels=['Irrigation extractions','Population extractions','Reused water from irrigation','Reused water from population'])
            + geom_text(text_df_1, aes(x='X', y='Y', label='label'), ha='right')
            + geom_text(text_df_2, aes(x='X', y='Y', label='label'), ha='left')
            + expand_limits(x=len(set(highs['Cluster'])) + 4) + labs(x='Cluster', y='Million cubic meters of water per year (Mm3/yr)'))
#        print(p)  
        if cluster[scenario] == 'y':
            p = (p + scale_x_discrete(labels=[admin_names.loc[int(float(x)),'NAME_1'] +
                                              ' (' + admin_names.loc[int(float(x)),'ISO'] + ')' for x in order])
                + labs(x='Province'))
            
        p.save('Water plot - ' + scenario.replace('\n', '') + '.pdf', height=8, width=7)
    
def tech_plot(data, pop_dic, agri_dic, order):
    '''
    '''
    order = order[1:]
    df_end = pd.DataFrame(columns=['scenario', 'value', 'type', 'tech'])
    for scenario, tech in data.items():
        pop_data = tech.groupby('PopulationLeastCostTechnology')['PopulationReclaimedWater'].sum()
        agri_data = tech.groupby('IrrigationLeastCostTechnology')['IrrigationReclaimedWater'].sum()
        pop_df = pd.DataFrame({'value': pop_data, 'type': 'Population\nwastewater', 'tech': [pop_dic[str(x)] for x in pop_data.index]})
        agri_df = pd.DataFrame({'value': agri_data, 'type': 'Irrigation\ntailwater', 'tech': [agri_dic[str(x)] for x in agri_data.index]})
        df = pop_df.append(agri_df)
        df['scenario'] = scenario
        df['value'] = df['value'].replace(0, np.nan)
        df = df.dropna(subset=['value'])
        df_end = df_end.append(df)
    df_end['category'] = df_end['scenario'] + df_end['type']
    df_end['total'] = df_end['category'].map(df_end.groupby('category')['value'].sum())
    
    df_end['scenario'] = df_end['scenario'].astype('category')
    df_end['scenario_cat'] = df_end['scenario'].cat.reorder_categories(order, ordered=True)
    
    p = (ggplot(df_end) + geom_bar(aes(x='type', y='value', fill='tech'), stat='identity', position='fill', width=0.8, size=0.3, color='gray') +
        geom_text(aes(x='type', y='value/total', fill='tech', label='[str(int(x/t * 100)) + "%" if x/t>0.05 else "" for x, t in zip(value,total)]'), size=7, color='#F7F7F7', position = position_stack(vjust=0.5)) +
        labs(y='Percentage of water treated by technology') + facet_wrap('~scenario') +
        scale_fill_brewer(type='div', palette='Brbg') + 
        scale_y_continuous(labels=lambda l: ["%d%%" % (v * 100) for v in l]) +
#        scale_y_continuous(expand = [0, 0]) +
        theme_minimal() +
#        theme_classic() +
        theme(legend_title=element_blank(),
              axis_title_x=element_blank(),
              axis_title_y=element_text(color='black'),
              legend_position = 'top'))
#    print(p)
    p.save('Technology plot - ' + scenario.replace('\n', '') + '.pdf', height=5, width=7)
    
def centroid(data):
#    length = data.df.groupby('Cluster')['X'].count()
    centroids = data.df.groupby('Cluster').agg({'X':'mean','Y':'mean','Cluster':'first'})
    centroids.rename(columns = {'Cluster': 'Centroid'}, inplace=True)
    return centroids

    
    
    
    
    
    
    
    
    
    