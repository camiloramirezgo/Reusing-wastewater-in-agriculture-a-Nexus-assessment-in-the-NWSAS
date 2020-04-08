#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:29:27 2018

@author: Camilo Ramirez Gomez
"""
from swtrt import *
import time

# Column names of the input data table, containing all layers information,
# this names must match the names of such columns, or if the dataframe is beign created,
# then the names of each layer file must match the ones efined here.
FN_REGION = 'Region'
FN_X_COORDINATE = "X" # X coordinate field name
FN_Y_COORDINATE = "Y" # Y coordinate field name
FN_ELEVATION = "Elevation" # Elevation layer field name
FN_GWD = "GroundwaterDepth" # GWD layer field name
FN_LANDCOVER = "Landcover" # Landcover layer field name
FN_POPULATION = "Population" # Population layer field name
FN_TDS = "TDS" # TDS
FN_TDS_CI = "TDS_CI" # TDS in the CI layer field name
FN_TDS_CT = "TDS_CT" # TDS in the CT layer field name
FN_ROAD_DIST = "RoadDistance" # Road distance layer field name
FN_POP_CALIB = "PopCalibrated" # Population calibrated layer field name
FN_POP_FUTURE = 'PopulationFuture' # Future population layer field name
FN_SLOPE = 'Slope' # Slope layer field name
FN_URBAN = 'IsUrban' # Site type layer field name. Idicates whether the site is urban (0 or 1)
FN_IRR_AREA = 'IrrigatedArea'
FN_IRR_WATER = 'IrrigationWater'
FN_POP_WATER = 'PopulationWater'
FN_TOT_WITHDRAWALS = 'TotalWithdrawals'
FN_GWS = 'GroundwaterStress'
FN_GWS_END = 'GroundwaterStressEnd'
FN_CLUSTER = 'Cluster'

# Columns in the specs file must match these exactly
SPE_REGION = 'Region'
SPE_START_YEAR = 'StartYear'
SPE_END_YEAR = 'EndYear'
SPE_POP = 'PopulationStartYear'  # The actual population in the base year
SPE_URBAN = 'UrbanRatioStartYear'  # The ratio of urban population (range 0 - 1) in base year
SPE_POP_FUTURE = 'PopulationEndYear'
SPE_URBAN_FUTURE = 'UrbanRatioEndYear'
SPE_URBAN_MODELLED = 'UrbanRatioModelled'  # The urban ratio in the model after calibration (for comparison)
SPE_URBAN_CUTOFF = 'UrbanCutOff'  # The urban cutoff population calirated by the model, in people per km2
SPE_URBAN_GROWTH = 'UrbanGrowth'  # The urban growth rate as a simple multplier (urban pop future / urban pop present)
SPE_RURAL_GROWTH = 'RuralGrowth'  # Same as for urban
SPE_URBAN_WATER = 'UrbanPopWaterUni'
SPE_RURAL_WATER = 'RuralPopWaterUni'
SPE_IRRIGATED_AREA = 'IrrigatedAreaHa'
SPE_IRRIGATED_AREA_FUTURE = 'IrrigatedAreaFutureHa'
SPE_IRRIGATION_WATER = 'IrrigationWaterPerHa'
SPE_IRRIGATION_HOURS = 'IrrigationHours'
SPE_TOTAL_WITHDRAWALS = 'TotalWithdrawals'
SPE_RECHARGE_RATE = 'RechargeRate'
SPE_ENVIRONMENAL_FLOW = 'EnvironmentalFlow'
SPE_GW_DIAMETER = 'GroundwaterPipeD'
SPE_GW_DENSITY = 'GroundwaterDensity'
SPE_GW_VISCOSITY = 'GroundwaterViscosity'
SPE_GW_ROUGHNESS = 'GroundwaterRoughness'
SPE_PUMP_EFFICIENCY = 'PumpEfficiency'
SPE_ELEC_PRICE = 'ElectricityPrice'  # Grid price of electricity in USD/kWh
SPE_CLUSTERING = 'Clustering'
SPE_LAT = 'lat'
SPE_ELEVATION = 'elevation'
SPE_WIND = 'wind'
SPE_SRAD = 'srad'
SPE_TMIN = 'tmin'
SPE_TMAX = 'tmax'
SPE_TAVG = 'tavg'
SPE_ETO = 'eto'
SPE_TDS_THRESHOLD = 'tdsThreshold'
#SPE_MAX_ROAD_DIST = 'MaxRoadDist'
#SPE_INCOME_PER_HA = 'IncomePerHa'

SPE_MIN_POP = 30
SPE_MIN_IRRIGATED = 3
SPE_CLUSTER_NUM = 40
POP_WATER_FRACTION = 0.7
POP_REUSED_WATER = 0.9
AGRI_WATER_FRACTION = 0.3
AGRI_NON_RECOVERABLE = 0.2
AGRI_WATER_REQ = 8500

lyr_names = {"Region": FN_REGION,
             "X": FN_X_COORDINATE,
             "Y": FN_Y_COORDINATE,
             "Elevation": FN_ELEVATION,
             "GWD": FN_GWD,
             "Landcover": FN_LANDCOVER,
             "Population": FN_POPULATION,
             "TDS_CI": FN_TDS_CI,
             "TDS_CT": FN_TDS_CT,
             "TDS": FN_TDS,
             "RoadDistance": FN_ROAD_DIST,
             "PopCalibrated": FN_POP_CALIB,
             'PopulationFuture': FN_POP_FUTURE,
             'Slope': FN_SLOPE,
             'IsUrban': FN_URBAN,
             'IrrigatedArea': FN_IRR_AREA,
             'IrrigationWater': FN_IRR_WATER,
             'PopulationWater': FN_POP_WATER,
             'TotalWithdrawals': FN_TOT_WITHDRAWALS,
             'GroundwaterStress': FN_GWS,
             'GroundwaterStressEnd': FN_GWS_END,
             'Cluster': FN_CLUSTER}

urban_wastewater = {"ss": 900,
                    "N": 40,
                    "P": 20,
                    "BOD": 500,
                    "COD": 500}

treated_wastewater = {"ss": 100,
                      "N": 10,
                      "P": 2,
                      "BOD": 50,
                      "COD": 50}

func_variables = {"a": 0,
                  "ss": round((urban_wastewater['ss'] - treated_wastewater['ss']) / urban_wastewater['ss'], 2),
                  "N": round((urban_wastewater['N'] - treated_wastewater['N']) / urban_wastewater['N'], 2),
                  "P": round((urban_wastewater['P'] - treated_wastewater['P']) / urban_wastewater['P'], 2),
                  "BOD": round((urban_wastewater['BOD'] - treated_wastewater['BOD']) / urban_wastewater['BOD'], 2),
                  "COD": round((urban_wastewater['COD'] - treated_wastewater['COD']) / urban_wastewater['COD'], 2),
                  "x": "population",
                  "v": "water",
                  "E": 'exp'}


module = int(input('Select module: 1) Scenario analysis 2) Graphics 3) Cancel: '))

if module == 1:
    # specs_path = str(input("Enter the path for the excel file containing all scenarios specifications: "))
    specs_path = 'Scenarios.xlsx'
    xls_specs = pd.ExcelFile(specs_path)
    
    input_scenarios = str(input('The following scenarios were found, {}, please write the number of the scenarios that you want to run separated by commas, or type "all" to run all scenarios: '.format(', '.join([str(i + 1) + ') ' + x for i, x in enumerate(xls_specs.sheet_names)]))))
    
    scenarios = multiple_sheet_excel(xls_specs, input_scenarios)
    
    pop_treatment_systems_path = 'Treatment Systems - population.xlsx'
    agri_treatment_systems_path = 'Treatment Systems - agriculture.xlsx'
    # pop_treatment_systems_path = str(input("Enter the path for the excel file containing all population treatment systems specifications: "))
    # agri_treatment_systems_path = str(input("Enter the path for the excel file containing all agricultural treatment systems specifications: "))
    
    xls_pop_treatment_systems = pd.ExcelFile(pop_treatment_systems_path)
    xls_agri_treatment_systems = pd.ExcelFile(agri_treatment_systems_path)
    pd.options.display.max_colwidth = 100
    
    treatment_systems_pop = multiple_sheet_excel(xls_pop_treatment_systems, 'all')
    treatment_systems_agri = multiple_sheet_excel(xls_agri_treatment_systems, 'all')
    
    #specs = pd.read_excel(specs_path)
    create_dataframe = int(input('1) Create dataframe from separate layer files, 2) Load dataframe from file, 3) Cancel: '))
    
    #    dir_files = str(input('Enter the file directory of all GIS layers: '))
    if create_dataframe == 1:
    #    dir_files = "/Users/camo/Box Sync/SEE Master/Master Thesis/QGIS Analysis/Output Data/Rescaled 1km"
######## Low GWD: ###############
        # sensitivity_vars = {FN_GWD: -10}
        # sensitivity_func = 'sum'
############################################
######## High GWD: ###############
        # sensitivity_vars = {FN_GWD: 10}
        # sensitivity_func = 'sum'
############################################
######## Low TDS: ###############
        # sensitivity_vars = {FN_TDS: 0.5}
        # sensitivity_func = 'times'
############################################
######## High TDS: ###############
        # sensitivity_vars = {FN_TDS: 1.5}
        # sensitivity_func = 'times'
############################################
######## No sensitivity case: ###############
        sensitivity_vars = {}
        sensitivity_func = None
############################################
        # dir_files = str(input("Enter the path for the folder containing all layers in csv format: "))
        dir_files = 'Test data'
        cell_area = int(input('Enter the cell area size in km: '))
        file_name = str(input("Enter the name for the file: "))
        main_data = DataFrame(create_dataframe = True, dir_files = dir_files, lyr_names = lyr_names, 
                              file_name = file_name, save_csv = True, cell_area = cell_area, 
                              sensitivity_vars = sensitivity_vars, sensitivity_func = sensitivity_func)
    elif create_dataframe == 2:
        # file_name = str(input('Enter the name of the input file: '))
        file_name = 'nwsas_10km_low_tds'
        cell_area = int(input('Enter the cell area size in km: '))
        main_data = DataFrame(create_dataframe = False, lyr_names = lyr_names, input_file = file_name, save_csv = False, cell_area = cell_area)
    elif create_dataframe == 3:
        exit()
    
    run_all = 0
    all_procedures = False
    for scenario, specs in scenarios.items():
        data = DataFrame(empty = True)
        data.copy(main_data)
        
        if run_all == 0:
            choice = int(input('1) Run the "{}" scenario 2) Run all procedures on all scenarios 3) Skip scenario: '.format(scenario)))
            if choice == 2:
                run_all = 1
                choice = 1
                all_procedures = True
                calibrate_pop = 'all'
        
        if choice == 1:
            print("\n------------------------------\nRunning {} scenario\n------------------------------".format(scenario))
            start_time = time.time()
            clustering = str(specs.loc[0, SPE_CLUSTERING])
            if not all_procedures:
                calibrate_pop = str(input('Calibrate urban and rural population? (y/n) (type "all" to run all procedures): ')).lower()
            
            if calibrate_pop == "all":
                all_procedures = True
                enter_irrigation_system = "y"
                calculate_population_water = "y"
                enter_groundwater_stress = "y"
                calculate_groundwater_pumping = "y"
                calculate_desalinisation = "y"
                calculate_energy_costs = "y"
                calculate_treatment = "y"
                create_csv_tables = "y"
            else:
                enter_irrigation_system = str(input('Calibrate irrigated area and calculate irrigation water needs? (y/n): ')).lower()
                calculate_population_water = str(input('Calculate population water needs? (y/n): ')).lower()
                enter_groundwater_stress = str(input('Calculate Groundwater Stress indicator? (y/n): ')).lower()
                calculate_groundwater_pumping = str(input('Calculate Groundwater pumping energy? (y/n): ')).lower()
                calculate_desalinisation = str(input('Calculate desalination energy? (y/n): ')).lower()
                calculate_energy_costs = str(input('Calculate irrigation energy costs and share in the income? (y/n): ')).lower()
                calculate_treatment = str(input('Calculate treatment system? (y/n): ')).lower()
                create_csv_tables = str(input('Save results? (y/n): '))
            
            if calibrate_pop == "y" or calibrate_pop == "all":
                print('\nCalibrating population...')
                data.df[lyr_names["IsUrban"]] = 0
                data.df[lyr_names["PopulationFuture"]] = None
                for region in specs[SPE_REGION]:
                    print('    - Region {}...'.format(region))
                    pop_actual = float(specs.loc[specs[SPE_REGION] == region, SPE_POP])
                    pop_future = float(specs.loc[specs[SPE_REGION] == region, SPE_POP_FUTURE])
                    urban_current = float(specs.loc[specs[SPE_REGION] == region, SPE_URBAN])
                    urban_future = float(specs.loc[specs[SPE_REGION] == region, SPE_URBAN_FUTURE])
                    urban_cutoff = 0
                    
                    urban_cutoff, urban_modelled = data.calibrate_pop_and_urban(region, pop_actual, pop_future, urban_current,
                                                                                urban_future, urban_cutoff)
                    specs.loc[specs[SPE_REGION] == region, SPE_URBAN_CUTOFF] = urban_cutoff
                    specs.loc[specs[SPE_REGION] == region, SPE_URBAN_MODELLED] = urban_modelled
                    
        
            if enter_irrigation_system == 'y':
                print('\nCalibrating irrigation area and calculating irrigation water needs...')
                data.df['IrrigatedAreaFuture'] = None
                for region in specs[SPE_REGION]:
                    print('    - Region {}...'.format(region))
                    total_irrigated_area = float(specs.loc[specs[SPE_REGION] == region, SPE_IRRIGATED_AREA])
                    irrigation_per_ha = float(specs.loc[specs[SPE_REGION] == region, SPE_IRRIGATION_WATER])
                    irrigated_area_growth = float(specs.loc[specs[SPE_REGION] == region, SPE_IRRIGATED_AREA_FUTURE]) / \
                                            float(specs.loc[specs[SPE_REGION] == region, SPE_IRRIGATED_AREA])
                            
                    data.calculate_irrigation_system(region, total_irrigated_area, irrigation_per_ha, irrigated_area_growth)
            
            
            if calculate_population_water == 'y':
                print('\nCalculating population water needs...')
                for region in specs[SPE_REGION]:
                    print('    - Region {}...'.format(region))
                    urban_uni_water = float(specs.loc[specs[SPE_REGION] == region, SPE_URBAN_WATER])
                    rural_uni_water = float(specs.loc[specs[SPE_REGION] == region, SPE_RURAL_WATER])
                    data.calculate_population_water(region, urban_uni_water, rural_uni_water)
            
            
            if enter_groundwater_stress == 'y':
                print('\nCalculating Groundwater Stress indicator...')
                data.total_withdrawals()
                data.df['RechargeRate'] = 0
                data.df['EnvironmentalFlow'] = 0
                for region in specs[SPE_REGION]:
                    print('    - Region {}...'.format(region))
                    recharge_rate = float(specs.loc[specs[SPE_REGION] == region, SPE_RECHARGE_RATE])
                    environmental_flow = float(specs.loc[specs[SPE_REGION] == region, SPE_ENVIRONMENAL_FLOW])
                    data.recharge_rate(region, recharge_rate, environmental_flow)
                    data.groundwater_stress(region, data.df['TotalWithdrawals'])
                    
            
            if calculate_groundwater_pumping == 'y':
                print('\nCalculating groundwater pumping energy intensity...')
                for region in specs[SPE_REGION]:
                    print('    - Region {}...'.format(region))
                    groundwater_diameter = float(specs.loc[specs[SPE_REGION] == region, SPE_GW_DIAMETER])
                    groundwater_roughness = float(specs.loc[specs[SPE_REGION] == region, SPE_GW_ROUGHNESS])
                    groundwater_density = float(specs.loc[specs[SPE_REGION] == region, SPE_GW_DENSITY])
                    groundwater_viscosity = float(specs.loc[specs[SPE_REGION] == region, SPE_GW_VISCOSITY])
                    pump_efficiency = float(specs.loc[specs[SPE_REGION] == region, SPE_PUMP_EFFICIENCY])
                    irrigation_hours = float(specs.loc[specs[SPE_REGION] == region, SPE_IRRIGATION_HOURS])
                    groundwater_pipe = PipeSystem(groundwater_diameter, groundwater_roughness)
                    data.groundwater_pumping_energy(region = region, hours = irrigation_hours, 
                                                    density = groundwater_density, delivered_head = 0, pump_efficiency = pump_efficiency, calculate_friction = False,
                                                    viscosity = groundwater_viscosity, pipe = groundwater_pipe)
                    
            
            if calculate_desalinisation == 'y':
                print('\nCalculating groundwater desalination energy intensity...')
                for region in specs[SPE_REGION]:
                    print('    - Region {}...'.format(region))
                    osmosis_system = ReverseOsmosis(0.95, 0.85)
                    data.df['GroundwaterSolutes'] = 'NaCl'
                    data.df['GroundwaterTemperature'] = 25
                    threshold = float(specs.loc[specs[SPE_REGION] == region, SPE_TDS_THRESHOLD])
                    data.reverse_osmosis_energy(region, threshold, osmosis_system)
            
            
            if calculate_energy_costs == 'y':
                print('\nCalculating irrigation energy needs...')
    #            data.df['IrrigationEnergyCost'] = 0
    #            data.df['IrrigationWaterCost'] = 0
    #            data.df['IncomeEnergyCostShare'] = 0
                for region in specs[SPE_REGION]:
                    print('    - Region {}...'.format(region))
                    data.total_irrigation_energy()
    #                electricity_price = float(specs.loc[specs[SPE_REGION] == region, SPE_ELEC_PRICE])
    #                income_per_ha = float(specs.loc[specs[SPE_REGION] == region, SPE_INCOME_PER_HA])
    #                data.total_energy_irrigation_cost(region, electricity_price)
    #                data.income_energy_cost_share(region, income_per_ha)
                    
            if clustering == 'y':
                print('\nRunning clustering algorithm for population and irrigated land areas...')
                try:
                    del data.df['Cluster']
                except:
                    print('Creating clusters...')
                data.clustering_algorithm(population_min = SPE_MIN_POP, irrigated_min = SPE_MIN_IRRIGATED, cluster_num = SPE_CLUSTER_NUM, clusterize = True)
                data.df = data.newdf
            else:
                data.df.loc[data.df['Cluster'] == 0,'Cluster'] = None
                data.clustering_algorithm(population_min = SPE_MIN_POP, irrigated_min = SPE_MIN_IRRIGATED, cluster_num = SPE_CLUSTER_NUM, clusterize = False)
                
            data.df['PopulationPerCluster'] = None
            data.df['PopulationFuturePerCluster'] = None
            data.df['IrrigatedAreaPerCluster'] = None
            data.df['IrrigatedAreaFuturePerCluster'] = None
            data.df['PopulationWaterPerCluster'] = None
            data.df['PopulationWaterFuturePerCluster'] = None
            data.df['IrrigationWaterPerCluster'] = None
            
            print('\nCalculating per-cluster data...')
            for cluster in set(data.df['Cluster'].dropna()):
                data.calculate_per_cluster(cluster, 'Population', 'PopulationFuture', SPE_MIN_POP)
                data.calculate_per_cluster(cluster, 'PopulationFuture', 'PopulationFuture', SPE_MIN_POP)
                data.calculate_per_cluster(cluster, 'IrrigatedArea', 'IrrigatedArea', SPE_MIN_IRRIGATED)
                data.calculate_per_cluster(cluster, 'IrrigatedAreaFuture', 'IrrigatedArea', SPE_MIN_IRRIGATED)
                data.calculate_per_cluster(cluster, 'PopulationWater', 'PopulationFuture', SPE_MIN_POP)
                data.calculate_per_cluster(cluster, 'IrrigationWater', 'IrrigatedArea', SPE_MIN_IRRIGATED)
            
            print('\nCalculating final water extractions and reuse share...')
            data.calculate_reclaimed_water(POP_WATER_FRACTION, AGRI_WATER_FRACTION)
            data.get_eto(SPE_ETO, SPE_LAT, SPE_ELEVATION, SPE_WIND, SPE_SRAD, SPE_TMIN, SPE_TMAX, SPE_TAVG)
            data.get_storage(leakage=0.9, area_percent=0.02, storage_depth=3, agri_water_req=AGRI_WATER_REQ, agri_non_recoverable=AGRI_NON_RECOVERABLE)
            data.reused_water(pop_percentage_of_reuse = POP_REUSED_WATER)
            
            if calculate_treatment == 'y':
                years = float(specs.loc[0, SPE_END_YEAR] - specs.loc[0, SPE_START_YEAR])
                data.df['PopulationGrowthPerCluster'] = (data.df['PopulationFuturePerCluster'] / \
                                                         data.df['PopulationPerCluster'])**(1./years) - 1
                data.df['IrrigatedGrowthPerCluster'] = (data.df['IrrigatedAreaFuturePerCluster'] / \
                                                         data.df['IrrigatedAreaPerCluster'])**(1./years) - 1
                
                data.population_opex = {}
                for name, system in treatment_systems_pop.items():
                    print('\nCalculating {} treatment system CAPEX, OPEX and energy requirements...'.format(name))
                    data.calculate_capex(treatment_system_name = name + 'CAPEX', treatment_system = system, values = func_variables,
                                         parameter = 'CAPEXFunction', variable = 'PopulationFuture', limit = 'Limit', limit_func = 'Population')
                    data.population_opex[name] = data.calculate_opex(treatment_system_name = name + 'OPEX', treatment_system = system, values = func_variables,
                                        water_fraction = POP_WATER_FRACTION, parameter = 'OPEXFunction', variable = 'Population', years = years)
                    data.calculate_treatment_energy(treatment_system_name = name + 'Energy', treatment_system = system, values = func_variables, 
                                               parameter = 'EnergyFunction', variable = 'Population')
                    
                data.irrigation_opex = {}
                for name, system in treatment_systems_agri.items():
                    print('\nCalculating {} treatment system CAPEX, OPEX and energy requirements...'.format(name))
                    data.calculate_capex(treatment_system_name = name + 'CAPEX', treatment_system = system, values = func_variables,
                                         parameter = 'CAPEXFunction', variable = 'IrrigationFuture', limit = 'Limit', limit_func = 'Irrigation')
                    data.irrigation_opex[name] = data.calculate_opex(treatment_system_name = name + 'OPEX', treatment_system = system, values = func_variables,
                                        water_fraction = AGRI_WATER_FRACTION, parameter = 'OPEXFunction', variable = 'Irrigation', years = years)
                    data.calculate_treatment_energy(treatment_system_name = name + 'Energy', treatment_system = system, values = func_variables, 
                                               parameter = 'EnergyFunction', variable = 'Irrigation')
                
                
            print('\nCalculating LCOWs for each treatment technology:')
            calculate_lcows(data, clusters = SPE_CLUSTER_NUM, variable = 'Population', water_fraction = POP_WATER_FRACTION, 
                            degradation_factor = 0.01, income_tax_factor = 1, years = years, discount_rate = 0.04, 
                            treatment_systems = treatment_systems_pop, variable_present = 'Population',
                            opex_data = data.population_opex)
            calculate_lcows(data, clusters = SPE_CLUSTER_NUM, variable = 'Irrigation', water_fraction = AGRI_WATER_FRACTION, 
                            degradation_factor = 0.01, income_tax_factor = 1, years = years, discount_rate = 0.04, 
                            treatment_systems = treatment_systems_agri, variable_present = 'IrrigatedArea', 
                            opex_data = data.irrigation_opex)
            
            print('\nChoosing the least-costs treatment system based on LCOW...')
            systems_pop = [a + 'LCOW' for a in list(treatment_systems_pop.keys())]
            class_name_pop = data.least_cost_technology(systems_pop, 'PopulationLeastCost')
            systems_agri = [a + 'LCOW' for a in list(treatment_systems_agri.keys())]
            systems = list(systems_agri)
            systems.append('IrrigationWaterCost')
            class_name_agri = data.least_cost_technology(systems_agri, 'IrrigationLeastCost')
            class_name = data.least_cost_system(['PopulationLeastCostTechnology','IrrigationLeastCostTechnology'],
                                                class_name_pop, class_name_agri)
    #        data.least_cost_option()
            
            print('\nCalculating final energy requirements...')
            data.calculate_final_energy(class_name_pop, class_name_agri)
            
            print('\nCalculating final Groundwater Stress indicator...')
            for region in specs[SPE_REGION]:
                print('    - Region {}...'.format(region))
                data.groundwater_stress(region, data.df['FinalWaterWithdrawals'], 'GroundwaterStressEnd')
            
            if create_csv_tables == 'y':
                print('\nCreating results files...')
        #        folder_name = str(input('Enter folder name: ')) 
#                if clustering == 'y': 
#                    clust_text = ' (per cluster)'
#                else:
#                    clust_text = ' (by province)'
                
                folder_name = file_name + ' - Results' #+clust_text
                scenario_folder = folder_name + '/' + scenario
                if not os.path.isdir(folder_name):
                    os.makedirs(folder_name)
                    os.makedirs(scenario_folder)
                    os.makedirs(scenario_folder + '/CSV')
                    os.makedirs(scenario_folder + '/Rasters')
                else:
                    if not os.path.isdir(scenario_folder):
                        os.makedirs(scenario_folder)
                        if not os.path.isdir(scenario_folder + '/CSV'):
                            os.makedirs(scenario_folder + '/CSV')    
                        if not os.path.isdir(scenario_folder + '/Rasters'):
                            os.makedirs(scenario_folder + '/Rasters')
                if (len(os.listdir(scenario_folder + '/CSV')) > 0) & (not all_procedures):
                    replace_files = str(input('The CSV folder already contains files, do you want to replace them? (y/n): '))
                    if replace_files == 'y':
                        delete_files(scenario_folder + '/CSV')
                if (len(os.listdir(scenario_folder + '/Rasters')) > 0) & (not all_procedures):
                    replace_files = str(input('The Rasters folder already contains files, do you want to replace them? (y/n): '))
                    if replace_files == 'y':
                        delete_files(scenario_folder + '/Rasters')
                
    #            urban_rural_pop = data.df.loc[:, ['X', 'Y', 'IsUrban']]
    #            urban_rural_pop['IsUrban'] = urban_rural_pop['IsUrban'] * 1
    #            urban_rural_pop.to_csv(scenario_folder + "/CSV/UrbanPopulation.csv", index = False)
                
    #            convert_m3_to_mm(data.cell_area, scenario_folder, data.df, 
    #                             'TotalWithdrawals',
    #                             'PopulationWater',
    #                             'IrrigationWater')
                if clustering == 'y':
                    centroids = centroid(data)
                    save_layers(scenario_folder, centroids, 'Centroid')
                    
                save_layers(scenario_folder, data.df,
                            'GroundwaterStress',
                            'GroundwaterStressEnd',
                            'LeastCostSystem',
                            'Cluster',
                            'IrrigatedArea')
                
    #            for system in treatment_systems.keys():
    #                save_layers(scenario_folder, data.df,
    #                        system + 'Energy',
    #                        system + 'CAPEX',
    #                        system + 'OPEX')
        
                print('    - Saving {} scenario dataframe...'.format(scenario))
                data.df.to_csv(scenario_folder + '/' + scenario + '.gz', index = False)
            
            print('\nPopulation classes:')
            print(set(data.df['PopulationLeastCostTechnology'].dropna()))
            print(class_name_pop)
            print('\nAgricultural classes:')
            print(set(data.df['IrrigationLeastCostTechnology'].dropna()))
            print(class_name_agri)
            print('\nSystem classes:')
            print(set(data.df['LeastCostSystem'].dropna()))
            print(class_name)
    #        print('\nLeast-cost options classes:')
    #        print(set(data.df['LeastCostOption'].dropna()))
    #        class_name_option = dict(class_name)
    #        class_name_option['-1'] = 'GroundwaterExtraction'
    #        print(class_name_option)
            
            end_time = time.time()
            print('\nTotal enlapsed time')
            print(str(np.floor((end_time - start_time) / 60)) + ' min, ' + str(round((end_time - start_time) % 60, 2)) + ' sec')
            
        elif choice == 3:
            print("\n...")
            
elif module == 2:
#    number_of_files = int(input('Enter the amount of files to load: '))
    # folder_path = str(input('Enter the path of the folder with the scenarios: '))
    folder_path = 'nwsas_10km_full - Results'
    admin_names = pd.read_csv('admin1 names.csv')
    admin_names = admin_names.set_index('Province')
#    folder_path = '/Users/camo/Box Sync/Master Thesis/Python Model/NWSAS_10km - Results'
    os.chdir(folder_path)
    # for root, dirs, files in os.walk('.', topdown=False):
        # for name in files:
           # print(os.path.join(root, name))
        # for name in dirs:
           # print(os.path.join(root, name))
    scenarios = np.array([x[1] for x in os.walk('.')][0])
    input_scenarios = str(input('The following scenarios were found, {}, please write the number of the scenarios that you want to plot separated by commas: '.format(', '.join([str(i + 1) + ') ' + x for i, x in enumerate(scenarios)]))))
    scenarios = scenarios[[int(x)-1 for x in input_scenarios.split(',')]]
    print(scenarios)
    is_baseline = int(input('Select the scenario from which the baseline will be extracted:\n    {}\nInput: '.format('\n    '.join([str(i + 1) + ') ' + x for i, x in enumerate(scenarios)]))))
    keep_scenario = input('Plot the {} scenario too?(y/n): '.format(scenarios[is_baseline-1]))
    data_list = []
    cluster = {}
    cell_area = int(input('Enter the cell area size in km: '))
    
    for scenario in scenarios:
        cluster[scenario.replace('- ','\n')] = str(input('Load cluster names for {} scenario?(y/n): '.format(scenario)))
        
    for scenario in scenarios:
#        file_name = str(input('Enter the path of the input file: '))
        print('\nLoading {} scenario...'.format(scenario))
        os.chdir(scenario)
        file_name = glob.glob("*.gz")
        file_name = file_name[0].split('.')[0]
        data_list.append(DataFrame(create_dataframe = False, lyr_names = lyr_names, input_file = file_name, save_csv = False,  cell_area = cell_area))
        os.chdir('..')
        
    #scenarios_name = str(input('Enter the names of the scenarios: '))
    #scenarios_name = 'Baseline, Water reuse \n(clustering), Water reuse \n(by province)'
#    scenarios_name = 'Baseline, Water reuse \n(by province), Water reuse \n(per cluster)'   
#    scenarios_name = [x.strip().replace('\\n','\n') for x in scenarios_name.split(',')]
    scenarios_name = ['Baseline']
    scenarios_name.extend([x.replace('- ','\n') for x in scenarios])
    
    energy_start = (scenarios_name[0], data_list[is_baseline - 1].df[['IrrigationDesalinationEnergy', 'IrrigationPumpingEnergy']].sum())
    gws_start = data_list[is_baseline - 1].df['TotalWithdrawals'].sum() / \
                (data_list[is_baseline - 1].df.loc[data_list[is_baseline - 1].df['CI'] == 1,'RechargeRate'].sum() - data_list[is_baseline - 1].df.loc[data_list[is_baseline - 1].df['CI'] == 1,'EnvironmentalFlow'].sum())
    
    energy_per_cluster = []
    energy_end = {}
    #scenarios_name[i]: energy_start
    gws_values = [gws_start]
#    total_withdrawals = data_list[0].df['TotalWithdrawals'].sum()
    withdrawals_per_cluster =  {}
    withdrawals_total = {}
    withdrawals_baseline = {}
    tech = {}
    for i, data in enumerate(data_list):
        energy_per_cluster.append(data.df.groupby('Cluster').agg({'FinalDesalinationEnergy': 'sum', 
                                                    'FinalPumpingEnergy': 'sum', 
                                                    'FinalTreatmentEnergy': 'first'}))
        energy_end[scenarios_name[i + 1]] = data.df.agg({'FinalDesalinationEnergy': 'sum', 'FinalPumpingEnergy': 'sum'})
        energy_end[scenarios_name[i + 1]]['FinalTreatmentEnergy'] = energy_per_cluster[i]['FinalTreatmentEnergy'].sum()
        gws_end = data.df['FinalWaterWithdrawals'].sum() / (data.df.loc[data.df['CI'] == 1,'RechargeRate'].sum() - data.df.loc[data.df['CI'] == 1,'EnvironmentalFlow'].sum())
        gws_values.append(gws_end)
        withdrawals_per_cluster[scenarios_name[i + 1]], withdrawals_total[scenarios_name[i + 1]], withdrawals_baseline[scenarios_name[i + 1]] = data.get_water_stats()
        withdrawals_per_cluster[scenarios_name[i + 1]]['Cluster'] = withdrawals_per_cluster[scenarios_name[i + 1]].index
        
        tech[scenarios_name[i + 1]] = data.df.groupby('Cluster').agg({'PopulationLeastCostTechnology': 'max', 'IrrigationLeastCostTechnology': 'max',
                                          'PopulationReclaimedWater': 'first', 'IrrigationReclaimedWater': 'first'})
    
    withdrawals_total['Baseline'] = withdrawals_baseline[scenarios_name[is_baseline]]
    
#    sensitivity_path = str(input('Enter the path of the folder with the scenarios: '))
    sensitivity_path = {'TDS': 'nwsas_10km_tds_sensitivity',
                        'Depth': 'nwsas_10km_gwd_sensitivity'}
    sensitivity_variables = {}
    os.chdir('..')
    for key, value in sensitivity_path.items():
        os.chdir(value)
        sensitivity_variables[key] = [x[1] for x in os.walk('.')][0]  
        os.chdir('..')
    data_list.clear()
    sensitivity_energy = {}
    
    for scenario in scenarios:
        sensitivity_energy[scenario.replace('- ','\n')] = pd.DataFrame({'Desalination energy': [], 'Pumping energy': []})
        if scenario.replace('- ','\n') == scenarios_name[is_baseline]:
            sensitivity_energy['Baseline'] = pd.DataFrame({'Desalination energy': [], 'Pumping energy': []})
        for key, item in sensitivity_variables.items():
            print('\nLoading {} sensitivity for {} scenario...'.format(key,scenario))
            for value in item:
                os.chdir(sensitivity_path[key] + '/' + value + '/' + scenario)
                file_name = glob.glob("*.gz")
                file_name = file_name[0].split('.')[0]
                sensitivity_data = DataFrame(create_dataframe = False, lyr_names = lyr_names, input_file = file_name, save_csv = False,  cell_area = cell_area)
                sensitivity_energy[scenario.replace('- ','\n')] = sensitivity_energy[scenario.replace('- ','\n')].append(pd.DataFrame({'Desalination energy': sensitivity_data.df.agg({'FinalDesalinationEnergy': 'sum'}).values,
                                                                              'Pumping energy': sensitivity_data.df.agg({'FinalPumpingEnergy': 'sum'}).values,
                                                                              'Treatment energy': sensitivity_data.df.groupby('Cluster').agg({'FinalTreatmentEnergy': 'first'}).sum(),
                                                                              'SensitivityVar': key}),
                                                                ignore_index=True)
                if scenario.replace('- ','\n') == scenarios_name[is_baseline]:
                    sensitivity_energy['Baseline'] = sensitivity_energy['Baseline'].append(pd.DataFrame({'Desalination energy': sensitivity_data.df.agg({'IrrigationDesalinationEnergy': 'sum'}).values,
                                                                              'Pumping energy': sensitivity_data.df.agg({'IrrigationPumpingEnergy': 'sum'}).values,
                                                                              'Treatment energy': [0],
                                                                              'SensitivityVar': key}),
                                                                ignore_index=True)
                    sensitivity_energy['Baseline']['Scenario'] = 'Baseline'
                    boolean_vec = sensitivity_energy['Baseline'].duplicated(subset=['Desalination energy'], keep=False)
                    sensitivity_energy['Baseline'].loc[boolean_vec, 'Desalination energy'] = None
                    boolean_vec = sensitivity_energy['Baseline'].duplicated(subset=['Pumping energy'], keep=False)
                    sensitivity_energy['Baseline'].loc[boolean_vec, 'Pumping energy'] = None
                    boolean_vec = sensitivity_energy['Baseline'].duplicated(subset=['Treatment energy'], keep=False)
                    sensitivity_energy['Baseline'].loc[boolean_vec, 'Treatment energy'] = None
    
                sensitivity_energy[scenario.replace('- ','\n')]['Scenario'] = scenario.replace('- ','\n')
                boolean_vec = sensitivity_energy[scenario.replace('- ','\n')].duplicated(subset=['Desalination energy'], keep=False)
                sensitivity_energy[scenario.replace('- ','\n')].loc[boolean_vec, 'Desalination energy'] = None
                boolean_vec = sensitivity_energy[scenario.replace('- ','\n')].duplicated(subset=['Pumping energy'], keep=False)
                sensitivity_energy[scenario.replace('- ','\n')].loc[boolean_vec, 'Pumping energy'] = None
                boolean_vec = sensitivity_energy[scenario.replace('- ','\n')].duplicated(subset=['Treatment energy'], keep=False)
                sensitivity_energy[scenario.replace('- ','\n')].loc[boolean_vec, 'Treatment energy'] = None
                os.chdir('../../..')
                
        
    class_name_pop = {'0': 'NaN', '1': 'Extended aeration', '2': 'Membrane bioreactor',
                      '3': 'Sequencing batch reactor', '4': 'Rotating biological contractors',
                      '5': 'Intermittent sand filter', '6': 'Trickling filter',
                      '7': 'Moving bed biofilm reactor'}
    class_name_agri = {'0': 'NaN', '1': 'Pond system', '2': 'Wetlands'}
    
    # order = ['Baseline','WWR FreeWapha *','WWR SubWapha *','WWR PrivWapha *','WWR HighWapc *','WWR LowWapc *','WWR per cluster','WWR by province']
    # order = ['Baseline','WWR by province','WWR per cluster']
    
    if keep_scenario == 'n':
        gws_values.pop(is_baseline)
        scenario = scenarios_name.pop(is_baseline)
        energy_end.pop(scenario)
        sensitivity_energy.pop(scenario)
        withdrawals_total.pop(scenario)
        withdrawals_per_cluster.pop(scenario)
        tech.pop(scenario)
    order = scenarios_name
    
    os.chdir(folder_path)
    gws_plot(gws_values, scenarios_name, order)
    energy_plot(energy_start, energy_end, sensitivity_energy, order)
    water_plot(withdrawals_total, order)
    water_plot_per_cluster(withdrawals_per_cluster, cluster, admin_names)
    tech_plot(tech, class_name_pop, class_name_agri, order)
    
#    path = '/Users/camo/Documents/SEE Master/Master Thesis/Python Model/NWSAS_1km - Results/Water reuse - (per cluster)'
##    centroids = centroid(data_list[1])
##    save_layers(path, centroids, 'Centroid') 
#    save_layers(path, data_list[1].df, 'IrrigatedArea')
    
    for scenario in order[1:]:
        sensTDS = sensitivity_energy[scenario].loc[sensitivity_energy[scenario]['SensitivityVar'] == 'TDS','Desalination energy']
        enerTDS = energy_end[scenario].loc['Desalination energy']
        
        sensDepth = sensitivity_energy[scenario].loc[sensitivity_energy[scenario]['SensitivityVar'] == 'Depth','Pumping energy']
        enerDepth = energy_end[scenario].loc['Pumping energy']
        
        enerTreat = energy_end[scenario].loc['Treatment energy']
        
        print(scenario + ': \n    - Desalination energy: ' +
              str(round((sensTDS[1] - enerDepth - enerTreat - enerTDS)/enerTDS * 100,2)) + '% to ' +
              str(round((sensTDS[0] - enerDepth - enerTreat - enerTDS)/enerTDS* 100,2)) + '%')
        
        print('    - Pumping energy: ' +
              str(round((sensDepth[3] - enerDepth - enerTreat)/enerDepth *100,2)) + '% to ' +
              str(round((sensDepth[2] - enerDepth - enerTreat)/enerDepth *100,2)) + '%')
        print('')
    




















     