'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''
import pandas as pd
import geopandas as gpd
from get_data import stations
import plot
import methods
import flowsignatures
import geoprocessing
import clustering

"""
STEP 1 - GET AND PLOT ALL FLOW STATIONS DATA
"""
bhsf = gpd.read_file(r'shapefiles\BHSF_WGS1984.shp')
brazil_flu = gpd.read_file(r'shapefiles\Brazil_FlowStations_WithData.shp') #File provided by the HidroData repository from github.com/wallissoncarvalho/HidroData
bhsf_flu = brazil_flu[brazil_flu.geometry.within(bhsf.geometry[0])] #Selecting the stations within the São Francisco basin
observed_flows_consisted = stations(list(bhsf_flu['Code']),'3',only_consisted=True) #Downloading all consisted stations data
observed_flows_consisted.to_pickle(r'data/all_observed_consisted_flows.pkl') #Saving all stations data

"""
STEP 2 - REMOVING AFFECTED STATIONS BY RESERVOIRS
"""
affected_stations = pd.read_pickle(r'data\affected_stations.pkl').melt()['value'].dropna().to_list()
affected_stations = affected_stations + ['44950000'] #Adding an inconsistent station to drop
affected_stations = affected_stations + ['44740000'] # The station 44740000 is the same that 44090000
observed_flows_consisted.drop(affected_stations,axis=1,inplace=True) #Drop selected stations data

"""
STEP 3 - PLOTTING AVALIABLE DATA
"""
plot.gantt(observed_flows_consisted,r'graphics/All Stations Data Avaibility Consisted') #Plotting all stations data avaiability
asy = plot.available_stations_year(observed_flows_consisted) #Plotting station availability per year

"""
STEP 4 - FILTERING STATIONS
"""
observed_flows = methods.stations_filter(observed_flows_consisted,start_date='01/01/1995',end_date='31/12/2014')
observed_flows.to_pickle(r'data/observed_flows.pkl') #Saving filtered data
observed_flows_shapefile=bhsf_flu.loc[bhsf_flu['Code'].isin(list(observed_flows.columns))]
observed_flows_shapefile.to_file(r'shapefiles\observed_flows.shp')

"""
STEP 5 - FLOW SIGNATURES GENERATION
"""
signatures = flowsignatures.all_signatures(observed_flows)
signatures.to_pickle(r'data/signatures.pkl') #Saving signatures data

"""
STEP 6 - PHYSIOGRAFIC DATA
"""
physiographic_data = geoprocessing.physiographic_data(r'shapefiles\Watersheds.shp',r'shapefiles\LongestFlowPath.shp',r'raster/mdt_23s.tif')
physiographic_data.to_pickle(r'data/physiographic_data.pkl') #Saving physiographic data

"""
STEP 7 - GENERATING AND STANDARIZING ALL PARAMETERS
"""
all_parameters = pd.concat([physiographic_data,signatures],axis=1, sort=True)
all_parameters['Q90'] = all_parameters['Q90']/all_parameters['AreaKm2']
all_parameters['Q10'] = all_parameters['Q10']/all_parameters['AreaKm2']
all_parameters['Qmean'] = all_parameters['Qmean']/all_parameters['AreaKm2']
all_parameters=all_parameters.rename(columns={'AreaKm2':'Area', 'Slp1085':'$S_{10-85}$', 'Elev_Mean':'$E_{mean}$',
                                              'Elev_std':'$E_{std}$', 'Q90':'$Q_{90}$','Qmean':'$Q_{mean}$',
                                              'Q10':'$Q_{10}$','RBF':'$RB_{Flash}$','IBF':'$I_{BF}$'}) #Renaming columns to plot
all_parameters = methods.standard_data(all_parameters)
all_parameters.to_pickle(r'data/all_parameters.pkl') #Saving all parameters
"""
STEP 8 - ANALYZING CORRELATIONS
"""
#Calculate Correlations - All Parameters
correlation = all_parameters.corr(method='spearman')
calc_correlations = methods.calc_high_correlations(correlation)
plot.correlation_matrix(all_parameters)

#Removing High Correlations
final_parameters = all_parameters.drop(['Area','AC','CV'],axis=1)
final_parameters.to_pickle(r'data/final_parameters.pkl') #Saving final parameters

#Calculate Correlations - Final Parameters
correlation_final = all_parameters.corr(method='spearman')
calc_correlations_final = methods.calc_high_correlations(correlation_final)
plot.correlation_matrix(final_parameters)

"""
STEP 9 - CLUSTERING ASSESSMENT
"""
cluster = clustering.kmeans_ward_evaluation(all_parameters)
plot.cluster_evaluation(final_parameters)
plot.dendogram(final_parameters)
