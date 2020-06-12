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
#import geoprocessing
#import clustering

"""
STEP 1 - GET AND PLOT ALL FLOW STATIONS DATA
"""
bhsf = gpd.read_file(r'shapefiles\BHSF_WGS1984.shp')
brazil_flu = gpd.read_file(r'shapefiles\Brazil_FlowStations_WithData.shp') #File provided by the HidroData repository from github.com/wallissoncarvalho/HidroData
#Selecting the stations within the São Francisco basin
bhsf_flu = brazil_flu[brazil_flu.geometry.within(bhsf.geometry[0])]
observed_flows_consisted = stations(list(bhsf_flu['Code']),'3',only_consisted=True) #Downloading all consisted stations data
observed_flows_consisted.to_pickle(r'data/all_observed_consisted_flows.pkl') #Saving all stations data

"""
STEP 2 - REMOVING AFFECTED STATIONS BY RESERVOIRS
"""
affected_stations = pd.read_pickle(r'data\affected_stations.pkl').melt()['value'].dropna().to_list()
affected_stations = affected_stations + ['44950000'] #UNCONSISTED STATION
observed_flows_consisted.drop(affected_stations,axis=1,inplace=True)

"""
STEP 3 - PLOTTING AVALIABLE DATA
"""
plot.gantt(observed_flows_consisted,r'graphics/All Stations Data Avaibility Consisted') #Plotting all stations data avaiability
asy = plot.available_stations_year(observed_flows_consisted)

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
signatures.to_pickle(r'data/signatures.pkl')

"""
STEP 6 - PHYSIOGRAFIC DATA
"""
