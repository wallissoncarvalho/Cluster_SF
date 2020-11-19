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
from sklearn.preprocessing import StandardScaler
import hydrobr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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

# Verifying the areas
stations_area = hydrobr.get_data.ANA.list_flow_stations()
stations_area = stations_area[stations_area.Code.isin(physiographic_data.index.to_list())]
stations_area.index = stations_area.Code
stations_area = pd.concat([stations_area[['DrainageArea']], physiographic_data[['AreaKm2']].rename(index={'Name':'Code'})], axis=1)
stations_area['AreaError'] = 100*abs((stations_area['AreaKm2'] - stations_area['DrainageArea'])/stations_area['AreaKm2'])
stations_area=stations_area[stations_area['AreaError']>25]

"""
STEP 7 - GENERATING ALL PARAMETERS
"""
signatures = pd.read_pickle(r'data/signatures.pkl')
physiographic_data = pd.read_pickle(r'data/physiographic_data.pkl')
all_parameters = pd.concat([physiographic_data,signatures],axis=1, sort=True)
all_parameters = all_parameters.loc[~all_parameters.index.isin(stations_area.index.to_list())]
all_parameters.to_pickle(r'data/all_parameters.pkl') #Saving all parameters

"""
STEP 8 - STANDARIZING ALL PARAMETERS
"""
all_parameters['Q90'] = all_parameters['Q90']/all_parameters['AreaKm2']
all_parameters['Q10'] = all_parameters['Q10']/all_parameters['AreaKm2']
all_parameters['Qmean'] = all_parameters['Qmean']/all_parameters['AreaKm2']
all_parameters = all_parameters.drop(['AreaKm2','PD'],axis=1)

#scaled_parameters = StandardScaler().fit_transform(all_parameters)
#scaled_parameters = pd.DataFrame(scaled_parameters, index=all_parameters.index, columns=all_parameters.columns)

scaled_parameters = MinMaxScaler().fit_transform(all_parameters)
scaled_parameters = pd.DataFrame(scaled_parameters, index=all_parameters.index, columns=all_parameters.columns)

'''
STEP 9 - PCA ANALYSIS
'''
plot.pca_components(scaled_parameters)
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(scaled_parameters)
principalComponents = pd.DataFrame(data = principalComponents,
                           columns = ['principal 1', 'principal 2','principal 3','principal 4'])


"""
STEP 10 - CLUSTERING ASSESSMENT
"""
cluster = clustering.kmeans_ward_evaluation(principalComponents)
plot.cluster_evaluation(principalComponents)
plot.dendogram(principalComponents)


