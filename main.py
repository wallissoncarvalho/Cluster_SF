'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import pandas as pd
import methods
import geoprocessing
import flowsignatures
import clustering
import plot





observed_flows = pd.read_pickle(r'data\observed_flows.pkl')


##GENERATING FLOW STATIONS SHAPEFILE
flow_stations = geoprocessing.shape_estacoes_area(observed_flows,r'data/stations_hidroweb.xls',r'shapefiles/BHSF_WGS1984.shp')
affected_stations = pd.read_pickle(r'data\affected_stations_final.pkl')

##STATIONS AFFECTED BY RESERVOIRS
list_affected = []
for column in affected_stations.columns:
    list_affected = list_affected+list(affected_stations[column].dropna())
flow_stations=flow_stations.drop(list(map(int,list_affected)))
flow_stations.to_file(r'shapefiles/flow_stations.shp')
list_affected = list_affected+ ['46105000']
##GENERATION FLOW SIGNATURES
signatures = flowsignatures.all_signatures(observed_flows)
signatures = signatures.drop(list_affected) #REMOVING STATIONS AFFECTEDS BY RESERVOIRS
signatures=pd.read_pickle(r'data/signatures.pkl')



##GENERATING Physiographic Data
physiographic_data = geoprocessing.physiographic_data(r'shapefiles\Watersheds.shp',r'shapefiles\LongestFlowPath.shp',r'raster/mdt_23s.tif')
physiographic_data.to_pickle(r'data/physiographic_data.pkl')
physiographic_data = pd.read_pickle(r'data/physiographic_data.pkl')


#GENERATING ALL PARAMETERS
all_parameters = pd.concat([physiographic_data,signatures],axis=1, sort=True)
all_parameters['Q90'] = all_parameters['Q90']/all_parameters['AreaKm2']
all_parameters['Q10'] = all_parameters['Q10']/all_parameters['AreaKm2']
all_parameters['Qmean'] = all_parameters['Qmean']/all_parameters['AreaKm2']
all_parameters=all_parameters.dropna()
all_parameters=all_parameters.rename(columns={'AreaKm2':'Area','Slp1085':'$S_{10-85}$','Elev_Mean':'$E_{mean}$','Elev_std':'$E_{std}$',
                                              'Q90':'$Q_{90}$','Qmean':'$Q_{mean}$','Q10':'$Q_{10}$','RBF':'$RB_{Flash}$',
                                              'SFDC':'$S_{FDC}$','IBF':'$I_{BF}$'})
all_parameters.to_pickle(r'data/all_parameters.pkl')


#ANALYZING CORRELATIONS
correlation = all_parameters.corr(method='spearman')
calc_correlations = methods.calc_high_correlations(correlation)
plot.correlation_matrix(all_parameters)


#REMOVING HIGH CORRELATIONS
all_parameters.pop('Area')
all_parameters.pop('$S_{FDC}$')
all_parameters.pop('AC')


#Standardizing ALL PARAMETERS
all_parameters = methods.standard_data(all_parameters)


##CLUSTERING PROCESS
cluster = clustering.kmeans_ward_evaluation(all_parameters)
plot.cluster_evaluation(all_parameters)

##PLOT WARD DENDOGRAM
plot.dendogram(all_parameters)

#CLUSTERING ASSESSMENT
all_parameters = pd.read_pickle(r'data/all_parameters.pkl')


