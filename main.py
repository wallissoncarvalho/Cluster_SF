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
natural_flows = pd.read_pickle(r'data\natural_flows.pkl')

#GENERATING FLOW STATIONS SHAPEFILE
flow_stations = geoprocessing.shape_estacoes_area(observed_flows,r'data/stations_hidroweb.xls',r'shapefiles/BHSF_WGS1984.shp')
affected_stations = pd.read_pickle(r'data\affected_stations_final.pkl')
list_affected = []
for column in affected_stations.columns:
    list_affected = list_affected+list(affected_stations[column].dropna())
flow_stations=flow_stations.drop(list(map(int,list_affected)))
flow_stations.to_file(r'shapefiles/flow_stations.shp')

#GENERATION FLOW SIGNATURES
observed_flows = pd.read_pickle(r'data\observed_flows.pkl')
natural_flows = pd.read_pickle(r'data\natural_flows.pkl')
all_flows = pd.concat([observed_flows,natural_flows],axis=1)
signatures = flowsignatures.all_signatures(all_flows)
signatures = signatures.drop(list_affected)



#GENERATING Physiographic Data
physiographic_data = geoprocessing.physiographic_data(r'shapefiles\Watersheds.shp',r'shapefiles\LongestFlowPath.shp',r'raster/mdt_23s.tif')


#GENERATING ALL PARAMETERS
all_parameters = pd.concat([physiographic_data,signatures],axis=1, sort=True)
all_parameters['Q90'] = all_parameters['Q90']/all_parameters['AreaKm2']
all_parameters['Q10'] = all_parameters['Q10']/all_parameters['AreaKm2']
all_parameters['Qmean'] = all_parameters['Qmean']/all_parameters['AreaKm2']
all_parameters=all_parameters.dropna()
all_parameters.pop('AreaKm2')
correlation = all_parameters.corr(method='spearman')
plot.plot_correlation_matrix(all_parameters)

#NORMALIZING ALL PARAMETERS
for column in all_parameters.columns:
    all_parameters[column] = (all_parameters[column]-all_parameters[column].mean())/all_parameters[column].std()

signatures = signatures.dropna()
for column in signatures.columns:
    signatures[column] = (signatures[column]-signatures[column].mean())/signatures[column].std()

#CLUSTERING PROCESS
cluster = clustering.kmeans_ward_evaluation(signatures)
#cluster_points=clustering.kmeans_ward_shape(standard_data_cluster,2,r'shapefiles\Watersheds.shp')
#cluster_points.to_file(r'shapefiles\cluster_points.shp')
