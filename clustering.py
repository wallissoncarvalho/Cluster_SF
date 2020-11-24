'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import sklearn
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import hydrobr

def kmeans_ward_evaluation(dados):
    classes =[]
    db = []
    for i in range(9):
        classes.append(i+2)
        kmeans = KMeans(n_clusters=i+2).fit(dados)
        labels = kmeans.labels_
        db.append(sklearn.metrics.davies_bouldin_score(dados, labels))
    db_k = pd.DataFrame({'NClusters':classes,'DB - K means':db})
    db_k.index = db_k.NClusters
    db_k.pop('NClusters')
    
    classes =[]
    si = []
    for i in range(9):
        classes.append(i+2)
        kmeans = KMeans(n_clusters=i+2).fit(dados)
        labels = kmeans.labels_
        si.append(sklearn.metrics.silhouette_score(dados, labels))
    si_k = pd.DataFrame({'NClusters':classes,'SC(i) - K means':si})
    si_k.index = si_k.NClusters
    si_k.pop('NClusters')
    
    classes =[]
    db = []
    for i in range(9):
        classes.append(i+2)
        ward = sklearn.cluster.AgglomerativeClustering(n_clusters=i+2).fit(dados)
        labels = ward.labels_
        db.append(sklearn.metrics.davies_bouldin_score(dados, labels))
    db_w = pd.DataFrame({'NClusters':classes,'DB - Ward':db})
    db_w.index = db_w.NClusters
    db_w.pop('NClusters')

    classes =[]
    si = []
    for i in range(9):
        classes.append(i+2)
        ward = sklearn.cluster.AgglomerativeClustering(n_clusters=i+2).fit(dados)
        labels = ward.labels_
        si.append(sklearn.metrics.silhouette_score(dados, labels))
    si_w = pd.DataFrame({'NClusters':classes,'SC(i) - Ward':si})    
    si_w.index = si_w.NClusters
    si_w.pop('NClusters')
    
    df = pd.concat([db_k, db_w, si_k, si_w],axis=1)
    return df

def kmeans_ward_shape(scaled_parameters,k):
    kmeans = KMeans(n_clusters=k).fit(scaled_parameters)
    label_km = kmeans.labels_
    ward = AgglomerativeClustering(n_clusters=k).fit(scaled_parameters)
    label_w = ward.labels_
    labels = pd.DataFrame({'Code':scaled_parameters.index,'Label_Kmeans':label_km, 'Label_Ward':label_w})
    labels.index = labels.Code
    labels.index = map(str, labels.index)
    labels.pop('Code') 
    def replace(x):
        if x==0:
            return 2
        elif x==1:
            return 1
        elif x==2:
            return 0
    labels['Label_Ward_New'] = labels['Label_Ward'].apply(replace)
    labels['Label_Ward'] = labels['Label_Ward_New']
    labels.pop('Label_Ward_New')

    stations = hydrobr.get_data.ANA.list_flow_stations()    
    stations.index = stations.Code
    stations.index = map(str, stations.index)   
    stations=stations[['Name','Latitude','Longitude']]
    stations = stations.join(labels)
    stations = stations.dropna()
    
    points=[Point(x) for x in zip(stations.Longitude,stations.Latitude)]
    crs={'proj':'latlong','ellps':'WGS84','datum':'WGS84','no_def':True} #SC WGS 
    stations=gpd.GeoDataFrame(stations,crs=crs,geometry=points)
    stations.to_file(r'shapefiles/Clustered_Stations.shp')
    return stations

