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

def kmeans_ward_shape(dados,n, shape_area_dir):
    kmeans = KMeans(n_clusters=n).fit(dados)
    label_km = kmeans.labels_
    ward = sklearn.cluster.AgglomerativeClustering(n_clusters=n).fit(dados)
    label_w = ward.labels_
    labels = pd.DataFrame({'Codigo':dados.index,'Label_Kmeans':label_km, 'Label_Ward':label_w})
    labels.index = labels.Codigo
    labels.index = map(str, labels.index)
    labels.pop('Codigo')
    shapefile = gpd.read_file(shape_area_dir)
    shapefile.index = shapefile['Name']
    shapefile = shapefile.join(labels)
    shapefile = shapefile.dropna()
    return shapefile

