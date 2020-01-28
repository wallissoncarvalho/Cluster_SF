import sklearn
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import geopandas as gpd


def kmeans_ward_evaluation(dados):
    classes =[]
    db = []
    for i in range(20):
        classes.append(i+2)
        kmeans = KMeans(n_clusters=i+2).fit(dados)
        labels = kmeans.labels_
        db.append(sklearn.metrics.davies_bouldin_score(dados, labels))
    db_k = pd.DataFrame({'classes':classes,'DB_KM':db})
    db_k.index = db_k.classes
    db_k.pop('classes')
    
    classes =[]
    si = []
    for i in range(20):
        classes.append(i+2)
        kmeans = KMeans(n_clusters=i+2).fit(dados)
        labels = kmeans.labels_
        si.append(sklearn.metrics.silhouette_score(dados, labels))
    si_k = pd.DataFrame({'classes':classes,'SI_KM':si})
    si_k.index = si_k.classes
    si_k.pop('classes')
    
    classes =[]
    db = []
    for i in range(20):
        classes.append(i+2)
        ward = sklearn.cluster.AgglomerativeClustering(n_clusters=i+2).fit(dados)
        labels = ward.labels_
        db.append(sklearn.metrics.davies_bouldin_score(dados, labels))
    db_w = pd.DataFrame({'classes':classes,'DB_W':db})
    db_w.index = db_w.classes
    db_w.pop('classes')

    classes =[]
    si = []
    for i in range(20):
        classes.append(i+2)
        ward = sklearn.cluster.AgglomerativeClustering(n_clusters=i+2).fit(dados)
        labels = ward.labels_
        si.append(sklearn.metrics.silhouette_score(dados, labels))
    si_w = pd.DataFrame({'classes':classes,'SI_W':si})    
    si_w.index = si_w.classes
    si_w.pop('classes')
    
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

