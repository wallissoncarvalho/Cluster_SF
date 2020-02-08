'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import sklearn
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


def filter_stations(data,n_years=10,missing_percent=5,start_date=False,end_date=False):
    '''
    For this function, you need a DataFrame structure with flow stations on columns and the measurements at each time on the rows. 
    This function selects stations with at least n_years of data between the first and the last one registered flow.
    The selected stations can be filtered by a maximum missing data percentage (from 0% to 100%) inside a period of n_years.
    Moreover, you can choose a specific starting date and/or end date to clip your data.
    '''
    #CLIPING DATA BETWEEN DATES
    if start_date!=False and end_date!=False:
        data=data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    elif start_date!=False:
        data=data.loc[pd.to_datetime(start_date):]
    elif end_date!=False:
        data=data.loc[:pd.to_datetime(end_date)]
    
    #Selects the stations with at least n_years of data:
    stations = []
    for column in data.columns:
        series = data[column]
        series_drop = series.dropna()
        if len(series_drop)>0:
            anos = (series_drop.index[-1]-series_drop.index[0])/np.timedelta64(1,'Y')
            if anos>=n_years:
                stations.append(column)
    data=data[stations]
    
    #FILTRO 2: Checks if there is at least one period with 10 years of registered records and max missing_percent.
    stations=[]
    state = 0
    for column in data.columns:
        print('Loading {}%'.format(round(state/len(data.columns)*100,1)))
        series = data[column]
        series_drop = series.dropna()    
        periodos = []
        Start1 = series_drop.index[0]
        Finish1 = 0
        for i in range(len(series_drop)):
            if i!=0 and (series_drop.index[i]-series_drop.index[i-1])/np.timedelta64(1,'D') != 1:
                Finish1=series_drop.index[i-1]
                periodos.append(dict(Start=Start1,Finish=Finish1,Intervalo=(Finish1-Start1)/np.timedelta64(1,'Y')))
                Start1 = series_drop.index[i]
                Finish1 = 0
        Finish1 = series_drop.index[-1]
        periodos.append(dict(Start=Start1,Finish=Finish1,Intervalo=(Finish1-Start1)/np.timedelta64(1,'Y')))
        periodos=pd.DataFrame(periodos)
        if len(periodos[periodos['Intervalo']>=n_years])>0:
             stations.append(column)
        else:
            j=0
            aux=0
            while j<len(periodos) and aux==0:
                j+=1
                if periodos['Start'][j] + relativedelta(years=n_years) <= periodos['Finish'][periodos.index[-1]]:
                    series_period=series.loc[periodos['Start'][j]:periodos['Start'][j] + relativedelta(years=n_years)]
                    falhas=series_period.isnull().sum()/len(series_period)
                    if falhas<=missing_percent/100 and aux==0:
                        aux=1
                        stations.append(column)
                else:
                    aux=1
        state+=1
    print('Fully Completed')
    data=data[stations]
    return data

def calc_high_correlations(correlation):
    correlations = []
    for column in correlation.columns:
        value = 0
        for i in range(len(correlation[column])):
            if abs(correlation[column][i])>=.8 and correlation[column][i]!=1:
                value +=1
        correlations.append(value)
    correlations = pd.DataFrame({'High Correlations':correlations},index=correlation.columns)
    return correlations

def standard_data(data):
    for column in data.columns:
        data[column] = (data[column]-data[column].mean())/data[column].std()
    return data

def return_labels(dados,n_cluters):
    kmeans = KMeans(n_clusters=n_cluters).fit(dados)
    label_km = kmeans.labels_
    ward = sklearn.cluster.AgglomerativeClustering(n_clusters=5).fit(dados)
    label_w = ward.labels_
    labels = pd.DataFrame({'Codigo':dados.index,'Label_Kmeans':label_km, 'Label_Ward':label_w})
    return labels
