'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


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
    
    #FILTRO 2: Verifica se há pelo menos uma janela com 10 anos de registro com até no máximo missing_percent.
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


## Deprecated method 
def naturalize_flow(observed_flows,natural_flows,reservoirs_discharge,affected_stations):
    to_sum = pd.DataFrame()
    for column in natural_flows.columns:
        to_sum = pd.concat([to_sum,pd.DataFrame({column:(natural_flows[column] - reservoirs_discharge[column])})],axis=1)
    
    for column in affected_stations.columns:
        stations = list(affected_stations[column].dropna())
        for station in stations:
            observed_flows[station] = observed_flows[station] + to_sum[column]
###


