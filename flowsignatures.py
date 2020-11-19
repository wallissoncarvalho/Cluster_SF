'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import pandas as pd
import numpy as np

def quantiles(data):
    return pd.DataFrame({'Q90':data.quantile(.1),'Qmean':data.mean(),'Q10':data.quantile(.9)})

def coefficient_variation(data):
    return pd.DataFrame({'CV':(data.std()/data.mean())})

def rbf(data):
    values = []
    for column in data.columns:
        denominator = 0
        numerator = 0
        for i in range(len(data[column])-1):
            if pd.notna(data[column][i]) and pd.notna(data[column][i+1]):
                numerator += abs(data[column][i+1]-data[column][i])
                denominator += data[column][i]
        if pd.notna(data[column][-2]) and pd.notna(data[column][-1]):
            denominator += data[column][-1]
        rbf = numerator/denominator
        values.append(rbf)
    df = pd.DataFrame({'RBF':values}, index=data.columns)
    return df
   
def baseflow_index(data):
    values = []
    for column in data.columns:
        values.append(data[column].rolling(7).mean().min()/data[column].mean())
    df = pd.DataFrame({'IBF':values}, index=data.columns)
    return df

#def peak_distribution(data):
#    values = []
#    for column in data.columns:
#        try:
#            values.append((data[column].quantile(.9)-data[column].quantile(.5))/.4)
#        except:
#            values.append(np.nan)
#    df = pd.DataFrame({'PD':values}, index=data.columns)
#    return df

def auto_correlation(data):
    values = []
    for column in data.columns:
        values.append(data[column].autocorr())
    df = pd.DataFrame({'AC':values}, index=data.columns)
    return df


def all_signatures(data):
    df = pd.DataFrame()
    df = pd.concat([df, quantiles(data)],axis=1)
    df = pd.concat([df, coefficient_variation(data)],axis=1)
    df = pd.concat([df, rbf(data)],axis=1)
    df = pd.concat([df, baseflow_index(data)],axis=1)
    df = pd.concat([df, peak_distribution(data)],axis=1)
    df = pd.concat([df, auto_correlation(data)],axis=1)
    return df