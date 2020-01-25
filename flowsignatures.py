import pandas as pd
import numpy as np

def quantiles(data):
    df = pd.DataFrame(index=data.columns.values, columns=['q90','Mean','q10'])
    for column in list(data.columns.values):
        serie = pd.Series(data[column], name=column)
        q90=serie.quantile(0.1)
        q10=serie.quantile(0.9)
        mean=serie.mean()
        df['q90'].loc[column] = q90
        df['q10'].loc[column] = q10
        df['Mean'].loc[column] = mean
    return df

def coefficient_variation(data):
    CV = data.std()/data.mean()
    return pd.Series(CV.values, index=CV.index, name='CV')

def rbf(data):
    df = pd.DataFrame(index=data.columns.values, columns=['RBF'])
    denominator = 0
    for column in list(data.columns.values):
        numerator=0
        serie = pd.Series(data[column], name=column).dropna()
        for i in range(len(serie)-1):
            if serie.iloc[i+1] != np.nan:
                numerator += abs(serie.iloc[i+1]-serie.iloc[i])
                denominator +=serie.iloc[i]
        rbf = numerator/denominator
        df['RBF'].loc[column] = rbf
    return df

def slope_fdc(data):
    import math
    df = pd.DataFrame(index=data.columns.values, columns=['SFDC'])
    for column in list(data.columns.values):
        serie = pd.Series(data[column], name=column).dropna()
        sfdc = (math.log(serie.quantile(0.66))-math.log(serie.quantile(0.33)))/0.33
        df['SFDC'].loc[column] = sfdc
    return df
    
def baseflow_index(data):
    df = pd.DataFrame(index=data.columns.values, columns=['IBF'])
    for column in list(data.columns.values):
        serie = pd.Series(data[column], name=column).dropna()
        ibf = serie.rolling(7).mean().min()/serie.mean()
        df['IBF'].loc[column] = ibf
    return df

def all_signatures(data):
    df = pd.DataFrame()
    df = pd.concat([df, quantiles(data)],axis=1)
    df = pd.concat([df, coefficient_variation(data)],axis=1)
    df = pd.concat([df, rbf(data)],axis=1)
    df = pd.concat([df, slope_fdc(data)],axis=1)
    df = pd.concat([df, baseflow_index(data)],axis=1)
    return df