'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import pandas as pd
import numpy as np
from plotly.offline import plot
import plotly.figure_factory as ff

def gantt(dados,nome_grafico,mensal=True): 
    """
    For this function, you need a DataFrame structure with flow stations on columns and the measurements at each time on the rows. 
    The index must be a datetime.
    Plots the Gantt graph for passed data, showing temporal data availability;
    """
    postos_com_dados=[]
    periodos = []
    for column in dados.columns:
        serie = dados[column]
        if mensal==True:
            falhas=serie.isnull().groupby(pd.Grouper(freq='1MS')).sum().to_frame()
            serie_drop=falhas.loc[falhas[column] < 7] #UM MÊS COM AUSÊNCIA DE 7 DADOS É CONSIDERADO COM FALHA
            DELTA='M'
        else:
            serie_drop = serie.dropna() 
            DELTA='D'
        if serie_drop.shape[0]>1:
            postos_com_dados.append(column)
            Task1=column
            Resource1='Periodo com Dados'      
            Start1 = str(serie_drop.index[0].year)+'-'+str(serie_drop.index[0].month)+'-'+str(serie_drop.index[0].day)
            Finish1 = 0
            for i in range(len(serie_drop)):
                if i!=0 and round((serie_drop.index[i]-serie_drop.index[i-1])/np.timedelta64(1,DELTA),0) != 1:
                    Finish1=str(serie_drop.index[i-1].year)+'-'+str(serie_drop.index[i-1].month)+'-'+str(serie_drop.index[i-1].day)
                    periodos.append(dict(Task=Task1,Start=Start1,Finish=Finish1,Resource=Resource1))
                    Start1 = str(serie_drop.index[i].year)+'-'+str(serie_drop.index[i].month)+'-'+str(serie_drop.index[i].day)
                    Finish1 = 0
            Finish1 = str(serie_drop.index[-1].year)+'-'+str(serie_drop.index[-1].month)+'-'+str(serie_drop.index[-1].day)
            periodos.append(dict(Task=Task1,Start=Start1,Finish=Finish1,Resource=Resource1))
        else:
            print('Posto {} não possui meses com dados significativos'.format(column))
    periodos=pd.DataFrame(periodos)
    colors={'Periodo com Dados': 'rgb(0,191,255)'}
    fig = ff.create_gantt(periodos, colors=colors, index_col='Resource', show_colorbar=True,showgrid_x=True, showgrid_y=True, height=(200+len(postos_com_dados)*30),width=1800, group_tasks=True)
    plot(fig,filename=nome_grafico +'.html')
    return
