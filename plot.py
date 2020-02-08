'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from plotly.offline import plot
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import clustering


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

def correlation_matrix(data):
    import seaborn as sns
    corr = data.corr(method='spearman')
    ax = sns.heatmap(corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        fontsize=7,
        horizontalalignment='right'
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=7,
    )
    return

def cluster_evaluation(data):
    clusters=clustering.kmeans_ward_evaluation(data)
    db = clusters[['DB - K means','DB - Ward']].rename(columns={'DB - K means':'K means','DB - Ward':'Ward'})
    si = clusters[['SC(i) - K means','SC(i) - Ward']].rename(columns={'SC(i) - K means':'K means','SC(i) - Ward':'Ward'})
    
    fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True)
    fig.subplots_adjust(hspace=0.7)
    ax1.plot(db)
    ax1.title.set_text('Davies Boldin Index')

    ax1.set_xlabel('Number of Clusters')

    ax2.plot(si)
    ax2.title.set_text('Silhouette Coefficient')

    ax2.set_xlabel('Number of Clusters')
    plt.xticks(list(range(2,21)))
    plt.show()

def dendogram(X):
    def create_dendogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram
    
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
    
        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)
    
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    
    model = model.fit(X)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    create_dendogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()