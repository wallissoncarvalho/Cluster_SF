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
import plotly.graph_objects as go
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import clustering
from sklearn.decomposition import PCA

def gantt(data, graph_name='gant_plot', monthly=True):
    """
    Make the Gantt plot. This graphic shows the temporal data availability for each station.
    :param data:A Pandas daily DataFrame with DatetimeIndex where each column corresponds to a station.
    :param graph_name: str, optional, default: True
    Defines the name of the exported graph
    :param monthly: boolean, optional, default: True
    Defines if the availability count of the data will be monthly to obtain a more fluid graph.
    """
    stations_with_data = []
    periods = []
    date_index = pd.date_range(data.index[0], data.index[-1], freq='D')
    data = data.reindex(date_index)
    for column in data.columns:
        series = data[column]
        if monthly:
            missing = series.isnull().groupby(pd.Grouper(freq='1MS')).sum().to_frame()
            series_drop = missing.loc[missing[column] < 7]  # A MONTH WITHOUT 7 DATA IS CONSIDERED A MISSING MONTH
            DELTA = 'M'
        else:
            series_drop = series.dropna()
            DELTA = 'D'
        if series_drop.shape[0] > 1:
            stations_with_data.append(column)
            task = column
            resource = 'Available data'
            start = str(series_drop.index[0].year) + '-' + str(series_drop.index[0].month) + '-' + str(
                series_drop.index[0].day)
            finish = 0
            for i in range(len(series_drop)):
                if i != 0 and round((series_drop.index[i]-series_drop.index[i - 1])/np.timedelta64(1, DELTA), 0) != 1:
                    finish = str(series_drop.index[i - 1].year) + '-' + str(series_drop.index[i - 1].month) + '-' + str(
                        series_drop.index[i - 1].day)
                    periods.append(dict(Task=task, Start=start, Finish=finish, Resource=resource))
                    start = str(series_drop.index[i].year) + '-' + str(series_drop.index[i].month) + '-' + str(
                        series_drop.index[i].day)
                    finish = 0
            finish = str(series_drop.index[-1].year) + '-' + str(series_drop.index[-1].month) + '-' + str(
                series_drop.index[-1].day)
            periods.append(dict(Task=task, Start=start, Finish=finish, Resource=resource))
        else:
            print('Station {} has no months with significant data'.format(column))
    periods = pd.DataFrame(periods)
    start_year = periods['Start'].apply(lambda x:int(x[:4])).min()
    finish_year = periods['Start'].apply(lambda x:int(x[:4])).max()
    colors = {'Available data': 'rgb(0,191,255)'}
    fig = ff.create_gantt(periods, colors=colors, index_col='Resource', show_colorbar=True, showgrid_x=True,
                          showgrid_y=True, height=(200+len(stations_with_data)*30),width=1800, group_tasks=True)
    
    fig.layout.xaxis.tickvals = pd.date_range('1/1/'+str(start_year),'12/31/'+str(finish_year+1), freq='2AS')
    fig.layout.xaxis.ticktext = pd.date_range('1/1/'+str(start_year),'12/31/'+str(finish_year+1), freq='2AS').year
    fig.layout.xaxis.title = 'Year'
    fig.layout.yaxis.title = 'Station Code'
    fig = go.FigureWidget(fig)
    fig.update_layout(font=dict(family="Courier New, monospace",size=15))
    plot(fig,filename=graph_name + '.html')
    stations_with_data = {'Station Code': stations_with_data}
    stations_with_data = pd.DataFrame(data=stations_with_data)
    return stations_with_data

def pca_components(data_rescaled):
    pca = PCA().fit(data_rescaled)
    plt.rcParams["figure.figsize"] = (10,8)
    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots()
    xi = np.arange(1, len(data_rescaled.columns)+1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)   
    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='-', color='b')
    
    plt.xlabel('Number of Components',fontsize=8)
    plt.xticks(np.arange(1, len(data_rescaled.columns)+1, step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)',fontsize=8)
    plt.title('The number of components needed to explain variance', fontsize=8)
    
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.text(0.5, 0.85, '80% cut-off threshold', color = 'red', fontsize=6)
    ax.grid(axis='x')
    plt.show()


def available_stations_year(data):
    """
    Make a bar plot for the number of stations with available data on each year. The plotting regards three possibilities
    of missing data on each station, which are until 5%, 10%, and 15%.
    :param data: A Pandas daily DataFrame with DatetimeIndex where each column corresponds to a station.
    """
    years = list(set(data.index.year))
    list_5, list_10, list_15 = [], [], []
    for year in years:
        series = data.loc[data.index.year == year]
        missing = series.isnull().sum().to_frame()
        list_5.append(len(list(missing.loc[missing[missing.columns[0]] < 19].index)))
        list_10.append(len(list(missing.loc[missing[missing.columns[0]] < 37].index)))
        list_15.append(len(list(missing.loc[missing[missing.columns[0]] < 55].index)))
    cut = 0
    while list_5[cut] == 0 and list_10[cut] == 0 and list_15[cut] == 0:
        cut += 1
#    years = years[cut + 1:-1]
#    list_5 = list_5[cut + 1:-1]
#    list_10 = list_10[cut + 1:-1]
#    list_15 = list_15[cut + 1:-1]
    years = years[cut:]
    list_5 = list_5[cut:]
    list_10 = list_10[cut:]
    list_15 = list_15[cut:]
    plt.figure(num=None, figsize=(12, 9), dpi=300, facecolor='w', edgecolor='k')
    #_ = plt.bar(years, list_15, .8)
    #_ = plt.bar(years, list_10, .8)
    _ = plt.bar(years, list_5, .8)
    plt.ylabel('Number of stations', fontsize=14)
    plt.xlabel('Years', fontsize=14)
    plt.xticks(np.arange(min(years), max(years)+1, 3),rotation='vertical')
    plt.yticks(np.arange(0, max(list_15)+1, 10))
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    #plt.legend(('15%', '10%', '5%'))
    plt.show()
    asy = pd.DataFrame({'15%':list_15,'10%':list_10,'5%':list_5},index=years)
    return asy


def correlation_matrix(data):
    corr = data.corr(method='spearman')
    plt.figure(num=None, figsize=(12, 12), dpi=300, facecolor='w', edgecolor='k')
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(),rotation=45, fontsize=10)
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

def dendogram(X, level=3):
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
    create_dendogram(model, truncate_mode='level', p=level)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()