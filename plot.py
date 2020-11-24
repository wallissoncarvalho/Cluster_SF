'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.patheffects as pe

def pca_ncomponents(data_rescaled):
    pca = PCA().fit(data_rescaled)
    plt.rcParams["figure.figsize"] = (10,8)
    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots()
    xi = np.arange(1, len(data_rescaled.columns)+1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)   
    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='-', color='b')
    plt.bar(xi, pca.explained_variance_ratio_)
    
    plt.xlabel('Number of Components',fontsize=8)
    plt.xticks(np.arange(1, len(data_rescaled.columns)+1, step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)',fontsize=8)
    plt.title('The number of components needed to explain variance', fontsize=8)
    
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.axvline(x=4, color='r', linestyle='--')
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

#def cluster_evaluation(data):
#    clusters=clustering.kmeans_ward_evaluation(data)
#    db = clusters[['DB - K means','DB - Ward']].rename(columns={'DB - K means':'K means','DB - Ward':'Ward'})
#    si = clusters[['SC(i) - K means','SC(i) - Ward']].rename(columns={'SC(i) - K means':'K means','SC(i) - Ward':'Ward'})
#    
#    fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True)
#    fig.subplots_adjust(hspace=0.7)
#    ax1.plot(db)
#    ax1.title.set_text('Davies Boldin Index')
#
#    ax1.set_xlabel('Number of Clusters')
#
#    ax2.plot(si)
#    ax2.title.set_text('Silhouette Coefficient')
#
#    ax2.set_xlabel('Number of Clusters')
#    plt.xticks(list(range(2,21)))
#    plt.show()    

def pca_components(scaled_parameters, n_components=4, k=3):
    plt.rcParams["figure.figsize"] = (20,20)
    plt.rcParams['font.size'] = 6
    pca = PCA()
    principalComponents = pca.fit_transform(scaled_parameters)
    
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(scaled_parameters.shape[1])]
    most_important_names = [scaled_parameters.columns.to_list()[most_important[i]] for i in range(scaled_parameters.shape[1])]
    df = pd.DataFrame({'PC {}'.format(i+1): most_important_names[i] for i in range(scaled_parameters.shape[1])}.items(),columns=['PC','Feature'])
    
    score=principalComponents[:,0:4]
    coeff=np.transpose(pca.components_[0:4, :])
    labels=scaled_parameters.columns.to_list()
    
    pca_c = PCA(n_components=n_components)
    principal_c= pca_c.fit_transform(scaled_parameters)
    kmeans = KMeans(n_clusters=k).fit(principal_c)
    ward = AgglomerativeClustering(n_clusters=k).fit(principal_c)
    label_w = ward.labels_
    label_w = pd.DataFrame({'Label_Ward':label_w})
    def replace(x):
        if x==2:
            return 0
        elif x==0:
            return 1
        elif x==1:
            return 2
    label_w['Label_Ward_New'] = label_w['Label_Ward'].apply(replace)
    yw = label_w['Label_Ward_New'].array   
    yk = kmeans.labels_
    

    xs1 = score[:,0]
    ys1 = score[:,1]
    scalex1 = 1.0/(xs1.max() - xs1.min())
    scaley1 = 1.0/(ys1.max() - ys1.min())
    
    xs2 = score[:,2]
    ys2 = score[:,3]
    scalex2 = 1.0/(xs2.max() - xs2.min())
    scaley2 = 1.0/(ys2.max() - ys2.min())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.scatter(xs1 * scalex1,ys1 * scaley1, c=yk, zorder=5, edgecolor='white')
    n = coeff.shape[0]
    for i in range(n):
        ax1.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r', zorder=15)
        ax1.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center',
                 path_effects=[pe.withStroke(linewidth=1, foreground="white")],zorder=20)    
    ax1.grid()
    ax1.set_title('PC with points clustered by Kmeans')
    ax1.set_xlim(-1,1)
    ax1.set_ylim(-1,1) 
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    
    ax2.scatter(xs1 * scalex1,ys1 * scaley1, c=yw, zorder=5, edgecolor='white')
    n = coeff.shape[0]
    for i in range(n):
        ax2.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r', zorder=15)
        ax2.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center',
                 path_effects=[pe.withStroke(linewidth=1, foreground="white")],zorder=20)     
    ax2.grid()
    ax2.set_title('PC with points clustered by Ward')
    ax2.set_xlim(-1,1)
    ax2.set_ylim(-1,1) 
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    
    
    ax3.scatter(xs2 * scalex2,ys2 * scaley2, c=yk, zorder=5, edgecolor='white')
    n = coeff.shape[0]
    for i in range(n):
        ax3.arrow(0, 0, coeff[i,2], coeff[i,3],color = 'r', zorder=15)
        ax3.text(coeff[i,2]* 1.15, coeff[i,3] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center',
                 path_effects=[pe.withStroke(linewidth=1, foreground="white")],zorder=20)    

    ax3.grid()
#    ax3.set_title('PC with points clustered by Kmeans')
    ax3.set_xlim(-1,1)
    ax3.set_ylim(-1,1) 
    ax3.set_xlabel('PC 3')
    ax3.set_ylabel('PC 4')
    
    ax4.scatter(xs2 * scalex2,ys2 * scaley2, c=yk, zorder=5, edgecolor='white')
    n = coeff.shape[0]
    for i in range(n):
        ax4.arrow(0, 0, coeff[i,2], coeff[i,3],color = 'r', zorder=15)
        ax4.text(coeff[i,2]* 1.15, coeff[i,3] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center',
                 path_effects=[pe.withStroke(linewidth=1, foreground="white")],zorder=20)
    ax4.grid()
#    ax4.set_title('PC with points clustered by Ward')
    ax4.set_xlim(-1,1)
    ax4.set_ylim(-1,1) 
    ax4.set_xlabel('PC 3')
    ax4.set_ylabel('PC 4')
   
    return df


def pca_components_2(scaled_parameters, n_components=4, method='Kmeans', k=3):
    plt.rcParams["figure.figsize"] = (12,12)
    plt.rcParams['font.size'] = 8
    pca = PCA()
    principalComponents = pca.fit_transform(scaled_parameters)
    
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(scaled_parameters.shape[1])]
    most_important_names = [scaled_parameters.columns.to_list()[most_important[i]] for i in range(scaled_parameters.shape[1])]
    df = pd.DataFrame({'PC {}'.format(i+1): most_important_names[i] for i in range(scaled_parameters.shape[1])}.items(),columns=['PC','Feature'])
    
    score=principalComponents[:,0:4]
    coeff=np.transpose(pca.components_[0:4, :])
    labels=scaled_parameters.columns.to_list()
    
    xs = score[:,2]
    ys = score[:,3]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    
    pca_c = PCA(n_components=4)
    principal_c= pca_c.fit_transform(scaled_parameters)
    
    if method=='Kmeans':
        kmeans = KMeans(n_clusters=k).fit(principal_c)
        y = kmeans.labels_
        plt.title('Principal Components with points clustered by Kmeans')
    elif method=='Ward':
        ward = AgglomerativeClustering(n_clusters=k).fit(principal_c)
        label_w = ward.labels_
        label_w = pd.DataFrame({'Label_Ward':label_w})
        def replace(x):
            if x==2:
                return 0
            elif x==0:
                return 1
            elif x==1:
                return 2
        label_w['Label_Ward_New'] = label_w['Label_Ward'].apply(replace)
        y = label_w['Label_Ward_New'].array
        plt.title('Principal Components with points clustered by Ward')
    
    plt.scatter(xs * scalex,ys * scaley, c=y, zorder=5, edgecolor='white')
    n = coeff.shape[0]
    for i in range(n):
        plt.arrow(0, 0, coeff[i,2], coeff[i,3],color = 'r', zorder=15)
        plt.text(coeff[i,2]* 1.15, coeff[i,3] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center',
                 path_effects=[pe.withStroke(linewidth=1, foreground="white")],zorder=20)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC {}".format(3))
    plt.ylabel("PC {}".format(4))
    plt.grid()
    plt.show()
    return df
       
def cluster_sc(data):
    cluster=clustering.kmeans_ward_evaluation(data)
    si = cluster[['SC(i) - K means','SC(i) - Ward']].rename(columns={'SC(i) - K means':'K means','SC(i) - Ward':'Ward'})

    plt.rcParams["figure.figsize"] = (10,8)
    plt.rcParams["figure.dpi"] = 300

    plt.plot(si['K means'], color='blue', marker='o', linestyle='-')
    #plt.axvline(x=si['K means'].idxmax(), color='blue', linestyle='--')
    
    plt.plot(si['Ward'], color='orange', marker='o', linestyle='-')
    plt.axvline(x=si['Ward'].idxmax(), color='black', linestyle='--')
    
    plt.xticks(np.arange(2, 11, step=1)) 
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Optimal number of clusters: \nSilhouette Method')
    plt.legend(('K-means','Ward'))
#    plt.xticks(list(range(2,21)))
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
    plt.yticks([])
    plt.show()