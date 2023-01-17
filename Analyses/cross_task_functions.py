from sklearn.cluster import BisectingKMeans, AgglomerativeClustering, KMeans, DBSCAN
from sklearn.model_selection import KFold
from umap import UMAP

import numpy as np
import pandas as pd


def umap_cluster(data, z_score=False, n_clusters=None, umap_params_in=None, cluster_params_in=None, random_state=999):
    """
    first uses umap to change dimensions, and then clusters the umap dimensions. Returns umap coordinates,
    and the cluster ID. in a data frame.
    :param data:
    :param z_score: bool. if to zscore the rows
    :param n_clusters: int. number of clusters, defaults to 3 if zscore is False, 4 if zscore is True
    :param umap_params_in: dict, umap params
    :param cluster_params_in: dict, clustering params
    :return: data frame with umap dimensions and cluster result
    """

    np.random.seed(0)
    X = data.copy()
    if z_score:
        X = X.sub(X.mean(axis=1).values, axis=0)
        X = X.div(X.std(axis=1).values, axis=0)

        if random_state is None:
            random_state = 5
        if n_clusters is None:
            n_clusters = 4
    else:
        if random_state is None:
            random_state = 32
        if n_clusters is None:
            n_clusters = 3

    umap_params = {'n_components': 2, 'n_neighbors': 20, 'min_dist': 0.1, 'random_state': random_state}
    if umap_params_in is not None:
        umap_params.update(umap_params_in)
    umap_coords = umap_cell_clusters(X, **umap_params)

    cluster_params = {'n_clusters': n_clusters, 'method': 'agg', 'return_all': False}
    if cluster_params_in is not None:
        cluster_params.update(cluster_params_in)

    labels = cluster_data(umap_coords, **cluster_params)

    n_umap_components = umap_params['n_components']

    umap_coords_column_names = [f'UMAP-{ii}' for ii in range(1, n_umap_components + 1)]
    dat = pd.DataFrame(columns=umap_coords_column_names + ['Cluster'])
    dat['Cluster'] = labels
    dat[umap_coords_column_names] = umap_coords

    return dat


def umap_kmeans_cluster_error_table(data, z_score=False):
    """
    Experiment to find optimal number of clusters
    :param data: dataframe, table of data to be clustered
    :param z_score: bool, if True, zscores the rows
    :return:
    dataframe of results
    """
    np.random.seed(0)

    umap_neighbors = np.arange(15, 35, 5)
    n_neighbor_expt = len(umap_neighbors)
    n_folds = 5
    n_repeats = 3

    clusters_range = np.arange(1, 7)
    n_cluster_range = len(clusters_range)
    n_split = 2

    clusters_table = pd.DataFrame(index=np.arange(n_repeats * n_cluster_range * n_neighbor_expt * n_folds * n_split),
                                  columns=['repeat', 'n_clusters', 'umap_neighbor', 'fold', 'split',
                                           'cosine', 'MSE'])

    X = data.copy()
    if z_score:
        X = X.sub(X.mean(axis=1).values, axis=0)
        X = X.div(X.std(axis=1).values, axis=0)

    cnt = 0
    for rr in range(n_repeats):
        for ii in range(n_neighbor_expt):
            X2 = umap_cell_clusters(X, **{'n_components': 2, 'n_neighbors': umap_neighbors[ii], 'min_dist': 0.1})

            for kk in clusters_range:
                kf = KFold(n_splits=n_folds, shuffle=True)

                fold = 0
                for train_idx, test_idx in kf.split(X):
                    Xtrain = X2[train_idx]
                    Xtest = X2[test_idx]

                    train_labels, centroids, model = cluster_data(Xtrain, n_clusters=kk, method='bikmeans',
                                                                  return_all=True)
                    test_labels = model.predict(Xtest)

                    train_cs = cosine_clustering_score(Xtrain, train_labels)
                    test_cs = cosine_clustering_score(Xtest, test_labels)

                    train_mse = MSE_clustering_score(Xtrain, train_labels)
                    test_mse = MSE_clustering_score(Xtest, test_labels)

                    clusters_table.loc[cnt] = rr, kk, umap_neighbors[ii], fold, 'train', \
                                              train_cs, train_mse
                    cnt += 1

                    clusters_table.loc[cnt] = rr, kk, umap_neighbors[ii], fold, 'test', \
                                              test_cs, test_mse
                    cnt += 1

                    fold += 1


    return clusters_table


def umap_cell_clusters(data, n_umap_clusters=2, **umap_params):
    params = {'n_components': n_umap_clusters, 'n_neighbors': 25, 'min_dist': 0.1}
    params.update(umap_params)
    return UMAP(**params).fit_transform(data)


def cluster_data(data, n_clusters=3, method='kmeans', return_all=False, **cluster_params):
    params = {}
    if method == 'bikmeans':
        params.update(cluster_params)
        params['n_clusters'] = n_clusters
        func = BisectingKMeans(**params)

    elif method == 'agg':
        params.update(cluster_params)
        params['n_clusters'] = n_clusters
        func = AgglomerativeClustering(**params)

    elif method == 'dbscan':
        params.update(cluster_params)
        func = AgglomerativeClustering(**params)

    else:
        params.update(cluster_params)
        params['n_clusters'] = n_clusters
        func = KMeans(**params)

    if method in ['bikmeans', 'kmeans']:
        model = func.fit(data)
        labels = model.predict(data)
        centroids = model.cluster_centers_
    else:
        labels = func.fit_predict(data)
        centroids = None
        model = None

    if return_all:
        return labels, centroids, model

    return labels


def cosine_clustering_score(data, labels):
    n_clusters = len(np.unique(labels))
    centroids_by_samp = np.zeros_like(data) * np.nan
    for jj in range(n_clusters):
        idx = labels == jj
        if idx.sum() > 0:
            centroids_by_samp[idx] = data[idx].mean(axis=0)

    d = sample_cosine_dist(data, centroids_by_samp)
    return np.mean(d)


def sample_cosine_dist(x, y):
    xn = np.linalg.norm(x, ord=2, axis=1)
    yn = np.linalg.norm(y, ord=2, axis=1)
    return 1 - (x * y).sum(axis=1) / (xn * yn)


def cluster_MSE(x, mu):
    return ((x - mu) ** 2).sum(axis=1).mean()


def MSE_clustering_score(X, labels, weighting='sample_weighted_mean'):

    label_names = np.unique(labels)
    cluster_scores = pd.DataFrame(index=label_names, columns=['score', 'n_samps'])
    for g in label_names:
        idx = np.where(labels == g)[0]
        mu = X[idx].mean(axis=0)
        score = cluster_MSE(X[idx], mu)
        cluster_scores.loc[g] = score, idx.sum()

    if weighting == 'median':
        error = np.median(cluster_scores['score'])
    elif weighting == 'mean':
        error = np.mean(cluster_scores['score'])
    elif weighting == 'sample_weighted_mean':
        error = np.mean(cluster_scores['score']*cluster_scores['n_samps'])/cluster_scores['n_samps'].sum()
    elif weighting == 'sample_weighted_median':
        error = np.median(cluster_scores['score']*cluster_scores['n_samps'])
    else:
        error = cluster_scores['score'].sum()

    return error
