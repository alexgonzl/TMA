from pathlib import Path
import time
import traceback
from importlib import reload

import numpy as np
import scipy.stats as stats
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.covariance import MinCovDet

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from itertools import combinations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option('display.max_rows', 50)
sns.set(style='whitegrid', palette='muted')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def dim_reduction(data, method, random_seed=0, **method_kwargs):
    """
    reduce data dimensions
    :param data: samples x features
    :param method: ['tsne', 'pca', 'umap']
    :param random_seed:
    :param method_kwargs:
        specific parameters for the method, see the method [sklearn - pca, tsne; umap-UMAP] for additional details
    :return:
        data: samples x n_components (usually 2-3)
    """
    if data.ndim > 2:
        raise ValueError("data must be 2 dimensional")

    np.random.seed(random_seed)

    if method == 'tsne':
        default_params = {'n_components': 2, 'perplexity': 20, 'n_jobs': -1}
        for k, v in default_params.items():
            if k not in method_kwargs.keys():
                method_kwargs[k] = v

        model = TSNE(**method_kwargs)

    elif method == 'umap':
        default_params = {'n_components': 2, 'n_neighbors': 30, 'min_dist': 0}
        for k, v in default_params.items():
            if k not in method_kwargs.keys():
                method_kwargs[k] = v

        model = UMAP(**method_kwargs)

    elif method == 'pca':
        default_params = {'n_components': 2}
        for k, v in default_params.items():
            if k not in method_kwargs.keys():
                method_kwargs[k] = v

        model = PCA(**method_kwargs)

    else:
        print("Invalid method")
        return

    return model.fit_transform(data)


def get_2d_confidence_ellipse(mu=None, cov=None, n_std=2, n_points=100,
                              x=None, y=None, robust_est=False):
    """
    Creates ellipsoid object from mean & covariance (if given) or from x,y coordinates.
    Returns shapely object.
    :param mu: array with x, y location
    :param cov:  array 2 x 2
    :param n_std: [int or float] std distance to outline the ellipse
    :param n_points: number of points to use for the parameitization
    :param x:
    :param y:
    :param robust_est: only used if mu or cov are not provided, see estimate moments for details.
    :return:
        Polygon object [shapely library] with the desired location and covariance.

    Example:
        f, ax = plt.subplots()
        ellipse = get_2d_confidence_ellipse( mu=np.zeros(2), cov=np.eye(2))
        patch = PolygonPatch(ellipse, facecolor="tab:red", alpha=0.3)
        ax.add_patch(patch)

    ref: https://cookierobotics.com/007/
    """

    if (mu is None) and (cov is None):
        mu, cov = estimate_moments(np.array((x, y)), robust_est=robust_est)

    # get eigen values and eigen vectors of covariance
    eig_values, eig_vectors = np.linalg.eig(cov)

    # Below we use a parametric formulation for a 2d ellipse
    # parameterized points
    t = np.linspace(0, 2 * np.pi, n_points)
    v = np.array([n_std * np.sqrt(eig_values[0]) * np.cos(t),
                  n_std * np.sqrt(eig_values[1]) * np.sin(t)])

    # rotation
    xy = eig_vectors @ v

    # translation by coordinate
    xy += mu[:, np.newaxis]

    return Polygon(xy.T)


def get_3d_confidence_ellipsoid_surface_points(x=None, y=None, z=None, n_std=2,
                                               n_points=100, mu=None, cov=None, robust_est=True):
    """
    Creates ellipsoid object from mean & covariance (if given) or from x,y,z coordinates
    :param mu: array with x, y, z locations
    :param cov:  array 3 x 3
    :param n_std: [int or float] std distance to outline the ellipse
    :param n_points: number of points to use for the parameitization
    :param x:
    :param y:
    :param z:
    :param robust_est: only used if mu or cov are not provided, see estimate moments for details.
    :return:
        x,y,z srufaces to be used in a 3d plotting function, like plt.plot_surface

    # ref: https://cookierobotics.com/007/
    """

    if (mu is None) and (cov is None):
        mu, cov = estimate_moments(np.array((x, y, z)), robust_est=robust_est)

        # get eigen values and eigen vectors of covariance
    eig_values, eig_vectors = np.linalg.eig(cov)

    # get radii
    r = n_std * np.sqrt(eig_values)

    # parameterized points
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)

    x = r[0] * np.outer(np.cos(u), np.sin(v))
    y = r[1] * np.outer(np.sin(u), np.sin(v))
    z = r[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(n_points):
        for j in range(n_points):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], eig_vectors) + mu

    return x, y, z


def estimate_moments(x, robust_est=True):
    """
    estimates mean and covariance
    :param x: samples x n_dims
    :param robust_est: if true, uses MinCovDet from sklearn.covariance which computes covariance exluding outliers
        else computes empirical covariance
    :return:
        mean - [first moment] - array(n_dims)
        covariance - [second moment] - array(n_dims x n_dims)
    """

    # compute robust covariance:
    if robust_est:
        fit = MinCovDet(random_state=0).fit(x)
        cov = fit.covariance_
        mu = fit.location_
    else:
        cov = np.cov(x.T)
        mu = np.mean(x, axis=0)

    return mu, cov


def kld_mvg(m1, s1, m2, s2):
    """
    Kullback–Leibler divergence for 2 multivariate gaussians [note this is differential KLD]
    :param m1: mean of distribution1
    :param s1: covariance of d1
    :param m2: mean of d2
    :param s2: covariance of d2
    :return:
        KL distances in bits np.array:
            distance from d1 to d2
            distance from d2 to d1

    ref:https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
    """
    assert m1.shape == m2.shape, "inputs do not match"
    assert s2.shape == s2.shape, "inputs do not match"

    if (m1.ndim == 1) & (m2.ndim == 1):
        k = len(m1)
        m1 = m1[:, np.newaxis]
        m2 = m2[:, np.newaxis]
    elif (m1.dim == 2) & (m2.ndim == 2):
        if (m1.shape[1] > 1) or (m2.shape[1] > 1):
            print("invalid input")
            return np.nan * np.ones(2)
        k = m1.shape[0]
    else:
        print("invalid input")
        return np.nan * np.ones(2)

    inv_s1 = np.linalg.inv(s1)
    inv_s2 = np.linalg.inv(s2)

    det_s1 = np.linalg.det(s1)
    det_s2 = np.linalg.det(s2)

    kld12 = 0.5 * (np.log(det_s2 / det_s1) - k + np.trace(inv_s2 @ s1) + (m2 - m1).T @ inv_s2 @ (m2 - m1)) / np.log(2)
    kld21 = 0.5 * (np.log(det_s1 / det_s2) - k + np.trace(inv_s1 @ s2) + (m1 - m2).T @ inv_s1 @ (m1 - m2)) / np.log(2)

    return np.array([kld12.flatten(), kld21.flatten()])


def entropy_mvg(sigma):
    """
    Computes differential entropy of a multivatiate gaussian (see ref for limitations)
    :param sigma: covariance matrix
    :return:
        h [float] - differential entropy in bits (can be negative)

    ref: https://en.wikipedia.org/wiki/Differential_entropy
    """
    det = np.linalg.det(sigma)
    k = sigma.shape[0]
    return 0.5 * (k + k * np.log(2 * np.pi) + np.log(det)) / np.log(2)


def get_clusters_moments(data, labels, robust_est=True):
    """
    wrapper function for estimate moments. with longform data and cluster labels [ints] indicating the id
    of each cluster, returns the means and covariances for each cluster

    :param data: 2d array n_samps x n_dims
    :param labels: array of ints, n_samps
    :param robust_est: if true, estimates robust covariance, see estimate_moments for details
    :return:
        mu - array n_clusters x n_dims
        cov - array n_clusters x n_dims x n_dims
    """

    cluster_nums = np.unique(labels)
    n_clusters = len(cluster_nums)
    k = data.shape[1]
    mu = np.zeros((n_clusters, k))
    cov = np.zeros((n_clusters, k, k))

    for cluster in cluster_nums:
        x = data[labels == cluster, :]
        mu[cluster], cov[cluster] = estimate_moments(x, robust_est=robust_est)
    return mu, cov


def get_clusters_dists(clusters_loc, clusters_cov, method='kld', **kwargs):
    """
    Wrapper function for estimating distances between distributions / computed clusters.
    :param clusters_loc: list or array of cluster mean locations
    :param clusters_cov: list or array of cluster covariances
    :param method: one of:
        'kld' -  kull leibler divergence [0, inf]
        'he' - hellinger distance [0, 1]
        'pe' - custom - avg prob of misclassification error [-inf, inf]
    :param kwargs:
        additional arguments needed for method pe, ignored ow.
        data - 2d array n_samps x n_dim; n_dim must match the dims of given covariance
        lables - 1d array n_samps; integers labeling each sample, must match the order of clusters in
                clusters_loc clusters_cov inputs
    :return:
        2d array n_clusters x n_clusters of distances
    """

    if clusters_loc.shape[0] != clusters_cov.shape[0]:
        raise ValueError("number of clusters for moments inputs do not match")

    n_clusters = clusters_loc.shape[0]
    dist_mat = np.zeros((n_clusters, n_clusters))

    if method == 'pe':
        if ('data' in kwargs.keys()) and ('labels' in kwargs.keys()):
            for cl1 in range(n_clusters):
                m = clusters_loc[cl1]
                s = clusters_cov[cl1]
                for cl2 in range(n_clusters):
                    x = kwargs['data'][kwargs['labels'] == cl2]
                    dist_mat[cl1, cl2] = mean_samp_prob_gaussian(x, m, s)
            dist_mat = 1 - dist_mat / np.diag(dist_mat)
        else:
            print(f" cond_prob method needs the data samples and cluster labels")
            return None
    else:

        if method == 'kld':
            dist_func = kld_mvg
        elif method == 'he':
            dist_func = hellinger_dist_mvg
        else:
            print("method not supported")
            return

        for cl1 in range(n_clusters - 1):
            m1 = clusters_loc[cl1]
            s1 = clusters_cov[cl1]

            for cl2 in range(cl1, n_clusters):
                m2 = clusters_loc[cl2]
                s2 = clusters_cov[cl2]

                dist_mat[cl1, cl2], dist_mat[cl2, cl1] = dist_func(m1, s1, m2, s2)

    return dist_mat


def mean_samp_prob_gaussian(x, mu, sigma):
    " for a mvg of mean mu and covariance sigma, returns the pdf evaluated at samples x"
    return stats.multivariate_normal(mu, sigma).pdf(x).mean()


def hellinger_dist_1d_gaussians(m1, s1, m2, s2):
    """ hellinger distance for 1d gaussian
    :param m1: mean of distribution1
    :param s1: variance of d1
    :param m2: mean of d2
    :param s2: variance of d2
    :return:
        Hellinger distance

    """

    exp_factor = -1 / 4 * (m1 - m2) ** 2 / (s1 ** 2 + s2 ** 2)
    h2 = 1 - np.sqrt(2 * s1 * s2 / (s1 ** 2 + s2 ** 2)) * np.exp(exp_factor)
    return np.sqrt(h2)


def hellinger_dist_mvg(m1, s1, m2, s2):
    """ hellinger distance for mvg
    :param m1: mean of distribution1
    :param s1: covariance of d1
    :param m2: mean of d2
    :param s2: covariance of d2
    :return:
        Hellinger distance array:
            distance d1 to d2
            distance d2 to d1

    ref: # https://en.wikipedia.org/wiki/Hellinger_distance
    """

    assert m1.shape == m2.shape, "inputs do not match"
    assert s2.shape == s2.shape, "inputs do not match"

    if (m1.ndim == 1) & (m2.ndim == 1):
        m1 = m1[:, np.newaxis]
        m2 = m2[:, np.newaxis]
    elif (m1.dim == 2) & (m2.ndim == 2):
        if (m1.shape[1] > 1) or (m2.shape[1] > 1):
            print("invalid input")
            return np.nan * np.ones(2)
    else:
        print("invalid input")
        return np.nan * np.ones(2)

    s1s2_2 = (s1 + s2) / 2

    det_s1 = np.linalg.det(s1)
    det_s2 = np.linalg.det(s2)
    det_s1s2_2 = np.linalg.det(s1s2_2)

    inv_s1s2_2 = np.linalg.inv(s1s2_2)

    exp_factor = -1 / 8 * (m1 - m2).T @ inv_s1s2_2 @ (m1 - m2)

    h2 = 1 - (det_s1 * det_s2) ** (1 / 4) / (det_s1s2_2) ** (1 / 2) * np.exp(exp_factor)
    h = np.sqrt(h2)

    return np.array([h, h])


def get_clusters_entropy(clusters_cov):
    """
    Wrapper function to compute the entropy of mvg given a list of covariances.
    :param clusters_cov:
    :return: entropy for each cluster
    """
    n_clusters = len(clusters_cov)

    h = np.zeros(n_clusters)
    for cl in range(n_clusters):
        h[cl] = entropy_mvg(clusters_cov[cl])

    return h


def get_clusters_all_dists(clusters_loc, clusters_cov, data=None, labels=None,
                           normalize_kld=True, make_symmetric=True):
    """
    Get all clusters distances, wrapper for clusters_dists
    :param clusters_loc: list of locations for each cluster
    :param clusters_cov: list of covariances for each cluster
    :param data: needed for 'pe' dist calculation. array of n_samps x n_dims
    :param labels: needed for 'pe' dist, array of n_samps ints of cluster ids
    :param normalize_kld: if true, divides the KLD result by the entropy of a MVG(0, 1) ~ 4 bits
    :param make_symmetric: distances are made symmetric
    :return:
        dictionary of distances ['he', 'kld', 'pe']
    """

    dist_metrics = ['he', 'kld', 'pe']

    dists_mats = {'he': get_clusters_dists(clusters_loc, clusters_cov, method='he'),
                  'kld': get_clusters_dists(clusters_loc, clusters_cov, method='kld'),
                  'pe': get_clusters_dists(clusters_loc, clusters_cov, method='pe',
                                           data=data, labels=labels)}

    if normalize_kld:
        k = clusters_loc.shape[1]  # num data dims
        dists_mats['kld'] /= entropy_mvg(np.eye(k))

    if make_symmetric:
        for metric in dist_metrics:
            dists_mats[metric] = 0.5 * (dists_mats[metric] + dists_mats[metric].T)

    return dists_mats


def plot_2d_cluster_ellipsoids(clusters_loc, clusters_cov, data=None, std_levels=[1, 2],
                               labels=None, ax=None, legend=False, cl_names=None, cl_colors=None):

    n_levels = len(std_levels)
    if isinstance(clusters_loc, list):
        n_clusters = len(clusters_loc)
    elif isinstance(clusters_loc, np.ndarray):  # not supper robust here
        n_clusters = clusters_loc.shape[0]
    elif isinstance(clusters_loc, dict):
        n_clusters = len(clusters_loc)
    else:
        print("Invalid input")
        return

    cluster_ellipsoids = np.zeros((n_clusters, n_levels), dtype=object)

    for cl in range(n_clusters):
        for jj, level in enumerate(std_levels):
            cluster_ellipsoids[cl, jj] = \
                get_2d_confidence_ellipse(mu=clusters_loc[cl], cov=clusters_cov[cl], n_std=level)

    if cl_colors is None:
        cl_colors = colors
    elif isinstance(cl_colors, str):
        cl_colors = [cl_colors]

    n_colors = len(cl_colors)

    if ax is None:
        f, ax = plt.subplots()

    label_patch = []
    if data is not None:
        ax.scatter(data[:, 0], data[:, 1], c=np.array(colors)[labels], alpha=0.2)
        facecolors = ['grey'] * n_clusters
    else:
        facecolors = cl_colors

    if cl_names is None:
        cl_names = ['cl' + str(cl) for cl in range(n_clusters)]

    for cl in range(n_clusters):
        for jj, level in enumerate(std_levels):
            patch = PolygonPatch(cluster_ellipsoids[cl, jj], facecolor=facecolors[np.mod(cl, n_colors)], alpha=0.3)
            ax.add_patch(patch)

        label_patch.append(mpatches.Patch(color=facecolors[np.mod(cl, n_colors)], label=cl_names[cl], alpha=0.7))

    if legend:
        ax.legend(handles=label_patch, frameon=False, loc=(1.05, 0))

    _ = ax.axis('scaled')

    return ax


def find_all_cluster_matches(distance_mat, thr):
    labels = distance_mat.index.values
    assert np.all(
        labels == distance_mat.columns.values), "distance mat data frame must be squared and have matching index/columns"

    n_cl = len(labels)
    matches = {}

    for jj in np.arange(n_cl):
        label1 = labels[jj]
        matches[label1] = []
        for ii in np.arange(n_cl):
            label2 = labels[ii]
            if labels[ii].split("_")[0] != labels[jj].split("_")[0]:
                if distance_mat.loc[label1, label2] < thr:
                    matches[label1].append(label2)

    return matches


def find_session_cl_matches(distance_mat, thr, select_lower=True, exclude_multi_matched=True, session_cl_sep='_'):
    """
    For a given labeled distance matrix data frame, obtain cluster matches in order of distance score.

    params:
        - distance_mat [pandas_df]: squared pandas data frame with labeled indices & columns as s#_cl#,
            where the important part is that sessions and clusters within that session are separtated
            by an underscore "_"
        - thr [float]: number indicating the thr for inclusion in matches
        - select_lower [bool]: if True (default) matches with <thr are included,
            otherwise matches>thr are included
        - exclude_multi_matched [bool]: if True (default) cl matches that repeat won't be rematched
            with another cluster.
        -  session_cl_sep [str]: on the labels, what char separates session and cluster

    returns:
        dict of matches indexed by label
    """

    labels = distance_mat.index.values
    assert np.all(
        labels == distance_mat.columns.values), \
        "distance mat data frame must be squared and have matching index/columns"

    n_cl = len(labels)

    # mask over threshold values & diagonal down [assumes symmetry]
    Y = np.array(distance_mat.to_numpy())
    if select_lower:
        Y[Y > thr] = np.nan
    else:
        Y[Y <= thr] = np.nan
    Y[np.tril_indices_from(Y)] = np.nan

    # sort distance matrix & exclude nan indices
    linear_sort_idx = np.argsort(Y.flatten())
    nan_sort_idx = np.isnan(Y.flatten()[linear_sort_idx])
    linear_sort_idx = linear_sort_idx[~nan_sort_idx]

    # get 2dims indices
    sort_match_idx = np.unravel_index(linear_sort_idx, (n_cl, n_cl))

    # get sessions
    sessions = []
    for ll in labels:
        s = ll.split(session_cl_sep)[0]
        if s not in sessions:
            sessions.append(s)

    # get possible cluster matches by session (excludes self)
    possible_cl_matches_session = {s: [] for s in sessions}
    for cl1 in labels:
        s1 = cl1.split(session_cl_sep)[0]
        for cl2 in labels:
            s2 = cl2.split(session_cl_sep)[0]
            # if sessions are different
            if s1 != s2:
                if cl2 not in possible_cl_matches_session[s1]:
                    possible_cl_matches_session[s1].append(cl2)

    # get possible sessions to be matched for each cluster, excludes own session
    possible_session_cl = {cl: [] for cl in labels}
    se_set = set(sessions)
    for cl in labels:
        possible_session_cl[cl] = list(se_set.difference({cl.split(session_cl_sep)[0]}))

    # pre-allocate
    matches = {cl: [] for cl in labels}
    cnt = 0

    # iterate over matches
    for ii, jj in zip(sort_match_idx[0], sort_match_idx[1]):
        # for each match, obtain cl label and session id
        cl1 = labels[ii]
        cl2 = labels[jj]

        s1 = cl1.split(session_cl_sep)[0]
        s2 = cl2.split(session_cl_sep)[0]

        # cl2 can be matched to a cluster in session1,
        if (cl2 in possible_cl_matches_session[s1]) & (cl1 in possible_cl_matches_session[s2]):
            # s2 can be matched to cl1
            if (s2 in possible_session_cl[cl1]) & (s1 in possible_session_cl[cl2]):

                # abort matching if any of the two clusters has already been
                cancel_match = False
                if exclude_multi_matched:
                    for clx in matches[cl1]:
                        if s2 not in possible_session_cl[clx]:
                            cancel_match = True
                            break
                    for clx in matches[cl2]:
                        if s2 not in possible_session_cl[clx]:
                            cancel_match = True
                            break

                    # if exclude multiple matches, cl2 is removed from possible s1 matches.
                    possible_cl_matches_session[s1].remove(cl2)
                    possible_cl_matches_session[s2].remove(cl1)

                if not cancel_match:
                    # add that cluster to the unique matches
                    matches[cl1].append(cl2)
                    matches[cl2].append(cl1)

                # s2 clusters can no longer be matched to cl1
                possible_session_cl[cl1].remove(s2)
                possible_session_cl[cl2].remove(s1)

    return matches


def matches_dict_to_unique_sets(matches, distance_mat,
                                select_lower=True, require_subsets=True):
    """
    Run after find_unique_session_cl_matches. This returns the unique sets given a unique matches dictionary.
    :param matches: output of find_unique_session_cl_matches
    :param distance_mat: distance matrix to be used for tie-breakers
    :param select_lower: bool, if true selects lowest score for tie-breaker.
        score is determine by the product of all submatches
        *note dm will be normalize such that max(dm) =1 for select_lower True
        if select_lower is False, dm will be normalize such that min(dm)=1
    :param require_subsets: bool, if true, a bigger set can only be included if the subsets are present **convservative
    :return:
        list of unique cluster sets

    -- bug found. repeated elements across sets is possible.
    """

    # subfunction to get all combinations
    def _get_subsets(big_set):
        return [set(subset) for subset in combinations(big_set, len(big_set) - 1)]

    def _recursive_valid_sets(all_sets, check_set):

        n_elements = len(check_set)

        # if only one element, set is valid
        if n_elements == 1:
            return True
        # if set tuple in all sets, valid
        elif n_elements == 2:
            return check_set in all_sets
        # if set is of 3 or more, all of the subsets must be in the list
        else:
            new_check_sets = _get_subsets(check_set)
            valid = True
            for set_i in new_check_sets:
                valid &= _recursive_valid_sets(all_sets, set_i)
            return valid

    def _set2dict(cl_sets):
        cl_sets_dict = {}
        for cl_set in cl_sets:
            for cl in cl_set:
                if cl not in cl_sets_dict.keys():
                    cl_sets_dict[cl] = [cl_set - {cl}]
                else:
                    cl_sets_dict[cl].append(cl_set - {cl})
        return cl_sets_dict

    ## main ##
    if select_lower: # normalize max to 1
        distance_mat = distance_mat/np.abs(distance_mat.values.max())
    else: # normalize min to 1
        distance_mat = 2 - distance_mat / distance_mat.values.max()

    # change matches to sets & including subsets
    all_matches_sets = []
    for cl, cl_matches in matches.items():
        all_matches_sets.append(set([cl] + cl_matches))
        if len(cl_matches) > 1:
            for clx in cl_matches:
                all_matches_sets.append(set([cl] + [clx]))

    # take out repeated sets
    cl_sets = []
    for cl_set in all_matches_sets:
        if cl_set not in cl_sets:
            cl_sets.append(cl_set)

    # verifies that larger sets contain the respective subsets
    # this is a conservative approach
    if require_subsets:
        valid_cl_sets = []
        for cl_set1 in cl_sets:
            if _recursive_valid_sets(cl_sets, cl_set1):
                valid_cl_sets.append(cl_set1)
        cl_sets = valid_cl_sets

    # change sets to a dict to verify there are no duplicate clusters
    cl_sets_dict = _set2dict(cl_sets)

    # iterate to get final set list
    valid_cl_sets = []
    for cl, clm in cl_sets_dict.items():
        n_matches = len(clm)
        if n_matches == 1:  # unique match
            cl_set = {cl} | clm[0]
        else:  # match with best scoring set
            match_vals = np.zeros(n_matches)
            for ii, clm_i in enumerate(clm):  # takes product of the set matches
                match_vals[ii] = distance_mat.loc[cl, [clx for clx in clm_i]].values.prod()

            if select_lower:
                ii = np.argmin(match_vals)
            else:
                ii = np.argmax(match_vals)

            # best scoring set
            cl_set = {cl} | clm[ii]

        # make sure it is not already on the list
        if cl_set not in valid_cl_sets:
            valid_cl_sets.append(cl_set)

    valid_cl_sets_dict = _set2dict(valid_cl_sets)

    return valid_cl_sets, valid_cl_sets_dict

