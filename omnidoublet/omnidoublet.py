import os
import sys
import argparse
import random
import numpy as np
import scanpy as sc
import scipy
import pandas as pd
import anndata
import gc
from sklearn.metrics import f1_score

from annoy import AnnoyIndex
from scipy.stats import gaussian_kde
# from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, isspmatrix_csc, vstack, csc_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

# generate doublet parents index
def gen_index(origin_num, rseed=123, sim_ratio=0.3, method='random', clusters=None):
    np.random.seed(rseed)

    doublet_num = int(origin_num * sim_ratio)

    if method == 'random':
        print('Randomly simulating doublets ...')
        idx1 = np.random.choice(origin_num, size=doublet_num, replace=True)
        idx2 = np.zeros_like(idx1)
        for i, idx in enumerate(idx1):
            avail_indices = np.setdiff1d(np.arange(origin_num), idx1[i], assume_unique=True)
            idx2[i] = np.random.choice(avail_indices)

    elif method == 'homo':
        print('Generating homotypic doublets ...')
        if clusters is None:
            raise ValueError('Error : please input pre-clustering results.')
        selected_clusters = np.random.choice(clusters, doublet_num)
        idx1 = np.array(
            [np.random.choice(np.where(clusters == cluster_id)[0], 1)[0] for cluster_id in selected_clusters])
        idx2 = np.array([np.random.choice(np.setdiff1d(np.where(clusters==cluster_id)[0], selected1), 1)[0] for selected1, cluster_id in zip(idx1, selected_clusters)])

    elif method == 'hetero':
        print('Generating heterotypic doublets ...')
        if clusters is None:
            raise ValueError('Error : please input pre-clustering results.')
        selected_clusters = np.random.choice(clusters, doublet_num)
        idx1 = np.array(
            [np.random.choice(np.where(clusters == cluster_id)[0], 1)[0] for cluster_id in selected_clusters])
        idx2 = np.array(
            [np.random.choice(np.where(clusters != cluster_id)[0], 1)[0] for cluster_id in selected_clusters])

    else:
        raise ValueError('Error : unrecognized method !')

    return idx1, idx2


# library size normalization
def bootstrap_normalize(combined_profiles, original_profiles, n_bootstrap=100):
    if issparse(original_profiles):
        original_library_sizes = np.array(original_profiles.sum(axis=1)).reshape(-1)
    else:
        original_library_sizes = np.sum(original_profiles, axis=1).reshape(-1)
    # print('original_library_sizes : ', original_library_sizes.shape, original_library_sizes)

    bootstrap_samples = np.random.choice(original_library_sizes, n_bootstrap, replace=True)
    sampled_library_sizes = np.random.choice(bootstrap_samples, combined_profiles.shape[0], replace=True)

    if issparse(combined_profiles):
        combined_library_sizes = np.array(combined_profiles.sum(axis=1)).reshape(-1)
    else:
        combined_library_sizes = np.sum(combined_profiles, axis=1).reshape(-1)
    # print('combind_library_sizes : ', combined_library_sizes.shape, combined_library_sizes)

    scale_factor = sampled_library_sizes/combined_library_sizes
    # print('scale_factor :', scale_factor.shape, scale_factor)

    # print('combined_profiles : ', combined_profiles.shape, combined_profiles)
    if issparse(combined_profiles):
        combined_profiles = combined_profiles.toarray()

    normalized_combined_profiles = combined_profiles * scale_factor[:, np.newaxis]
    # print('normalized_combined_profiles : ', normalized_combined_profiles.shape, normalized_combined_profiles)

    if issparse(original_profiles):
        normalized_combined_profiles = scipy.sparse.csc_matrix(normalized_combined_profiles)

    return normalized_combined_profiles

# create doublets
def create_doublets(scData, idx1, idx2, normalized=False):
    # print("scData.X shape : ", scData.X.shape)
    doublets = (scData.X[idx1, :] + scData.X[idx2, :])
    if normalized:
        doublets = bootstrap_normalize(doublets, original_profiles=scData.X, n_bootstrap=scData.shape[0])

    # print('doublets : ', doublets.shape, doublets)
    return doublets

# RNA process pipeline
def RNA_pp(adata, min_cells=3, target_sum=1e4, ntg=2000, n_nbs=None, n_pcs=40, min_dist=0.4):
    adata.var_names_make_unique()
    if n_nbs is None :
        n_nbs = int(round(0.5*np.sqrt(adata.shape[0])))
    sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    # hvg
    sc.pp.highly_variable_genes(adata, n_top_genes=ntg)

    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    # reduction
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_nbs, n_pcs=n_pcs)
    sc.tl.umap(adata, min_dist=min_dist)

    return adata


# scanpy-scRNA preprocess + leiden
def fast_cluster(adata, min_cells=3, target_sum=1e4, ntg=2000, n_nbs=None, n_pcs=40, resolution=0.5):
    adata.var_names_make_unique()
    if n_nbs is None:
        n_nbs = int(round(0.5 * np.sqrt(adata.shape[0])))
    sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=ntg)

    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_nbs, n_pcs=n_pcs)
    # 5. Run the Leiden clustering algorithm
    sc.tl.leiden(adata, resolution=resolution)

    # Extract the clusters as a NumPy array
    clusters = adata.obs['leiden'].to_numpy().astype(int)
    # print('clusters: ', clusters.shape, clusters)

    return clusters

# centered log-ratio transformation
def CLR_transform(df):
    '''
    implements the CLR transform used in CITEseq (need to confirm in Seurat's code)
    https://doi.org/10.1038/nmeth.4380
    '''
    logn1 = np.log(df + 1)
    T_clr = logn1.sub(logn1.mean(axis=1), axis=0)
    return T_clr


def ADT_pp(ADTadata):
    ADT_df = ADTadata.to_df()
    ADT_norm = CLR_transform(ADT_df)

    return ADT_norm

def ATAC_pp(adata, min_cells=None, pct_top_idf=0.2, n_nbs=None, n_pcs=40, min_dist=0.4):
    adata.var_names_make_unique()
    if min_cells is None:
        min_cells = round(adata.shape[0]*0.01)
    if n_nbs is None :
        n_nbs = int(round(0.5*np.sqrt(adata.shape[0])))
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.raw = adata.copy()

    # TF-IDF
    tfidf_transformer = TfidfTransformer()
    tfidf_data = tfidf_transformer.fit_transform(adata.X)
    tfidf_data = np.log1p(tfidf_data * 1e4)

    # keep top% peaks
    # n_top_idf = round(adata.X.shape[1] * pct_top_idf)
    # top_n_indices = np.argsort(tfidf_transformer.idf_)[-n_top_idf:]
    # top_idf_data = tfidf_data[:, top_n_indices]
    #
    # adata = adata[:, top_n_indices]

    # reduction
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_nbs, n_pcs=n_pcs)
    sc.tl.umap(adata, min_dist=min_dist)

    # lsi reduction
    svd = TruncatedSVD(n_components=n_pcs)
    adata.obsm['tfidf'] = tfidf_data
    adata.obsm['lsi'] = svd.fit_transform(tfidf_data)

    return adata

# annoy knn
# data : numpy array
def get_annoy_graph(data, k, n_tree, dist_metric='euclidean', rseed=123):
    # print('get_annoy_graph : ')
    if issparse(data):
        data = data.todense()
    data = np.array(data)
    n_samples = data.shape[0]
    n_features = data.shape[1]

    # initialize index
    annoy_index = AnnoyIndex(n_features, metric=dist_metric)
    annoy_index.set_seed(rseed)
    # add item to index
    for i, vec in enumerate(data):
        # print('i : ', i)
        # print('vec : ', vec.shape, vec)
        # print(type(vec))
        vec = vec.flatten()
        annoy_index.add_item(i, vec)

    annoy_index.build(n_tree)

    # knn info
    knn = []
    distance = []
    for i in range(n_samples):
        # Exclude the item itself from its knn
        neighbors = annoy_index.get_nns_by_item(i, k+1)[1:]
        knn.append(neighbors)
        # get distances
        nei_dist = []
        for nei in neighbors:
            nei_dist.append(annoy_index.get_distance(i, nei))
        distance.append(nei_dist)

    # print('knn : ', type(knn))
    # print('distance : ', type(distance))

    knn = np.array(knn)
    distance = np.array(distance)

    return knn, distance


# calculate jaccard distance between two modality knn
def calculate_jaccard_distance(knn1, knn2):
    jac_dist = []
    for neighbors1, neighbors2 in zip(knn1, knn2):
        intersection = np.intersect1d(neighbors1, neighbors2)
        union = np.union1d(neighbors1, neighbors2)
        dist = 1.0 - (intersection.shape[0]/union.shape[0])
        jac_dist.append(dist)

    jac_dist = np.array(jac_dist)
    # print('jac_dist : ', jac_dist.shape, jac_dist)

    return jac_dist

# calculate jaccard coefficient of each sample between two modality
def calculate_jaccard_coef(knn1, knn2):
    jac_coef = []
    for neighbors1, neighbors2 in zip(knn1, knn2):
        intersection = np.intersect1d(neighbors1, neighbors2)
        union = np.union1d(neighbors1, neighbors2)
        coef = float(intersection.shape[0])/union.shape[0]
        jac_coef.append(coef)

    jac_coef = np.array(jac_coef)
    # print('jac_coef : ', jac_coef.shape, jac_coef)

    return jac_coef

# normalize distance
def normalize_distance(distances):
    # closer neigbhor , bigger weight
    eps = np.finfo(distances.dtype).eps
    distances = np.maximum(distances, eps)
    inverted_distances = 1.0 / distances
    sum_distances = np.sum(inverted_distances, axis=1)
    normalized_weights =inverted_distances / sum_distances[:, np.newaxis]

    return normalized_weights

def min_max_normalize(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)

    # apply min-max normalization
    normalized_scores = (scores - min_score) / (max_score - min_score)

    return normalized_scores

class OmniDoublet ():
    def __init__(self, RNAadata, modality_adata, modality = 'ATAC', min_cells=None, target_sum=1e4, ntg=2000, n_nbs=None, n_pcs=40,
                 min_dist=0.4, pct_top_idf=0.2, rseed=123, sim_ratio=0.3, normalized=False, sim_method='random',
                 dist_metric='euclidean', n_tree=10, confi_threshold=0.5, posterior=False):
        # print('function : __init__')

        # inistialize parameters
        self.RNAadata = RNAadata
        self.modality_adata = modality_adata
        self.modality = modality
        self.num_cells = RNAadata.shape[0]
        self.posterior = posterior

        if min_cells is None:
            min_cells = round(self.num_cells * 0.01)
        if n_nbs is None:
            n_nbs = int(round(0.5 * np.sqrt(self.num_cells)))
        if n_tree is None:
            n_tree = int(round(0.5 * np.sqrt(self.num_cells)))
        if dist_metric == 'cosine':
            dist_metric = 'angular'

        # preprocess parameters
        self.min_cells = min_cells
        self.target_sum = target_sum
        self.ntg = ntg
        self.n_nbs = n_nbs
        self.n_pcs = n_pcs
        self.min_dist = min_dist
        self.pct_top_idf = pct_top_idf
        self.rseed = rseed
        self.sim_ratio = sim_ratio
        self.normalized = normalized
        self.sim_method = sim_method
        self.n_tree = n_tree
        self.dist_metric = dist_metric
        self.confi_threshold = confi_threshold


    # core function
    def core(self,):
        # print('function : core')
        self._filter()
        # doublet simulation
        # print('Simulating doublets ...')
        #pre cluster fast cluster
        if self.sim_method != "random":
            clusters = fast_cluster(self.RNAadata)
            self.clusters = clusters
        else:
            self.clusters = None

        RNA_simD, modality_simD = self.simulate_doublets()
        # concat dataset
        RNA_all = vstack((self.RNAadata.X, RNA_simD))
        modality_all = vstack((self.modality_adata.X, modality_simD))
        init_labels =  np.array([0] * self.num_cells + [1] * RNA_simD.shape[0])

        # data preprocess & dim reduction
        # get dimension-reduced features and perform kNN scoring
        # compute doublet scores for each modality separately, then apply Jaccard-based modality weighting
        
        # get dimension-reduced features
        RNA_embeds = RNA_all
        modality_embeds = modality_all

        # compute knn graph
        RNA_knn, RNA_dist = get_annoy_graph(RNA_embeds, k=self.n_nbs, n_tree=self.n_tree,
                                            dist_metric=self.dist_metric, rseed=self.rseed)

        modality_knn, modality_dist = get_annoy_graph(modality_embeds, k=self.n_nbs, n_tree=self.n_tree,
                                                      dist_metric=self.dist_metric, rseed=self.rseed)

        jac_coef = calculate_jaccard_coef(RNA_knn, modality_knn)

        # neighbor score matrix
        RNA_knn_scores = np.zeros_like(RNA_knn)
        modality_knn_scores = np.zeros_like(modality_knn)

        RNA_knn_jac = np.zeros_like(RNA_knn, dtype=float)
        modality_knn_jac = np.zeros_like(modality_knn, dtype=float)
        for i in range(RNA_knn.shape[0]):
            neighbor_indices = RNA_knn[i]
            RNA_knn_scores[i] = init_labels[neighbor_indices]
            RNA_knn_jac[i] = jac_coef[neighbor_indices]
        for i in range(modality_knn.shape[0]):
            neighbor_indices = modality_knn[i]
            modality_knn_scores[i] = init_labels[neighbor_indices]
            modality_knn_jac[i] = jac_coef[neighbor_indices]

        # normalize distance to weight
        RNA_knn_weights = normalize_distance(RNA_dist)
        modality_knn_weights = normalize_distance(modality_dist)
        # RNA_knn_weights = min_max_normalize(RNA_dist)
        # modality_knn_weights = min_max_normalize(modality_dist)

        # element-wise multiply  [N,K]
        RNA_weights = RNA_knn_jac * RNA_knn_weights
        modality_weights = modality_knn_jac * modality_knn_weights

        # final score by weighted sum
        final_score = (RNA_weights * RNA_knn_scores + modality_weights * modality_knn_scores).sum(axis=1)

        # normalization
        final_score = min_max_normalize(final_score)


        # threshold calcluation
        threshold, _, _ = auto_semi_supervised_cutoff(real_scores, sim_scores, true_labels=init_labels, n_init=5, rseed=self.rseed)
        cls = (origin_pred >= threshold).astype(int)

        omnid_res = pd.DataFrame({'score':origin_pred, 'class':cls}, index=self.RNAadata.obs.index)

        if self.posterior:
            # calculate posterior probabilities
            doublet_probs, labels, _ = calculate_posterior_probabilities(final_score, gmm, confidence_threshold=self.confi_threshold)
            doublet_probs1 = doublet_probs[:self.num_cells]
            labels1 = labels[:self.num_cells]

            omnid_res['posterior_prob'] = doublet_probs1
            omnid_res['posterior_label'] = labels1

        # omnid_res.to_csv('omnid_res.csv')

        return omnid_res

    def _filter(self):
        # print('function : )_filter'
        self.RNAadata.var_names_make_unique()
        self.modality_adata.var_names_make_unique()
        sc.pp.filter_genes(self.RNAadata, min_cells=self.min_cells)
        if not issparse(self.RNAadata.X):
            self.RNAadata.X = csc_matrix(self.RNAadata.X)
        elif not isspmatrix_csc(self.RNAadata.X):
            self.RNAadata.X = self.RNAadata.X.tocsc()

        if self.modality == 'ATAC':
            sc.pp.filter_genes(self.modality_adata, min_cells=self.min_cells)
            if not issparse(self.modality_adata.X):
                self.modality_adata.X = csc_matrix(self.modality_adata.X)
            elif not isspmatrix_csc(self.modality_adata.X):
                self.modality_adata.X = self.modality_adata.X.tocsc()


    def simulate_doublets(self,):
        # print('function : simulate_doublets')
        idx1, idx2 = gen_index(origin_num=self.RNAadata.shape[0], rseed=self.rseed,
                               sim_ratio=self.sim_ratio, method=self.sim_method, clusters=self.clusters)
        # simulate RNA & ATAC doublets
        RNA_simD = create_doublets(self.RNAadata, idx1, idx2, self.normalized)
        modality_simD = create_doublets(self.modality_adata, idx1, idx2, self.normalized)

        self.doublets_parents_ = np.column_stack((idx1, idx2))

        return RNA_simD, modality_simD

# singlet-doublet cutoff
def calculate_cutoff(scores, rseed=123):
    # fit gmm
    gmm = GaussianMixture(n_components=2, random_state=rseed)
    # gmm = CustomGMM(n_components=2)
    gmm.fit(scores.reshape(-1,1))
    scores_sorted = np.sort(scores)
    prob = gmm.predict_proba(scores_sorted.reshape(-1, 1))

    cutoff_idx = np.where(np.diff(np.argmax(prob, axis=1)))[0][0]
    cutoff_score = scores_sorted[cutoff_idx]

    return cutoff_score, gmm


def calculate_posterior_probabilities(scores, gmm, confidence_threshold=0.5):
    '''
    given GMM compute the posterior probability of each droplet being a doublet
    '''
    means = gmm.means_.flatten()
    doublet_component = np.argmax(means) # return the index of the distribution with the larger mean

    probs = gmm.predict_proba(scores.reshape(-1,1))
    doublet_probs = probs[:, doublet_component]

    # assign labels
    labels = (doublet_probs > confidence_threshold).astype(int)

    return doublet_probs, labels, probs

def semi_supervised_gmm_cutoff(real_scores, sim_scores, n_init=5, rseed=123):
    all_scores = np.concatenate([real_scores, sim_scores])
    labels = np.concatenate([np.zeros_like(real_scores), np.ones_like(sim_scores)])

    means_init = np.array([[np.mean(real_scores)], [np.mean(sim_scores)]])
    weights_init = np.array([len(real_scores) / len(all_scores), len(sim_scores)/len(all_scores)])
    gmm = GaussianMixture(n_components=2, means_init= means_init, weights_init = weights_init, 
                          random_state=rseed, n_init=n_init)
    gmm.fit(all_scores.reshape(-1,1))

    if gmm.means_[0] > gmm.means_[1]:
        doublet_component = 0
        singlet_component = 1
    else:
        doublet_component = 1
        singlet_component = 0

    assert np.mean(sim_scores) > np.mean(real_scores), "Expect doublet scores > singlet scores"
    assert gmm.means_[doublet_component] > gmm.means_[singlet_component], "GMM learned opposite mapping!"


    scores_sorted = np.sort(all_scores)
    probs = gmm.predict_proba(scores_sorted.reshape(-1, 1))

    cutoff_idx = np.where(np.diff(np.argmax(probs , axis=1)))[0][0]
    cutoff_score = scores_sorted[cutoff_idx]

    return cutoff_score, gmm

def auto_semi_supervised_cutoff(real_scores, sim_scores, true_labels=None, n_init=5, rseed=123):
    all_scores = np.concatenate([real_scores, sim_scores])
    initial_labels = np.concatenate([np.zeros_like(real_scores), np.ones_like(sim_scores)])

    means_init = np.array([[np.mean(real_scores)], [np.mean(sim_scores)]])
    weights_init = np.array([len(real_scores) / len(all_scores), len(sim_scores) / len(all_scores)])

    # 拟合 GMM
    gmm = GaussianMixture(n_components=2, means_init=means_init, weights_init=weights_init, random_state=rseed, n_init=n_init)
    gmm.fit(all_scores.reshape(-1, 1))

    if gmm.means_[0] > gmm.means_[1]:
        doublet_component = 0
        singlet_component = 1
    else:
        doublet_component = 1
        singlet_component = 0

    scores_sorted = np.sort(all_scores)
    probs = gmm.predict_proba(scores_sorted.reshape(-1, 1))

    cutoff_idx = np.where(np.diff(np.argmax(probs , axis=1)))[0][0]
    cutoff_gmm = scores_sorted[cutoff_idx]

    if true_labels is None:
        return cutoff_gmm, 'GMM', gmm

    best_cutoff = cutoff_gmm
    best_f1 = -1
    for thresh in np.linspace(0, 1, 200):
        preds = (all_scores > thresh).astype(int)
        f1 = f1_score(true_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_cutoff = thresh

    preds_gmm = (all_scores > cutoff_gmm).astype(int)
    f1_gmm = f1_score(true_labels, preds_gmm)

    if best_f1 > f1_gmm:
        return best_cutoff, 'F1_opt', gmm
    else:
        return cutoff_gmm, 'GMM', gmm
