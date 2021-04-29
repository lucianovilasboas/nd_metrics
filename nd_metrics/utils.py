'''
Created on 2021-04-29

@author: lucianovilas@gmail.com

'''



import numpy as np



def make_clusters(labels_true,labels_pred):
    
    n_samples = len(labels_true)
    
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)    


    return true_clusters, pred_clusters



def check_labels(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays"""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred


def check_clusters(labels_true_c, labels_pred_c):
    """ Check that labels_true_c and labels_pred_c have the same number of instances """
    
    if len(labels_true_c) == 0:
            raise ValueError("labels_true_c must have at least one instance")
    if len(labels_pred_c) == 0:
            raise ValueError("labels_pred_c must have at least one instance")    

    l1, l2 = 0, 0
    for k in labels_true_c: l1 += len(labels_true_c[k]) 
    for k in labels_pred_c: l2 += len(labels_pred_c[k]) 

    if l1 != l2:
        raise ValueError('Cluster labels are not the same number of instances')
    
    return labels_true_c, labels_pred_c

