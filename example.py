'''
Created on 2021-04-29

@author: lucianovilas@gmail.com

'''

import numpy as np

from nd_metrics.utils import make_clusters
from nd_metrics.metrics import all_metrics, ND_ClusterF, ND_KMetric, ND_BCubedMetric, ND_PairwiseFMetric


def test_all_metrics_function():
    # true and pred labels as list or numpy 1-d array
    y_true = np.array([1,1,1,2,2,3,3,3])
    y_pred = np.array([1,1,1,2,2,2,2,2])

    # true and pred clusters as dict of sets
    y_true_c, y_pred_c = make_clusters(y_true, y_pred)
    # y_true_c: {1: set({0, 1, 2}), 2: set({3, 4}), 3: set({5, 6, 7})}
    # y_pred_c: {1: set({0, 1, 2}), 2: set({3, 4, 5, 6, 7})}

    # print('T: ',y_true_c)
    # print('P:', y_pred_c)

    return all_metrics(y_true_c, y_pred_c)

def test_all_metrics_classes():
    # true and pred labels as list or numpy 1-d array
    y_true = np.array([1,1,1,2,2,3,3,3])
    y_pred = np.array([1,1,1,2,2,2,2,2])

    # true and pred clusters as set of sets
    cf_metric = ND_ClusterF(y_true, y_pred)()
    k_metric  = ND_KMetric(y_true, y_pred)()
    b3_metric = ND_BCubedMetric(y_true, y_pred)()
    p_metric  = ND_PairwiseFMetric(y_true, y_pred)()
    

    return {'ClusterFMetric': cf_metric,\
            'KMetric': k_metric,\
            'BCubedMetric': b3_metric,\
            'PairwiseFMetric': p_metric}


if __name__ == '__main__':
    
    all_metrics = test_all_metrics_function()

    print("\nall_metrics_functions\n")
    print( "{:<20s} {:<5s} {:<5s} {:<5s}".format( 'Metric', "P","R","M" ))
    for m in all_metrics:
        print( "{:<20s} {:.3f} {:.3f} {:.3f}".format( m, *all_metrics[m]) )



    all_metrics = test_all_metrics_classes()

    print("\nall_metrics_classes\n")
    print( "{:<20s} {:<5s} {:<5s} {:<5s}".format( 'Metric', "P","R","M" ))
    for m in all_metrics:
        print( "{:<20s} {:.3f} {:.3f} {:.3f}".format( m, *all_metrics[m]) )        