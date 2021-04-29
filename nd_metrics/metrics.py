'''
Created on 2021-04-29

@author: lucianovilas@gmail.com

'''


import numpy as np
from .utils import make_clusters, check_clusters, check_labels


# Auxiliar constants and functions

ND_Epsilon=1e-07 # A very small value to avoid division by zero


def f1(p,r):
    """ Harminic mean """
    return 2 * p * r / (p + r + ND_Epsilon)



### Api functions    

        
# K-Metric only calculate
def k_metric(T,P):
    """Compute: K-Metric
   
    Parameters
    ----------
    :param T: clusters containing the ground truth cluster labels.
    :param P: clusters containing the predicted cluster labels.
    
    Returns 
    -------
    :return float AAP: KMetric recall
    :return float ACP: KMetric precision
    :return float K: KMetric geometric mean
    Reference
    ---------
    Kim, J. "A fast and integrative algorithm for 
    clustering performance evaluation in author name disambiguation."
    Scientometrics (2019): 661-681, 120(2).    
    """
    
    pIndex, cSize = {}, {}
    instSum, aapSum, acpSum = 0,0,0
    for i,Pi in enumerate(P):
        for p in P[Pi]:
            pIndex[p]=i
        cSize[i]=len(P[Pi])
        
    instSum=0
    for Tj in T:
        len_Tj = len(T[Tj])
        instSum += len_Tj
        
        tMap = {}
        for t in T[Tj]:
            if pIndex[t] not in tMap:
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] += 1
        for k,v in tMap.items():
            aapSum += v**2 / len_Tj
            acpSum += v**2 / cSize[k]

    AAP = aapSum/instSum
    ACP = acpSum/instSum
    
    return ACP, AAP, np.sqrt(AAP*ACP)


# All-In-One metrics calculate
def all_metrics(T,P):
    """Compute all metrics for name disambiguation methods.
    Compute: ClusterF-Metric, K-Metric, B^3-Metric, SE&LE-Metric and PairwiseF-Metric
   
    Parameters
    ----------
    :param T: map of clusters containing the ground truth cluster labels.
    :param P: map of clusters containing the predicted cluster labels.
    
    Returns 
    -------
    :return float cR: ClusterFMetric recall
    :return float cP: ClusterFMetric precision
    :return float f_score: ClusterFMetric f_score
    
    :return float AAP: KMetric recall
    :return float ACP: KMetric precision
    :return float K: KMetric geometric mean
    
    :return float bR: BCubedMetric recall
    :return float bP: BCubedMetric precision
    :return float f_score: BCubedMetric f_score

    :return float SE: SELEMetric recall
    :return float LE: SELEMetric precision
    :return float f_score: SELEMetric f_score

    :return float pR: PairwiseFMetric recall
    :return float pP: PairwiseFMetric precision
    :return float f_score: PairwiseFMetric f_score

    Reference
    ---------
    Kim, J. "A fast and integrative algorithm for 
    clustering performance evaluation in author name disambiguation."
    Scientometrics (2019): 661-681, 120(2).    
    """
    
    pIndex = {} # Common to all measures
    cSize = {}  # ClusterF-Metric, K-Metric, B^3-Metric and SE&LE-Metric
    cMatch = 0  # ClusterF-Metric
    instSum, aapSum, acpSum = 0,0,0 #  K-Metric and B^3-Metric
    spSum, lmSum, instTrSum, instPrSum = 0,0,0,0 # SE&LE-Metric
    pairPrSum, pairTrSum, pairInstSum = 0,0,0 # PairwiseF-Metric
    
    for i,Pi in enumerate(P):
        for p in P[Pi]:
            pIndex[p]=i
        cSize[i]=len(P[Pi])
        pairPrSum += len(P[Pi])*(len(P[Pi])-1)/2.0
        
    instSum=0
    for Tj in T:
        len_Tj = len(T[Tj])
        instSum += len_Tj
        pairTrSum += len_Tj*(len_Tj-1)/2.0
        
        tMap = {}
        for t in T[Tj]:
            if pIndex[t] not in tMap:
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] += 1
        
        maxKey, maxValue = 0,0            
        for k,v in tMap.items():
            if v == len_Tj and cSize[k] == len_Tj:
                cMatch += 1
            aapSum += v**2 / len_Tj
            acpSum += v**2 / cSize[k]            
            if v > maxValue:
                maxValue = v
                maxKey = k
            pairInstSum += v*(v-1)/2.0
        
        spSum += len_Tj - maxValue
        lmSum += cSize[maxKey] - maxValue
        instTrSum += len_Tj
        instPrSum += cSize[maxKey]
    
    cR, cP   = cMatch / len(T), cMatch / len(P)
    AAP, ACP =  aapSum/instSum, acpSum/instSum
    bR, bP   = AAP, ACP 
    SE, LE   = spSum/instTrSum, lmSum/instPrSum
    pR, pP   = pairInstSum/pairTrSum, pairInstSum/pairPrSum
    
    return {'ClusterFMetric': (cP, cR, f1(cP,cR)),\
            'KMetric': (ACP, AAP, np.sqrt(ACP*AAP)),\
            'BCubedMetric': (bP, bR, f1(bP,bR)),\
            'SELEMetric': (SE, LE, f1(SE,LE)),\
            'PairwiseFMetric': (pP, pR, f1(pP,pR))}



### Api classes    
    
class ND_MetricBase():
    """
        Base class for initiate and check inputs...
    """
    
    def __init__(self, labels_true, labels_pred):

        self.T, self.P = self.valid_inputs(labels_true, labels_pred)
        self.pIndex = {} # Common to all measures

    def valid_inputs(self, true_inputs, pred_inputs):
        if isinstance(true_inputs, (np.ndarray, list)) and isinstance(pred_inputs, (np.ndarray, list)):
            return make_clusters( *check_labels(true_inputs, pred_inputs) )
        elif isinstance(true_inputs, (dict)) and isinstance(pred_inputs, (dict)):
            return check_clusters(true_inputs, pred_inputs)
        else:     
            raise ValueError('Mismatch formats')
        
    def score(self):
        pass
    
    def __call__(self):
        return self.score()    




class ND_ClusterF(ND_MetricBase):
    """
    Class ClusterF Metric 
    
    Reference
    ---------
    Kim, J. "A fast and integrative algorithm for 
    clustering performance evaluation in author name disambiguation."
    Scientometrics (2019): 661-681, 120(2).         

    """
    
    def __init__(self, labels_true, labels_pred):
        super().__init__(labels_true, labels_pred)

    def score(self):
        cSize = {}  # Cluster-F, K-Metric, B^3 and SE&LE
        cMatch = 0  # Cluster-F

        for i,Pi in enumerate(self.P):
            for p in self.P[Pi]:
                self.pIndex[p]=i
            cSize[i]=len(self.P[Pi])

        instSum=0
        for Tj in self.T:
            len_Tj = len(self.T[Tj])
            instSum += len_Tj
            
            tMap = {}
            for t in self.T[Tj]:
                if self.pIndex[t] not in tMap:
                    tMap[self.pIndex[t]] = 0
                tMap[self.pIndex[t]] += 1

            for k,v in tMap.items():
                if v == len_Tj and cSize[k] == len_Tj:
                    cMatch += 1

        cR, cP   = cMatch / len(self.T), cMatch / len(self.P)
        
        return cP, cR, f1(cP, cR)
    
    
        
class ND_KMetric (ND_MetricBase):
    """
    Class KMetric Metric 
    
    Reference
    ---------
    Kim, J. "A fast and integrative algorithm for 
    clustering performance evaluation in author name disambiguation."
    Scientometrics (2019): 661-681, 120(2).         

    """
    
    def __init__(self, labels_true, labels_pred):
        super().__init__(labels_true, labels_pred)

    def score(self):
        
        self.cSize = {}
        instSum, aapSum, acpSum = 0,0,0
        
        for i,Pi in enumerate(self.P):
            for p in self.P[Pi]:
                self.pIndex[p]=i
            self.cSize[i]=len(self.P[Pi])

        instSum=0
        for Tj in self.T:
            len_Tj = len(self.T[Tj])
            instSum += len_Tj
            
            tMap = {}
            for t in self.T[Tj]:
                if self.pIndex[t] not in tMap:
                    tMap[self.pIndex[t]] = 0
                tMap[self.pIndex[t]] += 1
            for k,v in tMap.items():
                aapSum += v**2 / len_Tj
                acpSum += v**2 / self.cSize[k]

        AAP = aapSum / instSum
        ACP = acpSum / instSum
        
        return ACP, AAP, np.sqrt(ACP*AAP)
    

class ND_BCubedMetric(ND_KMetric):
    """
    Class BCubedMetric Metric 
    
    Reference
    ---------
    Kim, J. "A fast and integrative algorithm for 
    clustering performance evaluation in author name disambiguation."
    Scientometrics (2019): 661-681, 120(2).         

    """   
        
    def __init__(self, labels_true, labels_pred):
        super().__init__(labels_true, labels_pred)
    
    def score(self):
        bP, bR, _ = super().score()
        
        return bP, bR, f1(bP, bR)
    


class ND_PairwiseFMetric (ND_MetricBase):
    """
    Class PairwiseFMetric Metric 
    
    Reference
    ---------
    Kim, J. "A fast and integrative algorithm for 
    clustering performance evaluation in author name disambiguation."
    Scientometrics (2019): 661-681, 120(2).         

    """
    
    def __init__(self, labels_true, labels_pred):
        super().__init__(labels_true, labels_pred)
    
    def score(self):
        
        pairPrSum=0
        pairTrSum=0
        pairInstSum=0  
        
        for i,Pi in enumerate(self.P):
            for p in self.P[Pi]:
                self.pIndex[p]=i
                len_Pi = len(self.P[Pi])
            pairPrSum += len_Pi * (len_Pi-1) / 2.0

        for Tj in self.T:
            len_Tj = len(self.T[Tj])
            pairTrSum += len_Tj * (len_Tj-1) / 2.0
            
            tMap = {}
            for t in self.T[Tj]:
                if self.pIndex[t] not in tMap:
                    tMap[self.pIndex[t]] = 0
                tMap[self.pIndex[t]] += 1

            for k,v in tMap.items():
                pairInstSum += v*(v-1)/2.0

        try:
            pP =  pairInstSum / pairPrSum
        except ZeroDivisionError:
            pP = 1.0

        try:
            pR = pairInstSum / pairTrSum 
        except ZeroDivisionError:
            pR = 1.0

        return pP, pR, f1(pP, pR)

