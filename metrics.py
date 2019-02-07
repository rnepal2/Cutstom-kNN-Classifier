import math
import numpy as np

# If we use this function with test data: 
def accuracy_score(y_true, y_pred):
    '''
    parameters:
        y_true: acutual target array
        y_pred: predicted target array
    '''
    assert len(y_true) == len(y_pred)
    
    if len(y_true) == 0: raise ValueError('Empty y_true!')
    
    correct = 0
    for y1, y2 in zip(y_true, y_pred):
        if y1 == y2: correct += 1
    return correct/len(y_true)
    
# Generalization error = 1 - accuracy_score


# 8. 
# precision
def precision_score(y_true, y_pred):
    '''
    parameters:
        y: actual target array (first input array should be actual target)
        y_: predicted target array
    
    precision_score = TP/(TP + FP) {TP: True Positive, FP: False Positive}
    Note that: this precision_score is valid only for binary classification.
    '''
    # some confirmations and coversion if required
    if len(np.unique(y_true)) != 2:
        raise ValueError('the target has more than two classes!')
    
    if min(y_true)!=0 and min(y_pred)!=0 and max(y_true)!=1 and max(y_pred)!=1:
        raise ValueError('target can only be either 0 or 1!')
    
    tp = 0; fp = 0
    for y1, y2 in zip(y_true, y_pred):
        if y1==1 and y2==1: tp += 1
        if y1==0 and y2==1: fp += 1
    if tp+fp==0: raise ValueError('TP+FP=0: precision cannot be defined.')
    return tp/(tp+fp)
        
# recall
def recall_score(y_true, y_pred):
    '''
    parameters:
        y_true: actual target array (first input array should be actual target)
        y_pred: predicted target array
    
    recall = TP/(TP + FN) {TP: True Positive, FN: False Negative}
    Note that: this recall is valid only for binary classification.
    '''
    # some confirmations and coversion if required
    if len(np.unique(y_true)) != 2:
        raise ValueError('the target has more than two classes!')
    
    if min(y_true)!=0 and min(y_pred)!=0 and max(y_true)!=1 and max(y_pred)!=1:
        raise ValueError('target can be only 0 and 1!')
    
    tp = 0; fn = 0
    for y1, y2 in zip(y_true, y_pred):
        if y1==1 and y2==1: tp += 1
        if y1==1 and y2==0: fn += 1
    if tp+fn==0: raise ValueError('TP+FN=0: precision cannot be defined.')
    return tp/(tp+fn)

# F1-score
def f1_score(y_true, y_pred):
    '''
    parameters:
        y_true: actual target array (first input array should be actual target)
        y_pred: predicted target array
    
    f1_score = 2 precision*recall/(precision + recall) 
    or f1_score = harmonic mean of precision and recall
    Note that: this f1_score is valid only for binary classification.
    '''
    precision, recall = precision_score(y_true, y_pred), recall_score(y_true, y_pred)
    return 2*precision*recall/(precision + recall)

# confusion matrix
# returns a confusion_matrix for n-classifier 
# works for binary or higher number of classes
def confusion_matrix(y_true, y_pred):  
    '''
    parameters:
        y_true: actual target array
        y_pred: predicted target array
    '''
    # confirm input sizes
    assert len(y_true) == len(y_pred)
    
    # number of classes
    length = len(np.unique(y_true))
    decisions = dict()
    for y1, y2 in zip(y_true, y_pred):
        
        if str(y1) in decisions.keys():
            decisions[str(y1)].append(y2)
        else:
            decisions[str(y1)] = [y2]
            
    # sorting and counting numbers in a new nested dictionary 
    nested_decisions = dict()
    for key in sorted(decisions.keys()):
        values = decisions[key]
        inner_dict = dict()
        # initialize inner_dict
        for value in range(length):
            inner_dict[str(value)] = 0
        # counting values
        for value in values:
            inner_dict[str(value)] += 1
        nested_decisions[key] = inner_dict
        
    # creating confusion matrix
    # initialize the confusion matrix
    confusion_matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            confusion_matrix[i][j] = nested_decisions[str(i)][str(j)]
    return confusion_matrix


# ROC curve
def roc_curve(y_true, y_score):
    '''
    paramters: 
        y_true: actual target array
        y_score: probability estimates of the positive target
        
        TPR = TP/(TP + FN)
        FPR = FP/(FP + TN)
    returns:
        fpr : array, shape = [>2]
            Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= thresholds[i].
        tpr : array, shape = [>2]
            Increasing true positive rates such that element i is the true
            positive rate of predictions with score >= thresholds[i].
        thresholds : array, shape = [n_thresholds]
            Decreasing thresholds on the decision function used to compute
            fpr and tpr. `thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`
    '''

    # changing y_true into a boolean array with 1 being true
    y_true = (y_true == 1)
    
    # sorting y_true and y_score on desceding y_score
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true, axis=None, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    
    # Adding an extra threshold position if necessary
    # to make sure that the curve starts at (0, 0)
    if len(tps) == 0 or fps[0] != 0 or tps[0] != 0:
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]
    
    return fps/max(fps), tps/max(tps), thresholds

# ROC curve implementation 2
def roc_curve2(y_true, y_score):
    '''roc_curve: returns fpr, tpr and thresholds'''
    assert len(y_true) == len(y_score)
    if max(y_true) > 1 or min(y_true) < 0:
        raise ValueError('roc_curve is implemented for only binary 0, 1 classes')
    
    num = len(np.unique(y_score))
    if num == 2:
        print('With y_labels/not probablity scores, will give only give 3 points.')
    
    tpr = []; fpr = []; thresholds = []
    
    for i in range(num):
        threshold = (1.0/num) * (i+1)
        y = (y_score > threshold).astype('int8')
        tn, fp, fn, tp = confusion_matrix(y_true, y).ravel()
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))
        thresholds.append(threshold)
    
    # Adding an extra threshold position if necessary
    # to make sure that the curve starts at (0, 0)
    if len(tpr) == 0 or fpr[0] != 0 or tpr[0] != 0:
        tpr = np.r_[0, tpr]
        fpr = np.r_[0, fpr]
        thresholds = np.r_[thresholds[0] + 1, thresholds]
        
    return fpr, tpr, thresholds


# area under the roc curve
def roc_auc_score(y_true, y_score):
    
    assert len(y_true) == len(y_score)
    
    if len(y_score) < 2:
        raise ValueError('at least 2 points are needed to calculate area, but y_score.shape: %s' % y_score.shape)
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # calculate area under roc_curve
    area = np.trapz(tpr, fpr)
    return area

# precision_recall implementation
def precision_recall_curve(y_true, y_score):
    '''precision_recall_curve: returns precisions, recalls and thresholds'''
    assert len(y_true) == len(y_score)
    if max(y_true) > 1 or min(y_true) < 0:
        raise ValueError('precision_recall_curve is implemented for only binary 0, 1 classes')
    
    num = len(np.unique(y_score))
    if num == 2:
        print('Without probablity scores -with only target classes, will give only 3 points.')
    
    precisions = []; recalls = []; thresholds = []
    for i in range(0, num):
        threshold = (1.0/num) * i
        y = (y_score > threshold).astype('int8')
        p = precision_score(y_true, y)
        r = recall_score(y_true, y)
        
        precisions.append(p)
        recalls.append(r)
        thresholds.append(threshold)
    
    # Adding an extra threshold position if necessary
    # to make sure that the curve starts at (0, 0)
    precisions = list(reversed(precisions))
    recalls = list(reversed(recalls))
    thresholds = list(reversed(thresholds))
    if len(precisions) == 0 or recalls[0] != 0 or precisions[0] != 0:
        precisions = np.r_[1, precisions]
        recalls = np.r_[0, recalls]
        thresholds = np.r_[thresholds[0] + 1, thresholds]
    return precisions, recalls, thresholds