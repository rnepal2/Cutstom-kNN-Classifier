'''
@author: Rabindra Nepal
Email: rnepal2@unl.edu
'''

import math
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
# from file
from metrics import accuracy_score, precision_score, recall_score, f1_score

# it takes data, labels and ith partition and returns 
# cross validation dataset for folds number of folds
def partition_data(data, labels, ith_fold, folds):
    '''
    parameters:
        data: [num_sample, num_features] or x_train
        labels: [num_sample] or y_train
        ith_fold: ith_fold number out ot folds
        folds: total number of folds
    
    '''
    assert len(data) == len(labels)
    
    size = int(len(data)/folds)
    test_indices = np.arange(ith_fold*size, (ith_fold+1)*size)
    train_indices = np.delete(np.arange(len(data)), test_indices)
    
    x_test, y_test = np.take(data, test_indices, axis=0), np.take(labels, test_indices)
    x_train, y_train = np.take(data, train_indices, axis=0), np.take(labels, train_indices)
    
    return (x_train, y_train), (x_test, y_test)



# this is a helper function to use for each of the fold in cross-validation
# this separate function is defined to parallize the cross_validation function 
# with each fold carried out parallely

def for_each_fold(fold, folds, data, labels, model, error_function):
    
    (x_train, y_train), (x_test, y_test) = partition_data(data, labels, fold, folds)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
        
    # Based on the error_function passed
    if error_function is None: # if None calculate precision
        error = precision_score(y_test, y_pred)
            
    elif error_function == 'precision':
        error = precision_score(y_test, y_pred)
        
    if error_function == 'accuracy':
        error = accuracy_score(y_test, y_pred)
            
    elif error_function == 'recall':
        error = recall_score(y_test, y_pred)
            
    elif error_function == 'f1':
        error = f1_score(y_test, y_pred)
    else:
        raise ValueError('%s error function is not defined.' % error_function)
        
    return {'expected labels': y_test, 
            'predicted labels': y_pred, 
            'errors': [error]}

# Cross-Validation function
def kfold_cross_validation(folds, data, labels, model, model_args=None, error_function=None):
    '''
    parameters:
        folds: an integer number of folds.
        data: a numpy array with rows representing data samples and columns representing features.
        labels: a numpy array with labels corresponding to each row of training_features.
        model: an object with the fit and predict functions. 
        model_args: a dictionary of arguments to pass to the classification algorithm.
        error_function: Returns error value between predicted and true labels. For example, 
                        accuracy, generalization error, precision, recall, and F1 score could
                        be used as error_function.
        verbose: 0: don't print any progress message
                 1: print the progress message
    returns:
        a dictionary with: expected labels, predicted labels, average error
    '''
    if model_args:
        if 'k' in model_args.keys():
            model.k = model_args['k']
        if 'weights' in model_args.keys():
            model.weights = model_args['weights']
        if 'distance_f' in model_args.keys():
            model.distance_f = model_args['distance_f']
        
    if error_function is None:
        error_function = 'precision'
        
    predictions = dict()
    # Parallelizing the jobs for each fold
    num_cores = multiprocessing.cpu_count()
    out = Parallel(n_jobs=num_cores)(
                delayed(for_each_fold)(fold, folds, data, labels, model, error_function) for fold in range(folds))
    
    # formatting the output
    for fold_out in out:
        for key in fold_out.keys():
            if key not in predictions.keys():
                predictions[key] = list(fold_out[key])
            else:
                predictions[key].extend(list(fold_out[key]))
    # to array
    for key in predictions.keys():
        predictions[key] = np.array(predictions[key])
    return predictions
