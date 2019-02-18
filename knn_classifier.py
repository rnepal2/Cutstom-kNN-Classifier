'''
@author: Rabindra Nepal
Email: rnepal2@unl.edu
'''

import math
import numpy as np
# scipy stats
from scipy import stats
from collections import Counter

from distances import euclidean_distance, manhattan_distance

# returns a weights array of a distances array
# if the distances contains any 0.: it returns 1 for 0 distances
# and 0. for the rest. Else returns 1./d for each distance elements
def get_weights(distances):
    distances = np.array(distances)
    
    # weights as an array of 1s
    weights = np.ones(distances.shape)    
    if np.any(distances==0.):
        for i, d in enumerate(distances):
            if d != 0.:
                weights[i] = 0.
    else:
        for i, d in enumerate(distances):
            weights[i] = 1./d
            
    return weights

# weighted mode: the actual target after inverse distance weighting
# the prediction based on the targets and weights
def weighted_mode(targets, weights):
    counter = Counter()
    for i, t in enumerate(targets):
        counter[t] += weights[i]
    return counter.most_common(1)[0][0]

# Used for probablity predictions: [0, 1] => [0.75, 0.25]
def weighted_counter(targets, weights):
    counter = Counter()
    for i, t in enumerate(targets):
        counter[t] += weights[i]
    return counter

# KNeighborClassifier Class

# knn classifier class
class KNeighborsClassifier:
    
    '''class initialization'''
    def __init__(self):
        '''  
        Initializing the fix method arguments: 
        the data will be passed later in fit method
        It's done this way to make each parameters available
        inside all the methods inside the class.
        '''
        self.training_features = None
        self.training_labels = None
        
        self.k = 5 # default value
        self.distance_f = 'euclidean'  # default distance function
        self.weights = 'uniform'       # default weights option


    # fitting with train data
    def fit(self, training_features, training_labels, k=None, distance_f=None, **kwargs):
        '''
        paramters:
            training_features: x_train: a 2D array (sample size * num of features)
            training_labels: y_train: a 1D array of targets (sample size, )
            k: number of neighbors
            distance_f: distance function to use: ['euclidean' or 'manhattan']
            **kwargs: dict of arguments to be passed for distance_f
        '''
        assert len(training_features) == len(training_labels)
        
        self.training_features = training_features
        self.training_labels = training_labels
        if k: self.k = k
        
        # if distance_f is passed use that otherwise (default: Euclidean distance)
        if distance_f: self.distance_f = distance_f
            
        # can pass other arguments also, if implemented
        # nothing to pass with kwargs for now
        if kwargs:
            pass 
    
    
    # returns a list of k nearest neighbors' distances and classes
    def kNearestNeighbors(self, x_train, y_train, test, k):
        '''
            parameters:
                x_train, y_train, test: single instance of x_test
                k: num of nearest neighbors
            retuns:
                a collections.Counter for k-nearest neighbors of of x_test array:
                    Counter({1: n1, 0: n0}): n0+n1 = num_test points
        '''
        # For uniform weights case
        
        distances, targets = list(), list()
        for i in range(len(x_train)):
            if self.distance_f == 'euclidean':
                dist = euclidean_distance(test, x_train[i, :])
            if self.distance_f == 'manhattan':
                dist = manhattan_distance(test, x_train[i, :])
            distances.append([dist, i])
        distances = sorted(distances)
        
        for i in range(k):
            dist, index = distances[i][0], distances[i][1]
            targets.append([dist, y_train[index]])
        # targets: contains k-nn each of the form [distance, neighbor class]
        return targets 
        
    
    # predict on test data
    def predict(self, test_features, weights=None):
        '''
        paramters:
            test_features: x_test 2D array of features data for test data
                           (test.size * num of features, same as in training data)
        return:
            a 1D array of predicted class for all test data
        '''
        
        # avoid running predict method before fitting the model
        if self.training_features is None or self.training_labels is None:
            raise ValueError('Model is not fitted yet, fit the model first!')
        
        # weights can take: ['uniform', 'distance'], defualt is 'uniform'
        if weights:
            self.weights = weights
        
        # list to save predictions on test data
        predictions = [] 
        for test in test_features:
            # k-nearest neighbors
            knn = self.kNearestNeighbors(self.training_features, self.training_labels, test, self.k)
            distances = np.array([knn[i][0] for i in range(len(knn))])
            targets = np.array([knn[i][1] for i in range(len(knn))]) 
            
            if self.weights == 'distance':
                weights = get_weights(distances)
                pred = weighted_mode(targets, weights)
                predictions.append(pred)
            
            if self.weights == 'uniform':
                pred = Counter(targets).most_common(1)[0][0]
                predictions.append(pred)
                
        return np.array(predictions)
    
    # prediction_probababilty for each class
    def predict_proba(self, test_features, weights=None):
        '''
        paramters:
            test_features: x_test 2D array of features data for test data
                           (test.size * num of features, same as in training data)
        return:
            a array of n test points predictions each with classes probabilities:
                [probability of 0, probability of 1]
        '''
        # avoiding prediction before fitting
        if self.training_features is None or self.training_labels is None:
            raise ValueError('Model is not fitted yet, fit the model first!')
        
        if weights:
            self.weights = weights
        
        # list to save predictions on test data
        predictions = []
        for test in test_features:
            knn = self.kNearestNeighbors(self.training_features, self.training_labels, test, self.k)
            distances = np.array([knn[i][0] for i in range(len(knn))])
            targets = np.array([knn[i][1] for i in range(len(knn))])
            
            if self.weights == 'uniform':
                count = Counter(targets)
                probabs = [count[0]/(count[0]+count[1]), count[1]/(count[0]+count[1])]
                predictions.append(probabs)
                
            if self.weights == 'distance':
                weights = get_weights(distances)
                count = weighted_counter(targets, weights)
                probabs = [count[0]/(count[0]+count[1]), count[1]/(count[0]+count[1])]
                predictions.append(probabs)
            
        return np.array(predictions)
