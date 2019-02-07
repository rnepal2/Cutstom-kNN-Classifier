import math
import numpy

# Euclidean distance between two vectors a and b
def euclidean_distance(a, b):
    
    assert len(a) == len(b)
    
    distance = 0.0
    for i in range(len(a)):
        distance += (a[i] - b[i])**2.0
    return distance**(1./2)

# Manhattan distance between two vectors a and b
def manhattan_distance(a, b):
    
    assert len(a) == len(b)
    
    distance = 0.0
    for i in range(len(a)):
        distance += abs(a[i] - b[i])
    return distance

# Minkowski
def minkowski(a, b, p):
    
    assert len(a) == len(b)
    
    distance = 0.0
    for i in range(len(a)):
        distance += abs(a[i] - b[i])**p
    return (distance)**(1./p)