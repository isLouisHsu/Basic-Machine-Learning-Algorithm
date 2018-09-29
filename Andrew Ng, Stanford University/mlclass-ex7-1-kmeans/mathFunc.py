import numpy as np
from numpy.linalg import norm

def distance(x1, x2, mode='Euclidean'):
    if mode == 'Manhattan':
        return np.sum(np.abs(x1 - x2))
    elif mode == 'Euclidean':
        return np.sqrt(np.sum((x1 - x2)**2))
    elif mode == 'Cosine':
        return x1.dot(x2) / (norm(x1)*norm(x2))