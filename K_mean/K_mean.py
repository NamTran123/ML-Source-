
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import  matplotlib.pyplot  as plt
from scipy.spatial.distance import cdist
import numpy as np  

np.random.seed(11)

means =  [[2 ,3 ] , [5,7] , [8,9]]
cov = [[1,0],[0,1]]
N =500  
X1 = np.random.multivariate_normal(means[0] , cov  , N)
X2 = np.random.multivariate_normal(means[1] , cov  , N)
X3 = np.random.multivariate_normal(means[2] , cov  , N)

X  = np.concatenate((X1,X2,X3) ,axis = 0)
K=3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

def kmeans_display(X, label):
    K = np.amax(label) + 1
    X1 = X[label == 0, :]
    X2 = X[label == 1, :]
    X3 = X[label == 2, :]
    
    plt.plot(X1[:, 0], X1[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X3[:, 0], X3[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
"""
 Khoi tao cac center  ban dau
"""
def kmeans_init_centers(X , k):
    return  X[np.random.choice(X.shape[0] , k ,  replace = False)]


"""
Gan cac diem gia tr vao cac center
"""
def kmeant_asign_labels(X , center):
    D =  cdist(X , center)
    a =np.argmin(D , axis=1)
    return a  

"""
Cap nhat cac center moi cho cac label 
"""

def kmeans_update_centers(X , labels  ,  K):
    centers = np.zeros((K, X.shape[1]))
    for k  in range(K):
         # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers 

"""
Kiem tra dieu kien dung cua bai toan
"""
def  has_converged(centers , new_center):
    return (set([tuple(a) for a   in  centers]) == set([tuple(b) for b  in new_center])) 
    

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeant_asign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])
# print(it)
# a=0
# for x  in labels[-1]:
#      print(x)
#      a=a+1
# print (a)
kmeans_display(X, labels[-1])