# generate data
# list of points 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 1000
X0 = np.random.multivariate_normal(means[0], cov, N).T  
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

def  kiem_tra_dau(w,x):
    return np.sign(np.dot(w.T,x))

def  kiem_tra_hoitu(w, X ,y):
    return np.array_equal(kiem_tra_dau(w,X),y)

def  perceptron (X, y ,w_init):
    D =  X.shape[1]
    d = X.shape[0]
    w =[w_init]
    mis_point =[]

    while True:
        mix_id = np.random.permutation(D)
        for  i in range(D):
            xi  = X[:,mix_id[i]].reshape(d,1)
            yi  = y[0,mix_id[i]]
            if kiem_tra_dau(w[-1],xi)[0] != yi:
                mis_point.append(mix_id[i])
                w_new = w[-1] + yi*xi
                w.append(w_new)
        if kiem_tra_hoitu(w[-1],X ,y):
            break ; 
    return (w , mis_point)

d  = X.shape[0]

w_init  = np.random.rand(d,1)

(w,p)= perceptron(X,y,w_init)

print(w[-1])
def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2], 'k')
    else:
        x10 = -w0/w1
        return plt.plot([x10, x10], [-100, 100], 'k')

draw_line(w[-1])