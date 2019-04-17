import  numpy as  np  
import matplotlib.pyplot  as  plt  

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

def  activate_function(w, x):
    return np.sign(np.dot(w.T ,x))

def has_converged (A , w , Y):
    return np.array_equal(activate_function(w , A), Y)

def perceptron_learning (w_init  , A , Y):
    w  = [w_init]
    while True:
        for item  in A:
            for yi in Y:
                if activate_function( w[-1] , item.reshape(d, 1)) != yi :
                    w_new  = w[-1] + yi*w[-1]
                    w.append(w_new)
        if has_converged(A ,w[-1] ,Y) :
            break
    return W

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w) = perceptron_learning( w_init ,X, y)
                
