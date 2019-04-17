import numpy as np
import matplotlib.pyplot as plt

np.random.seed()

# Data
X = np.random.rand(1000, 1)
y = 4+3*X + 0.2*np.random.rand(1000, 1)

# Add noise
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)


# def grad(w):
#     N = X_bar.shape[0]
#     return 1/N * X_bar.T.dot(X_bar.dot(w) - Y)


# def cost(w):
#     N = X_bar.shape[0]
#     return 0.5*N*np.linalg.norm(Y-X_bar.dot(w), 2)**2


# def gradient_descent(W_init, grad , eta):
#     W = [W_init]
#     quota = 1e-3
#     for i in range(100):
#         W_new = W[-1] - eta * grad(W[-1])
#         if np.linalg.norm(grad(W_new))/len(W_new) < quota:
#             break
#         W = W.append(W_new)
#     return (W, i)
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2

def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it) 

w_init = np.array([[2], [1]])
eta = 0.1
(w1, it1) = myGD(w_init, grad, 1)
print(w1[-1])
