import numpy as np
import sklearn.metrics
import hepper as hp
from autograd import grad
import numpy.random as rand
# Khoi tao data base

example = 1000
features = 100

D = (rand.randn(example, features), rand.randn(example))
print(D[1][1], D[0][1])

# Specify the network
#
n_layer1_unit = 10
n_layer2_unit = 1

# layer 1
W1 = rand.rand(features, n_layer1_unit)
b1 = rand.rand(n_layer1_unit)
# layer 2
W2 = rand.rand(n_layer1_unit, n_layer2_unit)
b2 = 0.0
# Set of parameters of the model.
theta = (W1, b1, W2, b2)

# Loss function


'''
Cross Entropy 
'''


def binary_cross_entropy(y, y_pre):
    return np.sum(-(y*np.log(y_pre)) + ((1-y)*np.log(1-y_pre)))

# Wraper  around  Neural Network


def neural_network(x, theta):
    W1, b1, W2, b2 = theta
    return np.tanh(np.dot((np.tanh(np.dot(x, W1) + b1)), W2) + b2)

# return  (yi-tanh(np.dot((np.tanh(np.dot(W1, x)+b1), W2)+b2))


def objective(theta, id):
    return hp.squared_loss(D[1][id], neural_network(D[0][id], theta))

# Update


def update_theta(theta, delta, alpha):
    w1, b1, w2, b2 = theta
    w1_delta, b1_delta, w2_delta, b2_delta = delta
    w1_new = w1 - alpha * w1_delta
    b1_new = b1 - alpha * b1_delta
    w2_new = w2 - alpha * w2_delta
    b2_new = b2 - alpha * b2_delta
    new_theta = (w1_new, b1_new, w2_new, b2_new)
    return new_theta


# Compute Gradient
grad_objective = grad(objective)
# Train the Neural Network
epochs = 10
for i in range(0, epochs):
    for j in range(0, example):
        delta = grad_objective(theta, j)
        theta = update_theta(theta, delta, 0.01)
rmse.append(sklearn.metrics.mean_squared_error(D[1], neural_network(D[0], theta)))
print("RMSE after training:", sklearn.metrics.mean_squared_error(D[1], neural_network(D[0],theta)))
print(rmse)
