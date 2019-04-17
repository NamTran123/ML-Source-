import numpy as np
import numpy.random as rand
import sklearn.metrics

# Khoi tao data base

example = 1000
features = 100

D = (rand.randn(example, features), rand.randn(features))
print(D)

# Specify the network
#
n_layer1_unit = 8
n_layer2_unit = 0

    # layer 1
W1 = rand.rand(features, n_layer1_unit)
b1 = rand.rand(features)
    # layer 2
W2 = rand.rand(n_layer1_unit, n_layer2_unit)
b2 = 0
    #Set of parameters of the model.
theta = (W1,b1  ,W2 ,b2)

