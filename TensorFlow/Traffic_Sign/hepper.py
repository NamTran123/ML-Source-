import numpy as np

'''
Loss function will be built to minimize the difference between the predicted output
a and real output y : 
Loss function using  squared  loss .
'''

def  squared_loss (y , y_pre):
    return np.dot((y-y_pre),(y-y_pre))

'''
Cross Entropy 
'''
def binary_cross_entropy (y , y_pre):
    return  np.sum(-(y*np.log(y_pre))+ ((1-y)*np.log(1-y_pre)))


def tanh_activate_function(s):
    return (np.exp(s) -np.exp(-s)/(np.exp(s)+np.exp(-s)))

def sigmoid (s):
    return (1/(1 + np.exp(-s)))

