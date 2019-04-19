'''
Perceptron  for classification  .

'''

import numpy as  np 
import matplotlib.pyplot  as  plt 


class  Perceptron :
    
    def __init__(self,eta,n_inter):
        self.eta  = eta
        self.n_inter  =  n_inter

    def net_input (self , X):
        return  np.dot(X, self.weight[1:])+self.weight[0]
    
    def predict  (self, X):
        return np.sign(self.net_input(X))

    def fit(self  , X ,y):
        self.weight  =  np.zeros(1  + X.shape[1])
        self.errors =  []
        for _ in range(self.n_inter):
            self.errors  =  0  
            for Xi , tanger  in zip(X,y):
                update  =  self.eta * (self.tanger - self.predict(Xi))
                self.weight[1:] = update*Xi
                self.weight[0] += update
                err += int(update != 0.0)
            self.errors.append(err)
        return self
    
    



        #Fit training data  


        