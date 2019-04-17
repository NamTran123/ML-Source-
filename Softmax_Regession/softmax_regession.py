import  numpy as  np
import  matplotlib.pyplot as plt 

np.random.seed()
#Data
x  =  np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
print(x.shape[1])
#Xbar  
X =  np.concatenate(np.ones((1,x.shape[1])) ,x)
print(X.shape[:,1].reshape(X.shape[0] ,1))
def simoig(z):
    return (1/(1+p.exp(-z)))

def update_W ():
    pass

# def logistic_simoig_regession(X , y  ,W_input , leaning_rate =0.001 ,tol  =  0.001 , count_max):
#     W  =  [W_input]
#     N  =  X.shape[1]
#     d  = X.shape[0]
