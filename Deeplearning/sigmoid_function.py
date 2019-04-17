'''
Sigmoid  Unit :

f(x) =  1/(1+exp(X))
'''
import  numpy  as  np  
import matplotlib.pyplot  as  plt
#Make data  using  arange  

X  =  np.arange(-10,10  , 0.1)

def  sigmoid_function(X):
    return (1/(1 + np.exp(X)))

f  =  sigmoid_function(X)

plt.plot(f ,'r')
plt.show()

