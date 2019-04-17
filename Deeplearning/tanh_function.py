import  numpy  as  np  
import matplotlib.pyplot  as  plt
#Make data  using  arange  

X  =  np.arange(-10,10  , 0.1)

def tanh_function(X):
    return ((np.exp(X) -1)/(np.exp(X)+1))

f  = tanh_function(X)

plt.plot(f)
plt.show()
