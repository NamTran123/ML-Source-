

import  math  
import numpy  as np  
import  matplotlib.pyplot  as  plt  

def  get_daoham (x):
    return  (2*x +  5*np.cos(x))

def  get_hamso(x):
    return  (x**2 + 5*np.sin(x))

def get_x_min(eta , x0):
    x=  [x0]
    for i in  range(100):
        if (get_daoham(x[-1])< 1e-3):
            break
        x_new  = x[-1]-  eta * get_daoham(x[-1])
        x.append(x_new)
    return  (x, i)

(x1 , stt) = get_x_min(0.1 , 5)
print  ("X  min  x1 = %f obtained after %d iterations  , and  cost  :  %f" , x1[-1] ,stt , get_hamso(x1[-1]) ) 