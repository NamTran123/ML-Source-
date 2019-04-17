
import  numpy as np  
import  math  

def getvalue_daoham(x):
    return (2*np.sin(x) +  np.sin(2*x))

def getvalue_y(x):
    return  (3-2*np.cos(x) - np.cos(2*x))

def get_x_min(eta,x_start ):
    x  = [x_start]
    for i  in range(2000):
        if (getvalue_daoham(x[-1])< 1e-9):
            break
        x_new = x[-1] - eta*getvalue_daoham(x[-1])
        x.append(x_new)
    return  (x , i)

[x,  i] = get_x_min(0.1 ,  2)
print(x[-1]) # 2.3714... so voi kq  2.355
print(i)
print(getvalue_daoham(x[-1]))