
import  numpy as np  
import  math  

def giatri_daoham(x):
    return (2*np.sin(x) +  np.sin(2*x))

def giatri_y(x):
    return  (3-2*np.cos(x) - np.cos(2*x))

def (eta,x_start ):
    x  = [x_start]
    for i  in range(200):
        if (giatri_daoham(x[-1]) < 1e-9):
            break
        x_new = x[-1] - eta*giatri_daoham(x[-1])
        x.append(x_new)
    return  (x , i)

[x,  i] = lay_x_nhonhat(0.1 ,  2)
print(x[-1]) # 2.3714... so voi kq  2.355
print(i)
