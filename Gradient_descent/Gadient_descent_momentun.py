#Gradient  descent  using momentum  

#f(x)   = x^2 + 10 * sin(x)

#grad =  f'(x)  = 2*x  + 10 cos(x)
import  numpy as   np  
import  math 
def  grad (x):
    return  (x**2 + 10*np.sin(x))

def  cost(x):
    return  (2*x  + 10*np.cos(x))

def has_converged(theta_new , grad):
    return np.linalg.norm(grad(theta_new))/len(theta_new) < 1e-3

def GD_momentun(theta_init  , grad   , eta ,  gamma ):
    theta  =  [theta_init]
    v_old  =  np.zeros_like(theta_init)
    for  node  in  range(100):
        v_new  =  gamma *  v_old +  eta *  grad(theta[-1])
        theta_new  = theta[-1] - v_new
        if ( has_converged(theta_new  ,  grad) ==  1):
            break
        theta.append(theta_new)
        v_old  = v_new
    return  theta ,  node 
theta_new  =  [2]
(theta , node ) = GD_momentun(theta_new  ,  grad  ,0.1 ,  0.9)

print (theta ,  node)

