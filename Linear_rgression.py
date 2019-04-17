import numpy as np  
import matplotlib.pyplot as plt 

# height (cm)
x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

#Visuaize data  
def display(x, y):
    plt.plot(y, x  , 'ro')
    plt.axis([ 45,75 ,140,190 ])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (cm)')
    plt.show() 
display(x, y)

#Building  Xbar  
one =  np.ones((x.shape[0] , 1))
Xbar  =  np.concatenate((one  ,  x)  ,  axis  = 1  )

#Caculating  weights   of the  fiting  line  
A  =  np.dot(Xbar.T , Xbar)
b  =  np.dot(Xbar.T ,  y)
W  =  np.dot(np.linalg.pinv(A) ,b  )    
print (W)

def guess ( xd,w):
    y0  =  w[1][0]*xd + w[0][0]
    return  y0 
#Predict weight of person with height 157 cm :
print (guess(157 , W))

def print_fitting_line(w  , x , y ):
    w0  = w[0][0]
    w1  = w[1][0]
    x0  =  np.linspace(145,185 , 2)
    y0  =  w1*x0 + w0
    plt.plot(x.T , y.T,'ro')
    plt.plot(x0 ,y0) 
    plt.axis([140, 190, 45, 75])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()

print_fitting_line(W ,x,y)
