import numpy as  np  
import  matplotlib.pyplot as  plt  

#Generate  toy data  
x =  np.linspace(-1,1,100)
signal   = 2*x + 2*x**2
noise  =  np.random.normal(0, 0.1,100)

y  = signal + noise
plt.plot(signal , 'b')
plt.plot(noise ,'r')
plt.plot(y ,'g')
plt.ylabel('Y')
plt.xlabel('x')
plt.legend(['Signal' ,'Noise' ,'Y Value'] , loc  = 2)
plt.show()

x_train  = x[0:80] 
y_train  = y[0:80]
print(x_train.transpose())
#Model with  degree    = 1  
#RMSE  = ROOT  MEAN  SQUARE  ERROR

X_train  = X_train = np.column_stack([np.power(x_train,i) for i in range(0,2)])
degree  =2
model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)),X_train.transpose()),y_train)
plt.plot(x,y,'g')
plt.xlabel("x")
plt.ylabel("y")
predicted = np.dot(model, [np.power(x,i) for i in range(0,degree)])
plt.plot(x, predicted,'r')
plt.legend(["Actual", "Predicted"], loc = 2)
train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80], y_train -
predicted[0:80])))
test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:], y[80:] -
predicted[80:])))
print("Train RMSE (Degree = 1)", train_rmse1)
print("Test RMSE (Degree = 1)", test_rmse1)
# Model with degree 2
plt.figure()
degree = 3
X_train = np.column_stack([np.power(x_train,i) for i in range(0,degree)])
model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)),
X_train.transpose()),y_train)
plt.plot(x,y,'g')
plt.xlabel("x")
plt.ylabel("y")
predicted = np.dot(model, [np.power(x,i) for i in range(0,degree)])
plt.plot(x, predicted,'r')
plt.legend(["Actual", "Predicted"], loc = 2)
train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80],
y_train - predicted[0:80])))
test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:],
y[80:] - predicted[80:])))
print("Train RMSE (Degree = 2)", train_rmse1)
print("Test RMSE (Degree = 2)", test_rmse1)
# Model with degree 8
plt.figure()
degree = 9
X_train = np.column_stack([np.power(x_train,i) for i in range(0,degree)])
model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)),
X_train.transpose()), y_train)
plt.plot(x, y,'g')
plt.xlabel("x")
plt.ylabel("y")
predicted = np.dot(model, [np.power(x,i) for i in range(0,degree)])
plt.plot(x, predicted,'r')
plt.legend(["Actual", "Predicted"], loc = 3)
train_rmse2 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80],
y_train - predicted[0:80])))
test_rmse2 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:],
y[80:] - predicted[80:])))
print("Train RMSE (Degree = 8)", train_rmse2)
print("Test RMSE (Degree = 8)", test_rmse2)
