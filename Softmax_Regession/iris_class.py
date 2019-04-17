import numpy as np 
from sklearn import datasets  as dt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model  import LogisticRegression  
from scipy import sparse #for one hot code 
import matplotlib.pyplot  as  plt  

iris  = dt.load_iris()

iris_X = iris.data
iris_y = iris.target 

x0  =  iris_X[iris_y==0 ,:]
print(x0[:5,])

x_train ,x_test,y_train,y_test = train_test_split(iris_X ,iris_y , test_size = 50)

print(x_train[:5,])

print(y_train[:5,])

classifier = LogisticRegression (multi_class='ovr')
classifier.fit(x_train  ,y_train)


y_pred = classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score

print ("Accuracy :" , (100*accuracy_score(y_test , y_pred)))
plt.plot(x_test[y_pred == 0 ,:] , 'r')
plt.plot(x_test[y_pred == 1 ,:] , 'g')
plt.plot(x_test[y_pred == 2 ,:] , 'b')
plt.show()
