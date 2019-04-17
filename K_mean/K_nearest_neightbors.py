import  numpy  as  np 
from sklearn   import  neighbors  , datasets  
import  matplotlib.pyplot  as plt  


#Loading  iris dataset 
data_iris = datasets.load_iris()

data_X  =  data_iris.data
data_Y  =  data_iris.target

print ("Number  class  in  dataset  :",len(np.unique(data_Y)))
print  ("Number  point  data  in  dataset :",len(data_X))


#Sample for  class  in dataset   
X0  = data_X[data_Y ==0,:]
print ("Sample  for class 0" , X0[:5,:])
X1 = data_X[data_Y == 1, :]
print ("Sample data for  class 1 ", X1[:5 , :])
X2  = data_X[data_Y==2,:]
print ("Sample data for class 2", X2[:5,:])

#Detach data  for train  and test   using  model_selection  
from  sklearn.model_selection import train_test_split

X_train  , X_test , Y_train  , Y_test  =  train_test_split(data_X , data_Y  , test_size =  50)

data_sample_train  =  X_train[:5,:]
print(data_sample_train)

#Su dung k_neaightnors  trong  sklearn de gan label cho  cac diem trong (Test Set hay  X_test )
#Voi  n_neightbors  = 10 va  p =2  
value  =  neighbors.KNeighborsClassifier(n_neighbors= 10 , p=2).fit(X_train ,Y_train)
Y_predict  = value.predict(X_test)

#Ket qua thu duoc 
print "Print results for 20 test data points:"
print "Predicted labels: ", Y_predict[20:40]
print "Ground truth    : ", Y_test[20:40]

#Danh gia  (Evaluation  Method)
from  sklearn.metrics import accuracy_score 

print ("Accuracy of 10 NN  : " , 100*accuracy_score(Y_test , Y_predict))