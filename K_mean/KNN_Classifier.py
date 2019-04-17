import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# Assigning features and label variables
# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Label or target varible
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#Creat  Label Encode 

LabelEncode = preprocessing.LabelEncoder()

#Converting String  Labels  into Number 

weather_encode  =  LabelEncode.fit_transform(weather)
print(weather_encode)
# 0:Overcast,1:Rainy' ,2:Sunny

temp_encode = LabelEncode.fit_transform(temp)
print(temp_encode)

laber = LabelEncode.fit_transform(play)
print(laber)
# 0:Hot,1:Cool ,2:Mild

#Combining Features

features  =  list(zip(weather_encode ,temp_encode))

#Generating  Model  
model  =  KNeighborsClassifier(n_neighbors=3)

model.fit(features , laber)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print(predicted)