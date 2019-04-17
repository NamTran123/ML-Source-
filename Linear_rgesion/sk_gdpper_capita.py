import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

#Load  data 
oecd = pd.read_csv("oecd_bli_2015.csv", thousands=',')

gdd_per   = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',encoding='latin1', na_values="n/a")

#Prepare  the data  
oecd_gdd = [oecd ,gdd_per]
country_stats = pd.concat(oecd_gdd)

X = np.c_[country_stats["2015"]]
y = np.c_[country_stats["Value"]]

country_stats.plot(kind='scatter', x='2015' , y='Value' )
plt.show()
lin_reg_model = sklearn.linear_model.LinearRegression()
lin_reg_model.fit(X, y)
X_new = [[22587]] # Cyprus' GDP per capita
print(lin_reg_model.predict(X_new)) # outputs [[ 5.96242338]]