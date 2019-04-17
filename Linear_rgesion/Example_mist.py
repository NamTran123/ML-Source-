# Su dung kmean phan nhom chu so  viet tay  

import  numpy as np 
import  matplotlib.pyplot as plt  
from mnist import MNIST # require  pip  install python-mnist 
from  sklearn.cluster  import  KMeans  


from   Kmean_display_network  import  display_network as display  

mndata  =  MNIST('/home/sink-all/Desktop/ML Source')
X = mndata.tesk_images()

kmeans = KMeans(n_clusters=3 ).fit(X)
pred_label = kmeans.predict(X) 
print(type(kmeans.cluster_centers_.T))
print(kmeans.cluster_centers_.T.shape)