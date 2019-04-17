
"""
# Image Compression (Nen anh va du lieu )
# Van de : Mot pixel anh co the nhan  1 trong 255^3 mau   -> 24 bit tren mot diem anh 
# - > Ta muon luu diem anh voi mot so bit nho hon va chiu mat di du lieu o mot muc nao do  .

# Idea :   Chung ta se phan 255^3 mau tren thanh :2, 5 or 10 or 20  cluster  sau do  
# dung cac center cua moi cluster thay the cho cac diem anh do 
#   
"""
import matplotlib.image as imgP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img  =  imgP.imread('/home/sink-all/Desktop/ML Source/lieu.jpg')
plt.imshow(img)

# Plot image import  
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()
print (img)
#chuyen  thanh ma tran  co img.shape[0]*img.shape[1] hang  , moi hang la 1 pixel 
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))  

for K in [2]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img4 = np.zeros_like(X)
    # Thay the cac pixel bang cac center cua no
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2])) 
    print (img5.dim)
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()