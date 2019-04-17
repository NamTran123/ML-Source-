from   sklearn.cluster import  KMeans 
from sklearn  import  datasets 

import  matplotlib.pyplot  as plt
iris = datasets.load_iris()
def kmeans_display(X, label):
    X1 = X[label == 0, :]
    X2 = X[label == 1, :]
    X3 = X[label == 2, :]
    
    plt.plot(X1[:, 0], X1[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X3[:, 0], X3[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()

X = iris.data   
kmeans  =   KMeans(n_clusters=3 ,random_state= 0 ).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)   