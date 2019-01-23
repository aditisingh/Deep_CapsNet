import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

X=np.load('vector_train.npy')
F=np.load('files_train.npy')

Y=[f[44] for f in F]
X1=X.reshape(-1,16*2)

X_embedded = TSNE(n_components=2).fit_transform(X1)
kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(X_embedded)

plt.scatter(X_embedded[:,0],X_embedded[:,1],c=Y)
for i, y in enumerate(Y):
     plt.annotate(kmeans[i],(X_embedded[i,0],X_embedded[i,1]))



plt.show()



