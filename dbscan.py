
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler



# #############################################################################
# Generate sample data

x=np.random.uniform(low=50,high=100,size=100)
x1=np.random.uniform(size=100)
y=np.random.uniform(low=50,high=100,size=100)
y1=np.random.uniform(size=100)

l=zip(x,y)
l1=zip(x1,y1)
l.extend(l1)
l=np.array(l)
#X = StandardScaler().fit_transform(l)
# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=20, min_samples=5).fit(l)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    x=l[class_member_mask]

    if x.size>0 and k!=-1:
    	print "pendiente ajuste cluster", k ,np.poly1d(np.polyfit(x[:,0], x[:,1], 1))[1]
    	plt.plot(np.unique(x[:,0]), np.poly1d(np.polyfit(x[:,0], x[:,1], 1))(np.unique(x[:,0])))

    xy = l[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=18)

    xy = l[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=2)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
