def assignment06():

	''' Assignment 65 for Introduction to Machine Learning, Sp 2018
		Classic Multidimensional Scaling (MDS), ISOMAP, Locally Liner Embedding (LLE)
	''' 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import eig
from numpy.matlib import repmat
from mpl_toolkits import mplot3d
from time import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

# Load Swissroll
swissroll = np.loadtxt('swissroll.txt') # 3-dimensional data set (N=500)

# Load Clusters 
clusters = np.loadtxt('clusters.txt') # 10-dimensional data set (N=600)
clusters_labels = clusters[:,10]
clusters = clusters[:,0:10]

#Load Halfmoons
halfmoons = np.loadtxt('halfmoons.txt') 
halfmoons_labels = halfmoons[:,3] 
halfmoons = halfmoons[:,0:3]

fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(131, projection='3d')
ax.plot3D(swissroll[:,0],swissroll[:,1], swissroll[:,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Swissroll Data Set')

ax = fig.add_subplot(132, projection='3d')
ax.plot3D(clusters[clusters_labels==1,0],clusters[clusters_labels==1,1], clusters[clusters_labels==1,2], '.r')
ax.plot3D(clusters[clusters_labels==2,0],clusters[clusters_labels==2,1], clusters[clusters_labels==2,2], '.g')
ax.plot3D(clusters[clusters_labels==3,0],clusters[clusters_labels==3,1], clusters[clusters_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Clusters Data Set')

ax = fig.add_subplot(133, projection='3d')
ax.plot3D(halfmoons[halfmoons_labels==1,0],halfmoons[halfmoons_labels==1,1], halfmoons[halfmoons_labels==1,2], '.r')
ax.plot3D(halfmoons[halfmoons_labels==2,0],halfmoons[halfmoons_labels==2,1], halfmoons[halfmoons_labels==2,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Halfmoons Data Set')

plt.show()

# Implement Classic MDS, ISOMAP and LLE
    


##### Locally Linear Embedding (LLE) #####

#def LLE(dataset, k):
    
### Get matrix of neighbors.
# N = Number of data points
dataset = swissroll
k = 20
N = np.size(dataset, 0)
print('Swissroll dataset has', N, 'data points.')
# We want k nearest neighbors. Use k+1 because NearestNeighbors counts 
# a datum as its own neighbor.
nbrs = NearestNeighbors(k+1, metric='euclidean').fit(dataset)
# indices tells the location of each datum's nearest k neighbors.
distances, indices = nbrs.kneighbors(dataset)

# Initialize W matrix.
W = np.zeros( (N,N) )

for i in range(N):

    # Make Vi, a matrix of the k neighbors of datum xi.
    # Each neighbor is put into a row.
    Vi = dataset[indices[i,1::], :] # The k indexed neighbors of datum xi are grabbed
    # Vi is now a k x D matrix
    # We want each neighbor to be in its own column, so transpose V.
    Vi = Vi.T # Vi is D x k
    
    ### Find the Gram matrix
    # Make a matrix of the ith data point in k columns.
    duplidata = np.matlib.repmat(dataset[i], k, 1).T # duplidata is D x k
    # G = Gram matrix
    G = (duplidata - Vi).T @ (duplidata - Vi) # G is k x k
    
    ### Find the weights associated with data point xi:
    wi = np.linalg.inv(G)@np.ones( (k,1) ) # wi is k x 1
    sum_wi = np.sum(wi)
    # Normalize wi:
    wi = wi/(sum_wi)
    # Remove a column to make broadcasting work.
    wi = np.squeeze(wi[:, np.newaxis])

### Build W matrix
    W[indices[i,1::],i] = wi
        
### Find M matrix.
M = ( np.identity(N) - W )@( np.identity(N) - W ).T
eigvals_M, eigvecs_M = np.linalg.eig(M)
# We only care about the magnitude of the eigenvalues, so use abs.
eigval_indices = np.argsort( (eigvals_M) ) # Sort from lowest to highest, using indices
# Grab the 2 eigenvectors corresponding to the smallest 2 eigenvalues
eigvecs_M_sorted = eigvecs_M[:, eigval_indices[1:3]]

fig = plt.figure()
axes = fig.add_subplot(111)

x = eigvecs_M_sorted[:, 0]
y = eigvecs_M_sorted[:, 1]

axes.plot(x, y, '.b')
axes.set_title('LLE Performed on Swissroll Data Set')
axes.set_xlabel('Eigenvector associated with\n2nd Smallest Eigenvalue')
axes.set_ylabel('Eigenvector associated with\n3rd Smallest Eigenvalue')

plt.show()

###!!! For some reason, some of M's eigenvalues are negative. Also don't have
###!!! eigenvalue of 0, or eigenvector [1,1,1,1,...,1]

#LLE(swissroll, 7)
    
##### MDS #####
    
    
##### ISOMAP #####

if __name__ == '__main__':
	assignment06()
