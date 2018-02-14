#####################################
##### Multi-Dimensional Scaling #####
#####################################
def MDS(dataset, reduced_dimension):

    distances = squareform(pdist(dataset, metric='euclidean'))
    N = distances.shape[1]
    """print(distances)
    for i in np.arange(N):
        print(np.mean(distances[:,i]))
        distances[:,i] = distances[:,i] - np.mean(distances[:,i])

    print(distances)"""

    J = np.identity(N) - 1/N * (np.ones((N,1)) @ np.ones((1,N)))

    B = -1/2 * J @ np.square(distances) @ J

    eigvals, eigvecs = np.linalg.eig(B)
    sorted_index = eigvals.argsort()[::-1]
    eigvals = ((np.diag(eigvals[sorted_index]))[:reduced_dimension, :reduced_dimension])
    eigvecs = (eigvecs[:, sorted_index])[:,:reduced_dimension]
    eigvals_sqrt = eigvals ** (1/2)

    result = eigvecs @ eigvals_sqrt

    return result

#####################################
##### Isometric Feature Mapping #####
#####################################
def ISO(dataset, reduced_dimension, K):

    N = dataset.shape[0]

    neighbors = NearestNeighbors(K + 1, metric= 'euclidean').fit(dataset)

    distance = neighbors.kneighbors_graph(dataset, K + 1).toarray() * squareform(pdist(dataset, metric='euclidean'))

    distance[distance == 0] = 10 ** 50
    np.fill_diagonal(distance, 0)

    for i in range(0, N):
        for m in range(0, N):
            for h in range(0, N):
                distance[m,h] = min(distance[m,h], distance[m, i] + distance[i, h])

    J = np.identity(N) - 1/N * (np.ones((N,1)) @ np.ones((1,N)))

    B = -1/2 * J @ np.square(distance) @ J

    eigvals, eigvecs = np.linalg.eig(B)
    sorted_index = eigvals.argsort()[::-1]
    eigvals = ((np.diag(eigvals[sorted_index]))[:reduced_dimension, :reduced_dimension])
    eigvecs = (eigvecs[:, sorted_index])[:,:reduced_dimension]
    eigvals_sqrt = eigvals ** (1/2)

    result = eigvecs @ eigvals_sqrt

    return result, K

#####################################
##### Locally Linear Embedding ######
#####################################
def LLE(dataset, k):
### Name the appropriate dataset.
    if np.array_equal(dataset, swissroll):
        title = "Swissroll"
    elif np.array_equal(dataset, clusters):
        title = "Clusters"
    else:
        title = "Halfmoons"

    ### Get matrix of neighbors.
    # N = Number of data points
    N = np.size(dataset, 0)
    # We want k nearest neighbors. Use k+1 because NearestNeighbors counts
    # a datum as its own neighbor.
    nbrs = NearestNeighbors(k+1, metric= 'euclidean').fit(dataset)
    # Use this if needed: metric='euclidean'
    # indices tells the location of each datum's nearest k neighbors.
    distances, indices = nbrs.kneighbors(dataset)

    # Initialize W matrix.
    W = np.zeros( (N,N) )

    ### Build the weight column matrix
    for i in range(N):

        ### Make Vi, a matrix of the k neighbors of datum xi.
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
        wi = np.linalg.inv( G + 0.001*np.identity(k) )@np.ones( (k,1) ) # wi is k x 1
        sum_wi = np.sum(wi)
        # Normalize wi:
        wi = wi/sum_wi
        # Remove a column to make broadcasting work.
        wi = np.squeeze(wi[:, np.newaxis])

        ### Build W matrix
        W[indices[i,1::], i] = wi

    ### Find M matrix.
    M = ( np.identity(N) - W )@( np.identity(N) - W ).T
    eigvals_M, eigvecs_M = np.linalg.eig(M)
    eigval_indices = np.argsort( (eigvals_M) ) # Sort from lowest to highest, using indices
    # Grab the 2nd and 3rd smallest eigenvectors (the first one is all 1's)
    result = eigvecs_M.real[:, eigval_indices[1:3]]

    return result, k, N


################
##### Main #####
################
def assignment06():
    ''' Assignment 6 for Introduction to Machine Learning, Sp 2018
    Classic Multidimensional Scaling (MDS), ISOMAP, Locally Linear Embedding (LLE)
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


##########################
##### Function Calls #####
##########################
swissroll2DPCA = MDS(swissroll, 2)
swissroll2DISO,swissrollISOk = ISO(swissroll, 2, 10)
swissrollLLE, swissrollk, swissrollN = LLE(swissroll, 8)

clusters2DPCA = MDS(clusters, 2)
clusters2DISO, clustersISOk = ISO(clusters, 2, 210)
clustersLLE, clustersk, clustersN = LLE(clusters, 200)

halfmoons2DPCA = MDS(halfmoons, 2)
halfmoons2DISO, halfmoonsISOk = ISO(halfmoons, 2, 60)
halfmoonsLLE, halfmoonsk, halfmoonsN = LLE(halfmoons, 8)


#############################
##### Original 3D Plots #####
#############################
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


##############################
##### Plotting Swissroll #####
##############################
fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(swissroll2DPCA[:,0], swissroll2DPCA[:,1], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('MDS on Swissroll\nN = %d data points' % (swissrollN) )
ax = fig.add_subplot(132)
ax.plot(swissroll2DISO[:,0], swissroll2DISO[:,1], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('ISOMAP on Swissroll\nk = % Neighbors' % (swissrollISOk))
ax = fig.add_subplot(133)
ax.plot(swissrollLLE[:,0], swissrollLLE[:,1], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('LLE on Swissroll\nk = %d Neighbors' % (swissrollk) )

plt.show()


#############################
##### Plotting Clusters #####
#############################
fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(clusters2DPCA[clusters_labels==1,0], clusters2DPCA[clusters_labels==1,1], '.r')
ax.plot(clusters2DPCA[clusters_labels==2,0], clusters2DPCA[clusters_labels==2,1], '.g')
ax.plot(clusters2DPCA[clusters_labels==3,0], clusters2DPCA[clusters_labels==3,1], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('MDS on Clusters\nN = %d data points' % (clustersN) )
ax = fig.add_subplot(132)
ax.plot(clusters2DISO[clusters_labels==1,0], clusters2DISO[clusters_labels==1,1], '.r')
ax.plot(clusters2DISO[clusters_labels==2,0], clusters2DISO[clusters_labels==2,1], '.g')
ax.plot(clusters2DISO[clusters_labels==3,0], clusters2DISO[clusters_labels==3,1], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('ISOMAP on Clusters\nk = % Neighbors' % (clustersISOk) )
ax = fig.add_subplot(133)
ax.plot(clustersLLE[clusters_labels==1,0], clustersLLE[clusters_labels==1,1], '.r')
ax.plot(clustersLLE[clusters_labels==2,0], clustersLLE[clusters_labels==2,1], '.g')
ax.plot(clustersLLE[clusters_labels==3,0], clustersLLE[clusters_labels==3,1], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('LLE on Clusters\nk = %d Neighbors' % (clustersk) )

plt.show()


##############################
##### Plotting Halfmoons #####
##############################
fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(halfmoons2DPCA[halfmoons_labels==1,0], halfmoons2DPCA[halfmoons_labels==1,1], '.r')
ax.plot(halfmoons2DPCA[halfmoons_labels==2,0], halfmoons2DPCA[halfmoons_labels==2,1], '.g')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('MDS on Halfmoons\nN = %d data points' % (halfmoonsN))
ax = fig.add_subplot(132)
ax.plot(halfmoons2DISO[halfmoons_labels==1,0], halfmoons2DISO[halfmoons_labels==1,1], '.r')
ax.plot(halfmoons2DISO[halfmoons_labels==2,0], halfmoons2DISO[halfmoons_labels==2,1], '.g')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('ISOMAP on Halfmoons\nk = % Neighbors' % (halfmoonsISOk))
ax = fig.add_subplot(133)
ax.plot(halfmoonsLLE[halfmoons_labels==1,0], halfmoonsLLE[halfmoons_labels==1,1], '.r')
ax.plot(halfmoonsLLE[halfmoons_labels==2,0], halfmoonsLLE[halfmoons_labels==2,1], '.g')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('LLE on Halfmoons\nk = %d Neighbors' % (halfmoonsk))

plt.show()


if __name__ == '__main__':
	assignment06()
