def assignment05():

	''' Assignment 05 for Introduction to Machine Learning, Sp 2018
		Principal Components Analysis (PCA)
	'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d


#Load Data G1
DS1 = np.loadtxt('G1.txt') # 10-dimensional data set (N=1500)
G1 = DS1[:,0:10]
G1_labels = DS1[:,10]

# Load Data G2
DS2 = np.loadtxt('G2.txt') # 3-dimensional data set (N=1500)
G2 = DS2[:,0:3]
G2_labels = DS2[:,3]

#Load Data G3
DS3 = np.loadtxt('G3.txt') # 3-dimensional data set (N=1500)
G3 = DS3[:,0:3]
G3_labels = DS3[:,3]

# Load Data Halfmoons
DS4 = np.loadtxt('halfmoons.txt') # 3-dimensional data set (N=1500)
halfmoons = DS4[:,0:3]
halfmoons_labels = DS4[:,3]

# Load Data Swissroll
swissroll = np.loadtxt('swissroll.txt') # 3-dimensional data set (N=1500)

# Load Data Seeds
DS6 = np.loadtxt('seeds.txt') # 7-dimensional data set (N=210)
seeds = DS6[:,0:7]
seeds_labels = DS6[:,7]



def pca(data):
# data = data set contained in an N x D matrix (N data points, D features)
# Count number of features in data set.
    N = np.size(data, 0) # number of data points
    D = np.size(data, 1)
    data_mean = np.mean(data, axis=0) # axis=0 averages by ROWS; axis=1 by columns
    deviation = data - data_mean
    covar = 1/np.size(data, 0)*deviation.T@deviation  # np.size gives us number of rows (N)!

# Create matrix of eigenvalues and matrix of eigenvectors.
    eigenvals, eigenvecs = np.linalg.eig(covar)

# Sort eigenvalues, highest to lowest.
    sorted_eigvals = np.argsort(-abs(eigenvals))
# Sort the eigenvectors, using the sorted eigenvalues, then update eigenvector matrix.
    eigenvecs = eigenvecs.T[sorted_eigvals]
    # Take transpose because linalg.eig puts eigenvectors in COLUMNS.

# Make an eigenvector matrix, keeping only the d highest dimensions
# Use the sorted eigenvalues to grab only the d highest dimensions we want to keep
# d = number of dimensions onto which we want to project our data

    d = 2
    largest_eigenvecs = eigenvecs[0:d]

# Project data onto our largest eigenvectors
    projected_data = largest_eigenvecs@data.T
# Transpose in order to make contraction work.
    return projected_data, data 



########## Plot G1 data ##########
projected_data, dataset = pca(G1)

D = np.size(dataset, 1)
d = 2

fig = plt.figure(1)
fig.tight_layout()
ax = fig.add_subplot(131, projection='3d')
ax.plot3D(G1[G1_labels==1,0],G1[G1_labels==1,1], G1[G1_labels==1,2], '.r')
ax.plot3D(G1[G1_labels==2,0],G1[G1_labels==2,1], G1[G1_labels==2,2], '.g')
ax.plot3D(G1[G1_labels==3,0],G1[G1_labels==3,1], G1[G1_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('G1 Data Set')

sp1 = plt.subplot(132)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_ylabel('Second Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimensions' % (D, d))
plt.plot(projected_data[0,G1_labels==1], projected_data[1,G1_labels==1], '.r')
plt.plot(projected_data[0,G1_labels==2], projected_data[1,G1_labels==2], '.g')
plt.plot(projected_data[0,G1_labels==3], projected_data[1,G1_labels==3], '.b')

sp1 = plt.subplot(133)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimension' % (D, d-1))
# Array of zeros
zeros = np.zeros_like(projected_data[0,G1_labels==1])
plt.plot(projected_data[0,G1_labels==1], zeros, '.r')
plt.plot(projected_data[0,G1_labels==2], zeros, '.g')
plt.plot(projected_data[0,G1_labels==3], zeros, '.b')

description = (""" 
               Principal components analysis did effectively retain 
               the distinct classes that the G1 data had originally.
               It is easy to see, even by eye, that the data were 
               and still are in three distinct classes. PCA simply
               reduced the dimensionality of the data.
               """)
plt.figtext(0.05, 0.01, description, weight=400, wrap=True)

plt.show()
    


########## Plot G2 data ##########
projected_data, dataset = pca(G2)

D = np.size(dataset, 1)
d = 2

fig = plt.figure(1)
fig.tight_layout()
ax = fig.add_subplot(131, projection='3d')
ax.plot3D(G2[G2_labels==1,0],G2[G2_labels==1,1], G2[G2_labels==1,2], '.r')
ax.plot3D(G2[G2_labels==2,0],G2[G2_labels==2,1], G2[G2_labels==2,2], '.g')
ax.plot3D(G2[G2_labels==3,0],G2[G2_labels==3,1], G2[G2_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('G2 Data Set')

sp1 = plt.subplot(132)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_ylabel('Second Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimensions' % (D, d))
plt.plot(projected_data[0,G2_labels==1], projected_data[1,G2_labels==1], '.r')
plt.plot(projected_data[0,G2_labels==2], projected_data[1,G2_labels==2], '.g')
plt.plot(projected_data[0,G2_labels==3], projected_data[1,G2_labels==3], '.b')

sp1 = plt.subplot(133)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimension' % (D, d-1))
# Array of zeros
zeros = np.zeros_like(projected_data[0,G2_labels==1])
plt.plot(projected_data[0,G2_labels==1], zeros, '.r')
plt.plot(projected_data[0,G2_labels==2], zeros, '.g')
plt.plot(projected_data[0,G2_labels==3], zeros, '.b')

plt.show()



########## Plot G3 data ##########
projected_data, data = pca(G3)

D = np.size(data, 1)
d = 2

fig = plt.figure(1)
fig.tight_layout()
ax = fig.add_subplot(131, projection='3d')
ax.plot3D(G3[G3_labels==1,0],G3[G3_labels==1,1], G3[G3_labels==1,2], '.r')
ax.plot3D(G3[G3_labels==2,0],G3[G3_labels==2,1], G3[G3_labels==2,2], '.g')
ax.plot3D(G3[G3_labels==3,0],G3[G3_labels==3,1], G3[G3_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('G3 Data Set')

sp1 = plt.subplot(132)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_ylabel('Second Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimensions' % (D, d))
plt.plot(projected_data[0,G3_labels==1], projected_data[1,G3_labels==1], '.r')
plt.plot(projected_data[0,G3_labels==2], projected_data[1,G3_labels==2], '.g')
plt.plot(projected_data[0,G3_labels==3], projected_data[1,G3_labels==3], '.b')

sp1 = plt.subplot(133)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimension' % (D, d-1))
# Array of zeros
zeros = np.zeros_like(projected_data[0,G3_labels==1])
plt.plot(projected_data[0,G3_labels==1], zeros, '.r')
plt.plot(projected_data[0,G3_labels==2], zeros, '.g')
plt.plot(projected_data[0,G3_labels==3], zeros, '.b')

plt.show()



########## Plot Halfmoons data ##########
projected_data, dataset = pca(halfmoons)

D = np.size(dataset, 1)
d = 2

fig = plt.figure(1)
fig.tight_layout()
ax = fig.add_subplot(131, projection='3d')
ax.plot3D(halfmoons[halfmoons_labels==1,0],halfmoons[halfmoons_labels==1,1], halfmoons[halfmoons_labels==1,2], '.r')
ax.plot3D(halfmoons[halfmoons_labels==2,0],halfmoons[halfmoons_labels==2,1], halfmoons[halfmoons_labels==2,2], '.g')
ax.plot3D(halfmoons[halfmoons_labels==3,0],halfmoons[halfmoons_labels==3,1], halfmoons[halfmoons_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Halfmoons Data Set')

sp1 = plt.subplot(132)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_ylabel('Second Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimensions' % (D, d))
plt.plot(projected_data[0,halfmoons_labels==1], projected_data[1,halfmoons_labels==1], '.r')
plt.plot(projected_data[0,halfmoons_labels==2], projected_data[1,halfmoons_labels==2], '.g')
plt.plot(projected_data[0,halfmoons_labels==3], projected_data[1,halfmoons_labels==3], '.b')

sp1 = plt.subplot(133)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimension' % (D, d-1))
# Array of zeros
zeros = np.zeros_like(projected_data[0,halfmoons_labels==1])
plt.plot(projected_data[0,halfmoons_labels==1], zeros, '.r')
plt.plot(projected_data[0,halfmoons_labels==2], zeros, '.g')

plt.show()



########## Plot Swissroll data ##########
projected_data, dataset = pca(swissroll)

D = np.size(dataset, 1)
d = 2

fig = plt.figure(1)
fig.tight_layout()

ax = fig.add_subplot(131, projection='3d')
ax.plot3D(swissroll[:,0],swissroll[:,1],swissroll[:,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Swissroll Data Set')

sp1 = plt.subplot(132)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_ylabel('Second Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimensions' % (D, d))
plt.plot(projected_data[0,halfmoons_labels==1], projected_data[1,halfmoons_labels==1], '.b')
#plt.plot(projected_data[0,halfmoons_labels==2], projected_data[1,halfmoons_labels==2], '.g')
#plt.plot(projected_data[0,halfmoons_labels==3], projected_data[1,halfmoons_labels==3], '.b')

sp1 = plt.subplot(133)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimension' % (D, d-1))
# Array of zeros
zeros = np.zeros_like(projected_data[0,halfmoons_labels==1])
plt.plot(projected_data[0,halfmoons_labels==1], zeros, '.b')
#plt.plot(projected_data[0,halfmoons_labels==2], zeros, '.g')

plt.show()


########## Plot Seeds data ##########
projected_data, data = pca(seeds)

D = np.size(data, 1)
d = 2

fig = plt.figure(1)
fig.tight_layout()
ax = fig.add_subplot(131, projection='3d')
ax.plot3D(seeds[seeds_labels==1,3],seeds[seeds_labels==1,5], seeds[seeds_labels==1,2], '.r')
ax.plot3D(seeds[seeds_labels==2,3],seeds[seeds_labels==2,5], seeds[seeds_labels==2,2], '.g')
ax.plot3D(seeds[seeds_labels==3,3],seeds[seeds_labels==3,5], seeds[seeds_labels==3,2], '.b')
ax.set_xlabel('Dimension 4')
ax.set_ylabel('Dimension 6')
ax.set_zlabel('Dimension 3')
ax.set_title('Seeds Data Set')

sp1 = plt.subplot(132)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_ylabel('Second Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimensions' % (D, d))
plt.plot(projected_data[0,seeds_labels==1], projected_data[1,seeds_labels==1], '.r')
plt.plot(projected_data[0,seeds_labels==2], projected_data[1,seeds_labels==2], '.g')
plt.plot(projected_data[0,seeds_labels==3], projected_data[1,seeds_labels==3], '.b')

sp1 = plt.subplot(133)
sp1.set_xlabel('Largest Eigenvector (variance)')
sp1.set_title('Reducing %s dimensions to %s dimension' % (D, d-1))
# Array of zeros
zeros = np.zeros_like(projected_data[0,seeds_labels==1])
plt.plot(projected_data[0,seeds_labels==1], zeros, '.r')
plt.plot(projected_data[0,seeds_labels==2], zeros, '.g')
plt.plot(projected_data[0,seeds_labels==3], zeros, '.b')

plt.show()

print()

if __name__ == '__main__':
	assignment05()
