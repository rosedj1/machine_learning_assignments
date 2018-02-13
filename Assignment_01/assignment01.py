def assignment01():

	''' Assignment 01 for Introduction to Machine Learning, Sp 2018
		Introduction to Python and Git assignment
	''' 
import numpy as np
import matplotlib.pyplot as plt

data = np.load('dataX.npy')
print('The original data set is:')
print(data)

##### Compute the mean 2-D vector of the data set (mean of the 1000 data points) #####

# slice 1st column from data set
data_1 = data[:, 0]
# slice 2nd column
data_2 = data[:, 1]

# take average of all values in each column
mean_1 = np.average(data_1)
mean_2 = np.average(data_2)

# mean 2-D vector
mean_vec = np.array([mean_1, mean_2])
print()
print('The mean 2-D vector of the data set is:')
print(mean_vec)

##### Subtract mean from each point in data set #####

deviation = data - mean_vec
print()
print('The deviation of the data set from the mean is:')
print(deviation)

##### Normalize! #####

# first strip the columns of deviation into individual arrays
dev_1 = deviation[:, 0]
dev_2 = deviation[:, 1]

# find norm of each data point
norm = np.sqrt(dev_1 ** 2 + dev_2 ** 2)

# extend norm into a (1000,2) array to allow broadcasting with deviation array
norm_2 = np.array([norm, norm])

# find normalized deviation data 
norm_dev_data = deviation / norm_2.T
print()
print('Subtracting the mean from the data and normalizing gives:')
print(norm_dev_data)

##### Plot #####

f, axarr = plt.subplots(3)
axarr[0].scatter(data_1, data_2, s=10)
axarr[0].set(ylabel = 'Original\nData')
#axarr[0].set_title('Original Data Set')
axarr[1].scatter(dev_1, dev_2, s=10, c='m')
axarr[1].set(ylabel = 'Deviation\nfrom the\nMean')
axarr[2].scatter(norm_dev_data[:, 0], norm_dev_data[:, 1], s=10, c='g')
axarr[2].set(ylabel = 'Normalized\nDeviation')

print()
print('And now we show some nifty plots!')
plt.show()

if __name__ == '__main__':
	assignment01()
