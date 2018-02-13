import numpy as np
import matplotlib.pyplot as plt
import math 
import textwrap
import sys
from numpy.linalg import inv 


def generateUniformData(N, l, u, gVar):
        '''generateUniformData(N, l, u, gVar): Generate N UNIFORMLY SPACED data points in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
        # x = np.random.uniform(l,u,N)
        step = (u-l)/(N);       #Makes sense. (Upper-Lower)/N = number of steps
        x = np.arange(l+step/2,u+step/2,step) #This is an array of x data points, equally spaced. Why divide by 2?
        e = np.random.normal(0,gVar,N) #Randomly draw N samples from Gaussian centered at 0 with stdev == gVar
        t = np.sin(2*math.pi*x) + e #True value follows a sinusoidal model with Gaussian error
        return x,t #x and t are both arrays!

def generateRandData(N, l, u, gVar):
        '''generateRandData(N, l, u, gVar): Generate N uniformly RANDOM data points in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
        x = np.random.uniform(l,u,N)
        e = np.random.normal(0,gVar,N)
        t = np.sin(2*math.pi*x) + e
        return x,t

def fitdata(x,t,M):      #Builds X_matrix, then determines w! Use the 10 training x values
        '''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t)'''     
        X_matrix = np.array([x**m for m in range(M+1)])    #Note that X_matrix is equal to Prof.'s X.transpose!
        w = (inv(X_matrix@X_matrix.T)@X_matrix)@t       #We derived this in class! This is the best w which minimizes the error.
        return w
        
def plotPoly(x,t,w,l,u,subplotloc):
        '''plotPoly(x,t,w,l,u,subplotloc): Plot data (x,t) and the polynomial with parameters w across the range [l,u) in a sub figure at location subplotloc'''
        xrange = np.arange(l,u,0.001)  #get equally spaced points in the xrange
        y = np.sin(2*math.pi*xrange) #compute the true function value
        X = np.array([xrange**m for m in range(w.size)]).T
        esty = X@w #compute the predicted value

        #plot everything
        plt.subplot(subplotloc) #identify the subplot to use
        plt.tight_layout()
        p1 = plt.plot(xrange, y, 'g') #plot true value
        p2 = plt.plot(x, t, 'bo') #plot training data
        p3 = plt.plot(xrange, esty, 'r') #plot estimated value

        #add title, legend and axes labels
        plt.ylabel('t') #label x and y axes
        plt.xlabel('x')
        plt.rcParams["axes.titlesize"] = 10
        myTitle = 'Plot of data, true function, and estimated polynomial with order M = ' + str(w.size-1) + ' and N =' + str(x.size)
        fig.add_subplot(subplotloc).set_title("\n".join(textwrap.wrap(myTitle, 50)))
        plt.legend((p1[0],p2[0],p3[0]),('True Function', 'Training Data', 'Estimated\nPolynomial'), fontsize=6)
        
if __name__ == '__main__':
        x, t = generateUniformData(10, 0, 1, 0.2)
        w1 = fitdata(x, t, 0)
        print('w1 is:', w1)
        
        w2 = fitdata(x, t, 1)
        print('w2 is:', w2)
        
        w3 = fitdata(x, t, 3)
        print('w3 is:', w3)
        
        w4 = fitdata(x, t, 9)
        print('w4 is:', w4)
        
        fig = plt.figure()
        plotPoly(x, t, w1, 0, 1, 221)
        plotPoly(x, t, w2, 0, 1, 222)
        plotPoly(x, t, w3, 0, 1, 223)
        plotPoly(x, t, w4, 0, 1, 224)

        plt.show()        
