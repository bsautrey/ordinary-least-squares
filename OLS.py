# Implement ordinary least squares from Andrew Ng's CS229 course: http://cs229.stanford.edu/notes/cs229-notes1.pdf. Stochastic gradient descent is used to learn the parameters, i.e. minimize the cost function.
from copy import copy

import numpy as np
import matplotlib.pyplot as plot

# alpha - The learning rate.
# dampen - Factor by which alpha is dampened on each iteration. Default is no dampening, i.e. dampen = 1.0
# tol - The stopping criteria
# theta - The parameters to be learned.

class OLS():
    
    def __init__(self):
        self.X = None
        self.Y = None
        self.alpha = None
        self.dampen = None
        self.tol = None
        self.theta = None
        
    def set_X(self,X):
        self.X = X
    
    def set_Y(self,Y):
        self.Y = Y
        
    def set_alpha(self,alpha=0.001,dampen=1.0):
        self.alpha = alpha
        self.dampen = dampen
        
    def set_tolerance(self,tol=0.0001):
        self.tol = tol
        
    def initialize_theta(self,theta=None):
        if not theta:
            number_of_parameters = self.X.shape[1]
            theta = copy(self.X[0,:])
            theta.resize((1,number_of_parameters))
            
        self.theta = theta
            
    def run_SGD(self,max_iterations=10000):
        old_theta = copy(self.theta)
        iterations = 0
        number_of_rows = self.X.shape[0]
        number_of_columns = self.X.shape[1]
        while True:
            for i in xrange(number_of_rows):
                x = self.X[i,:]
                y = self.Y[i,:][0]
                x.resize((number_of_columns,1))
                for j in xrange(number_of_columns):
                    theta_j = self.theta[0][j]
                    x_j = x[j][0]
                    dot = np.dot(self.theta,x)
                    new_theta_j = theta_j + self.alpha*(y - dot)*x_j
                    self.theta[0][j] = new_theta_j
            
                iterations = iterations + 1
                percent = self._calculate_convergence(old_theta)
                old_theta = copy(self.theta)
                self.alpha = self.alpha*self.dampen
                print iterations,percent,self.theta
                if percent < self.tol or iterations > max_iterations:
                    return
                
    def _calculate_convergence(self,old_theta):
        diff = old_theta - self.theta
        diff = np.dot(diff,diff.T)**0.5
        length = np.dot(old_theta,old_theta.T)**0.5
        percent = 100.0*diff/length
        return percent
        
    def generate_example(self,sample_size=1000):
        # assemble data
        mean = np.array([5,5])
        cov = np.array([[1,0.95],[0.95,1]])
        res = np.random.multivariate_normal(mean,cov,sample_size)
        intercept = np.ones((sample_size))
        X = np.column_stack((res[:,0],intercept))
        Y = np.array(res[:,1])
        Y.resize((sample_size,1))
        
        # initialize
        self.set_X(X)
        self.set_Y(Y)
        self.set_alpha(alpha=0.001,dampen=0.9999)
        self.set_tolerance()
        self.initialize_theta()
        self.run_SGD()
        
        # predict
        Y_hat = []
        number_of_rows = self.X.shape[0]
        for i in xrange(number_of_rows):
            x = self.X[i,:]
            x.resize((2,1))
            y_hat = np.dot(self.theta,x)[0][0]
            Y_hat.append(y_hat)
            
        Y_hat = np.array(Y_hat)
        
        # plot
        plot.scatter(self.X[:,0],self.Y,s=0.5)
        plot.plot(self.X[:,0],Y_hat)
        plot.show()
        