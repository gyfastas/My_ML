'''
Machine Learning HW1

Gradient Descent Linear Model
Author: 郭远帆
'''

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MaxAbsScaler
class GDLinear():
    def __init__(self,alpha = 0.0001,L = 0,max_iter = 10000,min_abs = 0.005,beta1 = 0.5,beta2 = 0.5 ):
        '''
        Initialize the model
        Paras:
        alpha: Learning rate (must be >0)
        L: Regularization, L=0:no regularization L=1: L1 regularization, L=2: L2 regularization
        max_iter: Maximum iteration, None means no limitation
        min_abs: the min absolute error for finishing training
        beta1: L1 Regularization ratio
        beta2: L2 Regularization ratio
        '''

        self.alpha = alpha
        self.L = L
        self.max_iter = max_iter
        self.min_abs = min_abs
        self.beta1 = beta1
        self.beta2 = beta2
    def fit(self,X,Y):
        '''

        :param self:
        :param X: Numpy Array of Training set X of shape(n_samples,n_features)
        :param Y: Numpy Array of Target of shape(n_samples,1)
        :return: nothing
        '''
        X = np.array(X)
        Y = np.array(Y)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_target = Y.shape[1]
        #Generate b and w by random value
        self.b = np.random.rand(1)
        self.w = np.random.rand(n_features,1)


        #Training(gradient descent)
        if self.max_iter==None:
            while(1):
                h_x = np.matmul(X,self.w) + self.b
                self.b = self.b - self.alpha/n_samples*(np.sum(h_x-Y))

                for j in range(0,n_features):
                    self.w[j,0] = self.w[j,0] - self.alpha/n_samples*(np.sum((h_x-Y)*X[:,j]))

                if self.L==0:
                    M = np.sum(np.abs(h_x-Y))
                    M = M/n_samples
                elif self.L==1:
                    M = 0.5/n_samples*np.sum(np.abs(h_x-Y)) + self.beta1*np.sum(np.abs(self.w))

                elif self.L==2:
                    M = 0.5/n_samples*np.sum(np.abs(h_x-Y)) + self.beta2*np.sqrt(np.sum(self.w*self.w))
                else:
                    M = np.sum(np.abs(h_x-Y))
                if M <= self.min_abs:
                    break

        else:
            #gradient descent
            for k in range(0,self.max_iter):
                h_x = np.matmul(X,self.w) + self.b
                self.b = self.b - self.alpha/n_samples*(np.sum((h_x-Y)))
                for j in range(0,n_features):
                    xj = X[:,j]
                    self.w[j,0] = self.w[j,0] - self.alpha/n_samples*(np.sum((h_x-Y)*xj))

                #cost function
                if self.L==0:
                    M = np.sum(np.abs(h_x-Y))
                    M = M/n_samples
                elif self.L==1:
                    M = 0.5/n_samples*np.sum(np.abs(h_x-Y)) + self.beta1*np.sum(np.abs(self.w))

                elif self.L==2:
                    M = 0.5/n_samples*np.sum(np.abs(h_x-Y)) + self.beta2*np.sqrt(np.sum(self.w*self.w))
                else:
                    M = np.sum(np.abs(h_x-Y))
                print(M)
                if M <= self.min_abs:
                    break


    def predict(self,X):
        '''
        :param self:
        :param X: Numpy Array of shape(n_samples,n_features)
        :return: Prediction Y
        '''
        Y = self.b + np.matmul(X,self.w)

        return Y

    def set_para(self,alpha,L,max_iter,min_abs,beta1,beta2):
        '''
        :param self:
        :param alpha: Learning rate
        :param L: Regularization
        :param min_abs: min error for stop training
        :param max_iter: Max iteration, None is no limitation on iteration
        :param beta1: L1 ratio
        :param beta2: L2 ratio
        :return:
        '''
        self.alpha = alpha
        self.L = L
        self.max_iter = max_iter
        self.min_abs = min_abs
        self.beta1 = beta1
        self.beta2 = beta2

    def get_coefficient(self):
        '''

        :return: coefficient of the model
        '''

        return self.b,self.w




