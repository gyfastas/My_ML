'''
Machine Learning HW1

Gradient Descent Linear Model
Author: 郭远帆
'''

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
class GDLinear():
    def __init__(self,alpha = 1,L = 0,max_iter = None,min_abs = 0.001,beta1 = 0.5,beta2 = 0.5 ):
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
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_target = Y.shape[1]
        #Generate b and w by random value
        self.b = np.random.rand(1)
        self.w = np.random.rand(n_features,1)


        #Training(gradient descent)
        if self.max_iter==None:
            while(1):
                h_x = np.matmul(X,self.w) + b
                self.b = self.b - self.alpha/n_samples*(np.sum((h_x-Y))
                for j in range(0,n_features):
                    self.w[j,0] = self.w[j,0] - self.alpha/n_samples*(np.sum((h_x-Y)*X[:,j]))

                if L==0:
                    M = mean_squared_error(h_x,Y)
                elif L==1:
                    M = 0.5/n_samples*mean_squared_error(h_x,y) + self.beta1*np.sum(np.abs(self.w))

                elif L==2:
                    M = 0.5/n_samples*mean_squared_error(h_x,y) + self.beta2*np.sqrt(np.sum(self.w*self.w)))
                else:
                    M = mean_squared_error(h_x,Y)
                if M <= self.min_abs:
                    break

        else:
            for k in range(0,self.max_iter):
                h_x = np.matmul(X,self.w) + b
                self.b = self.b - self.alpha/n_samples*(np.sum((h_x-Y))
                for j in range(0,n_features):
                    self.w[j,0] = self.w[j,0] - self.alpha/n_samples*(np.sum((h_x-Y)*X[:,j]))

                if L==0:
                    M = mean_squared_error(h_x,Y)
                elif L==1:
                    M = 0.5/n_samples*mean_squared_error(h_x,y) + self.beta1*np.sum(np.abs(self.w))

                elif L==2:
                    M = 0.5/n_samples*mean_squared_error(h_x,y) + self.beta2*np.sqrt(np.sum(self.w*self.w)))
                else:
                    M = mean_squared_error(h_x,Y)
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

    def set_para(self,alpha,L,max_iter,min_abs):
        '''
        :param self:
        :param alpha: Learning rate
        :param L: Regularization
        :param min_abs: min absolute error for stop training
        :param max_iter: Max iteration, None is no limitation on iteration
        :return:
        '''
        self.alpha = alpha
        self.L = L
        self.max_iter = max_iter
        self.min_abs = min_abs
    def get_coefficient(self):
        '''

        :return: coefficient of the model
        '''




