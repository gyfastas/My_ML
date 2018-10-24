'''
Machine Learning HW1

Gradient Descent Linear Model
Author: 郭远帆
'''

import numpy as np

class GDLinear():
    def __init__(self,alpha = 1,L = 0):
    '''
    Initialize the model
    Paras:
    alpha: Learning rate (must be >0)
    L: Regularization, L=0:no regularization L=1: L1 regularization, L=2: L2 regularization
    '''

        self.alpha = alpha
        self.L = L
    def fit(self,X,Y):
        '''

        :param self:
        :param X: Numpy Array of Training set X of shape(n_samples,n_features)
        :param Y: Numpy Array of Target of shape(n_samples,n_target)
        :return: nothing
        '''
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_target = Y.shape[1]
        self.w = np.zeros([n_features,n_target])




    def predict(self,X):
        '''

        :param self:
        :param X: Numpy Array of shape(n_samples,n_features)
        :return: Prediction Y
        '''

    def set_para(self,alpha,L):
        '''

        :param self:
        :param alpha: Learning rate
        :param L:  Regularization
        :return: nothing
        '''
        self.
    def get_coefficient(self):