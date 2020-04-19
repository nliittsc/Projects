# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:05:40 2020

@author: acros
"""

#%% Imports
import numpy as np
import pandas as pd


#%% algorithm

""" If we have M time series of length T contained in an MxT matrix X
    our goal is to learn a factorization of X = U.T.dot(V) where U is dxM
    and V is dxT
"""
import scipy.linalg as linalg


class OnlineMatrixFactorizer():
    """ Zero-Tolerance Implmentation from Gultekin and Paisley.
    Inputs:
        d = dimension of the latent space
        r0 = scalar giving the level of noise in prior diagonal covariance
        rho_v = constant for numerical stabilization. should be small
        p = number of lags for the AR model
    """
    def __init__(self, d, r0, rho_v, p):
        self.d = d
        self.r0 = r0
        self.rho_v = rho_v
        self.p = p
        
    
    def fit(self, X, max_ite=15):
        """ Fit method for the model.
        Inputs:
            X = MxT matrix, where M = num time series, T = length time
            max_ite = number of EM iterations to run
        Outputs:
            V = The latent stochastic process
            U = The latent traits for each time series
            Xpreds = The one step ahead predictions
            theta = the learned weights for the AR model
            MAE = the mean absolute error of the one-step ahead forecasts
        """
        d = self.d
        r0 = self.r0
        rho_v = self.rho_v
        p = self.p
        T = X.shape[1]
        M = X.shape[0]
        
        #initialize quantities to be estimated
        U = np.random.randn(d, M) * (2/d)
        V = np.random.randn(d, T) * (2/d)
        v = np.random.randn(d, 1)
        I = np.eye(p, p)
        rlt = r0 * I
        rrt = np.zeros((p, 1))
        theta = np.zeros((p, 1))
        
        #to store results
        Xpreds = np.zeros((M, T))
        P = np.random.randn(d, p)
        MAE = []
        for t in range(T):
            #get priors for time t-1
            if t == 0:
                vprior = np.zeros((d, 1))
                Uprior = np.zeros((d, M))
            elif t > 0 and t <= p:
                vprior = v.copy()
                Uprior = U.copy()
            else:
                Uprior = U.copy()
                vprior = P.dot(theta)
                
            #one step ahead forecast to time t
            x_fcast = Uprior.T.dot(vprior).reshape(-1)
            Xpreds[:, t] = x_fcast
            #observe data point and loss
            error = np.sum(np.abs(X[:,t] - x_fcast))# / len(x_fcast)
            MAE.append(error)
            x = X[:, t].reshape((-1, 1))
            #Update to posterior
            for i in range(max_ite):
                #updates v
                M1 = rho_v * np.eye(d, d) + U.dot(U.T)
                M2 = rho_v * vprior + U.dot(x)
                v = linalg.solve(M1, M2)
                
                #find lambda
                M1 = Uprior.T.dot(v) - x
                M2 = v.T.dot(v)
                lam = M1 / M2.item()
                U = Uprior - v.dot(lam.T)
            
            #currently assuming all observations
            V[:,t] = v.reshape(-1)
            if t >= p:
                P = V[:, t-p:t]    #get last p vectors
                rlt = rlt + P.T.dot(P)   #update cov
                rrt = rrt + P.T.dot(v)   #update cov
                theta = linalg.solve(rlt, rrt)    #get weights
            
        self.U = U
        self.V = V
        self.Xpreds = Xpreds
        self.theta = theta
        self.MAE = np.mean(MAE)
    
    
            
                
#%% Load Electricity Data and Preprocessing

path = r"C:\Users\acros\.spyder-py3\LD2011_2014.txt"
data = pd.read_csv(path, sep=';', decimal=',')
X = data.values[:, 1:].astype(float)
X = X.T / X.max(axis=1)
X = X.T
#%% Fit model

d = 5
r0 = 1
rho_v = 1e-4
p = 24
max_ite = 15

model = OnlineMatrixFactorizer(d, r0, rho_v, p)
model.fit(X, max_ite=max_ite)
print("One Step Ahead MAE: {}".format(model.MAE))







