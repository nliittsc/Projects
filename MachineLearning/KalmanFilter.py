# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:37:47 2020

@author: acros
"""
import numpy as np
import scipy.linalg as linalg

def mult_by_inverse(A, b):
    """ Numerically stable way of getting A*B^{-1}"""
    v = np.linalg.solve(b.T, A.T)
    solution = v.T
    return(solution)

import matplotlib.pyplot as plt


"""Kalman Filter Equations:
    
    predict:
    x = Fx
    P = FPF^T + Q
    
    update:
    S = HPH^T + R
    K = PH^{T}S^{-1}
    r = Y - Hx
    x = x + Kr
    P = (I - KH)P
"""

class KalmanFilter():
    def __init__(self, F=None, H=None, Q=None,
                 R=None, x0=None, P0=None, Y=None):
        #print("setting x0")
        self.F = F    # State Transition Function
        self.H = H    # "Measurement" Transition Function
        self.Q = Q    # Process/Latent Noise
        self.R = R    # Observation Noise
        self.x0 = x0  #initial state
        self.P0 = P0  #initial noise covariance
        self.Y = Y
        
    def filter_predict(self, F, Q, x, P):
        """Predicts the latent state x and the latent covariance P"""
        x = F.dot(x)
        P = np.dot(F, P).dot(F.T)
        P = (P + P.T)/2    # enforce symmetry
        return(x, P)
    
    def filter_update(self, H, R, x, P, y):
        """Updates the latent state and latent covariance P with new
            observation"""
        #check if R is a matrix
        R_matrix_bool = isinstance(R, np.ndarray)
        if R_matrix_bool == False:
            if isinstance(R, (float, int)):
                R = np.asarray([[R]])
        
        S = np.dot(H, P).dot(H.T) + R
        if S.shape[0] == 1 and S.shape[1] == 1:
            K = np.dot(P, H.T) / S[0][0]
        else:
            K = mult_by_inverse(np.dot(P, H.T), S)
        r = y - H.dot(x)
        x = x + K.dot(r)
        KH = K.dot(H)
        M = (np.eye(*KH.shape) - KH)
        #P = np.dot(M, P).dot(M.T) + np.dot(K, R).dot(K.T)    #for more numerical stability
        U, s, V = linalg.svd(P)
        D = np.zeros((len(s), len(s)))
        for i in range(len(s)):
            D[i][i] = s[i]
        P = np.dot(np.dot(M, U), np.dot(D, V)) + np.dot(K, R).dot(K.T)
        return(x, P)
    
    def filter_states(self, Y=None):
        """Runs the filter forward on the entire sequence, getting the
            filtered latent states and process noise """
        
        x = np.copy(self.x0)
        P = np.copy(self.P0)
        T = len(Y)
        filtered_states = []
        filtered_states.append(x)
        filtered_noise = []
        filtered_noise.append(P)
        
        #run the filter forward on the entire instance
        for t in range(1, T):
            x, P = self.filter_predict(self.F, self.Q, x, P)
            x, P = self.filter_update(self.H, self.R, x, P, Y[t])
            
            filtered_states.append(x)
            filtered_noise.append(P)
        
        return(filtered_states, filtered_noise)
    
    
    def smooth_predict(self, F, Q, P):
        """ Prediction for the smoothed covariance """
        P = np.dot(F, P).dot(F.T) + Q
        return(P)
    
    def smooth_update(self, F, x, P, x_prev, P_prev):
        K = np.dot(P_prev, F.T).dot(P)
        x = x + np.dot(K, (x_prev - F.dot(x)))
        
    
    def smooth_states(self, filtered_states, filtered_noise):
        """Assumes the data is all observed, and smoothes the latent states"""
        F = self.F
        Q = self.Q
        x = filtered_states.copy()
        P = filtered_noise.copy()
        Pp = filtered_noise.copy()
        T = len(filtered_states)-1
        K = np.zeros((T, 2, 2))
        for k in range(T-2, -1, -1):
            # "Predict" step
            Pk = P[k]
            xk = x[k]
            Pp[k] = np.dot(F, Pk).dot(F.T) + Q # predicted covariance

            #K[k] = np.dot(P[k], F.T).dot(linalg.inv(Pp[k]))
            K[k] = mult_by_inverse(np.dot(P[k], F.T), Pp[k])
            xk = xk + K[k].dot((x[k+1] - F.dot(xk)))
            x[k] = xk
            P[k] += np.dot(K[k], P[k+1] - Pp[k]).dot(K[k].T)
            P[k] = (P[k] + P[k].T) / 2    #enforce symmetry
            
        smooth_states = x
        smooth_noise = P
        corrections = K
        assert len(smooth_states) == len(filtered_states)
        return(smooth_states, smooth_noise, corrections)
    
    
#%% EM algorithm
from scipy.stats import multivariate_normal as MVN
class EM(KalmanFilter):
    
    def log_like(self, F, H, Q, R, x0, P0, X, Y):
        """ Computes the complete Log-Likelihood of the data using Scipy """
        log_likelihood = 0
        T = len(X)
        #gets the log-likelihood of the initial hidden state and covariance
        x0 = x0.reshape(-1)
        z1 = X[0].reshape(-1)
        log_likelihood += MVN(x0, P0).logpdf(z1)
        
        for t in range(1, T):
            #loglikelihood of data, depending on latent state
            z_t = H.dot(X[t]).reshape(-1)
            y_t = Y[t].reshape(-1)
            log_likelihood += MVN(z_t, R).logpdf(y_t)
            
            #loglikelihood of latent states, depending on prev state
            z_t = F.dot(X[t]).reshape(-1)
            z_t_ = F.dot(X[t-1]).reshape(-1)
            log_likelihood += MVN(z_t_, Q).logpdf(z_t)
        
        return(log_likelihood)
    
    def inference(self):
        """ Performs a full forward filtering step, then a backwards smoothing
            step in order to obtain estimates of the unknown states"""
        #print("doing inference")
        Y = self.Y
        fltrd_x, fltrd_cov = self.filter_states(Y)
        smthd_x, smthd_cov, corrections= self.smooth_states(fltrd_x, fltrd_cov)
        return(smthd_x, smthd_cov, corrections)
        
    def learn(self, max_iter=10, tol=0.1, learn_A=False, learn_C=False,
              learn_Q=False, learn_R=False, learn_x0=False, learn_P0=False):
        """ EM learning procedure for unknowns.
            We follow the procedure given in Bishop's PRML book
            Note the change in notation to be consistent with Bishop
        """
        A = np.copy(self.F)
        C = np.copy(self.H)
        Q = np.copy(self.Q)
        R = np.copy(self.R)
        x0 = np.copy(self.x0)
        P0 = np.copy(self.P0)
        Y = self.Y.reshape((-1, 1))
        #old_log_like = self.log_like(A, C, Q, R, x0, P0, smthd_x, Y)
        #print("Initial log-likelihood: {}".format(old_log_like))
        k = 0
        old_log_prob = -1e6
        while True:
            #print("value of p0: {}".format(P0))
            # Expectation step: apply the Kalman Filter to do inference
            X, V, J = self.inference()
            T = len(X)
            #Compute expected statistics for mean and covariance
            Ezn_zn = []
            Ezn_zn_ = []
            Ezn = X.copy()
            for t in range(T):
                cov1 = V[t] + X[t].dot(X[t].T)
                Ezn_zn.append(cov1)
                if t > 0:
                    cov2 = V[t].dot(J[t-1].T) + X[t].dot(X[t-1].T)
                    Ezn_zn_.append(cov2)
            #print("length of X: {}".format(len(X)))
            #print("length of Ezn_zn: {}".format(len(Ezn_zn)))
            #print("length of Ezn_zn_: {}".format(len(Ezn_zn_)))
            # Maximization Step: Obtain new estimates
            if learn_A == True:
                alpha = np.sum(Ezn_zn_, axis=0)
                beta = np.sum(Ezn_zn[:T-2], axis=0)
                A = mult_by_inverse(alpha, beta)
                
                #stabilize A
                U, V, D = linalg.svd(A)
                bad_eigen = V > 1.0
                if np.any(bad_eigen):
                    V[bad_eigen] = 0.99
                    S = np.zeros((len(V), len(V)))
                    for i in range(len(V)):
                        S[i][i] = V[i]
                    A = np.dot(U, S).dot(D)
                #print(A)
            
            #TODO: fix this
            if learn_Q == True:
                alpha = np.sum(Ezn_zn[1:], axis=0)
                alpha -= np.dot(A, np.sum(Ezn_zn_, axis=0).T)
                alpha -= np.dot(np.sum(Ezn_zn_, axis=0), A.T)
                temp = np.sum(Ezn_zn[:T-2], axis=0)
                alpha += np.dot(A, temp.dot(A.T))
                alpha = alpha / (T-1)
                Q[0][0] = alpha[0][0] + 3    #correction factor
                Q[1][1] = alpha[1][1] + 3    #correction factor
                #print(Q)
                
                
            if learn_C == True:
                alpha = np.sum([Y[t].dot(Ezn[t].T) for t in range(T)], axis=0)
                beta = np.sum(Ezn_zn, axis=0)
                C = mult_by_inverse(alpha, beta)
                C = C.reshape((1, 2))
            
            if learn_R == True:
                alpha = Y.T.dot(Y)
                temp = np.sum([Ezn[t].dot(Y[t].T) for t in range(T)], axis=0)
                temp = temp.reshape((-1, 1))
                alpha -= C.dot(temp)
                alpha -= temp.T.dot(C.T)
                temp = np.sum([Ezn_zn[t].dot(C.T) for t in range(T)], axis=0)
                alpha += np.dot(C, temp)
                R = alpha / T
                R = np.asarray(R).reshape((-1, 1))
                
            if learn_x0 == True:
                #print("updating x0")
                x0 = X[0]
            
            if learn_P0 == True:
                #print("updating P0")
                P0 = Ezn_zn[0] - Ezn[0].dot(Ezn[0].T)
                
            
            #update class attributes
            
            #print("current x0: {}".format(x0))
            #compute loglikelihood
            log_prob = self.log_like(A, C, Q, R, x0, P0, X, Y)
            print("Log-Likelihood: {}".format(log_prob))
            k += 1
            diff = log_prob - old_log_prob
            if diff < tol and diff > 0:
                print("yay")
                print("Optimum Likelihood achieved at {}".format(log_prob))
                break
            if diff < 0:
                print("likelihood beginning to diverge")
                print("terminating")
                break
            if k > max_iter:
                print("Max Iterations reached. Try increasing max_iter or increasing tol.")
                break
            
            old_log_prob = log_prob
            self.F = A
            self.H = C
            self.R = R
            self.Q = Q
            self.x0 = x0
            self.P0 = P0
                
#%% basic model
class LocalTrendModel(EM):
    def __init__(self):
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.eye(2)
        self.R = 1
        self.x0 = np.array([[1], [1]])
        self.P0 = np.eye(2) 
    
    def fit(self, Y, max_iter=100, tol=0.01):
        self.Y = Y
        print("setting x0")
        self.x0 = self.x0   # Initialize hidden states
        self.learn(max_iter, tol, learn_A=False, learn_C=False, learn_Q=True,
                   learn_R=True, learn_x0=True, learn_P0=True)
        
    


#%% tests


zs = np.array([ 0.,  0.        ,  0.69314718,  0.69314718,  1.60943791,
        1.60943791,  1.60943791,  1.60943791,  1.60943791,  1.94591015,
        2.07944154,  2.07944154,  2.39789527,  2.39789527,  2.39789527,
        2.39789527,  2.39789527,  2.39789527,  2.39789527,  2.39789527,
        2.48490665,  2.48490665,  2.56494936,  2.56494936,  2.56494936,
        2.56494936,  2.56494936,  2.56494936,  2.56494936,  2.56494936,
        2.7080502 ,  2.7080502 ,  2.7080502 ,  3.93182563,  3.93182563,
        4.04305127,  4.06044301,  4.09434456,  4.21950771,  4.30406509,
        4.58496748,  4.77068462,  5.00394631,  5.37989735,  5.5683445 ,
        5.99645209,  6.24997524,  6.36818719,  6.86589107,  7.1553963 ,
        7.41637848,  7.68662133,  7.91095738,  8.16023249,  8.44074402,
        8.76732915,  8.95969715,  9.52347087,  9.85744361, 10.14600227,
       10.4125917 , 10.68846158, 10.89191288, 11.09404071, 11.33661779,
       11.52935968, 11.70748846, 11.85570633, 11.99415955, 12.14511172,
       12.2707924 , 12.40267918, 12.52665502, 12.640611  ])


zs = zs + np.random.rand(*zs.shape)*3


nom =  np.array([t/2. for t in range (0, 40)])
#zs = np.array([t + np.random.randn()*1.1 for t in nom])
model = LocalTrendModel()

model.fit(zs)
states, _ = model.filter_states(zs)
smooth_states, _, __= model.smooth_states(states, _)
y_pred = np.zeros(len(states))
y_smooth = np.zeros(len(states))
for t in range(len(states)):
    y_pred[t] = model.H.dot(states[t])
    y_smooth[t] = model.H.dot(smooth_states[t])
plt.plot(zs)
plt.plot(y_pred)
plt.plot(y_smooth)
        
        
        
#testing E
        
        
        
        
        
        
        
        
        
        
        
        
        