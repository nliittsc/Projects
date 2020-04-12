# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:05:40 2020

@author: acros
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta


path = r"https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"
data = pd.read_csv(path)

path1 = r"https://raw.githubusercontent.com/ninoaf/baseline_epi_predict/master/submission_samples/2day_prediction_2020-03-15.csv"
predictions = pd.read_csv(path1)
cols = ['N', 'low95N', 'high95N','R', 'low95R', 'high95R', 'D', 'low95D', 'high95D', 'T', 'low95T',
       'high95T', 'M', 'low95M', 'high95M', 'C', 'low95C', 'high95C']
for col in cols:
    predictions[col] = np.nan

predictions['Province/State'] = predictions['Country']

def embed_ts(data, countries, h, num_lags):
### TODO: output test point for algorithm 
    embedding1 = []
    embedding2 = []
    embedding3 = []
    for country in countries:
        df = data[data['Country/Region'] == country].groupby('Date').sum()
        cntry_vals = np.repeat(country, repeats=len(df)).reshape((-1, 1))
        df = df.groupby('Date').sum()
        confirmed = df.Confirmed#.pct_change().fillna(0)
        #inf_mask = (confirmed == np.inf)
        #confirmed[inf_mask] = 0.0
        #inf_mask = (confirmed == -np.inf)
        #confirmed[inf_mask] = 0.0
        #confirmed += 1
        time = np.arange(len(confirmed)).reshape((-1, 1))
        recovered = df.Recovered
        deaths = df.Deaths
        x1 = []
        x2 = []
        x3 = []
        x1.append(confirmed.values.reshape((-1, 1)))
        x2.append(deaths.values.reshape((-1, 1)))
        x3.append(recovered.values.reshape((-1, 1)))
        for i in range(num_lags):
            x1.append(confirmed.shift(h+i, fill_value=0).values.reshape((-1, 1)))
            x2.append(deaths.shift(h+i, fill_value=0).values.reshape((-1, 1)))
            x3.append(recovered.shift(h+i, fill_value=0).values.reshape((-1, 1)))
        #stacks the data columnwise
        x1 = np.hstack(x1)
        x1 = np.append(x1, time, axis=1)
        x1 = np.append(x1, cntry_vals, axis=1)
        embedding1.append(x1)
        
        x2 = np.hstack(x2)
        x2 = np.append(x2, time, axis=1)
        x2 = np.append(x2, cntry_vals, axis=1)
        embedding2.append(x2)
        
        x3 = np.hstack(x3)
        x3 = np.append(x3, time, axis=1)
        x3 = np.append(x3, cntry_vals, axis=1)
        embedding3.append(x3)
    
    
    ohe_map = dict()
    
    #create design matrix for confirmed cases
    X1 = np.vstack(embedding1)    #form design matrix
    time = X1[:,-2]    #get column holding time indicators
    X1 = np.delete(X1, X1.shape[1]-2, axis=1) #delete time column from X1
    cats = pd.get_dummies(X1[:,-1]).astype(int)
    print(cats.columns)
    X1 = X1[:,:-1]    #remove country strings from X1
    #make the ohe_map
    i = 0
    for c in cats.columns:
        ohe_map[c] = X1.shape[1] + i - 1
        i+=1
    X1 = np.hstack([X1, cats])
    y1 = X1[:,0].astype(float)
    X1 = X1[:,1:].astype(float)
    time = time.astype(float)
    
    #create design matrix for deaths
    X2 = np.vstack(embedding2)    #form design matrix
    X2 = np.delete(X2, X2.shape[1]-2, axis=1) #delete time column from X2
    cats = pd.get_dummies(X2[:,-1]).astype(int)
    X2 = X2[:,:-1]    #remove country strings from X2
    X2 = np.hstack([X2, cats])
    y2 = X2[:,0].astype(float)
    X2 = X2[:,1:].astype(float)
    
    #create design matrix for deaths
    X3 = np.vstack(embedding3)    #form design matrix
    X3 = np.delete(X3, X3.shape[1]-2, axis=1) #delete time column from X2
    cats = pd.get_dummies(X3[:,-1]).astype(int)
    X3 = X3[:,:-1]    #remove country strings from X2
    X3 = np.hstack([X3, cats])
    y3 = X3[:,0].astype(float)
    X3 = X3[:,1:].astype(float)
    
    return(X1, y1, X2, y2, X3, y3, time, ohe_map)


def make_test_point(X, y, countries, ohe_map, num_lags):
    y_test = np.zeros((len(countries), X.shape[1]))
    for i, country in enumerate(countries):
        test_point = np.zeros((1, X.shape[1]))
        ohe_col = ohe_map[country]
        test_point[:,ohe_col] = 1.0
        mask = (X[:,ohe_col] == 1.0)
        y_subset = y[mask]
        for j in range(num_lags):
            test_point[0, j] = y_subset[-1-j]
    
        y_test[i] = test_point
        #print(y_test)
    
    return(y_test)
    
#%% Online learning Algorithm
def latent_predict(A, delta, z, P):
    #predict
    z = A.mm(z)
    Q = ((1 - delta) / delta) * P    #discounting
    P = A.mm(P).mm(A.T) + Q
    return(z, P)

def posterior_predict(C, R, z, P):
    #update step
    yhat = C.mm(z)
    S = C.mm(P).mm(C.T) + R
    return(yhat, S)

def update_step(C, yt, yhat, S, z, P):
    r = yt - yhat     #residual
    temp = P.mm(C.T)
    K, _ = torch.solve(temp.T, S.T)    #kalman gain
    K = K.T
    z = z + K.mm(r)
    KC = K.mm(C)
    M = torch.eye(*KC.shape) - KC
    P = M.mm(P).mm(M.T)
    P += K.mm(R).mm(K.T)
    return(z, P, r)

def smooth_step(A, C, z, P, t, states, covs, delta):
    z_t = states[t]
    P_t = covs[t]
    z_tplus1, P_tplus1 = latent_predict(A, delta, z_t, P_t)
    #J = P.mm(A.T)
    J, _ = torch.solve(P_t.T, P_tplus1.T)
    J = J.T
    #smooth states
    z = z_t + J.mm((z - z_tplus1))
    P = P_t + J.mm((P - P_tplus1)).mm(J.T)
    return(z, P, J)
            
#%% Data Preprocessing
     
num_lags = 4
h = 7
country_list = predictions.Country.unique()
X1, y1, X2, y2, X3, y3, time, ohe_map = embed_ts(data, country_list, h, num_lags)


#%% learning algorithm
from torch.distributions import MultivariateNormal as mvn

for i in range(1, 4):
    if i == 1:
        X = X1
        y = y1
    if i == 2:
        X = X2
        y = y2
    if i == 3:
        X = X3
        y = y3
        
    test = make_test_point(X, y, country_list, ohe_map, num_lags)
    
    d = X.shape[1]
    m = len(predictions.Country.unique())
    A = torch.eye(d, d).float()
    #for i in range(d):
    #    if i < d-1:
    #        A[i, i+1] = 1.0
    CT = torch.tensor(X).float()
    R = torch.eye(m, m).float()
    Q = torch.eye(d, d).float() * 1
    z = torch.ones(d, 1).float()
    P = torch.eye(d, d).float() * 1000
    y_train = torch.tensor(y).float()
    states = []
    covs = []
    noise = []
    
    T = time.max().astype(int)
    delta = 0.95
    max_iter = 200
    old_logprob = 0
    for k in range(max_iter):
        logprob = 0
        for t in range(T+1):
        
            #get features for time t
            mask = (time == t)
            yt = y_train[mask].view((-1, 1))
            C = torch.tensor(X[mask]).float()
            
            z, P = latent_predict(A, delta, z, P)
            yhat, S = posterior_predict(C, R, z, P)
            z, P, error = update_step(C, yt, yhat, S, z, P)
            
            #store updates
            states.append(z)
            covs.append(P)
            #noise.append(R)
        
        smooth_states = []
        smooth_states.append(z)
        smooth_covs = []
        smooth_covs.append(P)
        Js = []
        
        for t in range(T-1, -1, -1):
            mask = (time == t)
            C = torch.tensor(X[mask]).float()
            z, P, J = smooth_step(A, C, z, P, t, states, covs, delta)
            
            smooth_states.append(z)
            smooth_covs.append(P)
            Js.append(J)
        
        smooth_states.reverse()
        smooth_covs.reverse()
        
        
        #update parameters
        R = torch.zeros(*R.shape)
        #print(R)
        for t in range(T+1):
            mask = (time == t)
            yt = y_train[mask].view((-1, 1))
            C = torch.tensor(X[mask]).float()
            z = smooth_states[t]
            P = smooth_covs[t]
            R += yt.mm(yt.T)
            R -= C.mm(z).mm(yt.T)
            R -= yt.mm(z.T).mm(C.T)
            R += C.mm(z).mm(z.T).mm(C.T)
            
        R = R / T
        z = smooth_states[0]
        P = smooth_covs[0]
    
        logprob += mvn(yhat.view(-1), S).log_prob(yt.view(-1)).item()
        print("log prob: {}".format(logprob))
        
        if (k > 0):
            diff = abs(old_logprob - logprob)
            old_logprob = logprob
            print("difference: {}".format(diff))
            if diff < 0.001:
                break


#%% Validation Procedure




#%% Generate Forecast






    #generate predictive distribution
    C = torch.tensor(test).float()
    yhat_, S_ = posterior_predict(C, R, states[-1], covs[-1])
    
    samples = mvn(loc=yhat_.view(-1), covariance_matrix=S_).rsample((10000,))
    pred_mean = samples.mean(axis=0)
    pred_std = samples.std(axis=0)
    
    #write predictions to csv
    for j, country in enumerate(country_list):
        pred = int(pred_mean[j].item())
        std = int(pred_std[j].item())
        date = datetime.now() + timedelta(h)
        date = date.strftime('%Y-%m-%d')
        predictions.loc[predictions.Country == country, 'Target/Date'] = date
        
        if i == 1:
            predictions.loc[predictions.Country == country, 'N'] = pred
            predictions.loc[predictions.Country == country, 'low95N'] = pred - 2*std
            predictions.loc[predictions.Country == country, 'high95N'] = pred + 2*std
        if i == 2:
            predictions.loc[predictions.Country == country, 'D'] = pred
            predictions.loc[predictions.Country == country, 'low95D'] = pred - 2*std
            predictions.loc[predictions.Country == country, 'high95D'] = pred + 2*std
        else:
            predictions.loc[predictions.Country == country, 'R'] = pred
            predictions.loc[predictions.Country == country, 'low95R'] = pred - 2*std
            predictions.loc[predictions.Country == country, 'high95R'] = pred + 2*std


predictions['N'] = predictions.N.astype(int)
predictions['low95N'] = predictions.low95N.astype(int)
predictions['high95N'] = predictions.high95N.astype(int)
predictions['D'] = predictions.D.astype(int)
predictions['low95D'] = predictions.low95D.astype(int)
predictions['high95D'] = predictions.high95D.astype(int)
predictions['R'] = predictions.R.astype(int)
predictions['low95R'] = predictions.low95R.astype(int)
predictions['high95R'] = predictions.high95R.astype(int)

#very basic mortality estimates
for country in country_list:
    R = predictions[predictions['Country'] == country].R
    D = predictions[predictions['Country'] == country].D
    predictions.loc[predictions.Country == country, 'M'] = D / (D + R)
    
    R = predictions[predictions['Country'] == country].high95R
    D = predictions[predictions['Country'] == country].low95D
    predictions.loc[predictions.Country == country, 'low95M'] = D / (D + R)
    
    R = predictions[predictions['Country'] == country].low95R
    D = predictions[predictions['Country'] == country].high95D
    predictions.loc[predictions.Country == country, 'high95M'] = D / (D + R)
    




path = r"C:\Users\acros\.spyder-py3\\" + str(h) + "day_prediction_" + date + ".csv"
predictions.to_csv(path, index=False, columns=predictions.columns)






