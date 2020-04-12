A collection of things I tried to do regarding COVID-19. 

Largely I worked on the https://www.epidemicdatathon.com/

The goal was to build and supply predictive models for the COVID-19 pandemic, in order to try and aid in forecasting efforts.

`ARForecastCOVID.py` is an online (in principle) prediction algorithm I wrote in order to forecast COVID-19 Confirmed cases and COVID-19 Deaths. The algorithm is a work in progress. The algorithm currently predicts the case numbers two days in advance (so if today is day `T` then the algorithm predicts the cases at time `T+2`). The algorithm uses an autoregressive state space time series model. It is currently a simple model that makes forecasts as a linear combination of a number of previous data points, and some one-hot encoding for a few of the so-called "key" countries. Because coronavirus spreads at time `T` as a function of how many people are infected at times `T-k` for `k>=1`, an autoregressive model seemed appropriate here. As of `04/11/2019`, the model forecasts for only 6 countries. The model is estimated using a handwritten Kalman Filter in `Pytorch`, for the possibility of using their autograd tool in the future. A couple unknown parameters are fit to the data using EM. The model does NOT currently use an iterative forecast method, instead it directly learns to forecast the point estimate in the horizon directly. The parameters in the model currently evolve according to a random walk process.

`Goals` (roughly in order of importance):
`1.` Extend the program to make predictions at times `T+2`, `T+7`, and `T+30` automatically
`2.` Include more useful features
`3.` Find a better way to obtain prediction intervals other than assuming a normal distribution on the residuals (ultimately, learn the prediction interval?)
`4.` Find a way to allow observation noise to evolve over time
`5.` Remove parametric assumptions (though autoregression should be robust)
