import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
import pyDE as de

#simulated data parmaeters
mu = 10
sd = 2
data_size = 100

#number of mcmc samples
num_samples = 500

#simulate the data
data = norm.rvs(loc=mu,scale=sd,size=data_size)

# Define the model
theta = {
'mu':{'init':[5,15]},
'sd':{'init':[0.5,2]}
}

def log_prior(theta): 
    lp = norm.logpdf(x=theta['mu'], loc=10, scale=1)   
    lp += gamma.logpdf(x=theta['sd'], a=1, scale=1)
    return lp

def log_likelihood(data,theta):
    lp = sum(norm.logpdf(x=data, loc=theta['mu'], scale=theta['sd']))
    return lp


# initialize model object
model = de.Model(log_likelihood,log_prior,theta)

# run the model
samples = de.sample(data, model, num_samples=num_samples)

theta = samples['samples']
theta_mean = np.zeros([np.shape(theta)[1]])
for i in range(np.shape(theta)[1]):
    theta_mean[i] = np.mean(theta[:,i,300:400])
    d = dict(zip(model.theta,theta_mean))
print(d)

plt.plot(np.transpose(theta[:,0,:]))

