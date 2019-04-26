import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
import pyDE as de

def norm_subj():
    ##### Simulate data from normal distribution
    mu_mu = 10
    mu_sd = 1
    sd_mu = 1
    sd_sd = 1
    mu = norm.rvs(size=1,loc=mu_mu,scale=mu_sd)
    sd = de.truncnorm_rvs(size=1,loc=sd_mu,scale=sd_sd,a=0,b=float('inf'))
    sim_dat = norm.rvs(size=200,loc=mu,scale=sd)
    data = {'response':sim_dat}
    
    return data

data = []
num_subj = 10
for i in range(num_subj):
    data.append(norm_subj())

#### MCMC parameters
num_samples = 800
de_params={'b':.001, 
           'rand_phi':True,  # omit dependence between group and subject-level
           'migrate_start':300, # turn migration off by setting negative
           'migrate_end':400,
           'migrate_step':20}

#### Model parameters
#subject-level
theta = {
'mu':{'init':[1,10]},
'sd':{'init':[0.1,5]},
}
#group-level
phi = {
'mu_mu':{'init':[5,15],'block':1},
'mu_sd':{'init':[.1,5],'block':1},
'sd_a':{'init':[.1,5],'block':2},
'sd_scale':{'init':[.1,5],'block':2},         
}


#### Define model
def log_hyperprior(phi):
        
    lp = norm.logpdf(x=phi['mu_mu'], loc=3, scale=3)
    lp += gamma.logpdf(x=phi['mu_sd'], a=1, scale=1)
    
    lp += gamma.logpdf(x=phi['sd_a'], a=1, scale=1)
    lp += gamma.logpdf(x=phi['sd_scale'], a=1, scale=1)

    return lp

def log_prior(theta,phi):
       
    lp = norm.logpdf(x=theta['mu'], loc=phi['mu_mu'], scale=phi['mu_sd'])
    lp += gamma.logpdf(x=theta['sd'], a=phi['sd_a'], scale=phi['sd_scale'])
    
    return lp


def log_likelihood(data,theta):
    
    lp = sum(norm.logpdf(x=data['response'],loc=theta['mu'],scale=theta['sd']))

    return lp
           

model = de.Model(log_likelihood,log_prior,theta,log_hyperprior,phi)

samples = de.sample(data, model, num_samples=num_samples, de_params=de_params)

theta_samples = samples['theta_samples']
phi_samples = samples['phi_samples']

plt.plot(np.transpose(phi_samples[:,0,:]))
