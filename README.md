# pyDE - Bayesian Inference with Differential-Evolution MCMC in Python

### Installation

```shell
git clone https://github.com/jef324/pyDE
python setup.py install
```

### Non-hierarchical Template

```python
# Define the parameters of the model
pars = {
}

# Define the prior
def log_prior(theta): 

# Define the liklihood
def log_likelihood(data,theta):

# initialize model object
model = de.Model(log_likelihood,log_prior,theta)

# run the model
samples = de.sample(data, model, num_samples=500)
```

### Non-hierarchical Normal Example

In this example, we will sample from the following model:

mu ~ N(10,3)
sd ~ Gamma(1,1)
data ~ N(mu,sd)

We first import some additional packages for plotting (`matplotlib`), analysis (`numpy`),
and distributions (`scipy`). We then simulate some data from a normal distribution.

```python
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
```
Next, we define the model. The model takes three inputs, a parameters block (`theta` in this example),
a funciton for the log prior, and a function for the log likelihood.

```python
# Define the model
theta = {
'mu':{'init':[5,15]},
'sd':{'init':[0.5,2]}
}

def log_prior(theta): 
    lp = norm.logpdf(x=theta['mu'], loc=10, scale=3)   
    lp += gamma.logpdf(x=theta['sd'], a=1, scale=1)
    return lp

def log_likelihood(data,theta):
    lp = sum(norm.logpdf(x=data, loc=theta['mu'], scale=theta['sd']))
    return lp
```
Once the model is defined, it easy to sample from it. Just initialize the model
and run the DE sampler.

```python
# initialize model object
model = de.Model(log_likelihood,log_prior,theta)

# run the model
samples = de.sample(data, model, num_samples=num_samples)
```
We can then check the samples and plot them.
```python
theta = samples['samples']

# check the means of the samples
theta_mean = np.zeros([np.shape(theta)[1]])
for i in range(np.shape(theta)[1]):
    theta_mean[i] = np.mean(theta[:,i,300:400])
    d = dict(zip(model.theta,theta_mean))
print(d)

# plot the chains for mu
plt.plot(np.transpose(theta[:,0,:]))
```
![Alt text](images/norm_ind.jpg?raw=true "Example Output")

Next, we will show how to modify the non-hierarchical model to become a hierarchical model.

### Hierarchical Normal Example
The model we will be working with is as follows:

mu_mu ~ N(3,3)
mu_sd ~ Gamma(1,1)
sd_a ~ Gamma(1,1)
sd_scale ~ Gamma(1,1)
sd_{s} ~ Gamma(sd_a,sd_scale)
mu_{s} ~ N(mu_mu,mu_sd)
data_{s} ~ N(mu_{s},sd_{s})

First, we import packages as before and generate 10 simulated subjects, each
with 200 normally distributed data points.

```python
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
```
Setup the DE parameters and the number of samples to collect. `migrate_start` is a DE
parameter that indicates when migration should begin. It should ideally come after the chains 
have been burned in. `migrate_end` is when migration ends. `migrate_step` is the number
of itarations to wait between each migration step.

```python
#### MCMC parameters
num_samples = 1000
de_params={'b':.001, 
           'rand_phi':True,  # omit dependence between group and subject-level
           'migrate_start':400, # turn migration off by setting negative
           'migrate_end':600,
           'migrate_step':20}

```
We then define the model in terms of subject-level and group-level paramters.
Additionally, we specify a function that computes the log hyperprior.
```python
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
```      
Initialize the model as before and fun the DE sampler.     
```python
model = de.Model(log_likelihood,log_prior,theta,log_hyperprior,phi)

samples = de.sample(data, model, num_samples=num_samples, de_params=de_params)
```
We get the subject-level and group-level parameters in seperate arrays.
```python
theta_samples = samples['theta_samples']
phi_samples = samples['phi_samples']

plt.plot(np.transpose(phi_samples[:,0,:]))
```