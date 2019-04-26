import numpy as np
import pyDE as de

##### Simulate Data From LBA
sim_dat_1 = de.lba_rvs(n=400,b=2.5,A=2,v=[4,1],s=[1,1],t0=.3)
sim_dat_2 = de.lba_rvs(n=400,b=2.5,A=2,v=[2,1],s=[1,1],t0=.3)
sim_dat_1 = np.column_stack((sim_dat_1['resp'],sim_dat_1['rt']))
sim_dat_2 = np.column_stack((sim_dat_2['resp'],sim_dat_2['rt']))
sim_dat_1 = sim_dat_1[sim_dat_1[:,1]<6,:] #omit large RTs
sim_dat_2 = sim_dat_2[sim_dat_2[:,1]<6,:] #omit large RTs

#### Define Data Structure
data = {
'rt': np.concatenate((sim_dat_1[:,1],sim_dat_2[:,1])),
'condition': np.concatenate((np.repeat(0,len(sim_dat_1[:,0])),
                             np.repeat(1,len(sim_dat_2[:,0])))),
'response' : np.concatenate((sim_dat_1[:,0],sim_dat_2[:,0]))
}

#### Define Parameters and Initial Guess Interval [a,b] 
theta = {
'A':{'init':[.5,2]},
'b':{'init':[.2,2]},
't0':{'init':[.1,.75]},
'v1.0':{'init':[2,6]},
'v1.1':{'init':[1,3]},
'v2':{'init':[.5,2]},
'sv2':{'init':[.5,1.5]}
}

#### Define Model
def log_prior(pars):
    
    lp = de.truncnorm_logpdf(x=pars['A'], loc=1, scale=3)
    lp += de.truncnorm_logpdf(x=pars['b'], loc=.5, scale=3)
    lp += de.truncnorm_logpdf(x=pars['t0'], loc=.3, scale=.3)
    lp += de.truncnorm_logpdf(x=pars['v1.0'], loc=4, scale=3)
    lp += de.truncnorm_logpdf(x=pars['v1.1'], loc=2, scale=2)
    lp += de.truncnorm_logpdf(x=pars['v2'], loc=2, scale=2)
    lp += de.truncnorm_logpdf(x=pars['sv2'], loc=1, scale=1)
    
    return lp

def log_likelihood(data,theta):   
    rt = data['rt']
    response = data['response']
    cond = data['condition']  
    lp = 0
    for i in range(2):                         
        lp += sum(de.lba_logpdf(rt=rt[cond==i],response=response[cond==i],
                                b=theta['b']+theta['A'],
                                A=theta['A'],
                                v=[theta[de.idx('v1',i)],theta['v2']],
                                s=[1,theta['sv2']],
                                t0=theta['t0']))
    return lp

#### Initialize a pyDE Model object       
model = de.Model(log_likelihood, log_prior, theta)

#### Sample with DE-MCMC 
num_samples = 500
samples = de.sample(data, model, num_samples=num_samples)

# Check the means of each parameter. 
# Should be close to those used for the simulated data.
theta = samples['samples']
theta_mean = np.zeros([np.shape(theta)[1]])
for i in range(np.shape(theta)[1]):
    theta_mean[i] = np.mean(theta[:,i,300:400])
    d = dict(zip(model.theta,theta_mean))
print(d)