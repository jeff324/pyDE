import numpy as np
import pyDE as de

def lba_subj():
    ##### Simulate data from LBA
    b = de.truncnorm_rvs(size=1,loc=.5,scale=.1)
    A = de.truncnorm_rvs(size=1,loc=2,scale=.1)
    s = de.truncnorm_rvs(size=1,loc=1,scale=.1)
    t0 = de.truncnorm_rvs(size=1,loc=.3,scale=.1)
    v1_0 = de.truncnorm_rvs(size=1,loc=4,scale=.1)
    v1_1 = de.truncnorm_rvs(size=1,loc=1,scale=.1)
    v2 = de.truncnorm_rvs(size=1,loc=1,scale=.1)
    
    sim_dat_1 = de.lba_rvs(n=200,b=b+A,A=A,v=[v1_0,v2],s=[1,s],t0=t0)
    sim_dat_2 = de.lba_rvs(n=200,b=b+A,A=A,v=[v1_1,v2],s=[1,s],t0=t0)
    sim_dat_1 = np.column_stack((sim_dat_1['resp'],sim_dat_1['rt']))
    sim_dat_2 = np.column_stack((sim_dat_2['resp'],sim_dat_2['rt']))
    sim_dat_1 = sim_dat_1[sim_dat_1[:,1]<6,:] #omit large RTs
    sim_dat_2 = sim_dat_2[sim_dat_2[:,1]<6,:] #omit large RTs
    
    data = {'rt': np.concatenate((sim_dat_1[:,1],sim_dat_2[:,1])),
            'condition': np.concatenate((np.repeat(0,len(sim_dat_1[:,0])),np.repeat(1,len(sim_dat_2[:,0])))),
            'response': np.concatenate((sim_dat_1[:,0],sim_dat_2[:,0]))}
    
    return data

data = []
num_subj = 30
for i in range(num_subj):
    data.append(lba_subj())

#number of mcmc samples
num_samples = 3000
de_params={'b':.001,
           'rand_phi':True, 
           'migrate_start':600, 
           'migrate_end':800,
           'migrate_step':20}

#### Define parameters
#subject-level
theta = {
'A':{'init':[0.5,3]},
'b':{'init':[0.5,2]},
't0':{'init':[.1,.5]},
'v1.0':{'init':[2,6]},
'v1.1':{'init':[2,6]},
'v2':{'init':[.5,2]},
'sv2':{'init':[.5,2]}, 
}
#group-level
phi = {
'b_mu':{'init':[.5,3],'block':1},
'b_sd':{'init':[.5,3],'block':1},
'A_mu':{'init':[.5,3],'block':2},
'A_sd':{'init':[0.5,3],'block':2},
't0_mu':{'init':[.1,1],'block':3},
't0_sd':{'init':[.1,1],'block':3},
'v1_mu.0':{'init':[1,5],'block':4},
'v1_sd.0':{'init':[.5,3],'block':4},
'v1_mu.1':{'init':[1,5],'block':4},
'v1_sd.1':{'init':[.5,3],'block':4},
'v2_mu':{'init':[.5,3],'block':5},
'v2_sd':{'init':[.5,3],'block':5},
'sv2_mu':{'init':[.2,3],'block':6},
'sv2_sd':{'init':[.5,3],'block':6}            
}


#### Define model
def log_hyperprior(phi):
        
    lp = de.truncnorm_logpdf(x=phi['A_mu'], loc=2, scale=1)
    lp += de.truncnorm_logpdf(x=phi['A_sd'], loc=2, scale=1)
    
    lp += de.truncnorm_logpdf(x=phi['b_mu'], loc=.5, scale=1)
    lp += de.truncnorm_logpdf(x=phi['b_sd'], loc=.5, scale=1)
    
    lp += de.truncnorm_logpdf(x=phi['t0_mu'], loc=.3, scale=.5)    
    lp += de.truncnorm_logpdf(x=phi['t0_sd'], loc=.3, scale=.5)      
        
    lp += de.truncnorm_logpdf(x=phi['v2_mu'], loc=1, scale=1)
    lp += de.truncnorm_logpdf(x=phi['v2_sd'], loc=1, scale=1)    
    
    lp += de.truncnorm_logpdf(x=phi['sv2_mu'], loc=1, scale=1)    
    lp += de.truncnorm_logpdf(x=phi['sv2_sd'], loc=1, scale=1)
    
    for i in range(2):
        lp += de.truncnorm_logpdf(x=phi[de.idx('v1_mu',i)], loc=3, scale=1)
        lp += de.truncnorm_logpdf(x=phi[de.idx('v1_sd',i)], loc=3, scale=1)
    
    return lp

def log_prior(theta,phi):
       
    lp = de.truncnorm_logpdf(x=theta['A'], loc=phi['A_mu'], scale=phi['A_sd'])
    lp += de.truncnorm_logpdf(x=theta['b'], loc=phi['b_mu'], scale=phi['b_sd'])
    lp += de.truncnorm_logpdf(x=theta['t0'], loc=phi['t0_mu'], scale=phi['t0_sd'])
    lp += de.truncnorm_logpdf(x=theta['v2'], loc=phi['v2_mu'], scale=phi['v2_sd'])
    lp += de.truncnorm_logpdf(x=theta['sv2'], loc=phi['sv2_mu'], scale=phi['sv2_sd'])
    for i in range(2):            
        lp += de.truncnorm_logpdf(x=theta[de.idx('v1',i)], loc=phi[de.idx('v1_mu',i)], scale=phi[de.idx('v1_sd',i)])
                   
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
           

model = de.Model(log_likelihood,log_prior,theta,log_hyperprior,phi)

samples = de.sample(data, model, num_samples=num_samples)

theta = samples['theta_samples']
phi = samples['phi_samples']
