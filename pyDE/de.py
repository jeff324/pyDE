import math
import numpy as np
from pyDE import model as model_class

def crossover(cur_chain, n_chains, b, cur_theta):
     # set gamma parameter
     gamma = 2.38/math.sqrt(2*len(cur_theta[cur_chain,:]))
     pop = np.arange(0,n_chains)
     pop = np.delete(pop,cur_chain)
     index = np.random.choice(pop, size=2, replace=False)			
     x_update = cur_theta[cur_chain,:] + gamma*(cur_theta[index[0],:]-cur_theta[index[1],:]) + np.random.uniform(-b,b)   
     return x_update    

def accept(cur_weight,weight):
     try:
         alpha = math.exp(weight-cur_weight)
     except OverflowError:
         alpha = float('inf')
     if np.random.uniform() < alpha:							
         return True
     else:
         return False
           
def migrate(x,cur_weight):

    new_x = np.copy(x)
    new_weight = np.copy(cur_weight)
    n_chains = len(cur_weight)
    num_migration_chains = np.random.randint(n_chains)
    use_chains = np.random.choice(np.arange(0,n_chains),num_migration_chains,replace=False)

    for i in range(num_migration_chains):

        #get weight of random chain
        migration_cur_weight = cur_weight[use_chains[i]]
    
        #go to the previous chain in use_chains
        new_chain = i - 1 
        #if there is no previous chain, then go to the last chain
        if new_chain == -1:
            new_chain = num_migration_chains - 1
        
        migration_weight = cur_weight[use_chains[new_chain]]
        
        #move the chain given acceptance probability
        if accept(migration_cur_weight,migration_weight):          	
            new_x[use_chains[i],:] = new_x[use_chains[new_chain],:]
            new_weight[use_chains[i]] = new_weight[use_chains[new_chain]]
            
    return [new_x,new_weight]

    


def sample_hier(data, model, num_samples, n_chains, de_params, update):

    n_pars = len(model.theta)
    n_hpars = len(model.phi)
    n_subj = len(data)
    n_blocks = len(model.blocks)
    migrate_start = de_params['migrate_start']
    migrate_end = de_params['migrate_end']
    migrate_step = de_params['migrate_step']
    rand_phi = de_params['rand_phi']
    
    if n_chains is None:
        max_block = max(list(map(len,model.blocks)))
        if max_block > n_pars:
            n_chains = 2*max_block
        else:           
            n_chains = 2*n_pars
    theta = np.zeros([n_chains,n_pars,num_samples,n_subj])
    phi = np.zeros([n_chains,n_hpars,num_samples])
    weight_theta = np.zeros([n_chains,num_samples,n_subj]) - float('inf')
    weight_phi = np.zeros([n_chains,num_samples]) - float('inf')
    
    # initialize theta chains
    for s in range(n_subj):
        for k in range(n_chains):
            while weight_theta[k,0,s] == -float('inf'):
                for p in range(n_pars):
                        theta[k,p,0,s] = model.initializer(model.theta[p])                        
                x_theta = dict(zip(model.theta,theta[k,:,0,s]))
                weight_theta[k,0,s] = model.log_likelihood(data[s],x_theta)
    
    #initialize phi chains
    for k in range(n_chains):
        while weight_phi[k,0] == -float('inf'):
            for p in range(n_hpars):    
                phi[k,p,0] = model.initializer(model.phi[p])
            x_phi = dict(zip(model.phi,phi[k,:,0]))
            lp = 0
            for s in range(n_subj):
                x_theta = dict(zip(model.theta,theta[k,:,0,s]))
                lp += model.log_prior(x_theta,x_phi)
            weight_phi[k,0] = lp + model.log_hyperprior(x_phi)
                   
    # run DE-MCMC
    chain_idx = np.arange(0,n_chains)
    for i in range(1,num_samples):
        if (i+1) % update == 0:
            print('\n' + str(i+1) + '/' + str(num_samples))
        
        #sample phi
        if rand_phi:            
             chain_idx = np.arange(0,n_chains)
             chain_idx = np.random.choice(chain_idx, size=len(chain_idx), replace=False)     
             
        phi[:,:,i] = phi[:,:,i-1]
        for p in range(n_blocks):
            par_range = model.blocks[p]
            #migration step
            if (i > migrate_start) and (i < migrate_end) and (i % migrate_step == 0):
                phi_constant = np.copy(phi[:,:,i])
                weight_constant = []
                for k in range(n_chains):
                    #fix all parameters except current parameter block across all chains
                    all_pars = np.arange(len(phi[k,:,i]))
                    anti_par_block = np.delete(all_pars,par_range)
                    phi_constant[k,anti_par_block] = phi_constant[0,anti_par_block]
                    #update weight for each chain
                    x_phi = dict(zip(model.phi,phi_constant[k,:]))
                    lp = 0 
                    for s in range(n_subj):
                        x_theta = dict(zip(model.theta,theta[chain_idx[k],:,i-1,s]))
                        lp += model.log_prior(x_theta,x_phi)
                    weight_constant.append(lp + model.log_hyperprior(x_phi))              
                phi[:,par_range,i] = migrate(phi[:,par_range,i],weight_constant)[0]
                #if we have updated all blocks then
                #recompute weight for updated phi
                if p == (n_blocks-1):                    
                    for k in range(n_chains):
                        #update weight for each chain
                        x_phi = dict(zip(model.phi,phi[k,:,i]))
                        lp = 0 
                        for s in range(n_subj):
                            x_theta = dict(zip(model.theta,theta[chain_idx[k],:,i-1,s]))
                            lp += model.log_prior(x_theta,x_phi)
                        weight_phi[k,i] = lp + model.log_hyperprior(x_phi)                     
            else:
                #crossover step
                for k in range(n_chains):                
                    temp = np.copy(phi[k,:,i])
                    temp[par_range] = crossover(cur_chain=k, 
                                                  n_chains=n_chains, 
                                                  b=de_params['b'], 
                                                  cur_theta=phi[:,par_range,i])               
                    x_phi = dict(zip(model.phi,temp))
                    lp = 0 
                    for s in range(n_subj):
                        x_theta = dict(zip(model.theta,theta[chain_idx[k],:,i-1,s]))
                        lp += model.log_prior(x_theta,x_phi)
                    weight = lp + model.log_hyperprior(x_phi)
                    if accept(weight_phi[k,i-1],weight):
                        phi[k,par_range,i] = temp[par_range]
                        weight_phi[k,i] = weight
                    else:
                        weight_phi[k,i] = weight_phi[k,i-1]

        #sample theta
        if rand_phi:            
             chain_idx = np.arange(0,n_chains)
             chain_idx = np.random.choice(chain_idx, size=len(chain_idx), replace=False)     
             
        for s in range(n_subj):
            if (i > migrate_start) and (i < migrate_end) and (i % migrate_step == 0):
                m_out = migrate(theta[:,:,i-1,s],weight_theta[:,i-1,s])
                theta[:,:,i,s] = m_out[0]
                weight_theta[:,i,s] = m_out[1]
            else:
                for k in range(n_chains):                
                    temp = crossover(cur_chain=k, 
                                       n_chains=n_chains, 
                                       b=de_params['b'], 
                                       cur_theta=theta[:,:,i-1,s])
                    x_theta = dict(zip(model.theta,temp))                
                    x_phi = dict(zip(model.phi,phi[chain_idx[k],:,i]))
                    weight = model.log_likelihood(data[s],x_theta) + model.log_prior(x_theta,x_phi) 
                    if accept(weight_theta[k,i-1,s],weight):
                        theta[k,:,i,s] = temp
                        weight_theta[k,i,s] = weight
                    else:
                        theta[k,:,i,s] = theta[k,:,i-1,s]
                        weight_theta[k,i,s] = weight_theta[k,i-1,s]
            
    return {'theta_samples':theta,'phi_samples':phi}

def sample_ind(data,model,num_samples,n_chains,de_params,update):
    n_pars = len(model.theta)
    if n_chains is None:
        n_chains = 2*n_pars
    theta = np.zeros([n_chains,n_pars,num_samples])
    weight_theta = np.zeros([n_chains,num_samples]) - float('inf')   
    
    # initialize chains
    for i in range(n_chains):
        while weight_theta[0,i] == -float('inf'):
            for k in range(n_pars):
                    theta[i,k,0] = model.initializer(model.theta[k])
                    x = dict(zip(model.theta,theta[i,:,0]))
            weight_theta[0,i] = model.log_likelihood(data,x) + model.log_prior(x)
                   
    # run DE-MCMC
    for i in range(1,num_samples):
        if (i+1) % update == 0:
            print('\n' + str(i+1) + '/' + str(num_samples))
        for k in range(n_chains):
            temp = crossover(cur_chain=k, 
                               n_chains=n_chains, 
                               b=de_params['b'], 
                               cur_theta=theta[:,:,i-1])
            x = dict(zip(model.theta,temp))
            weight = model.log_likelihood(data,x) + model.log_prior(x) 
            if accept(weight_theta[k,i-1],weight):
                theta[k,:,i] = temp
                weight_theta[k,i] = weight
            else:
                theta[k,:,i] = theta[k,:,i-1]
                weight_theta[k,i] = weight_theta[k,i-1]
            
    return {'samples':theta}


def sample(data, model, num_samples=500, n_chains=None, update=10, de_params={'b':.001,'rand_phi':True, 
                                                                   'migrate_start':-1, 'migrate_end':-1,'migrate_step':-1}):

    if model.model_type == 'hierarchical':
        samples = sample_hier(data,model,num_samples=num_samples,n_chains=n_chains,de_params=de_params,update=update)
    
    if model.model_type == 'individual':
        samples = sample_ind(data,model,num_samples=num_samples,n_chains=n_chains,de_params=de_params,update=update)

    return samples

