from scipy.stats import uniform 
import numpy as np

class Model:
      
    def __init__(self, log_likelihood, log_prior, theta, log_hyperprior=None, phi=None):    
        
        # non-hierarchical model
        if log_hyperprior is None or phi is None:
            
            self.log_likelihood = log_likelihood
            self.log_prior = log_prior
            self.theta = list(theta.keys())
            self.start_point_bounds = theta      
            self.model_type = 'individual'
            
        # hierarchical model
        else:
            
            self.log_likelihood = log_likelihood
            self.log_prior = log_prior
            self.log_hyperprior = log_hyperprior
            self.theta = list(theta.keys())
            self.phi = list(phi.keys())
            self.model_type = 'hierarchical'  
            self.start_point_bounds = self._get_bounds(theta,phi)
            self.blocks = self._get_blocks(phi)

           
    def _get_blocks(self,phi):
        block = []
        for i in range(len(phi)):
            block += [phi[list(phi.keys())[i]]['block']]
            
        block_list = []
        unique_blocks = np.unique(block)
        idx = np.arange(len(block))
        for i in unique_blocks:
            block_list += [idx[block == i]]
            
        return block_list
                
                
    def _get_bounds(self,theta,phi):
        pars = dict.copy(theta)
        pars.update(phi)
        return pars
        

    def initializer(self,par_name):
        lb = self.start_point_bounds[par_name]['init'][0]
        ub = self.start_point_bounds[par_name]['init'][1]
        rv = uniform.rvs(size=1,loc=lb,scale=ub-lb)
        return rv
        
    
    
        

    
    

        
    