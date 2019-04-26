import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import truncnorm

def lba_logpdf(rt, response, b, A, v, s, t0):

    def fptpdf(z,x0max,chi,driftrate,sddrift):       
        if x0max<1e-10:
            out = (chi/np.power(z,2)) * norm.pdf(chi/z,loc=driftrate,scale=sddrift)
            return out
        zs = z*sddrift
        zu = z*driftrate
        chiminuszu = chi-zu
        chizu = chiminuszu/zs
        chizumax = (chiminuszu-x0max)/zs
        out = (driftrate*(norm.cdf(chizu)-norm.cdf(chizumax)) + sddrift*(norm.pdf(chizumax)-norm.pdf(chizu)))/x0max
        return out
        
    def fptcdf(z,x0max,chi,driftrate,sddrift):
        if x0max < 1e-10:
            return norm.cdf(chi/z,loc=driftrate,scale=sddrift)
        zs = z * sddrift
        zu = z * driftrate
        chiminuszu = chi - zu
        xx = chiminuszu - x0max
        chizu = chiminuszu / zs
        chizumax = xx / zs
        tmp1 = zs * (norm.pdf(chizumax)-norm.pdf(chizu))
        tmp2 = xx * norm.cdf(chizumax) - chiminuszu * norm.cdf(chizu)
        return 1 + (tmp1 + tmp2) / x0max
    
    def lba_pdf(t,x0max,chi,drift,sdI):
        G = 1-fptcdf(z=t,x0max=x0max[1],chi=chi[1],driftrate=drift[1],sddrift=sdI[1])
        out = G*fptpdf(z=t,x0max=x0max[0],chi=chi[0],driftrate=drift[0],sddrift=sdI[0])
        out = out / (1 - (norm.cdf(-drift[0]/sdI[0]) * norm.cdf(-drift[1]/sdI[1])))
        out[t<=0]=0
        return out
    
    def get_dens(rt, response, b, A, v, s, t0):
        out = np.zeros(len(rt))
        out[response==1] = lba_pdf(t=rt[response==1]-t0,x0max=[A,A],chi=[b,b],drift=v,sdI=s)
        out[response==2] = lba_pdf(t=rt[response==2]-t0,x0max=[A,A],chi=[b,b],drift=[v[1],v[0]],sdI=[s[1],s[0]])
        out = np.maximum(out,1e-10)
        return out
    
    return np.log(get_dens(rt, response, b, A, v, s, t0))
    

def lba_rvs(n,b,A,v,s,t0):
    drift_1 = norm.rvs(loc=v[0],scale=s[0],size=n)
    drift_2 = norm.rvs(loc=v[1],scale=s[1],size=n)        
    drift_1[drift_1 < 0] = 0
    drift_2[drift_2 < 0] = 0
    start_1 = np.array(uniform.rvs(loc=0,scale=A,size=n))
    start_2 = np.array(uniform.rvs(loc=0,scale=A,size=n))
    ttf_1 = (b-start_1) / drift_1
    ttf_2 = (b-start_2) / drift_2        
    rt = np.minimum(ttf_1,ttf_2) + t0
    ttf = np.column_stack((ttf_1,ttf_2))
    resp = np.argmin(ttf,axis=1) + 1 #1=v1 accumulator, 2=v2 accumulator
    return {'rt':rt,'resp':resp}


def truncnorm_logpdf(x,loc,scale,a=0,b=float('inf')):
    a = (a - loc) / (scale)
    b = (b - loc) / (scale)
    lp = truncnorm.logpdf(x=x,loc=loc,scale=scale,a=a,b=b)
    return lp

def truncnorm_rvs(size,loc,scale,a=0,b=float('inf')):
    a = (a - loc) / (scale)
    b = (b - loc) / (scale)
    rv = truncnorm.rvs(size=size,loc=loc,scale=scale,a=a,b=b)
    return rv