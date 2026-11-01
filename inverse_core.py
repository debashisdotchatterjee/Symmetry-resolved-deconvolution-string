"""Symmetry-resolved Fourier inversion; aligned with the accompanying manuscript.

No unrecorded labels are inferred. q_j=2p_j-1 must be externally known or
specified by a controlled masking design. Gaussian additive noise is assumed.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm

EPS=1e-12

def make_grid(y, sigma, n=321):
    y=np.asarray(y,float); sigma=np.asarray(sigma,float)
    r=float(max(np.quantile(np.abs(y), .995)*1.45, 4*np.median(sigma), .1))
    return np.linspace(-r,r,n)

def fourier_components(y, sigma, q, t, chunk=2500, method='proposed'):
    """Cosine/sine empirical CF, with distinct normal-equation denominators.

    method: proposed, ignore_sign, pooled_sign, invariant. Values for t >= 0.
    The invariant sector is estimated for all models, even when Q=0.
    """
    y=np.asarray(y,float); sigma=np.asarray(sigma,float); q=np.asarray(q,float)
    t=np.asarray(t,float)
    assert y.ndim==sigma.ndim==q.ndim==t.ndim==1
    assert len(y)==len(sigma)==len(q)>1 and t[0]>=0 and np.all(np.diff(t)>0)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(sigma)) and np.all(sigma>0)
    assert np.all(np.isfinite(q)) and np.all(np.abs(q)<=1+1e-12)
    ec=np.zeros(len(t)); es=np.zeros(len(t)); de=np.zeros(len(t)); do=np.zeros(len(t));
    # Avoid materializing n-by-grid matrices for very large datasets.
    for a in range(0,len(y),chunk):
        sl=slice(a,min(a+chunk,len(y)))
        tmat=t[None,:]
        psi=np.exp(-.5*(sigma[sl,None]*tmat)**2)
        phase=y[sl,None]*tmat
        ec += np.sum(psi*np.cos(phase), axis=0)
        de += np.sum(psi**2,axis=0)
        if method=='proposed':
            es+=np.sum((q[sl,None]*psi)*np.sin(phase),axis=0)
            do+=np.sum((q[sl,None]*psi)**2,axis=0)
        elif method=='ignore_sign':
            es+=np.sum(psi*np.sin(phase),axis=0)
            do+=np.sum(psi**2,axis=0)
        elif method=='pooled_sign':
            # An intentionally misspecified benchmark: ignores the joint sigma/q structure.
            es+=np.sum(psi*np.sin(phase),axis=0)
            do+=np.sum(psi**2,axis=0)
        elif method=='invariant':
            pass
        else:
            raise ValueError(method)
    c=np.divide(ec,de,out=np.zeros_like(ec),where=de>EPS)
    s=np.zeros_like(es)
    if method!='invariant':
        s=np.divide(es,do,out=np.zeros_like(es),where=do>EPS)
        if method=='pooled_sign':
            qbar=q.mean()
            s=s/qbar if abs(qbar)>1e-4 else np.zeros_like(s)
    c[0]=1.
    s[0]=0.
    return c,s,de,do

def reconstruct(c,s,t,x):
    # f_e(x)=(1/pi) integral_0^T c(t)cos(tx)dt
    # f_o(x)=(1/pi) integral_0^T s(t)sin(tx)dt
    phase=np.outer(t,x)
    fe=trapezoid(c[:,None]*np.cos(phase),t,axis=0)/np.pi
    fo=trapezoid(s[:,None]*np.sin(phase),t,axis=0)/np.pi
    return fe+fo,fe,fo

def choose_cutoff(sigma,q,tmax=12,beta=1.6,L=1.0,n_grid=75,sector='full'):
    """Pre-declared regularization rule: minimize *theoretical upper-risk proxy*.

    L is a user-selected smoothness-budget tuning constant (NOT data-estimated).
    Output includes the objective for full transparency. Hard upper bound is a
    convenient selection device, not a claim of adaptive minimaxity.
    """
    sigma=np.asarray(sigma,float);q=np.asarray(q,float)
    t=np.linspace(0,tmax,max(200,n_grid*4))
    ie=np.zeros_like(t); io=np.zeros_like(t)
    for a in range(0,len(q),5000):
        tmp=np.exp(-(sigma[a:a+5000,None]*t[None,:])**2)
        ie+=np.sum(tmp,axis=0)
        io+=np.sum(q[a:a+5000,None]**2*tmp,axis=0)
    var=1/np.maximum(ie,EPS)
    if sector=='full': var+=1/np.maximum(io,EPS)
    elif sector!='even':raise ValueError(sector)
    integ=np.r_[0,np.cumsum(.5*(var[1:]+var[:-1])*np.diff(t))]/np.pi
    obj=L*(1+t**2)**(-beta)+integ
    cand=(t>=.4)&(t<=tmax)
    j=np.flatnonzero(cand)[np.argmin(obj[cand])]
    return float(t[j]), {'T':float(t[j]),'beta_proxy':float(beta),'L_proxy':float(L),
                         'upper_proxy':float(obj[j]),'I_e_T':float(ie[j]),'I_o_T':float(io[j]),
                         'Q':float(np.sum(q*q))}

def estimate(y,sigma,q,x,T,nt=175,method='proposed'):
    t=np.linspace(0,float(T),int(nt))
    c,s,ie,io=fourier_components(y,sigma,q,t,method=method)
    f,fe,fo=reconstruct(c,s,t,x)
    return {'full':f,'even':fe,'odd':fo,'invariant_positive':2*fe,
            't':t,'c':c,'s':s,'Ie':ie,'Io':io}

def normal_mixture_pdf(x,mus,sd,weight):
    z=np.zeros_like(np.asarray(x,float))
    for a,b,c in zip(mus,sd,weight):z+=c*norm.pdf(x,loc=a,scale=b)
    return z

def ise(x,y,truth):return float(trapezoid((y-truth)**2,x))

def project_density(x,density):
    """Optional visual positivity projection (not part of proven estimator)."""
    den=np.clip(density,0,None)
    area=trapezoid(den,x)
    return den/area if area>EPS else np.zeros_like(den)

def sample_signs(q,rng):
    return np.where(rng.random(len(q))<(.5+.5*np.asarray(q)),1.,-1.)

def robust_scale(v):
    v=np.asarray(v,float)
    return max(np.std(v), np.median(np.abs(v-np.median(v)))*1.4826, 1e-8)
