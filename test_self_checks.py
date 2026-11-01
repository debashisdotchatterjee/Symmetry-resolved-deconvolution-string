"""Verification tests of key statistical identities; NO real data involved."""
import numpy as np
from inverse_core import (fourier_components,reconstruct,estimate,normal_mixture_pdf,
                          sample_signs,choose_cutoff,ise)

def run():
    rg=np.random.default_rng(99)
    n=7000
    U=rg.normal(loc=.75,scale=.45,size=n)
    t=np.linspace(0,2.5,51)
    q=np.full(n,.7);sigma=np.full(n,.18)
    Y=sample_signs(q,rg)*U+rg.normal(0,sigma)
    c,s,ie,io=fourier_components(Y,sigma,q,t)
    c_theory=np.exp(-.5*(.45*t)**2)*np.cos(.75*t)
    s_theory=np.exp(-.5*(.45*t)**2)*np.sin(.75*t)
    assert np.max(np.abs(c-c_theory))<.06,(c-c_theory).max()
    assert np.max(np.abs(s-s_theory))<.08,(s-s_theory).max()
    assert np.allclose(ie,n*np.exp(-(.18*t)**2),rtol=1e-10)
    assert np.allclose(io,.7**2*ie,rtol=1e-10)
    # Balanced case kills odd recovery irrespective of observed sample.
    z=estimate(Y,sigma,np.zeros(n),np.linspace(-2,2,201),T=2,nt=50)
    assert np.max(np.abs(z['odd']))<1e-12
    for sector in ('even','full'):
        T,meta=choose_cutoff(sigma,q,sector=sector)
        assert .4<=T<=12 and np.isfinite(meta['upper_proxy'])
    # Even/odd parity: f_e symmetric and f_o antisymmetric.
    x=np.linspace(-2,2,201);zz=estimate(Y,sigma,q,x,2.2,nt=80)
    assert np.max(abs(zz['even']-zz['even'][::-1]))<1e-10
    assert np.max(abs(zz['odd']+zz['odd'][::-1]))<1e-10
    print('PASS: sign model, Fourier weighting, information, parity, balanced identifiability, cutoffs')

if __name__=='__main__':run()
