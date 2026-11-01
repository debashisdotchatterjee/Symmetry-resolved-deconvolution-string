"""Independent simulation experiments: phase transition, heterogeneity, calibration."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from inverse_core import (sample_signs,normal_mixture_pdf,estimate,ise,
                          choose_cutoff)
from plotting import setup,savefig,table_csv_tex

MU=np.array([1.10,-0.55]);SD=np.array([.38,.28]);WT=np.array([.67,.33])

def latent(rng,n):
    a=rng.choice(len(MU),size=n,p=WT)
    return rng.normal(MU[a],SD[a])

def designs(n):
    j=np.arange(1,n+1,dtype=float)
    return {'fully_oriented':np.ones(n),
            'power_1_4':.7*j**(-.25),
            'power_1_2':.7*j**(-.50),
            'power_3_4':.7*j**(-.75),
            'triangular_boundary':np.full(n,n**(-.5)),
            'fully_balanced':np.zeros(n)}

def execute(out,profile='standard',seed=20261102):
    setup();out=Path(out);figdir=out/'figures';tabdir=out/'tables'
    cfg={'smoke':([90],3,85),'quick':([200,650],20,125),
         'standard':([250,750,1500],65,151),
         'full':([250,750,1500,3500],250,231)}
    ns,B,nt=cfg[profile]
    root=np.random.SeedSequence(seed)
    x=np.linspace(-2.6,2.6,311)
    f=normal_mixture_pdf(x,MU,SD,WT)
    fm=normal_mixture_pdf(-x,MU,SD,WT)
    f_even=.5*(f+fm); f_odd=.5*(f-fm)
    rows=[];allcurves={}
    for n in ns:
        design=designs(n)
        for name,q in design.items():
            # Precommitted same regularizer for all Monte Carlo repetitions within design.
            sigma=np.full(n,.32)
            if np.sum(q*q)<1e-8:
                T,info=choose_cutoff(sigma,q,beta=1.6,L=1.5,sector='even',tmax=9)
            else:
                T,info=choose_cutoff(sigma,q,beta=1.6,L=1.5,sector='full',tmax=9)
            seql=[]
            for b,ss in enumerate(root.spawn(B)):
                rng=np.random.default_rng(ss)
                u=latent(rng,n); y=sample_signs(q,rng)*u+rng.normal(0,.32,n)
                z=estimate(y,sigma,q,x,T,nt=nt)
                seql.append((ise(x,z['full'],f),ise(x,z['even'],f_even),
                             ise(x,z['odd'],f_odd)))
                if b==0: allcurves[(n,name)]=z['full']
            a=np.array(seql)
            for metric,idx in [('full_ise',0),('even_ise',1),('odd_ise',2)]:
                rows.append({'n':n,'design':name,'Q_n':np.sum(q*q),'cutoff_T':T,
                             'metric':metric,'mean':a[:,idx].mean(),
                             'mc_se':a[:,idx].std(ddof=1)/np.sqrt(B) if B>1 else np.nan,
                             'median':np.median(a[:,idx]),'replications':B})
        print(f'  Simulated n={n} ({B} replicates per design)',flush=True)
    res=pd.DataFrame(rows);table_csv_tex(res,tabdir,'S01_phase_transition_long')
    fig,ax=plt.subplots(1,3,figsize=(13.6,3.8),constrained_layout=True)
    palette=['#295a9a','#38a169','#d77c31','#8b57ac','#7f8791','#222222']
    order=list(designs(ns[-1]));metrics=['full_ise','even_ise','odd_ise']
    for k,metric in enumerate(metrics):
        for name,color in zip(order,palette):
            data=res[(res.design==name)&(res.metric==metric)]
            ax[k].errorbar(data.n,data['mean'],yerr=1.96*data.mc_se,marker='o',capsize=2,
                           label=name.replace('_',' '),color=color,lw=1.3)
        ax[k].set_xlabel('Number of observations');ax[k].set_ylabel('Integrated squared error')
        ax[k].set_title(['Oriented density','Even component','Odd component'][k]);
        ax[k].grid(alpha=.16)
    handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.06),fontsize=8)
    savefig(fig,figdir,'S01_information_phase_transition')
    n=ns[-1]; fig,axs=plt.subplots(2,3,figsize=(12.8,7),sharex=True,sharey=True,constrained_layout=True)
    for a,(name,_) in zip(axs.flat,designs(n).items()):
        a.plot(x,f,'k--',lw=2.2,label='True oriented density')
        a.plot(x,f_even,color='#607d8b',lw=1.5,label='Invariant component')
        a.plot(x,allcurves[(n,name)],color='#bd4c34',lw=1.4,label='Estimate (one replicate)')
        a.set_title(name.replace('_',' '));a.grid(alpha=.18)
    axs[-1,1].set_xlabel('Latent variable');axs[0,0].set_ylabel('Density');axs[1,0].set_ylabel('Density')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.035))
    savefig(fig,figdir,'S02_representative_reconstructions')
    # Heterogeneous errors: constant unknown target, varying sigma and known q.
    n=900 if profile not in ('smoke','quick') else 240
    B2={'smoke':3,'quick':30,'standard':110,'full':350}[profile]
    rg=np.random.default_rng(seed+17)
    j=np.arange(n); sigma=np.select([j%3==0,j%3==1],[.18,.34],default=.65).astype(float)
    q=np.select([j%3==0,j%3==1],[.88,-.45],default=.16).astype(float)
    T,_=choose_cutoff(sigma,q,beta=1.6,L=1.5,tmax=10)
    r=[];represent={}
    for b in range(B2):
        rng=np.random.default_rng(rg.integers(0,2**63-1))
        u=latent(rng,n); y=sample_signs(q,rng)*u+rng.normal(0,sigma)
        for meth in ('proposed','ignore_sign','pooled_sign','invariant'):
            z=estimate(y,sigma,q,x,T,nt=nt,method=meth)
            r.append({'rep':b,'method':meth,'full_ise':ise(x,z['full'],f),
                      'odd_ise':ise(x,z['odd'],f_odd),'T':T})
            if b==0:represent[meth]=z['full']
    het=pd.DataFrame(r);table_csv_tex(het,tabdir,'S02_heterogeneous_replications')
    hetgroup=het.groupby('method')[['full_ise','odd_ise']].agg(['mean','std']).reset_index()
    hetgroup.columns=['_'.join(j).strip('_') for j in hetgroup.columns]
    table_csv_tex(hetgroup,tabdir,'S03_heterogeneous_summary')
    fig,axs=plt.subplots(1,2,figsize=(11.7,4),constrained_layout=True)
    for meth,color in [('proposed','#096a89'),('ignore_sign','#cc5139'),
                        ('pooled_sign','#8567aa'),('invariant','#7c7e80')]:
        axs[0].plot(x,represent[meth],label=meth.replace('_',' '),color=color)
    axs[0].plot(x,f,'k--',lw=2,label='Truth');axs[0].set_title('Heterogeneous error: one replicate')
    axs[0].set_xlabel('Latent variable');axs[0].set_ylabel('Reconstructed density');axs[0].legend(fontsize=8)
    means=het.groupby('method').full_ise.mean().sort_values();std=het.groupby('method').full_ise.std().reindex(means.index)
    axs[1].barh(np.arange(len(means)),means,xerr=1.96*std/np.sqrt(B2),color='#54839b',alpha=.85)
    axs[1].set_yticks(np.arange(len(means)),[m.replace('_',' ') for m in means.index]);axs[1].invert_yaxis()
    axs[1].set_xlabel('Mean integrated squared error (95% MC interval)')
    axs[1].set_title('Comparison under identical cutoff')
    savefig(fig,figdir,'S03_heterogeneous_comparison')
    # Calibration misspecification; mathematically exact multiplier also tabulated.
    mis=[];nr=800 if profile not in ('smoke','quick') else 230
    j=np.arange(nr); sig=np.full(nr,.32);q=.2+.65*(j/nr)
    for fac in (.65,.85,1.,1.15,1.35):
        qt=np.clip(fac*q,-1,1);T,_=choose_cutoff(sig,qt,beta=1.6,L=1.5,tmax=8)
        for b in range({'smoke':2,'quick':15,'standard':70,'full':200}[profile]):
            rng=np.random.default_rng(seed+int(1000*fac)+b)
            u=latent(rng,nr); y=sample_signs(q,rng)*u+rng.normal(0,sig)
            z=estimate(y,sig,qt,x,T,nt=nt)
            mis.append({'factor':fac,'rep':b,'signed_ise':ise(x,z['odd'],f_odd),'T':T,
                        'calibration_factor_at_zero':np.sum(q*qt)/np.sum(qt*qt)})
    cal=pd.DataFrame(mis);table_csv_tex(cal,tabdir,'S04_calibration_sensitivity')
    calgroup=cal.groupby('factor',as_index=False).agg(mean_ise=('signed_ise','mean'),
        sd_ise=('signed_ise','std'),expected_amplitude=('calibration_factor_at_zero','mean'))
    table_csv_tex(calgroup,tabdir,'S05_calibration_summary')
    fig,ax=plt.subplots(figsize=(7.1,4),constrained_layout=True)
    ax.plot(calgroup.factor,calgroup.mean_ise,'o-',color='#245b8d')
    ax.axvline(1,color='k',ls='--',alpha=.6);ax.set_xlabel('Assumed / true channel imbalance')
    ax.set_ylabel('Odd-component ISE');ax.set_title('Sensitivity to channel calibration');ax.grid(alpha=.2)
    savefig(fig,figdir,'S04_calibration_sensitivity')
    return {'n_designs':len(designs(ns[-1])),'sample_sizes':ns,'repetitions':B,
            'heterogeneity_repetitions':B2,'note':'Simulation results are Monte Carlo estimates, not predetermined outcomes.'}
