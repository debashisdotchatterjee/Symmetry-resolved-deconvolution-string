"""Full real GWAS semi-synthetic application and reproducible sensitivity analysis.

All effect estimates and reported SEs are authentic; only sign masking is synthetic.
The available clean aligned summary statistics are noisy and not ground truth.
"""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from inverse_core import (estimate,choose_cutoff,sample_signs,make_grid,robust_scale,
                          ise)
from plotting import setup,savefig,table_csv_tex

def execute(gwas,out,seed=20261102,profile='standard'):
    setup();out=Path(out);rg=np.random.default_rng(seed+199)
    B={'smoke':2,'quick':12,'standard':45,'full':140}[profile]
    beta=gwas.beta.to_numpy(float); se=gwas.se.to_numpy(float)
    scale=robust_scale(beta)
    y0=beta/scale; sigma=se/scale;n=len(y0)
    x=make_grid(y0,sigma,361)
    # Fully aligned estimator used as a *noisy operational reference*, not f_X.
    Tref,ref_info=choose_cutoff(sigma,np.ones(n),beta=1.6,L=1.6,tmax=11)
    clean=estimate(y0,sigma,np.ones(n),x,Tref,nt=211)
    # Different design q_j, some of which correlate with the reported SE.
    ranks=pd.Series(sigma).rank(pct=True).to_numpy()
    designs={
       'fully_aligned':np.ones(n),
       'strongly_informative':np.full(n,.8),
       'moderately_informative':np.full(n,.45),
       'weakly_informative':np.full(n,.18),
       'fully_balanced':np.zeros(n),
       'heterogeneous_calibrated':.15+.8*(1-ranks),
       'mixed_channel_directions':np.where(ranks<.50,.68,-.55),
    }
    rows=[];represent={};cutoffs={}
    for des,q in designs.items():
        # One fixed conservative cutoff per design (no use of benchmark truth).
        sector='full' if np.sum(q*q)>1e-8 else 'even'
        T,info=choose_cutoff(sigma,q,beta=1.6,L=1.6,tmax=11,sector=sector)
        cutoffs[des]=info
        for b in range(B):
            yy=sample_signs(q,rg)*y0
            z=estimate(yy,sigma,q,x,T,nt=201)
            if b==0:represent[des]={'y':yy,'z':z,'T':T}
            rows.append({'design':des,'replicate':b,'n':n,'Q_n':np.sum(q*q),'cutoff_T':T,
                'mask_fraction_mean':(1-np.mean(q))/2,
                'reference_distance_L2':ise(x,z['full'],clean['full']),
                'even_reference_distance_L2':ise(x,z['even'],clean['even']),
                'odd_reference_distance_L2':ise(x,z['odd'],clean['odd']),
                'reconstructed_odd_L2':float(np.trapz(z['odd']**2,x)),
                'min_estimate_value':float(z['full'].min())})
        print(f'  GWAS masking: {des} ({B} replicates)',flush=True)
    exp=pd.DataFrame(rows);table_csv_tex(exp,out/'tables','R01_full_masking_experiment')
    agg=exp.groupby('design',as_index=False).agg(
        Q_n=('Q_n','first'),T=('cutoff_T','first'),
        mean_L2=('reference_distance_L2','mean'),sd_L2=('reference_distance_L2','std'),
        mean_even_L2=('even_reference_distance_L2','mean'),
        mean_odd_L2=('odd_reference_distance_L2','mean'),
        proportion_negative_minimum=('min_estimate_value',lambda s:float(np.mean(s<0))))
    table_csv_tex(agg,out/'tables','R02_masking_summary')
    fig,axs=plt.subplots(2,4,figsize=(15.3,7.2),constrained_layout=True,sharex=True)
    palette=['#1d6d94','#3e8c72','#ca7d39','#ad5e63','#697483','#765b9c','#4a8f9f']
    for a,(des,cc) in zip(axs.flat,zip(designs,palette)):
        z=represent[des]['z']
        a.plot(x,clean['full'],'k--',lw=2,label='Fully aligned deconvolution')
        a.plot(x,z['full'],color=cc,label='Mask + recover')
        a.plot(x,z['even'],color='#8a9a9d',ls=':',label='Even component')
        a.set_title(des.replace('_',' ')+'\n'+f"Q={np.sum(designs[des]**2):.0f}, T={represent[des]['T']:.2f}")
        a.grid(alpha=.17)
    axs[1,1].set_xlabel('Standardized signed effect');axs[0,0].set_ylabel('Density');axs[1,0].set_ylabel('Density')
    for unused in axs.flat[len(designs):]: unused.axis('off')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.035),fontsize=8)
    savefig(fig,out/'figures','R01_gwas_reconstructed_curves')
    fig,ax=plt.subplots(figsize=(9.4,4.3),constrained_layout=True)
    names=list(agg.design);ks=np.arange(len(names));w=.35
    ax.bar(ks-w/2,agg.mean_even_L2,w,label='Invariant component',color='#49769f')
    ax.bar(ks+w/2,agg.mean_odd_L2,w,label='Orientation-sensitive component',color='#c17d4f')
    ax.set_xticks(ks,[m.replace('_','\n') for m in names],fontsize=8)
    ax.set_ylabel('Squared L2 discrepancy relative to aligned noisy reference')
    ax.set_title('Real GWAS: discrepancy separated by symmetry sector');ax.legend()
    savefig(fig,out/'figures','R02_invariant_vs_odd_discrepancy')
    # Honest benchmark vs model misspecification under SAME cutoff and observations.
    bench=[];q=designs['heterogeneous_calibrated']
    T=cutoffs['heterogeneous_calibrated']['T']
    for b in range(B):
        yy=sample_signs(q,rg)*y0
        for meth in ('proposed','ignore_sign','pooled_sign','invariant'):
            z=estimate(yy,sigma,q,x,T,nt=201,method=meth)
            bench.append({'replicate':b,'method':meth,
                 'reference_distance_L2':ise(x,z['full'],clean['full']),
                 'odd_reference_distance_L2':ise(x,z['odd'],clean['odd'])})
    db=pd.DataFrame(bench);table_csv_tex(db,out/'tables','R03_benchmark_mask_replicates')
    benchsum=db.groupby('method',as_index=False).agg(
        mean_distance=('reference_distance_L2','mean'),
        sd_distance=('reference_distance_L2','std'),
        mean_odd_distance=('odd_reference_distance_L2','mean'))
    table_csv_tex(benchsum,out/'tables','R04_benchmark_summary')
    fig,ax=plt.subplots(figsize=(7.2,4),constrained_layout=True)
    ax.barh(benchsum.method,benchsum.mean_distance,color=['#307c9d','#b37a45','#628a62','#8f719d'])
    ax.set_xlabel('Mean squared L2 discrepancy to aligned deconvolution')
    ax.set_title('Matched-cutoff GWAS masking benchmarks');ax.invert_yaxis()
    savefig(fig,out/'figures','R03_methods_real_gwas')
    # Cutoff sensitivity: report raw inverse estimator (may be negative).
    rg2=np.random.default_rng(seed+455)
    q=designs['heterogeneous_calibrated'];yy=sample_signs(q,rg2)*y0
    Tbase=cutoffs['heterogeneous_calibrated']['T']
    mult=[.65,.85,1.,1.15,1.35]
    sens=[];plots={}
    for fac in mult:
        tt=max(.3,min(11.,Tbase*fac))
        z=estimate(yy,sigma,q,x,tt,nt=241)
        plots[fac]=z['full']
        sens.append({'factor':fac,'cutoff_T':tt,'distance_to_aligned_reference':ise(x,z['full'],clean['full']),
                     'negative_fraction_grid':float(np.mean(z['full']<0))})
    table_csv_tex(pd.DataFrame(sens),out/'tables','R05_cutoff_sensitivity')
    fig,ax=plt.subplots(figsize=(8.1,4.4),constrained_layout=True)
    ax.plot(x,clean['full'],'k--',label='Fully aligned reference',lw=2.5)
    for fac,cur in plots.items():ax.plot(x,cur,label=f'{fac:.2f} × cutoff',alpha=.8)
    ax.set_xlabel('Standardized signed effect');ax.set_ylabel('Estimated density')
    ax.set_title('Regularization sensitivity (same masked observations)');ax.legend(ncol=2,fontsize=8)
    savefig(fig,out/'figures','R04_cutoff_sensitivity')
    return {'source_rows':n,'beta_scale':scale,'replications_each_design':B,
            'aligned_reference_cutoff':Tref,'reference':'Fourier deconvolution of aligned observed GWAS coefficients, NOT latent truth.',
            'cautions':['Only sign masking is randomized; genetic effect estimates and SEs are real.',
                        'GIANT selected variants are not proven independent or LD-pruned.',
                        'No claim that genuine GWAS source omitted effect-allele metadata.',
                        'Gaussian standard-error approximation is not validated here.']}
