"""Independent introductory real-data experiment (GIANT height GWAS, no invented data)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from inverse_core import estimate,choose_cutoff, sample_signs,make_grid,robust_scale,ise
from plotting import setup,savefig,table_csv_tex

def execute(gwas,out,seed=20261102,profile='standard'):
    setup();out=Path(out);rng=np.random.default_rng(seed+35)
    d=gwas.copy(); beta=d.beta.to_numpy(float);se=d.se.to_numpy(float)
    # Standardized effect scale, preserving heteroscedastic SE ratios.
    scale=robust_scale(beta); uobs=beta/scale;s=se/scale
    x=make_grid(uobs,s,321)
    T,choice=choose_cutoff(s,np.ones(len(d)),beta=1.6,L=1.6,tmax=12,sector='full')
    aligned=estimate(uobs,s,np.ones(len(d)),x,T,nt=201)
    # Controlled masking - S artificially generated, NOT alleged to have occurred.
    qq=[1.,.6,0.]; curves={};rows=[]
    for qval in qq:
        q=np.full(len(d),qval)
        S=sample_signs(q,rng);yy=S*uobs
        z=estimate(yy,s,q,x,T,nt=201)
        curves[qval]=(yy,z)
        rows.append({'retained_orientation_probability':(1+qval)/2,'q':qval,'Q_n':sum(q*q),
                     'empirical_signed_mean':yy.mean(),
                     'estimated_odd_L2_norm_sq':float(np.trapz(z['odd']**2,x)),
                     'L2_distance_to_aligned_DECONV_REFERENCE':ise(x,z['full'],aligned['full'])})
    table_csv_tex(pd.DataFrame(rows),out/'tables','M01_intro_masks')
    fig,axs=plt.subplots(1,3,figsize=(13.4,3.9),constrained_layout=True)
    cols={1.:'#1c5e89',.6:'#c06a38',0.:'#667788'}
    for i,(q,(y,z)) in enumerate(curves.items()):
        axs[i].hist(y,bins=45,density=True,alpha=.22,color=cols[q],label='Actual (masked) GWAS betas')
        axs[i].plot(x,aligned['full'],'k--',lw=2.1,label='Aligned deconvolution reference')
        axs[i].plot(x,z['full'],color=cols[q],lw=2,label='Reflection-aware estimate')
        axs[i].plot(x,z['even'],color='#5a987b',ls=':',lw=1.5,label='Even component')
        axs[i].set_title(f'Known retention probability = {(1+q)/2:.2f}')
        axs[i].set_xlabel('Standardized height effect');axs[i].grid(alpha=.14)
    axs[0].set_ylabel('Density');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.06),fontsize=8)
    savefig(fig,out/'figures','M01_real_gwas_intro_motivation')
    fig,ax=plt.subplots(figsize=(7.5,4),constrained_layout=True)
    ax.hist(uobs,bins=80,density=True,alpha=.24,color='#3376a3',label='Aligned, noisy height estimates')
    ax.hist(-uobs,bins=80,density=True,alpha=.17,color='#cf7641',label='Same rows, convention reversed')
    ax.set_title('The meaning of sign depends on the effect-allele convention')
    ax.set_xlabel('Standardized signed estimate');ax.set_ylabel('Density');ax.legend(fontsize=8)
    savefig(fig,out/'figures','M02_gwas_allele_convention')
    return {'dataset_rows':len(gwas),'beta_scale':scale,'cutoff_T':T,
            'warning':'This is real GWAS data with artificial randomized sign labels. Reference is not latent truth.'}
