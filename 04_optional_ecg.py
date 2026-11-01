"""Optional *second* real dataset: PTB-XL+ signed lead-I scalar feature.

No fallback to fabricated ECG data. Exact feature column selection is recorded.
Uses reported scalar lead features, not reconstructed complete ECG traces.
"""
from pathlib import Path
import urllib.request, re, numpy as np,pandas as pd,matplotlib.pyplot as plt
from inverse_core import estimate,choose_cutoff,sample_signs,make_grid,robust_scale
from plotting import setup,savefig,table_csv_tex

SOURCE='https://physionet.org/files/ptb-xl-plus/1.0.1/features/12sl_features.csv'

def execute(out,cache,seed=20261102,ecg_file=None):
    setup();out=Path(out);cache=Path(cache);cache.mkdir(exist_ok=True,parents=True)
    raw=Path(ecg_file) if ecg_file else cache/'12sl_features.csv'
    if not raw.exists():
        print('Downloading actual PTB-XL+ 12SL CSV (~89 MB)...',flush=True)
        req=urllib.request.Request(SOURCE,headers={'User-Agent':'Mozilla/5.0 academic-research'})
        with urllib.request.urlopen(req,timeout=90) as f,open(raw,'wb') as g:
            while True:
                block=f.read(1<<20)
                if not block:break
                g.write(block)
    header=pd.read_csv(raw,nrows=0)
    columns=list(header.columns)
    # Only a signed, lead-I amplitude. An ambiguous column name causes explicit error.
    reg=[c for c in columns if re.search(r'(?i)(^|[_\s])i($|[_\s])',c) and
         re.search(r'(?i)(amp|amplitude)',c) and re.search(r'(?i)(r[\-_ ]?amp|r[\-_ ]?wave|r[\-_ ]?peak)',c)]
    if len(reg)!=1:
        (out/'ECG_FEATURE_COLUMNS.txt').write_text('\n'.join(columns),encoding='utf-8')
        raise ValueError('Cannot uniquely identify a signed lead-I R-amplitude feature; '
                         f'candidates={reg[:20]}. See ECG_FEATURE_COLUMNS.txt. '
                         'Do not silently substitute a magnitude or unrelated variable.')
    feature=reg[0]
    dat=pd.read_csv(raw,usecols=[feature])
    val=pd.to_numeric(dat[feature],errors='coerce').to_numpy()
    val=val[np.isfinite(val)]
    if len(val)<200:raise ValueError('Too few usable ECG amplitude observations.')
    rng=np.random.default_rng(seed+720)
    if len(val)>3500:val=rng.choice(val,size=3500,replace=False)
    # PTB-XL+ does not report an independent measurement-error SE per subject for this
    # specific amplitude field. Therefore do NOT fake one: observational comparison is
    # a controlled sign-masking demonstration without claiming deconvolution truth.
    scale=robust_scale(val);v=val/scale
    x=np.linspace(np.quantile(v,.003)-1,np.quantile(v,.997)+1,321)
    # Nonzero small noise proxy would be unsupported, so use plain KDE for this module.
    from scipy.stats import gaussian_kde
    aligned=gaussian_kde(v,bw_method='scott')(x)
    fig,ax=plt.subplots(figsize=(8.5,4.5),constrained_layout=True)
    ax.plot(x,aligned,'k--',lw=2.3,label='Recorded orientation: KDE')
    rows=[]
    for q,col in [(1.,'#24789b'),(.6,'#c17436'),(0.,'#756e90')]:
        yy=sample_signs(np.full(len(v),q),rng)*v
        kk=gaussian_kde(yy,bw_method='scott')(x)
        ax.plot(x,kk,label=f'Controlled sign mask, retention={(1+q)/2:.2f}',color=col)
        rows.append({'feature_column':feature,'n':len(v),'q':q,
                     'retention_probability':(1+q)/2,'sample_mean_after_mask':yy.mean()})
    ax.set_xlabel('Standardized signed lead-I amplitude');ax.set_ylabel('Density')
    ax.set_title('PTB-XL+: observational sign convention (no artificial SE)');ax.legend(fontsize=8)
    savefig(fig,out/'figures','E01_optional_ecg_polarity')
    table_csv_tex(pd.DataFrame(rows),out/'tables','E01_ecg_description')
    return {'rows':len(val),'feature':feature,'source':SOURCE,'data_doi':'10.13026/g6h6-7g88',
            'limitation':'KDE sign-mask demonstration only; no unobserved Gaussian noise variance is fabricated.'}
