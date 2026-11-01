"""GIANT (Wood et al. 2014) height GWAS access and audited sampling.

Use original primary-source summary statistics; NEVER simulate missing GWAS data.
No LD pruning is claimed. The selected data are a reproducible hash-thinned subset.
"""
from pathlib import Path
import os, hashlib, urllib.request, time, gzip
import numpy as np, pandas as pd

GIANT_URLS=[
 'https://giant-consortium.web.broadinstitute.org/images/0/01/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz',
 'https://portals.broadinstitute.org/collaboration/giant/images/0/01/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz']
DOI='10.1038/ng.3097' # Wood et al., Nature Genetics (2014)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def fetch_data(cache, override=None):
    """Download only once; user-supplied official copy is preferred if URLs fail.
    """
    if override:
        p=Path(override).expanduser().resolve()
        if not p.is_file():raise FileNotFoundError(f'GWAS input not found: {p}')
        return p,'user-provided-file'
    cache=Path(cache);cache.mkdir(exist_ok=True,parents=True)
    target=cache/'GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz'
    if target.exists() and target.stat().st_size>2_000_000:
        return target,'previously-cached-primary-source'
    errors=[]
    for url in GIANT_URLS:
        temp=target.with_suffix('.download')
        try:
            print('Downloading genuine GIANT height summary statistics:',url,flush=True)
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 academic-research'})
            with urllib.request.urlopen(req,timeout=65) as reader,open(temp,'wb') as dst:
                while True:
                    block=reader.read(1<<20)
                    if not block:break
                    dst.write(block)
            if temp.stat().st_size<2_000_000:
                raise ValueError('Downloaded file is unexpectedly small; refusing to use it.')
            with gzip.open(temp,'rb') as f:
                test=f.readline().decode('utf8',errors='replace')
            if not ('MarkerName' in test and ('SE' in test or '\tb\t' in test)):
                raise ValueError('Downloaded file header does not match GIANT height data.')
            temp.replace(target)
            return target,url
        except Exception as e:
            errors.append(f'{url}: {type(e).__name__}: {str(e)[:200]}')
            if temp.exists():temp.unlink()
            time.sleep(1)
    raise RuntimeError('Could not obtain the authentic GIANT GWAS file. No synthetic fallback will be used. '
        'Download the file manually from https://giant-consortium.web.broadinstitute.org/GIANT_consortium_data_files '
        'and supply --gwas-file /content/filename.gz. Attempts:\n'+'\n'.join(errors))

def select_gwas(filepath,out, max_variants=3500):
    """Select variants by stable SNP hash, not by association p-value.

    This is not LD-based thinning. Variant dependence may remain. Output is a
    selected set of *actual* GWAS observations (beta, SE) with allele metadata.
    """
    filepath=Path(filepath);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    first=pd.read_csv(filepath,sep=r'\s+',nrows=3,compression='infer')
    colmap={c.lower():c for c in first.columns}
    # Permissive for user-supplied harmonized files.
    candidates={
      'variant':['markername','snp','rsid','variant_id'],
      'beta':['b','beta','effect','effect_size','beta.exposure'],
      'se':['se','standard_error','stderr','se.exposure'],
      'allele1':['allele1','effect_allele','a1'],
      'allele2':['allele2','other_allele','a2'],
      'freq':['freq.allele1.hapmapceu','eaf','effect_allele_frequency','frequency'],
      'p':['p','p_value','pval'],
    }
    use={}
    for name,opts in candidates.items():
        col=next((colmap[key] for key in opts if key in colmap),None)
        if col:use[name]=col
    if not {'beta','se','variant'}.issubset(use):
        raise ValueError(f'No signed beta/SE/variant columns. Available={list(first.columns)}; mapped={use}')
    data=[]; read=0;valid=0
    for chunk in pd.read_csv(filepath,sep=r'\s+',compression='infer',chunksize=120000,
                             usecols=list(set(use.values())),low_memory=False):
        read+=len(chunk)
        chunk=chunk.rename(columns={v:k for k,v in use.items()})
        for cc in ('beta','se'):
            chunk[cc]=pd.to_numeric(chunk[cc],errors='coerce')
        good=np.isfinite(chunk.beta)&np.isfinite(chunk.se)&(chunk.se>0)
        # Remove A/T and C/G strand-ambiguous SNPs when alleles are present.
        if {'allele1','allele2'}.issubset(chunk):
            a=chunk.allele1.astype(str).str.upper();b=chunk.allele2.astype(str).str.upper()
            good &= (~(a+b).isin(['AT','TA','CG','GC']))
            good &= a.isin(list('ACGT')) & b.isin(list('ACGT')) & (a!=b)
        if 'freq' in chunk:
            chunk['freq']=pd.to_numeric(chunk['freq'],errors='coerce')
            good &= chunk.freq.between(.05,.95,inclusive='both')
        d=chunk.loc[good].copy();valid+=len(d)
        # Consistent hashing gives an auditable, order-independent random-like selection.
        h=pd.util.hash_pandas_object(d.variant.astype(str),index=False).to_numpy(dtype='uint64')
        d['hashkey']=h
        # keep a sparse set in memory before selecting the smallest keys.
        d=d.loc[(d.hashkey % np.uint64(173))<np.uint64(3)]
        if len(d):data.append(d)
    if not data:raise ValueError('No eligible GWAS rows after QC and stable hashing.')
    d=pd.concat(data,ignore_index=True).drop_duplicates(subset=['variant'])
    d=d.nsmallest(max_variants,'hashkey').sort_values('hashkey').reset_index(drop=True)
    if len(d)<100:raise ValueError(f'Only {len(d)} rows; need more eligible variants.')
    d.to_csv(out/'gwas_selected_summary.csv',index=False)
    meta={'input_file':str(filepath),'input_sha256':sha256(filepath),'original_rows':read,
          'rows_after_basic_qc':valid,'selected_rows':len(d),
          'source':'GIANT height 2014' if 'GIANT_HEIGHT' in filepath.name else 'provided GWAS summary statistics',
          'selection':'allele-QC + fixed-hash thinning; NOT LD pruning',
          'doi':DOI,'columns_mapped':use,
          'limitations':'Potential LD dependence, selection by allele availability and sample QC, beta is noisy, allele orientation is convention dependent.'}
    return d,meta
