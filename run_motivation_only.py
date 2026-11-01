"""Run a completely independent introductory GWAS real-data analysis in Colab.

python run_motivation_only.py --gwas-file /content/GIANT_...txt.gz
OR simply: python run_motivation_only.py  (download original GIANT data)

Produces a compact ZIP containing the introductory motivation figure, LaTeX table,
source metadata, the selected real summary statistics, and this code.
"""
import argparse,json,sys,zipfile,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from gwas_data import fetch_data,select_gwas
# Names starting with digits cannot be imported with ordinary `import` syntax.
import importlib.util
ROOT=Path(__file__).resolve().parent

def load_intro():
    spec=importlib.util.spec_from_file_location('intro_module',ROOT/'01_motivation_gwas.py')
    mod=importlib.util.module_from_spec(spec);sys.modules['intro_module']=mod
    spec.loader.exec_module(mod)
    return mod

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gwas-file',default=None)
    ap.add_argument('--out',default='/content/symmetry_motivation_only' if Path('/content').exists() else str(ROOT/'motivation_output'))
    ap.add_argument('--seed',type=int,default=20261102)
    ap.add_argument('--n',type=int,default=3200)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cache=Path('/content/gwas_cache' if Path('/content').exists() else ROOT/'_data_cache')
    infile,src=fetch_data(cache,args.gwas_file)
    d,prov=select_gwas(infile,out/'data_provenance',max_variants=args.n)
    prov['source_url_or_override']=src
    (out/'data_provenance'/'gwas_metadata.json').write_text(json.dumps(prov,indent=2)+'\n')
    ans=load_intro().execute(d,out/'MOTIVATION',args.seed)
    (out/'MOTIVATION_RESULTS_METADATA.json').write_text(json.dumps(ans,indent=2)+'\n')
    (out/'READ_ME_FIRST.txt').write_text(
        'Real source: original GIANT 2014 height GWAS effect estimates.\n'
        'Artificial experiment: sign masking is intentionally applied; the GIANT data do not have naturally missing effect-allele labels.\n'
        'Benchmark aligned deconvolution is NOT the true latent density.\n'
        'Variant subsampling by deterministic hash is NOT LD pruning.\n'
        'No unobserved clinical or genomic quantities have been invented.\n')
    target=out.parent/f'Symmetry_Motivation_GWAS_{args.seed}.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in out.rglob('*'):
            if p.is_file(): z.write(p,str(Path('RESULTS')/p.relative_to(out)))
        for item in ('01_motivation_gwas.py','gwas_data.py','inverse_core.py','plotting.py','run_motivation_only.py'):
            z.write(ROOT/item,str(Path('SOURCE')/item))
    print('Motivation ZIP:',target)
    return target

if __name__=='__main__':main()
