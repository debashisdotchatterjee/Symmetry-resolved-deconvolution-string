"""Colab/research workflow: simulation + GWAS motivation + full GWAS analysis.

Examples:
    python run_all_colab.py --profile standard
    python run_all_colab.py --profile quick --gwas-file /content/GIANT_HEIGHT_...gz
    python run_all_colab.py --profile smoke --skip-gwas  # software smoke test
    python run_all_colab.py --profile standard --ecg    # optional ECG validation
"""
import os,sys,json,traceback,zipfile,platform,hashlib,argparse,importlib.util,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy,pandas,scipy,matplotlib

ROOT=Path(__file__).resolve().parent

def load_module(filename,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def json_safe(x):
    if isinstance(x,dict):return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
    if isinstance(x,numpy.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x

def record_json(p,thing):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(json_safe(thing),indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def archive_tree(source,zip_path):
    source=Path(source); zip_path=Path(zip_path)
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(source.rglob('*')):
            if p.is_file():z.write(p,arcname=str(Path('RESULTS')/p.relative_to(source)))
        for filename in ('inverse_core.py','01_motivation_gwas.py','02_simulations.py',
                         '03_real_gwas.py','04_optional_ecg.py','gwas_data.py','run_all_colab.py',
                         'README.md','requirements.txt','test_self_checks.py','COLAB_RUN.ipynb','run_motivation_only.py'):
            p=ROOT/filename
            if p.exists():z.write(p,arcname=str(Path('SOURCE')/filename))
    with zipfile.ZipFile(zip_path) as z:
        fail=z.testzip()
    if fail:raise IOError('ZIP integrity check failed: '+fail)
    return str(zip_path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',choices=['smoke','quick','standard','full'],default='standard')
    ap.add_argument('--seed',type=int,default=20261102)
    ap.add_argument('--gwas-file',default=None,help='Authentic input GWAS file; do not use simulated data.')
    ap.add_argument('--skip-gwas',action='store_true',help='Developer/test only; simulations alone.')
    ap.add_argument('--strict-gwas',action='store_true',help='Exit with error if GWAS cannot load')
    ap.add_argument('--ecg',action='store_true',help='Run optional additional PTB-XL+ example (~89MB)')
    ap.add_argument('--ecg-file',default=None)
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    os.chdir(ROOT)
    out=Path(args.out).resolve() if args.out else Path('/content/symmetry_results' if Path('/content').exists() else ROOT/'results')
    if out.exists():shutil.rmtree(out)
    out.mkdir(parents=True)
    (out/'figures').mkdir();(out/'tables').mkdir();(out/'data_provenance').mkdir()
    status={'started_utc':datetime.now(timezone.utc).isoformat(),'profile':args.profile,'seed':args.seed,
        'python':platform.python_version(),'platform':platform.platform(),
        'versions':{'numpy':numpy.__version__,'pandas':pandas.__version__,
                    'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'sections':{},'failure_details':{}}
    cache=Path('/content/gwas_cache' if Path('/content').exists() else ROOT/'_data_cache')
    try:
        print('\n[1/3] Running reproducible simulations ...',flush=True)
        sim=load_module('02_simulations.py','simulation_module')
        status['sections']['simulation']=sim.execute(out/'SIMULATION',args.profile,args.seed)
        if args.skip_gwas:
            status['sections']['gwas']='SKIPPED BY USER; not real-data validation'
        else:
            print('\n[2/3] Retrieving genuine published GWAS summary data ...',flush=True)
            try:
                loader=load_module('gwas_data.py','gwas_module')
                source,source_url=loader.fetch_data(cache,args.gwas_file)
                gwas,metadata=loader.select_gwas(source,out/'data_provenance',max_variants=300 if args.profile=='smoke' else 1000 if args.profile=='quick' else 3200 if args.profile=='standard' else 6000)
                metadata['download_url_or_input']=source_url
                record_json(out/'data_provenance'/'gwas_metadata.json',metadata)
                print(f'Loaded {len(gwas)} summary-statistic rows from {source_url}.',flush=True)
                intro=load_module('01_motivation_gwas.py','intro_module')
                status['sections']['gwas_motivation']=intro.execute(gwas,out/'GWAS_MOTIVATION',args.seed,args.profile)
                print('\n[3/3] Running detailed GWAS sign-masking experiments ...',flush=True)
                app=load_module('03_real_gwas.py','real_module')
                status['sections']['gwas_main']=app.execute(gwas,out/'GWAS_FULL',args.seed,args.profile)
            except Exception as e:
                status['sections']['gwas']='FAILED - no synthetic substitute'
                status['failure_details']['gwas']=''.join(traceback.format_exception(type(e),e,e.__traceback__))
                print('\nWARNING: Genuine GWAS analysis could not be completed. '+str(e),flush=True)
                (out/'REAL_DATA_REQUIRES_ATTENTION.txt').write_text(status['failure_details']['gwas'])
                if args.strict_gwas: raise
        if args.ecg:
            print('\n[Optional] PTB-XL+...',flush=True)
            try:
                ecg=load_module('04_optional_ecg.py','ecg_module')
                status['sections']['ecg']=ecg.execute(out/'ECG_OPTIONAL',cache,args.seed,args.ecg_file)
            except Exception as e:
                status['sections']['ecg']='FAILED - no fabricated substitute'
                status['failure_details']['ecg']=''.join(traceback.format_exception(type(e),e,e.__traceback__))
                print('Optional ECG could not run:',str(e),flush=True)
    except Exception as e:
        status['failure_details']['fatal']=''.join(traceback.format_exception(type(e),e,e.__traceback__))
        print(status['failure_details']['fatal'],file=sys.stderr)
        status['fatal_error']=True
    status['finished_utc']=datetime.now(timezone.utc).isoformat()
    record_json(out/'RUN_MANIFEST.json',status)
    (out/'INTERPRETATION_AND_LIMITATIONS.md').write_text('''# Interpretation rules

- Genuine real data: GIANT 2014 height GWAS signed regression coefficients and reported SEs.
- Artificial intervention: randomized suppression of effect-allele sign with known probabilities.
- Do NOT describe controlled sign masking as natural loss of effect-allele labels.
- Aligned GWAS summary estimates are a NOISY REFERENCE, never true latent effects.
- Hash-based thinning is **not linkage-disequilibrium pruning**; independence may fail.
- Allele convention fixes an arbitrary orientation; sign changes are not genetic effect changes.
- All confidence displays based on repeated masking measure masking randomness only.
- Fourier inversions are unconstrained and can be negative; no hidden positivity clipping.
- Cutoffs use a prespecified upper-risk proxy with a non-estimated Sobolev constant L.
- No real-data results are claimed if source download or parsing failed.
- If the fully balanced channel is used, only the invariant density is identifiable.
''',encoding='utf8')
    archive=out.parent/f'Symmetry_Resolved_Results_{args.profile}_{args.seed}.zip'
    archive_tree(out,archive)
    print('\nOUTPUT DIRECTORY:',out, '\nRESULTS ZIP:',archive,'\nSections:',status['sections'],flush=True)
    if status.get('fatal_error'):sys.exit(1)
    return archive

if __name__=='__main__':main()
