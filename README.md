# Symmetry-resolved deconvolution: reproducible numerical study

**This package implements the precise model in the manuscript:**

\[Y_j=S_j U_j+\varepsilon_j,\quad \Pr(S_j=1)=\frac{1+q_j}{2},\quad \varepsilon_j\sim N(0,\sigma_j^2).\]

The real GWAS applications impose **known random signs on authentic, correctly oriented summary statistics**. They do *not* allege that the original study lost its allele annotations. The published signed effect estimates and standard errors are authentic; the observed GWAS estimates are already noisy, so the reference curves are not the latent true density.

## Files

- `inverse_core.py` — proposed Fourier/cosine/sine estimators; full, even, odd components; spectral cutoff proxy.
- `02_simulations.py` — phase-transition simulation, heterogeneity, method comparisons, calibration sensitivity.
- `gwas_data.py` — primary-source GIANT 2014 GWAS acquisition and QC, deterministic SNP-hash thinning.
- `01_motivation_gwas.py` — introductory real-data plot demonstrating why the observation mechanism matters.
- `03_real_gwas.py` — detailed real-data study, sector decomposition, masking replication, method benchmarks, cutoff sensitivity.
- `04_optional_ecg.py` — optional PTB-XL+ polarities demonstration, with strict feature-name validation. Does NOT fabricate subject-specific SEs.
- `run_all_colab.py` — runner: produces figures, CSV/LaTeX tables, full manifest, selected GWAS metadata, and final results ZIP.
- `test_self_checks.py` — verifies characteristic-function and parity identities before execution.
- `run_motivation_only.py` — a separate standalone program to generate ONLY the introductory GWAS motivation figure/tables and a compact ZIP.

## Data provenance

Primary authentic dataset: Wood et al. (2014), *Defining the role of common variation in the genomic and biological architecture of adult human height*, Nature Genetics, doi: **10.1038/ng.3097**. The GIANT Consortium file
`GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz`
contains columns `MarkerName`, `Allele1`, `Allele2`, `Freq.Allele1.HapMapCEU`, `b`, `SE`, `p`, `N`.

Primary-source mirror: https://giant-consortium.web.broadinstitute.org/GIANT_consortium_data_files

Direct URL used by script: https://giant-consortium.web.broadinstitute.org/images/0/01/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz

Original alternative: https://portals.broadinstitute.org/collaboration/giant/images/0/01/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz

The public file is about 34 MB compressed; the script caches it separately and saves its SHA-256. If automated download fails, manually obtain the *original* `.gz` and supply `--gwas-file PATH`. It never silently substitutes fake data.

Optional actual ECG dataset: PTB-XL+, DOI **10.13026/g6h6-7g88**, https://physionet.org/content/ptb-xl-plus/1.0.1/. The optional ECG script needs a uniquely named *signed* lead-I amplitude, and aborts if one cannot be identified. It provides only a sign-masking KDE illustration, not Gaussian-error deconvolution.

## Colab instructions

**Easiest:** upload `Symmetry_Resolved_Complete_Code.zip` to Colab; open `COLAB_RUN.ipynb` locally or execute the following Colab code:

```python
from google.colab import files
uploaded = files.upload()  # select Symmetry_Resolved_Complete_Code.zip
import zipfile, glob, os
package = next(k for k in uploaded if k.endswith('.zip'))
with zipfile.ZipFile(package) as z: z.extractall('/content/symmetry_code')
%cd /content/symmetry_code
!python test_self_checks.py
!python run_all_colab.py --profile standard
result = glob.glob('/content/Symmetry_Resolved_Results_standard_*.zip')
files.download(result[-1])
```

`--profile quick` gives a smaller initial test; `--profile full` increases replication counts substantially. `--profile standard` is the recommended first research run. Outputs depend on dataset access and Colab resources; no wall-time guarantee is made.

Manual source override:

```
python run_all_colab.py --profile standard --gwas-file /content/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz
```

Optional ECG (downloads ~89 MB):

```
python run_all_colab.py --profile standard --ecg
```

## Key research caveats

1. **GWAS variants are not proven independent.** Deterministic SNP-hash subsampling is *not* LD pruning and does not verify i.i.d. assumptions. A definitive population-level genetic claim requires LD-aware external variant filtering or region-restricted analyses.
2. The GWA coefficients are estimates, not latent individual truths. Report distances to aligned deconvolution **as benchmark discrepancies**, never true ISE.
3. Effects are aligned to the original published effect-allele convention; that convention does not confer an intrinsic biological sign.
4. The masking mechanism is deliberately generated. Report it as a **semi-synthetic analysis of real observations**, not naturally observed unknown orientation.
5. Unknown probabilities `p_j` are not estimated. The simulation includes misspecified channel calibrations.
6. The spectral cutoff uses an explicitly fixed smoothness proxy, not a proven optimal adaptive procedure. Output cutoff sensitivity results.
7. The Fourier estimator is allowed to be negative. Do not silently rebrand positivity-projected versions as the theoretical estimator.
8. A failed download must produce an explicit failure (and potentially simulation-only results), never invented real-data output.

## Recommended main figures

- Introduction: `GWAS_MOTIVATION/figures/M01_real_gwas_intro_motivation.pdf`
- Main numerical theory: `SIMULATION/figures/S01_information_phase_transition.pdf`
- Main real analysis: `GWAS_FULL/figures/R02_invariant_vs_odd_discrepancy.pdf`
- Supplement: all other plots, numerical tables, robustness studies.

## Suggested future declarations (only AFTER actual runs)

**Data availability:** GIANT height GWAS summary statistics, Wood et al. (2014), publicly available from the GIANT Consortium (URL and date accessed to be supplied). PTB-XL+ if used: PhysioNet, DOI 10.13026/g6h6-7g88.

**Code availability:** Add the actual persistent repository URL after uploading scripts, exact release identifier and computational environment. Do not invent a GitHub URL.

## Output

The final `Symmetry_Resolved_Results_*.zip` contains professional PNG/PDF/SVG figures, matching CSV and LaTeX tables, a provenance manifest and selected GWAS records, these source scripts and execution caveats. The original full GWAS source is cached separately and is not repackaged.

## Standalone real-data motivation (separate program)

Use this when writing the opening motivation before running the full paper analysis:

```
python run_motivation_only.py --n 3200
```

It independently downloads the authentic GIANT height summary statistics if necessary, and creates `Symmetry_Motivation_GWAS_20261102.zip` containing the introductory plot, table, retained original SNP records and execution notes.
