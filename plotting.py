from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'figure.dpi':125,'savefig.dpi':220,'axes.spines.top':False,
        'axes.spines.right':False,'lines.linewidth':1.9,'legend.frameon':False,
        'pdf.fonttype':42,'ps.fonttype':42})

def savefig(fig, out, basename):
    p=Path(out); p.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png','svg'):
        fig.savefig(p/f'{basename}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def table_csv_tex(df,folder,name,index=False):
    p=Path(folder);p.mkdir(parents=True,exist_ok=True)
    df.to_csv(p/f'{name}.csv',index=index,float_format='%.9g')
    (p/f'{name}.tex').write_text(df.to_latex(index=index, float_format='%.4g',escape=True),encoding='utf-8')
