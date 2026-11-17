"""Original visual conventions, with all downstream policy lists read from Results2."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd

from config import PROJECT_ROOT, OUTPUT_ROOT
TASK=PROJECT_ROOT/'results2'
ORANGE='#FA853E';BLUE='#1B6CA8'
CATEGORIES=['TechnologicalInfrastructural','Institutional','BehavioralCultural','NatureBased']
TOTAL=['APtotal','MPtotal','APtotal_dest','MPtotal_dest']
TYPE_LABELS=['Technological /\nInfrastructural AP','Institutional\nAP','Behavioral /\nCultural AP','Nature-\nbased AP',
             'Technological /\nInfrastructural MP','Institutional\nMP','Behavioral /\nCultural MP','Nature-\nbased MP']
SPECS=[('baseline','Main specification','main'),('RE','Random effects','re'),('winsor1','1% winsorizing','sensitivity'),
       ('trim5','5% trimming','sensitivity'),('sample80','Random 80% samples','sensitivity'),
       ('exclude2020','Excluding 2020 samples','sensitivity'),('lag1','One-year lag','lag'),('lag2','Two-year lag','lag')]
plt.rcParams.update({'font.family':'Arial','font.size':6.5,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,
                     'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,'xtick.major.size':2.5,
                     'ytick.major.size':2.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':True,'legend.frameon':False})

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plabel(p):return '<0.001' if p<.001 else f'{p:.3f}' if np.isfinite(p) else 'NA'
def number(x,_):return f'{0 if abs(x)<5e-8 else x:.3f}'
def name(v):
    side='destination' if v.endswith('_dest') else 'origin'
    base=v.removesuffix('_dest');p=base[:2];c=base[2:]
    c={'TechnologicalInfrastructural':'Technological / infrastructural','Institutional':'Institutional',
       'BehavioralCultural':'Behavioral / cultural','NatureBased':'Nature-based','total':'Total'}[c]
    return f'{c} {p} ({side})'

def read(path):
    d=pd.read_csv(path);d['source_file']=path.name;d['source_csv_row']=np.arange(len(d))+2;d['source_sha256']=sha(path)
    return d

def save(fig,stem,axes,out,source,selection_hash):
    qa=out/'qa';figdir=out/'figures';src=out/'figure_source_data'
    for p in [qa,figdir,src]:p.mkdir(exist_ok=True,parents=True)
    fig.canvas.draw()
    source=source.copy();source['selection_manifest_sha256']=selection_hash
    source.to_csv(src/f'{stem}.csv',index=False)
    fig.savefig(figdir/f'{stem}.pdf',facecolor='white')
    fig.savefig(figdir/f'{stem}.svg',facecolor='white')
    fig.savefig(figdir/f'{stem}.png',dpi=600,facecolor='white')
    plt.close(fig)

def types_plot(data,cov,out,selection_hash):
    d=data[(data.Specification=='baseline_types16')&(data.Model=='TWFE')&(data.Covariance==cov)].copy()
    assert len(d)==16
    fig,ax=plt.subplots(figsize=(183/25.4,94/25.4))
    fig.subplots_adjust(left=.088,right=.99,bottom=.225,top=.965)
    ax.axvspan(3.5,7.5,color='#F3F3F3',zorder=0);ax.axvline(3.5,color='#888888',lw=.4,zorder=1)
    ax.axhline(0,color='#777777',ls=(0,(3,2)),lw=.6,zorder=1)
    annotations=[]
    for j,variable in enumerate([p+c for p in ['AP','MP'] for c in CATEGORIES]):
        for side,off,color in [('',-.25,ORANGE),('_dest',.25,BLUE)]:
            row=d[d.Variable==variable+side];assert len(row)==1;r=row.iloc[0];x=j+off
            ax.vlines(x,r.CI95_lower,r.CI95_upper,color=color,lw=.6,alpha=.5,zorder=2)
            ax.vlines(x,r.CI90_lower,r.CI90_upper,color=color,lw=1.05,zorder=3)
            ax.plot(x,r.Coefficient,'o',ms=5,mfc='white',mec=color,mew=.9,zorder=4)
            last=j in [3,7] and side=='_dest'
            annotations.append(ax.annotate(plabel(r['P-value']),(x,r.Coefficient),xytext=(-4 if last else 4,2.5),
                textcoords='offset points',ha='right' if last else 'left',va='bottom',fontsize=5.6,zorder=5))
    lo,hi=min(d.CI95_lower.min(),0),max(d.CI95_upper.max(),0);span=hi-lo
    ax.set(xlim=(-.5,7.5),ylim=(lo-.12*span,hi+.13*span))
    fig.canvas.draw();zero=ax.transData.transform((0,0))[1]
    for a in annotations:
        box=a.get_window_extent(fig.canvas.get_renderer())
        if box.y0-1<=zero<=box.y1+1:a.set_position((a.get_position()[0],-3));a.set_va('top')
    ax.set_xticks(range(8),TYPE_LABELS,fontsize=6.4);ax.tick_params(axis='x',pad=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7));ax.yaxis.set_major_formatter(FuncFormatter(number))
    ax.set_ylabel('Migration association by policy type',fontsize=7.2)
    ax.spines[['top','right']].set_visible(False);ax.spines[['bottom','left']].set_linewidth(1.15)
    ax.legend(handles=[Line2D([],[],color=ORANGE,marker='o',mfc='white',lw=.8,ms=4,label='Origin'),
                      Line2D([],[],color=BLUE,marker='o',mfc='white',lw=.8,ms=4,label='Destination')],
              loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,fontsize=6.5,columnspacing=2.5,handlelength=2.5)
    save(fig,'Fig2_TypeEffects_'+('two_way' if cov=='origin_destination' else 'pair'),[ax],out,d,selection_hash)

def sensitivity_plot(data,cov,variables,out,selection_hash,aggregate=False):
    family='aggregate' if aggregate else 'types16'
    stem=('FigS6_Aggregate_' if aggregate else 'Fig3_SelectedSensitivity_')+('two_way' if cov=='origin_destination' else 'pair')
    parts=[]
    for order,(spec,label,kind) in enumerate(SPECS):
        model='RE' if spec=='RE' else 'TWFE';actual='baseline' if spec=='RE' else spec
        d=data[(data.Specification==actual+'_'+family)&(data.Model==model)&(data.Covariance==cov)&data.Variable.isin(variables)].copy()
        assert set(d.Variable)==set(variables)
        d['row_order_bottom_to_top']=order;d['display_specification']=label;d['display_family']=kind
        parts.append(d)
    d=pd.concat(parts,ignore_index=True)
    ncol=2 if len(variables)>1 else 1;nrow=max(1,int(np.ceil(len(variables)/ncol)))
    fig,axes=plt.subplots(nrow,ncol,squeeze=False,figsize=(183/25.4,61*nrow/25.4))
    fig.subplots_adjust(left=.162,right=.983,bottom=.093 if nrow==2 else .12,top=.972,wspace=.285,hspace=.42)
    active=[]
    if not variables:
        axes[0,0].text(.5,.5,'No type met the locked Results 2 selection rule',ha='center',va='center',transform=axes[0,0].transAxes)
        axes[0,0].axis('off');active=[axes[0,0]]
    for i,ax in enumerate(axes.flat):
        if i>=len(variables):
            if variables:ax.set_visible(False)
            continue
        variable=variables[i];active.append(ax);dd=d[d.Variable==variable].sort_values('row_order_bottom_to_top')
        ax.axhspan(1.5,5.5,facecolor='#EDF6F9',zorder=0);ax.axhspan(5.5,7.5,facecolor='#FFF4ED',zorder=0)
        for yy in [1.5,5.5,7.5]:ax.axhline(yy,color='#BDBDBD',lw=.35,zorder=1)
        main=float(dd.loc[dd.display_family=='main','Coefficient'].iloc[0])
        ax.axvline(0,color='#888888',ls=(0,(2.5,2)),lw=.55,zorder=1);ax.axvline(main,color='#BDBDBD',ls=':',lw=.45,zorder=1)
        annotations=[]
        for _,r in dd.iterrows():
            color={'main':'#808080','re':'#000000','sensitivity':BLUE,'lag':ORANGE}[r.display_family];yy=r.row_order_bottom_to_top
            ax.hlines(yy,r.CI95_lower,r.CI95_upper,color=color,lw=.55,alpha=.52,zorder=2)
            ax.hlines(yy,r.CI90_lower,r.CI90_upper,color=color,lw=.9,zorder=3)
            ax.plot(r.Coefficient,yy,'o',ms=3.7,color=color,mew=.65,mfc=color if r.display_family=='main' else 'white',zorder=4)
            annotations.append(ax.annotate(plabel(r['P-value']),(r.Coefficient,yy),xytext=(4.5,1),textcoords='offset points',
                                          ha='left',va='bottom',fontsize=5.5,annotation_clip=False,zorder=5))
        lo,hi=min(dd.CI95_lower.min(),0),max(dd.CI95_upper.max(),0);span=hi-lo
        ax.set_xlim(lo-.12*span,hi+.24*span);ax.set_ylim(-1,8)
        fig.canvas.draw();refs=[ax.transData.transform((v,0))[0] for v in [0,main]]
        for a in annotations:
            box=a.get_window_extent(fig.canvas.get_renderer())
            if any(box.x0-1<x<box.x1+1 for x in refs):
                a.set_position((-4.5,1));a.set_ha('right');box=a.get_window_extent(fig.canvas.get_renderer())
                if any(box.x0-1<x<box.x1+1 for x in refs):
                    a.xy=(max(a.xy[0],0,main),a.xy[1]);a.set_position((4.5,1));a.set_ha('left')
        ax.set_yticks(range(8));ax.set_yticklabels([s[1] for s in SPECS] if i%ncol==0 else [])
        ax.set_xlabel('\n'.join(textwrap.wrap(name(variable),width=37)),fontsize=6.8,labelpad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.xaxis.set_major_formatter(FuncFormatter(number))
        ax.spines[['top','right']].set_visible(False)
        ax.annotate(chr(97+i),(0,1),xycoords='axes fraction',xytext=(-78 if i%ncol==0 else -18,0),
                    textcoords='offset points',fontsize=11,weight='bold',ha='left',va='top',annotation_clip=False)
    save(fig,stem,active,out,d,selection_hash)

def moderation_plot(curves,cov,selected,out,selection_hash):
    d=curves[(curves.degree==2)&(curves.Covariance==cov)].copy() if len(curves) else curves
    assert set(d.Policy) == set(selected) if len(d) else not selected
    fig,axes=plt.subplots(2,4,figsize=(183/25.4,114/25.4))
    fig.subplots_adjust(left=.064,right=.825,bottom=.087,top=.972,wspace=.53,hspace=.22)
    mods=['income','aging','RD','EA'];labels=['Ln (income)','Ageing rate','Racial diversity','Educational attainment rate']
    colors=plt.get_cmap('tab10').colors
    for row,side in enumerate(['origin','dest']):
        policies=[p for p in selected if p.endswith('_dest')==(side=='dest')]
        for col,(mod,label) in enumerate(zip(mods,labels)):
            ax=axes[row,col];low=[0];high=[0]
            if policies:
                for i,policy in enumerate(policies):
                    c=d[(d.Policy==policy)&(d.Moderator==mod+'_'+side)].sort_values('Moderator_value')
                    assert len(c)==100 and np.all(np.diff(c.Moderator_value)>0)
                    ax.plot(c.Moderator_value,c.Effect,color=colors[i%10],lw=.65,zorder=3)
                    ax.fill_between(c.Moderator_value,c.CI95_lower,c.CI95_upper,color=colors[i%10],alpha=.2,lw=0,zorder=2)
                    low.extend([c.Effect.min(),c.CI95_lower.min()]);high.extend([c.Effect.max(),c.CI95_upper.max()])
                lo,hi=float(np.nanmin(low)),float(np.nanmax(high));span=hi-lo
                ax.set_ylim(lo-.04*span,hi+.05*span)
            else:
                ax.text(.5,.5,'No selected\n'+('origin' if side=='origin' else 'destination')+' type',ha='center',va='center',transform=ax.transAxes,fontsize=6)
            ax.axhline(0,color='#777777',lw=.5,ls='--',zorder=1)
            ax.set_xlabel(label,fontsize=6.4,labelpad=2);ax.set_ylabel('Migration association',fontsize=5.5,labelpad=1)
            ax.tick_params(labelsize=5.5);ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if col==0:ax.annotate('ab'[row],(0,1),xycoords='axes fraction',xytext=(-16,0),textcoords='offset points',fontsize=11,weight='bold',va='top')
        if policies:
            handles=[Line2D([],[],color=colors[i%10],lw=.8,label='\n'.join(textwrap.wrap(name(p),22))) for i,p in enumerate(policies)]
            axes[row,3].legend(handles=handles,loc='center left',bbox_to_anchor=(1.08,.5),fontsize=5.8,handlelength=2,labelspacing=1.5)
    save(fig,'Fig4_DynamicModeration_'+('two_way' if cov=='origin_destination' else 'pair'),axes.flat,out,d,selection_hash)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=OUTPUT_ROOT/'figures_aggregate_and_types')
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise SystemExit('Choose an empty output directory.')
    source=TASK/'data/figure_source_data'
    types=pd.read_csv(source/'Fig2_TypeEffects_pair.csv')
    totals=pd.read_csv(source/'FigS6_Aggregate_pair.csv')
    selection_hash=sha(TASK/'tables/results2_selected_pair8.csv')
    types_plot(types,'pair',out,selection_hash)
    sensitivity_plot(totals,'pair',TOTAL,out,selection_hash,aggregate=True)
    print(out)
if __name__=='__main__':main()
