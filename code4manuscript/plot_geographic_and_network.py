"""Redraw geographic figures and the migration network from retained source data.

Contract: preserve observations, class breaks, directed flows and colours;
use WGS84 longitude/latitude for maps and enlarge readable text. Python only.
"""
from pathlib import Path
import json
import math
import sys

from config import PROJECT_ROOT, OUTPUT_ROOT
ROOT = PROJECT_ROOT
PACKAGE = ROOT / 'results1'
import numpy as np
import pandas as pd
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import transform, unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
import statsmodels.api as sm
from scipy.optimize import minimize
FIG = OUTPUT_ROOT/'figures_geographic'
DATA = PACKAGE/'data/figure_source_data'
QA = OUTPUT_ROOT/'figures_geographic_qa'
FIG.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.linewidth':.7,
    'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,
    'axes.unicode_minus':True})
COLORS = ['#ca0020','#f4a582','#f7f7f7','#92c5de','#0571b0']
TYPES = ['TechnologicalInfrastructural','Institutional','BehavioralCultural','NatureBased']
TITLES = ['Technological / infrastructural','Institutional','Behavioural / cultural','Nature-based']
# The previous area scale was 2.25. Increase its circle diameter by 80%.
MAP_MARKER_AREA_SCALE = 2.25 * 1.8**2


def polygons(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for geom in geometry.geoms:
            yield from polygons(geom)


def boundaries():
    cache=PACKAGE/'data/boundaries/us_2020_epsg4326_plot.json'
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    folder=ROOT/'archive/original_project_2026-10-24/new_data4results5/raw/boundaries/tl_2020_us_county'
    source=CRS.from_wkt((folder/'tl_2020_us_county.prj').read_text())
    converter=Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)
    counties=[]; by_state={}
    sf=shapefile.Reader(str(folder/'tl_2020_us_county.shp'), encoding='utf-8')
    for sr in sf.iterShapeRecords():
        rec=sr.record.as_dict(); state=rec['STATEFP']
        if state in ['02','15'] or int(state)>56:
            continue
        geom=transform(converter.transform, shape(sr.shape.__geo_interface__))
        if not geom.is_valid:
            geom=geom.buffer(0)
        by_state.setdefault(state,[]).append(geom)
        for poly in polygons(geom.simplify(.008,preserve_topology=True)):
            counties.append(np.asarray(poly.exterior.coords).round(6).tolist())
    states=[]
    for geoms in by_state.values():
        geom=unary_union(geoms).simplify(.006,preserve_topology=True)
        for poly in polygons(geom):
            states.append(np.asarray(poly.exterior.coords).round(6).tolist())
    payload={'crs':'EPSG:4326','source_crs':source.to_string(),
        'source':'2020 TIGER/Line county boundaries','counties':counties,'states':states,
        'display_simplification_degrees':.008,'transform':converter.description,
        'scope':'Contiguous United States; Connecticut remains in the reference basemap'}
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
    return payload


def basemap(ax, geo):
    ax.add_collection(LineCollection(geo['counties'],colors='#dedede',linewidths=.16,zorder=0))
    ax.add_collection(LineCollection(geo['states'],colors='#888888',linewidths=.50,zorder=1))
    ax.set_xlim(-125.0,-66.3);ax.set_ylim(24.2,49.6)
    # Coordinates remain geographic EPSG:4326. This is the geographic display
    # aspect used for longitude/latitude plots, not a change of projection.
    ax.set_aspect(1/np.cos(np.deg2rad(37.0)),adjustable='box')
    ax.set_axis_off()


def point_coordinates(data, lon, lat):
    # Retained county internal-point coordinates use the TIGER geographic datum.
    converter=Transformer.from_crs(4269,4326,always_xy=True)
    x,y=converter.transform(data[lon].to_numpy(),data[lat].to_numpy())
    assert np.all((np.asarray(x)>-126)&(np.asarray(x)<-65))
    assert np.all((np.asarray(y)>24)&(np.asarray(y)<51))
    return x,y


def save(fig, stem, axes=None, panel_ids=None, row_groups=None, column_groups=None, exemptions=()):
    fig.canvas.draw()
    fig.savefig(FIG/f'{stem}.png',dpi=600,facecolor='white')
    fig.savefig(FIG/f'{stem}.svg',facecolor='white')
    fig.savefig(FIG/f'{stem}.pdf',facecolor='white')
    # A modest-size review image is also generated by Matplotlib.
    fig.savefig(QA/f'{stem}.preview.png',dpi=300,facecolor='white')
    plt.close(fig)


def overview(geo):
    county=pd.read_csv(DATA/'overview/Fig1a_county_means_and_coordinates.csv')
    cy=pd.read_csv(DATA/'overview/Fig1_origin_county_year_with_destination_partner_means.csv')
    stats=pd.read_csv(DATA/'overview/Fig1_scatter_statistics.csv')
    assert len(county)==107 and len(cy)==448
    breaks=json.loads((DATA/'overview/Fig1a_Jenks_breaks.json').read_text())
    fig=plt.figure(figsize=(183/25.4,136/25.4))
    grid=fig.add_gridspec(3,3,left=.065,right=.982,bottom=.095,top=.945,
        wspace=.40,hspace=.44,height_ratios=[.85,.85,1])
    amap=fig.add_subplot(grid[:2,:2]);basemap(amap,geo)
    x,y=point_coordinates(county,'lon','lat')
    amap.scatter(x,y,s=county.HPtotal*1.4*MAP_MARKER_AREA_SCALE,c=county.Jenks_class.map(lambda k:COLORS[int(k)-1]),
        edgecolors='#aaaaaa',linewidths=.35,zorder=3,alpha=.98,clip_on=False)
    amap.annotate('a',(0,1),xycoords='axes fraction',xytext=(-7,9),textcoords='offset points',fontsize=12,weight='bold')
    cbax=amap.inset_axes([.03,-.12,.59,.035])
    cmap=ListedColormap(COLORS);norm=BoundaryNorm(breaks,5)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal',ticks=breaks,spacing='uniform')
    cb.ax.set_xticklabels([f'{v:.3f}' for v in breaks]);cb.ax.tick_params(labelsize=6.3,length=1.5,pad=1)
    cb.set_label('Outflow rate',size=7,labelpad=2);cb.outline.set_linewidth(.4)
    # Concentric size legend, positioned in the open area above the Northeast.
    legend_values = [60, 30, 15]
    legend_sizes = np.asarray(legend_values) * 1.4 * MAP_MARKER_AREA_SCALE
    legend_x, legend_y = .72, .96
    amap.scatter([legend_x] * len(legend_values), [legend_y] * len(legend_values),
        s=legend_sizes, facecolors='none', edgecolors='#777777', linewidths=.55,
        transform=amap.transAxes, zorder=5, clip_on=False)
    fig.canvas.draw()
    bounds = amap.get_window_extent()
    outer_radius_pt = math.sqrt(legend_sizes[0] / math.pi)
    outer_radius_x = outer_radius_pt * fig.dpi / 72 / bounds.width
    outer_radius_y = outer_radius_pt * fig.dpi / 72 / bounds.height
    legend_label_x = legend_x + outer_radius_x + .012
    for value, y in zip([15, 30, 60],
                        [legend_y + .68 * outer_radius_y, legend_y,
                         legend_y - .68 * outer_radius_y]):
        amap.plot([legend_x + outer_radius_x + .002, legend_label_x - .006], [y, y],
            color='#777777', linewidth=.4, linestyle=(0, (1, 1.2)),
            transform=amap.transAxes, zorder=6, clip_on=False)
        amap.text(legend_label_x, y, str(value), transform=amap.transAxes,
            ha='left', va='center', fontsize=7, color='#333333', zorder=6)
    amap.text(legend_x, legend_y + outer_radius_y + .012, 'HP',
        transform=amap.transAxes, ha='center', va='bottom',
        fontsize=7.2, color='#333333', zorder=6)
    b=fig.add_subplot(grid[0,2]);b.axis('off')
    b.annotate('b',(0,1),xycoords='axes fraction',xytext=(-17,6),textcoords='offset points',fontsize=12,weight='bold')
    # Preserve the original separate AP/MP scales in panel b.
    bg=grid[0,2].subgridspec(1,2,wspace=.72)
    for bc,family,upper in [(0,'AP',21),(1,'MP',67)]:
        bx=fig.add_subplot(bg[bc]);origin=cy[family+'total'];dest=cy[family+'total_dest']
        bp=bx.boxplot([origin,dest],widths=.50,whis=(2.5,97.5),showfliers=False,patch_artist=True,
            medianprops={'color':'#555555','linewidth':.65},
            whiskerprops={'linewidth':.6,'color':'#777777'},capprops={'linewidth':.6,'color':'#777777'},
            boxprops={'linewidth':.6,'edgecolor':'#777777'})
        for patch,color in zip(bp['boxes'],['#1B6CA8','#FA853E']):patch.set_facecolor(color);patch.set_alpha(.65)
        bx.set_xticks([1,2],['Origin','Destination']);bx.set_ylim(0,upper)
        bx.set_title(family,fontsize=8,pad=4);bx.set_ylabel(family+'s',fontsize=7.4,labelpad=1)
        bx.tick_params(labelsize=6.3,length=2,pad=2);bx.spines[['top','right']].set_visible(False)
        bx.plot([1,2],[upper*.84]*2,color='#777777',lw=.6)
        bx.text(1.5,upper*.87,'***',ha='center',fontsize=8)
    axs={}
    placements={'c':grid[1,2],'d':grid[2,2],'e':grid[2,1],'f':grid[2,0]}
    labels={'APtotal':'APs of origin county','MPtotal':'MPs of origin county',
        'APtotal_dest':'APs of destination partners','MPtotal_dest':'MPs of destination partners'}
    for row in stats.itertuples():
        ax=fig.add_subplot(placements[row.panel]);axs[row.panel]=ax
        xx=cy[row.x].to_numpy();yy=cy.outflow_rate.to_numpy()
        fitted=sm.OLS(yy,sm.add_constant(xx)).fit()
        gx=np.linspace(xx.min(),xx.max(),100);pred=fitted.get_prediction(sm.add_constant(gx)).summary_frame()
        ax.scatter(xx,yy,s=9,color='#add0dd',edgecolors='#738d98',linewidths=.25,alpha=.55,zorder=2)
        ax.fill_between(gx,pred.mean_ci_lower,pred.mean_ci_upper,color='#d9d9d9',alpha=.75,zorder=0)
        ax.plot(gx,pred['mean'],color='#777777',linewidth=.6,zorder=1)
        ax.set_ylim(0,.047);ax.set_yticks([0,.015,.03,.045]);ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.set_xlabel(labels[row.x],fontsize=7.4,labelpad=3)
        ax.set_ylabel('Outflow rate',fontsize=7.4,labelpad=3);ax.tick_params(labelsize=6.8,length=2)
        ax.spines[['top','right']].set_visible(False)
        def stars(p):return '***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''
        text=f'r = {row.Pearson_r:.3f}{stars(row.Pearson_p)}\nβ = {row.Spearman_beta:.3f}{stars(row.Spearman_p)}\nρ = {row.OLS_rho:.4f}{stars(row.OLS_p)}'
        ax.text(.98,.98,text,transform=ax.transAxes,ha='right',va='top',fontsize=6.5,linespacing=1.05)
        ax.annotate(row.panel,(0,1),xycoords='axes fraction',xytext=(-17,6),textcoords='offset points',fontsize=12,weight='bold')
    # The map is intentionally a 2x2 hero panel. Quantitative bottom panels
    # share one rendered rectangle size, and the right column is aligned.
    save(fig,'Fig1',axes=[amap,b,axs['c'],axs['d'],axs['e'],axs['f']],panel_ids=['a','b','c','d','e','f'],
        row_groups=[['d','e','f']],column_groups=[['b','c','d']],
        exemptions=[{'panels':['a'],'checks':['row','column','panel-label'],
        'reason':'The WGS84 map spans four grid cells and has a geographic aspect ratio; its content rectangle is intentionally different from the quantitative panels.'}])


def policy_maps(geo,stem):
    data=pd.read_csv(DATA/f'maps_correlations_chord/{stem}_county_means.csv')
    breaks=pd.read_csv(DATA/'maps_correlations_chord/FigS2_FigS3_Jenks_breaks.csv')
    fig,axes=plt.subplots(4,2,figsize=(183/25.4,231/25.4))
    fig.subplots_adjust(left=.015,right=.985,top=.985,bottom=.105,wspace=.035,hspace=.17)
    suffix='' if stem=='FigS2' else '_dest'
    for r,typ in enumerate(TYPES):
        for c,family in enumerate(['AP','MP']):
            ax=axes[r,c];basemap(ax,geo);col=family+typ+suffix
            x,y=point_coordinates(data,'Longitude','Latitude')
            ax.scatter(x,y,s=17*MAP_MARKER_AREA_SCALE,c=data[col+'_Jenks_class'].map(lambda k:COLORS[int(k)-1]),
                edgecolors='#aaaaaa',linewidths=.3,zorder=3,clip_on=False)
            ax.text(.5,-.075,TITLES[r]+' '+family,transform=ax.transAxes,ha='center',va='top',fontsize=8.1)
    for col,family in enumerate(['AP','MP']):
        bb=breaks[(breaks.Figure==stem)&(breaks.Policy_class==family)].sort_values('Class')
        edges=[bb.Lower.iloc[0],*bb.Upper_inclusive.tolist()]
        cax=fig.add_axes([.04+col*.50,.035,.42,.012])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(edges,5),cmap=ListedColormap(COLORS)),cax=cax,orientation='horizontal',ticks=edges,spacing='uniform')
        cb.ax.set_xticklabels([f'{v:.3g}' for v in edges]);cb.ax.tick_params(labelsize=6.5,pad=1,length=1.5)
        cb.set_label('Mean policy-type label count (Jenks)',fontsize=7,labelpad=2);cb.outline.set_linewidth(.4)
    save(fig,stem,axes=list(axes.flat),panel_ids=[f'{r}{c}' for r in range(4) for c in range(2)])


def chord():
    sectors=pd.read_csv(DATA/'maps_correlations_chord/ExtendedDataFig1_sector_mapping.csv').sort_values('Order_clockwise')
    edges=pd.read_csv(DATA/'maps_correlations_chord/ExtendedDataFig1_all_observed_directed_state_edges.csv')
    ids=sectors.State_FIPS.astype(int).tolist();idx={f:i for i,f in enumerate(ids)};matrix=np.zeros((len(ids),len(ids)))
    for row in edges.itertuples():matrix[idx[row.origin_state_fips],idx[row.destination_state_fips]]=row.observed_n2
    incidence=matrix.sum(1)+matrix.sum(0)
    assert np.allclose(incidence,sectors.Sector_incidence_n2)
    degrees=sectors.Sector_degrees.to_numpy();gaps=sectors.Gap_degrees.to_numpy()
    starts=90-np.r_[0,np.cumsum(degrees[:-1]+gaps[:-1])];ends=starts-degrees
    factor=degrees/incidence
    out_stub={};in_stub={}
    for i in range(len(ids)):
        cursor=starts[i]
        for j in range(len(ids)):
            out_stub[i,j]=(cursor,cursor-matrix[i,j]*factor[i]);cursor-=matrix[i,j]*factor[i]
        for j in range(len(ids)):
            in_stub[j,i]=(cursor,cursor-matrix[j,i]*factor[i]);cursor-=matrix[j,i]*factor[i]
        assert abs(cursor-ends[i])<1e-8
    fig,ax=plt.subplots(figsize=(183/25.4,183/25.4));fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.985)
    ax.set_aspect('equal');ax.set_xlim(-1.66,1.66);ax.set_ylim(-1.66,1.66);ax.set_axis_off()
    def xy(angle,radius=1):
        angle=np.deg2rad(angle);return np.array([radius*np.cos(angle),radius*np.sin(angle)])
    for i in range(len(ids)):
        ax.add_patch(Wedge((0,0),1.02,ends[i],starts[i],width=.055,facecolor=sectors.Color.iloc[i],edgecolor='white',linewidth=.4))
    for row in edges.itertuples():
        i,j=idx[row.origin_state_fips],idx[row.destination_state_fips]
        a,b=out_stub[i,j];c,d=in_stub[i,j]
        ar=np.linspace(a,b,max(2,int(abs(a-b)/1.5)+2));br=np.linspace(c,d,max(2,int(abs(c-d)/1.5)+2))
        vertices=[xy(a,.96)];codes=[MplPath.MOVETO]
        for angle in ar[1:]:vertices.append(xy(angle,.96));codes.append(MplPath.LINETO)
        vertices.extend([xy(b,.10),xy(c,.10),xy(c,.96)]);codes.extend([MplPath.CURVE4]*3)
        for angle in br[1:]:vertices.append(xy(angle,.96));codes.append(MplPath.LINETO)
        vertices.extend([xy(d,.10),xy(a,.10),xy(a,.96)]);codes.extend([MplPath.CURVE4]*3)
        vertices.append(xy(a,.96));codes.append(MplPath.CLOSEPOLY)
        ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor=sectors.Color.iloc[i],edgecolor='none',alpha=.42))
    # Repel label angles while retaining clockwise order; leaders identify
    # the precise sector when a small sector's larger text is shifted.
    centres=(starts+ends)/2
    targets=-centres.copy()
    # Place all labels with a hard circular separation, without changing the
    # sector angles or ribbon data. A spring-only repeller can still collide.
    result=minimize(lambda a:np.sum((a-targets)**2),targets,
        jac=lambda a:2*(a-targets),method='SLSQP',
        constraints=[{'type':'ineq','fun':lambda a:np.diff(a)-8.2},
                     {'type':'ineq','fun':lambda a:360-(a[-1]-a[0])-8.2}],
        options={'maxiter':1000,'ftol':1e-10})
    if not result.success:raise RuntimeError(result.message)
    angles=result.x
    for i,row in enumerate(sectors.itertuples()):
        angle=-angles[i];point=xy(angle,1.22);true=xy(centres[i],1.025);end=xy(angle,1.16)
        ax.plot([true[0],end[0]],[true[1],end[1]],color='#999999',lw=.35,zorder=4)
        val=row.Sector_incidence_n2
        label=f'{val/1e6:.2f}M' if val>=1e6 else f'{val/1000:.1f}k'
        rotation=angle if point[0]>=0 else angle+180
        ha='left' if point[0]>=0 else 'right'
        ax.text(*point,f'{row.State}\n{label}',rotation=rotation,rotation_mode='anchor',ha=ha,va='center',fontsize=8.5,linespacing=1.08,zorder=5)
    save(fig,'ExtendedDataFig1',axes=[ax],panel_ids=['network'])
    (QA/'network_integrity.json').write_text(json.dumps({'directed_edges':len(edges),'states':len(sectors),'total_flow_events':int(matrix.sum()),'sector_incidence_matches':True,'font_points':8.5},indent=2),encoding='utf-8')


def main():
    geo=boundaries();overview(geo);print('Fig1 complete',flush=True)
    for stem in ['FigS2','FigS3']:policy_maps(geo,stem);print(stem+' complete',flush=True)
    chord();print('ExtendedDataFig1 complete',flush=True)
    (QA/'geographic_revision.json').write_text(json.dumps({'map_crs':'EPSG:4326','source_boundary_crs':geo['source_crs'],'figure1_counties':107,'supplementary_counties_per_role':107,'plotting_backend':'Python Matplotlib','data_exclusions_added':False},indent=2),encoding='utf-8')


if __name__=='__main__':main()
