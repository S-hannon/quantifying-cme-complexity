import os
import sys
import warnings
import csv
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image, ImageDraw
import scipy.stats as sps
import pandas as pd
from astropy.utils.exceptions import AstropyWarning
import bw_img_complexity as bw
from popfss_storm_front_workflow import POPFSSStormFrontWorkflow
import img_data as data
sys.path.insert(0, os.path.join('..', 'useful_code'))
from data_cme_complexity import CMEComplexity
from data_stereo_hi import STEREOHI
from data_helcats import HELCATS
from misc import find_nearest
from plotting_stereo import STEREOPlot
from config import data_loc
import misc
import data2df
fig_loc = os.path.join('..', '..', 'Plots')
warnings.simplefilter('ignore', category=AstropyWarning)

name_list = ['HCME_A__20090211_01', 'HCME_A__20110130_01',
             'HCME_B__20120831_01']
img_type_list = ['diff', 'diff_bg', 'diff_mask_cme', 'diff_mask_bg']
labels = {'diff' : 'CME Image',
          'diff_mask_cme' : 'Split: CME',
          'diff_mask_bg' : 'Split: Background',
          'diff_bg' : 'Background Image',
          'sta' : 'STEREO-A',
          'stb' : 'STEREO-B',
          'area' : 'Area',
          'slope' : 'Slope',
          'q' : 'Q',
          'newq' : 'Q for srange:',
          's' : 'V at scale-size S=',
          'complexity' : 'Visual Complexity',
          'time' : 'Time'}
colors = {'diff' : '#d73027',
          'diff_mask_cme' : '#fc8d59',
          'diff_mask_bg' : '#91bfdb',
          'diff_bg' : '#4575b4',
          'sta' : 'pink',
          'stb' : 'lightskyblue',
          'sta_means' : 'crimson',
          'stb_means' : 'navy'}
markers = {'diff' : 'o',
          'diff_mask_cme' : '^',
          'diff_mask_bg' : 's',
          'diff_bg' : '*',
          'sta' : 'o',
          'stb' : '*'}


def scale_size(helcats_name='HCME_B__20120831_01',
               s=128, img_type='diff', L=2):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    img = data.load_img(helcats_name, img_type)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.putalpha(200)
    draw = ImageDraw.Draw(img)
    N, M = img.size
    x, y = 32, 500
    draw.polygon([(x,y), (x+s,y), (x+s,y+s), (x,y+s)],
                 outline=None, fill=255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img, cmap='gray',  interpolation='nearest') 
    

def scale_size_demonstration(helcats_name='HCME_B__20120831_01',
                             s=64, img_type='diff', L=2):
    fig, ax = plt.subplots(1, 1)
    ax.set_title(labels[img_type] + '\n' + helcats_name, fontsize=12)
    img = data.load_img(helcats_name, img_type)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.putalpha(200)
    draw = ImageDraw.Draw(img)
    N, M = img.size
    s_list = [8, 64, 128]
    pix_between = (N - sum(s_list)) / (len(s_list) + 2)
    x, y = 32, 0
    for s in s_list:
        print(s)
        draw.polygon([(x,y), (x+s,y), (x+s,y+s), (x,y+s)],
                     outline=None, fill=255)
        y = y + pix_between + s
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img, cmap='gray',  interpolation='nearest')   


def plot_img(helcats_name, img_type, fig=None, ax=None):
    if fig == None:
        fig, ax = plt.subplots(1, 1)
        ax.set_title(labels[img_type] + '\n' + helcats_name, fontsize=16)
    else:
        ax.set_title(helcats_name, fontsize=16)
    img = data.load_img(helcats_name, img_type)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img, cmap='gray',  interpolation='nearest')   
    return fig, ax


def plot_img_types(helcats_name_list):
    """Plot of 3 cmes (columns) vs 5 rows: row 1 differenced images, row 2
    differenced images + classifications, row 3 differenced images + consensus
    fronts, row 4 masked differenced images, row 5 background of masked
    differenced images.
    """
    fig, axes = plt.subplots(5, 3, figsize=(10, 17))
    workflow = POPFSSStormFrontWorkflow()
    for i, helcats_name in enumerate(helcats_name_list):
        frame = workflow.load_storm_front(helcats_name)
        for j in [0, 1, 2]:
            fig, axes[j, i] = frame.plot_diff_img(fig, axes[j, i], data_src=1,
                                                  label=False)
        axes[0, i].set_title(helcats_name, size=16)
        fig, axes[1, i] = frame.add_storm_fronts(fig, axes[1, i], 'raw_classifications')
        fig, axes[2, i] = frame.add_storm_fronts(fig, axes[2, i], 'consensus_fronts')
        for k, invert in enumerate([False, True]):
            l = k + 3
            img = POPFSSStormFrontWorkflow.load_masked_img(helcats_name,
                                                           'diff',
                                                           data_src=1,
                                                           invert=invert)
            axes[l, i].imshow(img, cmap='gray',  interpolation='nearest')
            axes[l, i].set_xticks([])
            axes[l, i].set_yticks([])
    plt.tight_layout()


def plot_3_img(helcats_name_list, img_type):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for n, name in enumerate(helcats_name_list):
        fig, axes[n] = plot_img(name, img_type, fig=fig, ax=axes[n])
    plt.tight_layout()


def plot_s_vs_v_3cmes(helcats_name_list, xlim=(1, 10000), ylim=(1, 10000)):
    fig, axes = plt.subplots(1, len(helcats_name_list), figsize=(12, 8),
                             sharey=True)
    for n, img_type in enumerate(img_type_list):
        df = data.load_s_and_v_all_cmes(img_type)
        for m, helcats_name in enumerate(helcats_name_list):
            i = df.index[df['helcats_name'] == helcats_name][0]
            if m == 0:
                ylabel = True
            else:
                ylabel = False
            xlabel = True
            fig, axes[m] = bw.plot_s_vs_v(fig, axes[m], xlim=xlim, ylim=ylim,
                                          xlabel=xlabel, ylabel=ylabel,
                                          aspect=True, fontsize=16)
            fig, axes[m] = bw.add_s_vs_v(fig, axes[m],
                                         df['s_list'][i], df['v_list'][i],
                                         label=labels[img_type],
                                         color=colors[img_type],
                                         marker=markers[img_type],
                                         markersize=15)
            axes[m].set_title(helcats_name, fontsize=16)
            axes[m].set_xticks([1, 10, 100, 1000, 10000])
    plt.tight_layout()
    box = axes[len(helcats_name_list)-1].get_position()
    axes[len(helcats_name_list)-1].set_position([box.x0, box.y0, box.width,
                                                 box.height])
    axes[len(helcats_name_list)-1].legend(loc='center left',
                                          bbox_to_anchor=(1.05, 0.5),
                                          prop={'size': 16})
    
        
def get_label(name, s_lo=10, s_hi=40):
    if name.startswith('s') and name !='slope':
        return ('').join([labels['s'], name[1:len(name)]])
    elif name == 'q':
        return labels[name]
    else:
        return '%s (S=%s to S=%s)'%(labels[name], s_lo, s_hi)


def plot_x_vs_y(xname, yname, s_lo=10, s_hi=40, method=0, figsize=(10, 10),
                means=False, err='sem', corr=True, markersize=False, 
                ylim=False, xlim=False, ylog=False, ylim_list=False):
    """
    xname: 'complexity', 'time'
    yname: 'q', 'newq', 'area' or 'slope'
    """
    if ylim_list:
        fig, axes = plt.subplots(2, 2, sharex=True, figsize=figsize)     
    else:
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                                 figsize=figsize)           
    i = 0
    j = 0
    for n, img_type in enumerate(img_type_list):
        df = data.load_s_and_v_all_cmes(img_type)
        df = data.add(df=df, name=yname, s_lo=s_lo, s_hi=s_hi, method=method)
        if ylim_list:
            ylim1 = ylim_list[n]
        else:
            ylim1 = ylim
        if i == 0:
            ylabel = get_label(name=yname, s_lo=s_lo, s_hi=s_hi)
        else:
            ylabel = False
        if j == 1:
            xlabel = labels[xname]
        else:
            xlabel = False
        STEREOPlot.x_vs_y(df=df, col_x=xname, col_y=yname,
                          xlabel=xlabel, ylabel=ylabel, 
                          fig=fig, ax=axes[j, i], means=means, corr=corr,
                          xlim=xlim, ylim=ylim1, ylog=ylog, err=err,
                          markersize=markersize)
        axes[j, i].set_title(labels[img_type], fontsize=16)
        i = i + 1
        if i > 1:
            i = 0
            j = j + 1
    plt.tight_layout()
    

def c_vs_name_split(img_type, name, s_lo=10, s_hi=40, method=0,
                    ylim=False, ylog=False):
    alldf = data.load_s_and_v_all_cmes(img_type)
    dfs = data.split_into_n_groups(alldf, 2)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    plt.suptitle(labels[img_type], fontsize=16)
    for n, df in enumerate(dfs):
        df = data.add(df=df, name=name, s_lo=s_lo, s_hi=s_hi, method=method)
        if n == 0:
            ylabel = get_label(name, s_lo=s_lo, s_hi=s_hi)
        else:
            ylabel = False
        STEREOPlot.x_vs_y(df=df, col_x='complexity', col_y=name,
                          xlabel=labels['complexity'], ylabel=ylabel,
                          xlim=False, ylim=ylim, ylog=ylog,
                          fig = fig, ax=axes[n])


def plot_a_vs_b(name, s_lo=10, s_hi=40, lim=False, lim_list=False, log=False):
    if lim_list:
        sharex = False
        sharey = False
    else:
        sharex = True
        sharey = True
    fig, axes = plt.subplots(2, 2, figsize=[14, 14], sharex=sharex,
                             sharey=sharey)
    i = 0
    j = 0
    for n, img_type in enumerate(img_type_list):
        df = data.load_s_and_v_all_cmes(img_type)
        df = data.add(df=df, name=name, s_lo=s_lo, s_hi=s_hi)
        sta_vals, stb_vals = HELCATS.get_matched_data(df[name],
                                                      df['helcats_name'])
        sta_vals, stb_vals, sta_names, stb_names = HELCATS.get_matched_data(df[name],
                                                                            df['helcats_name'],
                                                                            return_names=True)
        if i == 0:
            ylabel = True
        else:
            ylabel = False
        if j == 1:
            xlabel = True
        else:
            xlabel = False
        if lim:
            this_lim = lim
        elif lim_list:
            this_lim = lim_list[n]
        else:
            this_lim = False
        axes[j, i] = STEREOPlot.a_vs_b(axes[j, i], sta_vals, stb_vals,
                                       label=get_label(name=name, s_lo=s_lo,
                                                       s_hi=s_hi),
                                       lim=this_lim, xlog=log, ylog=log,
                                       xlabel=xlabel, ylabel=ylabel)
        axes[j, i].set_aspect(1)
        axes[j, i].set_title(labels[img_type], fontsize=16)
        i = i + 1
        if i > 1:
            j = j + 1
            i = 0     
            
            
def plot_c_vs_v_3s(s_list=[2, 8, 64, 128]):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    titles = ['CME Image', 'Split: CME only', 'Split: Background only',
              'Background Image']
    for i, img_type in enumerate(['diff', 'diff_mask_cme', 'diff_mask_bg',
                                  'diff_bg']):
        df = data.load_s_and_v_all_cmes(img_type)
        for j, s in enumerate(s_list):
            df = data.add(df=df, name='s'+str(s))
            ids = find_nearest(df.s_list[0], s)
            # Then plot!
            if i == 0:
                ylabel = 'V (s=%s)'%(str(int(df.s_list[0][ids])))
            else:
                ylabel = None
            if j == 3:
                xlabel = 'Visual Complexity'
            else:
                xlabel = None
            if j == 0:
                axes[j, i].set_title(titles[i], fontsize=16)
            axes[j, i] = STEREOPlot.x_vs_y(df=df,
                                           col_x='complexity',
                                           col_y='s'+str(s),
                                           xlabel=xlabel, ylabel=ylabel,
                                           ylim=[1, 10000], ylog=True,
                                           fig=fig, ax=axes[j, i])
    plt.tight_layout()
 

def plot_s_vs_vc_corr():
    fig, ax = plt.subplots(1, 1, figsize=[14, 9])
    for n, img_type in enumerate(img_type_list):
        lens=[]
        df = data.load_s_and_v_all_cmes(img_type)
        corrs = {'sta' : [], 'stb' : []}
        cis = {'sta' : [], 'stb' : []}
        sfcisl = {'sta' : [], 'stb' : []}
        sfcisu = {'sta' : [], 'stb' : []}
        alpha = {'sta' : 0.2, 'stb' : 0.2}
        hatch = {'sta' : None, 'stb' : '|'}
        for craft in ['sta', 'stb']:
            dfs = df[df.craft == craft]
            for s in range(len(df.s_list[0])):
                vals = []
                cs = []
                for i in dfs.index:
                    if ~np.isnan(dfs.v_list[i][s]):
                        cs.append(dfs.complexity[i])
                        vals.append(dfs.v_list[i][s])
                if len(vals) > 0:
                    corr, pval = sps.spearmanr(cs, vals)
                    lens.append(len(vals))
                    corrs[craft].append(corr)
                    rcorrs = get_shuffled_pair_corrs(cs,
                                                     vals, runs=3)
                    ps = np.percentile(rcorrs, [2.5, 97.5])
                    cis[craft].append(ps)
                    
                    
                    # sfc, sfp = get_shuffled_corrs(cs, vals, runs=10)
                    # sfcps = np.percentile(sfc, [2.5, 97.5])
                    # sfcisl[craft].append(sfcps[0])
                    # sfcisu[craft].append(sfcps[1])
                    
                else:
                    corrs[craft].append(np.NaN)
                    cis[craft].append(np.array([np.NaN, np.NaN]))

                    # sfcisl[craft].append(np.NaN)
                    # sfcisu[craft].append(np.NaN)
                
            label = (' ').join([labels[craft], labels[img_type]])
            ax.scatter(df.s_list[0], corrs[craft], color=colors[img_type],
                       label=label, marker=markers[craft])
            ax.fill_between(df.s_list[0], *zip(*cis[craft]),
                            color=colors[img_type],
                            alpha=alpha[craft],
                            hatch=hatch[craft])

            # ax.scatter(df.s_list[0], sfcisl[craft], color='k')
            # ax.scatter(df.s_list[0], sfcisu[craft], color='k')
    ax.set_xscale('log')
    ax.axhline(0, ls='--', color='grey', zorder=0)
    ax.set_xlabel('Scale-size, S', fontsize=16)
    ax.set_ylabel('S.R. Correlation Coefficient all CMEs', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc=0, prop={'size': 12})


def plot_v_vs_s_bg_types():
    """
    'Ghost Ring' : 'HCME_A__20110320_01'
    """
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=3)
    bg_cs = {'Milky Way' : 'HCME_A__20130716_01',
             'Dust' : 'HCME_A__20100818_01',
             'Comet' : 'HCME_B__20130315_01',
             'CME, plain background' : 'HCME_A__20110125_01'}
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1])
    ax0 = plt.subplot(gs[:, 0])
    fig, ax0 = bw.plot_s_vs_v(fig, ax0,
                              xlim=[1, 10000], ylim=[1, 10000],
                              fontsize=16, aspect=True)
    for n, key in enumerate(bg_cs.keys()):
        helcats_name = bg_cs[key]
        ax0.set_xticks([1, 10, 100, 1000, 10000])
        if key == 'CME, plain background':
            img_type_list = ['diff']
        else:
            img_type_list = ['diff_mask_bg']
        for m, img_type in enumerate(img_type_list):
            df = data.load_s_and_v_all_cmes(img_type)
            cme = df[df['helcats_name'] == helcats_name].index[0]
            s_list = df['s_list'][cme]
            v_list = df['v_list'][cme]
            fig, ax0 = bw.add_s_vs_v(fig, ax0, s_list, v_list, key,
                                     cmap(norm(n)), marker='.',
                                     markersize=20)
        ax0.legend()
    # load imgs
    col = 1
    row = 0
    for n, key in enumerate(bg_cs.keys()):
        helcats_name = bg_cs[key]
        if key == 'CME, plain background':
            img = data.load_img(helcats_name, 'diff')
        else:
            img = POPFSSStormFrontWorkflow.load_masked_img(helcats_name,
                                                           'diff',
                                                           invert=True)
        ax = plt.subplot(gs[row, col])
        ax.imshow(img, cmap='gray',  interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(key, fontsize=16, color=cmap(norm(n)))
        col = col + 1
        if col > 2:
            col = 1
            row = row + 1
    plt.tight_layout()


def characterise_all(err='sem'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    for c, craft in enumerate(['sta', 'stb']):
        for n, img_type in enumerate(img_type_list):
            fig, axes[c] = bw.plot_s_vs_v(fig, axes[c],
                                          xlim=(1, 10000), ylim=(1, 10000),
                                          aspect=1)
            df = data.load_s_and_v_all_cmes(img_type)
            dfc = df[df.craft == craft]
            means, errs = data.find_means(dfc, err=err, s_lo=None, s_hi=None)
            fig, axes[c] = bw.add_s_vs_v(fig, axes[c],
                                         df['s_list'].values[0], means,
                                         labels[img_type], colors[img_type],
                                         markers[img_type])
            axes[c].errorbar(df['s_list'].values[0], means,
                             yerr=errs, fmt='none',
                             label=None, color=colors[img_type], marker=None,
                             zorder=0, capsize=3)
            axes[c].set_title(labels[craft], fontsize=16)
            axes[c].legend(loc=0)          
 

def characterise_ab(s_lo=10, s_hi=40, err='sem', matched_only=True):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 14))
    i = 0
    j = 0
    for img_type in img_type_list:
        for c, craft in enumerate(['sta', 'stb']):
            fig, axes[j, i] = bw.plot_s_vs_v(fig, axes[j, i], xlim=(1, 10000),
                                             ylim=(1, 10000), aspect=1)
            df = data.load_s_and_v_all_cmes(img_type)
            dfc = df[df.craft == craft]
            if matched_only:
                match_list = HELCATS.get_match_list(dfc['helcats_name'])
                dfc['match'] = pd.Series(match_list, index=dfc.index)
                dfc = dfc.dropna(subset=['match'])
            means, errs = data.find_means(dfc, err=err, s_lo=None, s_hi=None)
            fig, axes[j, i] = bw.add_s_vs_v(fig, axes[j, i],
                                            df['s_list'].values[0], means,
                                            labels[craft],
                                            colors[craft], markers[craft])
            axes[j, i].errorbar(df['s_list'].values[0], means,
                                yerr=errs, fmt='none',
                                label=None, color=colors[craft],
                                marker=None, zorder=0,
                                capsize=3)
            axes[j, i].legend(loc=0)      
        axes[j, i].legend()
        axes[j, i].set_title(labels[img_type], fontsize=16)
        for x in [s_lo, s_hi]:
            axes[j, i].axvline(x, ls='--', color='gray')
        i = i + 1
        if i > 1:
            i = 0
            j = j + 1       
    
    
def characterise_cmes(n=3, xlim=(1, 10000), ylim=(1, 10000), s_lo=10, s_hi=40,
                      err='sem'):
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=n)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    i = 0
    j = 0
    for img_type in img_type_list:
        df = data.load_s_and_v_all_cmes(img_type)
        dfs = data.split_into_n_groups(df, n)
        if j == 0:
            xlabel = False
        else:
            xlabel = True
        if i == 0:
            ylabel = True
        else:
            ylabel = False            
        fig, axes[j, i] = bw.plot_s_vs_v(fig, axes[j, i], xlim=xlim,
                                         ylim=ylim, xlabel=xlabel,
                                         ylabel=ylabel, aspect=True)
        for k in range(n):
            print(img_type, k)
            print(dfs[k]['s_list'].values[0][23])
            means, errs = data.find_means(dfs[k], err=err, s_lo=None, s_hi=None)
            print(means[23])
            label = str(k + 1)
            if k == 0:
                label = label + ' Least complex'
            elif k == n - 1:
                label = label + ' Most complex'
            fig, axes[j, i] = bw.add_s_vs_v(fig, axes[j, i],
                                            s_list=dfs[k]['s_list'].values[0],
                                            label=label,
                                            v_list=means,
                                            color=cmap(norm(k)),
                                            marker='o')
            axes[j, i].errorbar(dfs[k]['s_list'].values[0], means,
                                yerr=errs, fmt='none',
                                color=cmap(norm(k)), marker=None, zorder=0,
                                capsize=3)
        axes[j, i].legend(prop={'size': 12})
        axes[j, i].set_title(labels[img_type], fontsize=16)
        for x in [s_lo, s_hi]:
            axes[j, i].axvline(x, ls='--', color='gray')
        i = i + 1
        if i > 1:
            i = 0
            j = j + 1
    plt.tight_layout()


def get_shuffled_pair_corrs(x, y, runs=1000):
    corrs = []
    pairs = list(zip(x, y))
    for i in range(1, runs):
        xi, yi = zip(*random.choices(pairs, k=len(pairs)))
        corr, pval = sps.spearmanr(xi, yi, nan_policy='omit')
        corrs.append(corr)
    return corrs


def get_shuffled_corrs(x, y, runs=1000):
    corrs = []
    pvals = []
    for i in range(1, runs):
        random.shuffle(x)
        random.shuffle(y)
        corr, pval = sps.spearmanr(x, y, nan_policy='omit')
        corrs.append(corr)
        pvals.append(pval)
    return corrs, pvals


def plot_corr_significance(x, y, runs=1000, color='red', color2='black',
                           fig=None, ax=None, xlabel=True, ylabel=True):
    """
    How robust is this relationship? Assume no relationship between x and y.
    Shuffle the pairings 1000 times, and calculate correlation coefficient
    for each. If the actual correlation coefficients are far into the tails,
    we can conclude the correlation is likely significant.
    """
    if fig == None:
        fig, ax = plt.subplots(1, 1)
    # plot actual corr
    real_corr, real_pval = sps.spearmanr(x, y, nan_policy='omit')
    ax.axvline(real_corr, color='k', label="Actual correlation coefficient")
    # plot random pairings
    corrs, pvals = get_shuffled_corrs(x, y, runs=runs)
    ax.hist(corrs, color=color, label='%s randomised pairings'%(runs))
    ax.set_xlim((-1, 1))
    ax.legend(loc=0)
    if xlabel:
        ax.set_xlabel("S. R. Correlation Coefficient", fontsize=16)
    if ylabel:
        ax.set_ylabel("Frequency", fontsize=16)
    return fig, ax


def plot_c_vs_name_corr(name, s_lo=10, s_hi=37, method=0, runs=1000):
    fig, axes = plt.subplots(2, 2, figsize=[8, 8], sharex=True, sharey=True)
    if name.startswith('s') and name !='slope':
        yname = ('').join([labels['s'], name[1:len(name)]])
    elif name == 'q':
        yname = labels[name]
    elif s_lo != None and s_hi != None:
        yname = "%s (S=%s to S=%s)"%(labels[name], s_lo, s_hi)
    title = '%s vs. %s'%(labels['complexity'], yname)
    plt.suptitle(title, fontsize=16)
    i = 0
    j = 0
    for n, img_type in enumerate(img_type_list):
        df = data.load_s_and_v_all_cmes(img_type)
        df = data.add(df=df, name=name, s_lo=s_lo, s_hi=s_hi, method=method)
        if i == 0:
            ylabel = True
        else:
            ylabel = False
        if j == 1:
            xlabel = True
        else:
            xlabel = False
        fig, axes[j, i] = plot_corr_significance(x=df['complexity'].values,
                                                  y=df[name].values,
                                                  color=colors[img_type],
                                                  fig=fig, ax=axes[j, i],
                                                  xlabel=xlabel, ylabel=ylabel,
                                                  runs=runs)
        axes[j, i].set_title(labels[img_type], fontsize=16)
        i = i + 1
        if i > 1:
            j = j + 1
            i = 0


def significance_testing(s_lo=10, s_hi=37):
    for n, img_type in enumerate(img_type_list):
        print(labels[img_type])
        alldf = data.load_s_and_v_all_cmes(img_type)
        
        print('Complex vs Simple')
        dfs = data.split_into_n_groups(alldf, n=2)
        points0, slist = data.find_points(dfs[0], s_lo=s_lo, s_hi=s_hi)
        points1, slist = data.find_points(dfs[1], s_lo=s_lo, s_hi=s_hi)
        for s in range(len(points0)):
            stat, pval = sps.ttest_rel(points0[s], points1[s])
            print('s', slist[s], 'stat', stat, 'pval', pval)
        
        print('A vs B')
        dfa = alldf[alldf['craft'] == 'sta']
        dfb = alldf[alldf['craft'] == 'stb']
        pointsa, slist = data.find_points(dfa, s_lo=s_lo, s_hi=s_hi)
        pointsb, slist = data.find_points(dfb, s_lo=s_lo, s_hi=s_hi)
        for s in range(len(pointsa)):
            stat, pval = sps.ttest_ind(pointsa[s], pointsb[s])
            print('s', slist[s], 'stat', stat, 'pval', pval)
        print()


def srange_test(s_lo, s_hi):
    for n, img_type in enumerate(img_type_list):
        print(labels[img_type])
        df = data.load_s_and_v_all_cmes(img_type)
        for name in ['area', 'slope']:
            df = data.add(df=df, name=name, s_lo=s_lo, s_hi=s_hi)
            corr, pval = sps.spearmanr(df['complexity'].values,
                                       df[name].values,
                                       nan_policy='omit')
            print(name, 'corr', "{:.2f}".format(corr), 'pval',
                  "{:.2f}".format(pval))
        print()


def plot_s_corrs(img_type, name, xlabel=True, ylabel=True, fig=None, ax=None,
                 colorbar=True):
    # get the appropriate corr data
    df = data.get_c_corr_sranges()
    df = df[df['img_type'] == img_type]
    df = df[df['name'] == name]
    s_list = eval(df['s_list'][df.index[0]], {'nan' : np.NaN})
    C = eval(df['C'][df.index[0]], {'nan' : np.NaN})
    # make a plot if not given
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # actually do the plotting
    plot = ax.pcolormesh(s_list, s_list, C, vmin=-1, vmax=1)
    if colorbar:
        cb = fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label='S. R. Correlation Coefficient', fontsize=16)
    if xlabel:
        ax.set_xlabel('Upper Scale-Size, S', fontsize=16)
    if ylabel:
        ax.set_ylabel('Lower Scale-Size, S', fontsize=16)
    ax.set_title("%s of %s"%(labels[name], labels[img_type]), fontsize=16)
    ax.set_aspect(1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    return fig, ax


def plot_all_s_corrs(name):
    fig, axes = plt.subplots(2, 2, figsize=[14, 14], sharex=True, sharey=True)
    i = 0
    j = 0
    for n, img_type in enumerate(img_type_list):
        if i == 0:
            ylabel = True
        else:
            ylabel = False
        if j > 0:
            xlabel = True
        else:
            xlabel = False
        fig, axes[j, i] = plot_s_corrs(img_type, name, colorbar=False,
                                       xlabel=xlabel, ylabel=ylabel,
                                       fig=fig, ax=axes[j, i])
        i = i + 1
        if i > 1:
            j = j + 1
            i = 0
        cb = mpl.colorbar.ColorbarBase(fig.add_axes([0.85, 0.15, 0.05, 0.7]),
                                       cmap=mpl.cm.viridis,
                                       norm=mpl.colors.Normalize(vmin=-1,
                                                                 vmax=1))
        cb.set_label(label='S. R. Correlation Coefficient', fontsize=16)
        fig.subplots_adjust(right=0.8, wspace=0.1, hspace=0.01)
        

def cme_size():    
    diff = data.load_s_and_v_all_cmes('diff')
    diff_mask_cme = data.load_s_and_v_all_cmes('diff_mask_cme')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    diff = data.add(diff, 's128')
    diff_mask_cme = data.add(diff_mask_cme, 's128')
    axes[0].scatter(diff['s128'], diff_mask_cme['s128'], s=10)
    axes[0].set_xlim((0, 6000))
    axes[0].set_ylim((0, 6000))
    axes[0].set_xlabel('diff v at s=128', fontsize=16)
    axes[0].set_ylabel('diff_mask_cme v at s=128', fontsize=16)
    corr, pval = sps.spearmanr(diff['s128'], diff_mask_cme['s128'], nan_policy='omit')
    axes[0].set_title(corr)
    
    diff = data2df.add_width_to_df(diff)
    diff_mask_cme = data2df.add_width_to_df(diff_mask_cme)
    axes[1].scatter(diff['width'], diff['s128'], s=10)
    axes[1].set_xlim((0, 150))
    axes[1].set_ylim((0, 2000))
    axes[1].set_xlabel('diff width', fontsize=16)
    axes[1].set_ylabel('diff v at s=128', fontsize=16)
    corr, pval = sps.spearmanr(diff['width'], diff['s128'], nan_policy='omit')
    axes[1].set_title(corr)
    
    axes[2].scatter(diff['width'], diff_mask_cme['s128'], s=10)
    axes[2].set_xlim((0, 150))
    axes[2].set_ylim((0, 6000))
    axes[2].set_xlabel('diff width', fontsize=16)
    axes[2].set_ylabel('diff_mask_cme v at s=128', fontsize=16)
    corr, pval = sps.spearmanr(diff['width'], diff_mask_cme['s128'], nan_policy='omit')
    axes[2].set_title(corr)
        
    
def fit_lm():
    import statsmodels.api as sm
    diff = data.load_s_and_v_all_cmes('diff')
    diff = data.add(diff, 's128')
    diff = data2df.add_width_to_df(diff)
    diff_mask_cme = data.load_s_and_v_all_cmes('diff_mask_cme')
    diff_mask_cme = data.add(diff_mask_cme, 's128')
    y = diff['s128']
    xs = np.array([diff['width'], diff_mask_cme['s128']])
    xs = np.transpose(xs)
    # Fit and summarize OLS model
    # smf.ols(formula='Y_variable ~ X_variable', data=df)
    mod = sm.OLS(y, xs, missing='drop')
    res = mod.fit()
    print(res.summary())
    print(res.params)
    
    
###############################################################################
def plot_3_resized_images(img_list, s_list, L=2, resample=1,
                          title_list=[None, None, None], suptitle=None):
    for in_list in [s_list, img_list, title_list]:
        if len(in_list) != 3:
            raise ValueError("input list %s must have length 3"%(in_list))
    f, axes = plt.subplots(4, 3, figsize=(10, 14))
    if suptitle:
        f.suptitle(suptitle, fontsize=16)
    # append L of original image to the start
    s_list = [L] + s_list
    for i, img in enumerate(img_list):
        N, M = img.size
        # loop over each resized image
        for j, s in enumerate(s_list):
            # resize the image
            Nr = int(L * N / s)
            Mr = int(Nr / (N / M))
            img_r_img = img.resize((Nr, Mr), resample=resample)
            axes[j, i].imshow(img_r_img, interpolation="nearest", cmap=plt.cm.gray)
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
            title_str = "%s x %s pixels, S: %s"%(Nr, Mr, "{:.2f}".format(s))
            # if first row, add CME name too
            if j == 0:
                if title_list != [None, None, None]:
                    title_str = title_list[i] + "\n" + title_str
            axes[j, i].set_title(title_str, fontsize=16)
    plt.tight_layout()

            
def plot3vmaps(name_list, s_list, img_type, vmin=0.1, vmax=10000,
               L=2, resample=1, data_src=0):
    # if vmin is None:
    #     all_vmin = np.nanmin(df['vmin'].values)
    # else:
    #     all_vmin = vmin
    # if vmax is None:
    #     all_vmax = np.nanmax(df['vmax'].values)
    # else:
    #     all_vmax = vmax
    fig, axes = plt.subplots(4, 3, figsize=(10, 14), sharex=True, sharey=True)
    # loop over each resized image
    for j, s in enumerate(s_list):
        for i, helcats_name in enumerate(name_list):
            if j == 0:
                axes[j, i].set_title(helcats_name, fontsize=12)
            if img_type in ['diff', 'diff_bg']:
                img = data.load_img(helcats_name, img_type)
            else:
                invert = False
                if img_type == 'diff_mask_bg':
                    invert = True
                img = POPFSSStormFrontWorkflow.load_masked_img(helcats_name,
                                                               img_type,
                                                               invert=invert,
                                                               data_src=data_src)
            N, M, v_vals, vmini, vmaxi = data.get_img_v_data(img, helcats_name, s,
                                                             img_type, L=L,
                                                             resample=resample)
            fig, axes[j, i] = bw.plot_v_map(fig, axes[j, i], v_vals,
                                            vmin=vmin, vmax=vmax)


def plot3(img_list, name_list, s_list, img_type, func,
          sharex=False, sharey=False,
          xlim=None, ylim=None, vmin=None, vmax=None):
    """sharey refers to each s value having a different y.
    """
    fig, axes = plt.subplots(4, 3, figsize=(10, 14), sharex=sharex)
    # loop over each resized image
    for j, s in enumerate(s_list):
        for i, img in enumerate(img_list):
            # Add labels to the correct places
            if j == len(s_list) - 1:
                xlabel = True
            else:
                xlabel = False
            if i == 0:
                ylabel = True
            else:
                ylabel = False
            fig, axes[j, i] = func(fig, axes[j, i],
                                   img=img, s=s, name=name_list[i], 
                                   img_type=img_type,
                                   vmin=vmin, vmax=vmax,
                                   xlabel=xlabel, ylabel=ylabel,
                                   xlim=xlim, ylim=ylim)
            # if first row, add CME name too
            # if j == 0:
            #     axes[j, i].set_title(('\n').join([name_list[i],
            #                                       df_img_s['title'].values[0]]),
            #                           fontsize=12)
