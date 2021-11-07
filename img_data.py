import sys
import os
import csv
import numpy as np
import pandas as pd
import scipy.stats as sps
from PIL import Image, ImageDraw
import bw_img_complexity as bw
sys.path.insert(0, os.path.join('..', 'useful_code'))
from data_cme_complexity import CMEComplexity
from data_stereo_hi import STEREOHI
from data_helcats import HELCATS
from config import data_loc
from misc import find_nearest


########################## Load Images #######################################
def is_flipped(helcats_name):
    """Returns True if the image was flipped horizontally before being
    uploaded to the project, and False if not.
    """
    craft, time = HELCATS.get_cme_details(helcats_name)
    if craft == 'stb':
        return True
    elif time.year > 2015:
        return True
    else:
        return False


def load_img_file(helcats_name, img_type):
    craft, time = HELCATS.get_cme_details(helcats_name)
    flip = is_flipped(helcats_name)
    img = STEREOHI.load_img(helcats_name, craft, img_type, 'POPFSS', flip=flip)
    return img


def load_masked_img(helcats_name, img_type, invert=False, data_src=0):
    """Loads the image for cme with helcats_name with mask layer applied.
    img_type: 'diff', 'norm', 'diff_be'
    """
    # load the image
    craft, time = HELCATS.get_cme_details(helcats_name)
    img = STEREOHI.load_img(helcats_name, craft, img_type, 'POPFSS')
    # define the mask
    out_path = os.path.join(data_loc(), 'protect-our-planet-from-solar-storms',
                            'all_cme_mask_points.csv')
    df = pd.read_csv(out_path, converters={'mask_points' : str})
    i_cme = df.index[df['helcats_name'] == helcats_name][0]
    mask_points = eval(df['mask_points'][i_cme], {'nan' : np.NaN})
    # now add the mask layer
    if invert:
        mask_img = Image.new("L", img.size, 255)
        draw = ImageDraw.Draw(mask_img)            
        draw.polygon(mask_points, fill=0)
    else:
        mask_img = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask_img)            
        draw.polygon(mask_points, fill=255)     
    img.putalpha(mask_img)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img
  

def load_img(helcats_name, img_type):
    if img_type in ['norm', 'diff', 'diff_be', 'diff_bg']:
        return load_img_file(helcats_name, img_type)
    elif img_type == 'diff_mask_cme':
        return load_masked_img(helcats_name, 'diff')
    elif img_type == 'diff_mask_bg':
        return load_masked_img(helcats_name, 'diff', invert=True)
    else:
        raise ValueError("img_type %s not recognised"%(img_type))
  
    
############################ Get V and S data ################################
def get_cme_list(img_type):
    # get a list of all CMEs in complexity ranking
    if img_type in ['diff', 'norm', 'diff_be']:
        return CMEComplexity.load(img_type)
    elif img_type in ['diff_bg', 'diff_mask_cme', 'diff_mask_bg',
                      'diff_bg_half']:
        return CMEComplexity.load('diff')
    else:
        raise ValueError("img_type %s invalid"%(img_type))    
    
    
def save_s_and_v_all_cmes(img_type, i, j=100, L=2, resample=1, n=50,
                          mask_val=255):
    df = get_cme_list(img_type)
    if 'mask' in img_type:
        mask = True
    else:
        mask = False
    # only run code on a subset of this
    df = df[i:i+j]
    # set up output file
    file_name = os.path.join(data_loc(),
                             ("_").join(["s_v_list_all_cmes", img_type,
                                         str(i) + ".csv"]))
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['helcats_name', 'complexity', 'c_rank', 's',
                         'v_map_vals'])
        for k in df.index:
            img = load_img(df.helcats_name[k], img_type) 
            if img != None:
                N, M = img.size
                N_list, M_list = bw.get_img_size_list(N=N, M=M, L=L, n=n)
                for n, Nr in enumerate(N_list):
                    Mr = M_list[n]
                    img_r = img.resize((Nr, Mr), resample=resample)
                    v_list_r, coord_list_r = bw.find_img_v_list(img_r, L=L,
                                                                mask=mask,
                                                                mask_val=mask_val)
                    v_map_vals = bw.v_list_to_v_map_vals(v_list_r,
                                                         coord_list_r)
                    s = L * N / Nr
                    writer.writerow([df.helcats_name[k],
                                     df.complexity[k],
                                     df.c_rank[k],
                                     s,
                                     v_map_vals])


def load_s_and_v_all_boxes_all_cmes(img_type, d=None):
    file_info = img_type
    if d is not None:
        file_info = file_info + '_' + 'd' + str(d)
    df = pd.DataFrame()
    c = {"v_map_vals": str}
    for f in os.listdir(os.path.join(data_loc(), 'Complexity')):
        if f.startswith('s_v_list_all_cmes_' + file_info):
            end = f.split('s_v_list_all_cmes_' + file_info)[1]
            # make sure getting correct 'diff'
            end = end.split('.')[0].split('_')
            if len(end) == 2:
                file_path = os.path.join(data_loc(), 'Complexity', f)
                if df.empty:
                    df = pd.read_csv(file_path, delimiter=',', converters=c)
                else:
                    df2 = pd.read_csv(file_path, delimiter=',', converters=c)
                    df = df.append(df2, ignore_index=True)
    return df


def convert_v_all_boxes_to_means(img_type, include_zero=True, L=2):
    for f in os.listdir(os.path.join(data_loc(), 'Complexity')):
        if f.startswith('s_v_list_all_cmes_' + img_type):
            end = f.split('s_v_list_all_cmes_' + img_type + '_')[1]
            # make sure getting correct img_type
            # e.g. want "diff" but get "diff_bg" or something
            end = end.split('.')[0].split('_')
            if len(end) == 1:
                file_path = os.path.join(data_loc(), 'Complexity', f)
                print(file_path)
                df = pd.read_csv(file_path, delimiter=',',
                                 converters={"v_map_vals": str})
                print(len(df))
                all_helcats_name = []
                all_complexity = []
                all_c_rank = []
                all_s_list = []
                all_v_list = []
                print(np.unique(df['helcats_name']))
                for helcats_name in np.unique(df['helcats_name']):
                    dfh = df[df['helcats_name'] == helcats_name]
                    s_list = []
                    v_list = []
                    for s in np.unique(dfh['s']):
                        dfhs = dfh[dfh['s'] == s]
                        if len(dfhs) > 1:
                            raise ValueError("too many rows?")
                        v_map_vals = eval(dfhs['v_map_vals'][dfhs.index[0]],
                                          {'nan' : np.NaN})
                        mean_v = bw.find_img_av_v(v_list=v_map_vals, L=L,
                                                  include_zero=include_zero)
                        s_list.append(s)
                        v_list.append(mean_v)
                    all_helcats_name.append(helcats_name)
                    all_complexity.append(dfh['complexity'][dfh.index[0]])
                    all_c_rank.append(dfh['c_rank'][dfh.index[0]])
                    all_s_list.append(s_list)
                    all_v_list.append(v_list)
                new_df = pd.DataFrame({'helcats_name' : all_helcats_name,
                                       'complexity' : all_complexity,
                                       'c_rank' : all_c_rank,
                                       's_list' : all_s_list,
                                       'v_list' : all_v_list})
                file_path = os.path.join(data_loc(), 'Complexity',
                                         's_v_list_means_' + img_type + '_' + end[1] + '.csv')
                print(file_path)
                new_df.to_csv(file_path, mode='w', index=False, na_rep="nan")
 

def load_s_and_v_all_cmes(img_type, d=None):
    import data2df  
    file_info = img_type
    if d is not None:
        file_info = file_info + '_' + 'd' + str(d)
    df = pd.DataFrame()
    c = {"s_list": str, "v_list": str}
    for f in os.listdir(os.path.join(data_loc(), 'Complexity', 'means_only')):
        if f.startswith('s_v_list_all_cmes_' + file_info):
            end = f.split('s_v_list_all_cmes_' + file_info)[1]
            # make sure getting correct 'diff'
            end = end.split('.')[0].split('_')
            if len(end) == 2:
                file_path = os.path.join(data_loc(), 'Complexity',
                                         'means_only', f)
                if df.empty:
                    df = pd.read_csv(file_path, delimiter=',', converters=c)
                else:
                    df2 = pd.read_csv(file_path, delimiter=',', converters=c)
                    df = df.append(df2, ignore_index=True)   
    # add craft
    df = data2df.add_craft_and_time_to_df(df)
    # now convert s_list and v_list to arrays from string, sorting nans
    # can't work out how to use eval with extra arguments as a converter
    v_list = []
    s_list = []
    for i in df.index:
        v_list.append(eval(df["v_list"][i], {'nan' : np.NaN}))
        s_list.append(eval(df["s_list"][i], {'nan' : np.NaN}))
    df['v_list'] = pd.Series(v_list, index=df.index)
    df['s_list'] = pd.Series(s_list, index=df.index)
    return df
       

###################### Q, Area and Slope Calculations ########################
def add_v_at_s(df, name):
    s = float(name[1:len(name)])
    ids = find_nearest(df.s_list[0], s)
    v = []
    for i in df.index:
        v.append(df.v_list[i][ids])
    df[name] = pd.Series(v, index=df.index)
    return df


def add_q(df, method=0):
    df = df.sort_values(by='complexity')
    q_list = []
    for ind in df.index:
        q_list.append(bw.find_Q(df['s_list'][ind], df['v_list'][ind],
                                method=method))
    df['q'] = pd.Series(q_list, index=df.index)
    return df


def add_newq(df, s_lo=10, s_hi=37, method=0):
    df = df.sort_values(by='complexity')
    id_lo = find_nearest(df.s_list.values[0], s_lo)
    id_hi = find_nearest(df.s_list.values[0], s_hi) + 1
    q_list = []
    for ind in df.index:
        q_list.append(bw.find_Q(df.s_list[ind][id_lo:id_hi],
                                df.v_list[ind][id_lo:id_hi],
                                method=method))
    df['newq'] = pd.Series(q_list, index=df.index)
    return df


def add_area(df, s_lo=10, s_hi=37):
    id_lo = find_nearest(df.s_list.values[0], s_lo)
    id_hi = find_nearest(df.s_list.values[0], s_hi) + 1
    area_list = []
    for ind in df.index:
        integral = 0
        for m in range(id_lo + 1, id_hi):
            # integrate
            x = df['s_list'][ind][m] - df['s_list'][ind][m-1]
            y = (df['v_list'][ind][m] + df['v_list'][ind][m-1])/2
            new_area = x * y
            integral = integral + new_area
        area_list.append(integral)
    df['area'] = pd.Series(area_list, index=df.index)
    return df


def add_slope(df, s_lo=10, s_hi=37):
    df = df.sort_values(by='complexity')
    id_lo = find_nearest(df.s_list.values[0], s_lo)
    id_hi = find_nearest(df.s_list.values[0], s_hi) + 1
    slope = []
    for ind in df.index:
        out = sps.linregress(df.s_list[ind][id_lo:id_hi], df.v_list[ind][id_lo:id_hi])
        slope.append(out[0])
    df['slope'] = pd.Series(slope, index=df.index)
    return df


def add(df, name, s_lo=10, s_hi=37, method=0):
    if name.startswith('s') and name != 'slope':
        return add_v_at_s(df, name)
    elif name == 'q':
        return add_q(df, method=method)
    elif name == 'newq':
        return add_newq(df, s_lo=s_lo, s_hi=s_hi, method=method)
    elif name == 'area':
        return add_area(df, s_lo=s_lo, s_hi=s_hi)    
    elif name == 'slope':
        return add_slope(df, s_lo=s_lo, s_hi=s_hi)
    else:
        raise ValueError("name %s invalid"%(name))
    
    
##############################################################################
def split_into_n_groups(df, n):
    # make sure sorted by increasing complexity
    df = df.sort_values(by='complexity')
    # now split into n groups
    group_size = int(len(df) / n)
    group_range = range(0, len(df) - group_size + 1, group_size)
    dfs = [df.iloc[ind:ind+group_size] for ind in group_range]
    return dfs


def find_points(df, s_lo=None, s_hi=None):
    if s_lo != None and s_hi != None:
        id_lo = find_nearest(df.s_list.values[0], s_lo)
        id_hi = find_nearest(df.s_list.values[0], s_hi) + 1
    else:
        id_lo = 0
        id_hi = len(df)
    points_list = []
    s_list = []
    for i, s in enumerate(df['s_list'].values[0]):
        if i in range(id_lo, id_hi):
            vals = []
            for v_list in df['v_list']:
                vals.append(v_list[i])
            points_list.append(vals)
            s_list.append(s)
    return points_list, s_list
           

def find_means(df, err='std', s_lo=None, s_hi=None):
    means = []
    errs = []
    points_list, s_list = find_points(df=df, s_lo=s_lo, s_hi=s_hi)
    for vals in points_list:
        means.append(np.mean(vals))
        if err == 'std':
            errs.append(np.std(vals))
        elif err == 'sem':
            errs.append(np.std(vals)/np.sqrt(len(vals)))    
        else:
            raise ValueError('err must be std or sem')
    return means, errs


##############################################################################
def get_c_corr_sranges():
    filepath = os.path.join(data_loc(), 'Complexity',
                            'c_corr_s_ranges_data.csv')
    if os.path.exists(filepath):
        df_all = pd.read_csv(filepath, converters={'img_type' : str,
                                                   'name' : str,
                                                   'C' : str,
                                                   's_list' : str})
    else:
        df_all = pd.DataFrame(columns=['img_type', 'name', 'C', 's_list'])
        for img_type in ['diff', 'diff_bg', 'diff_mask_cme', 'diff_mask_bg']:
            df = load_s_and_v_all_cmes(img_type)
            for name in ['area', 'slope']:
                s_list = df['s_list'][df.index[0]]
                C = []
                for s_lo in s_list:
                    row = []
                    for s_hi in s_list:
                        corr = np.NaN
                        if s_lo < s_hi:
                            df = add(df=df, name=name, s_lo=s_lo, s_hi=s_hi)
                            if np.count_nonzero(~np.isnan(df[name])) >= 3:
                                corr, pval = sps.spearmanr(df['complexity'],
                                                           df[name],
                                                           nan_policy='omit')
                        row.append(corr)
                    C.append(row)
                df_all = df_all.append({'img_type' : img_type, 'name' : name,
                                        'C' : C, 's_list' : s_list},
                                       ignore_index=True)
        df_all.to_csv(filepath, index=False, na_rep="nan")
    return df_all


########################## Run on the Cluster ################################
# save_s_and_v_all_cmes(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
# convert_v_all_boxes_to_means(sys.argv[1], sys.argv[2], int(sys.argv[3]))
