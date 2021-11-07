import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from popfss_storm_front_workflow import POPFSSStormFrontWorkflow
import bw_img_complexity as bw
import img_data as data
from img_analysis import *
sys.path.insert(0, os.path.join('..', 'sswii_storm_front'))
from sswii_sf_frame import Frame
sys.path.insert(0, os.path.join('..', 'useful_code'))
from config import data_loc
from data_stereo_hi import STEREOHI
from data_cme_complexity import CMEComplexity
from data_helcats import HELCATS
import data2df

# #### For "Draw the Storm Front" data
# POPFSSStormFrontWorkflow.save_all_cmes_one_file(points_lim=[3,30],
#                                                 pixel_spacing=8,
#                                                 kernel="epanechnikov",
#                                                 bandwidth=200,
#                                                 thresh=10,
#                                                 data_src=1,
#                                                 print_n=True,
#                                                 kde=True,
#                                                 process=False)


# make sure correct fts files downloaded from hard drives
# df = CMEComplexity.load('diff')
# STEREOHI.check_disk_space(len(df)*2)
# for i in df.index:
#     helcats_name, time, img_type = Frame.get_frame_details(df['f_name'][i])
#     craft, date = HELCATS.get_cme_details(helcats_name)
#     STEREOHI.save_specific_fits_and_previous(time, craft, camera='hi1',
#                                              background_type=1)


# plots
helcats_name_list = ['HCME_A__20090211_01', 'HCME_A__20110130_01',
                      'HCME_B__20120831_01']

## Figure 1: 3 example CME images prior to CMEs appearing in FOV
# plot_3_img(helcats_name_list, 'diff_bg')

## Figure 2: 3 example CME images: differenced only, with 30 fronts, with
# consensus front, CME part only, background part only
# plot_img_types(helcats_name_list)

## Figure 3: 3 example CMEs resized to S=2, 8, 64 and 128
# cme_list = []
# for cme in helcats_name_list:
#     cme_list.append(data.load_img(cme, 'diff'))
# plot_3_resized_images(cme_list, [8, 64, 128], L=2, resample=1,
#                       title_list=helcats_name_list)

## Figure 4: S vs V for 3 example CMEs
# plot_s_vs_v_3cmes(helcats_name_list)

## Figure 5: Complexity vs V for each image type (4x4 subplots)
#plot_c_vs_v_3s()

## Figure 6: Spearman's Rank correlation for all CMEs
#plot_s_vs_vc_corr()

## Figure 7: V(S) for each image type, split into 3 complexity groups
characterise_cmes()

## Figure 8: Complexity vs area for each image type
# plot_x_vs_y('complexity', 'area', ylog=True, ylim=(100, 100000))

## Figure 9: Complexity vs slope for each image type
# plot_x_vs_y('complexity', 'slope', ylim_list=[(-10,25),(-10,15),(-25,80),(-8,5)])

## Figure 10: Time vs area for each image type
# plot_x_vs_y('time', 'area', corr=False, means=True, markersize=10,
#             ylog=True, ylim=(100, 100000), figsize=(10, 8))

## Figure 11: Time vs slope for each image type
# plot_x_vs_y('time', 'slope', ylim_list=[(-10,25),(-10,15),(-25,80),(-10,10)],
#             corr=False, means=True, markersize=10, figsize=(10, 8))

## Figure 12:
# plot_v_vs_s_bg_types() 
    
    
    
    

# plot_x_vs_y('complexity', 's128', ylog=True)
# plot_x_vs_y('complexity', 'q')
# plot_x_vs_y('complexity', 'newq')

# plot_x_vs_y('time', 's128', corr=False,means=True, markersize=10,
#             ylog=True, ylim=(0.1, 10000), figsize=(14,9))

# c_vs_name_split('diff', 'newq', s_lo=10, s_hi=40, method=0, ylim=(0.8, 1.0))

# plot_a_vs_b('slope', all_lim=(-60, 80))
# plot_a_vs_b('slope', lim_list=[(-10, 30),(-6,6),(-10,70),(-6,0)])
# plot_a_vs_b('area', lim=(100, 100000), log=True)
# plot_a_vs_b('area', lim_list=[(1000, 100000), (1000, 100000), (1000, 100000),
#                               (100, 10000)],
#             log=True)

# characterise_all()
# characterise_ab()

# plot_c_vs_name_corr('slope')
# plot_c_vs_name_corr('area')
# plot_c_vs_name_corr('q')
# plot_c_vs_name_corr('newq')

# significance_testing()
# srange_test(s_lo=10, s_hi=37)

# plot_all_s_corrs('slope')
# plot_all_s_corrs('area')

# cme_size()
# fit_lm()

###############################################################################
# bw.plot_3_resized_images(bg_mask, [8, 64, 128], L=2, resample=1,
#                          title_list=name_list)
#s_list = [8, 64, 128]
#img = data.load_img(img, 'diff')## remove
# bw.plot3(img_list=cme_list, name_list=name_list, s_list=s_list,
#          img_type='diff', func=bw.plot_v_map)
# bw.plot3(img_list=mask_list, name_list=name_list, s_list=s_list,
#          img_type='diff_mask', func=bw.plot_v_map, vmin=0.1, vmax=11000)
# bw.plot3(maskv, bw.histv, vmin=None, vmax=None, sharex=True, sharey=True,
#          ylim=(0, 0.3))
# bw.plot3(maskv, bw.mapv, vmin=0.1, vmax=11000)

# what do the resized images look like?
#plot_3_resized_images(helcats_name_list, s_list, L=2, resample=1,
#                      title_list=helcats_name_list)
#                          suptitle='Differenced CME Image')
# bw.plot_3_resized_images(bg_list, s_list, L=2, resample=1,
#                          title_list=name_list,
#                          suptitle='Differenced Background')
# bw.plot_3_resized_images(mask_list, s_list, L=2, resample=1,
#                          title_list=name_list,
#                          suptitle='Differenced CME Image Masked')
