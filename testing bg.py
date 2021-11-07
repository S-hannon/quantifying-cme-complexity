import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from popfss_storm_front_workflow import POPFSSStormFrontWorkflow
import bw_img_complexity as bw
import img_analysis as ima
sys.path.insert(0, os.path.join('..', 'sswii_storm_front'))
from sswii_sf_frame import Frame
sys.path.insert(0, os.path.join('..', 'useful_code'))
from config import data_loc
from data_stereo_hi import STEREOHI
from data_cme_complexity import CMEComplexity
from data_helcats import HELCATS

img = ima.load_img('HCME_A__20090211_01', 'diff_bg')
s_list0, v_list0 = bw.find_s_and_v(img)


def try_run(v_list, control=v_list0):
    plt.figure(figsize=(10, 10))
    plt.scatter(control, v_list)
    plt.xlim((0, 500))
    plt.ylim((0, 500))
    plt.plot([0, 500], [0, 500], ls='--', color='gray')
    print('control: %s'%(control))
    print('v_list: %s'%(v_list))
    print('difference: %s'%([control[i] - v_list[i] for i in range(len(v_list))]))


# try flipping image
# img1 = img.transpose(Image.FLIP_LEFT_RIGHT)
# s_list, v_list = bw.find_s_and_v(img1)
# try_run(v_list)

# edit mean calculation
# import bw_img_complexity as bw
# s_list, v_list = bw.find_s_and_v(img, N_list)
# try_run(v_list)

# remove constraint len(pixels) == L**2
# import bw_img_complexity as bw
# s_list, v_list = bw.find_s_and_v(img, N_list)
# try_run(v_list)


# try loading as colour
img2 = STEREOHI.load_img('HCME_A__20090211_01', 'sta', 'diff_bg', 'POPFSS',
                         flip=False, mode='RGB')
s_list, v_list = bw.find_s_and_v(img2)
try_run(v_list)

