import os
import sys
from datetime import timedelta
sys.path.insert(0, os.path.join('..', 'useful_code'))
from data_stereo_hi import STEREOHI
from data_helcats import HELCATS
from data_cme_complexity import CMEComplexity


df = CMEComplexity.load('diff')
for i in df.index:
    print(df.helcats_name[i], "%s of %s"%(i+1, len(df)))
    craft, date = HELCATS.get_cme_details(df.helcats_name[i])
    start_time, mid, end_time, mid_el = HELCATS.get_te_track_times(df.helcats_name[i])
    if start_time != None:
        STEREOHI.make_img(img_tag=df.helcats_name[i], craft=craft,
                          img_type='diff', project_tag='test',
                          start_time=start_time - timedelta(hours=2),
                          end_time=start_time,
                          data_src=1, img_type_suffix='bg')
        