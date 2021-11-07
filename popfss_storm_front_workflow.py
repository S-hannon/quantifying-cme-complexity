import os
import sys
import warnings
import h5py
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from PIL import Image, ImageDraw
import hi_processing.images as hip
sys.path.insert(0, os.path.join('..', 'sswii_storm_front'))
from sswii_sf_workflow import StormFrontWorkflow
from sswii_sf_frame import Frame
from sswii_sf_storm_front import RawClassification, ProcessedClassification, ConsensusFront
sys.path.insert(0, os.path.join('..', 'useful_code'))
from data_stereo_hi import STEREOHI
from data_helcats import HELCATS
from data_cme_complexity import CMEComplexity
from config import data_loc
fig_loc = os.path.join('..', '..', 'Plots')
warnings.simplefilter('ignore', category=AstropyWarning)
import matplotlib.pyplot as plt


class POPFSSStormFrontWorkflow(StormFrontWorkflow):
    def __init__(self):
        workflow_name = 'Draw the Storm Front'
        project_name = 'Protect Our Planet From Solar Storms'
        StormFrontWorkflow.__init__(self, workflow_name=workflow_name,
                                    project_name=project_name)

    
    @staticmethod
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


    @classmethod
    def save_all_cmes_one_file(cls, points_lim=[3,30], pixel_spacing=8,
                               kernel="epanechnikov", bandwidth=40,
                               thresh=10, data_src=0, print_n=False, kde=True,
                               process=True):
        """This function is designed for protect our planet from solar storms
        not suitable for all the data from solar stormwatch ii!
        """
        workflow = POPFSSStormFrontWorkflow()
        if process:
            workflow.process_classifications()
        # set up hdf5 file
        out_name = os.path.join(workflow.root,
                                ('-').join([workflow.project_savename,
                                            workflow.workflow_savename,
                                            'all-cme-storm-fronts.hdf5']))
        if os.path.exists(out_name):
            os.remove(out_name)
        out_file = h5py.File(out_name, 'w')
        
        # Make the cme from the classification data
        all_df = workflow.load_all_classifications()
        helcats_name_list = np.unique(all_df.helcats_name)
        # helcats_name_list = ['HCME_A__20090211_01', 'HCME_A__20110130_01',
        #                      'HCME_B__20120831_01']
        for i, helcats_name in enumerate(helcats_name_list):
            # is the image flipped?
            flip = workflow.is_flipped(helcats_name)
            if print_n:
                print((" ").join([str(i+1), 'of',
                                  str(len(helcats_name_list)),
                                  helcats_name]))
            df = all_df[all_df['helcats_name'] == helcats_name]
            cme = workflow.construct_cme(helcats_name, df,
                                         points_lim=points_lim,
                                         pixel_spacing=pixel_spacing,
                                         kernel=kernel, bandwidth=bandwidth,
                                         thresh=thresh, data_src=data_src,
                                         kde=kde, flip=flip)
            if cme != None:
                # append the CME data to file
                out_file.create_group(helcats_name)
                # Add cme attributes
                for cme_attr in ['helcats_name', 'craft', 'date_str']:
                    out_file[helcats_name].attrs.create(cme_attr,
                                                        str(getattr(cme,
                                                                    cme_attr)))
                for frame in cme.frames:
                    out_file[helcats_name].create_group(frame.name)
                    # add attributes to frame
                    for f_attr in ['craft', 'camera', 'time_str', 'img_type']:
                        out_file[helcats_name][frame.name].attrs.create(f_attr,
                                                                        str(getattr(frame,
                                                                                    f_attr)))    
                    for sf_type in ['raw_classifications',
                                    'processed_classifications',
                                    'consensus_fronts']:
                        for front_label in getattr(frame, sf_type).keys():
                            out_file.create_group((r'/').join([helcats_name,
                                                               frame.name,
                                                               sf_type,
                                                               front_label]))
                            for n, sf in enumerate(getattr(frame,
                                                           sf_type)[front_label]):
                                # convert pandas df to numpy array with column names
                                sf_dtype = []
                                for col in sf.coords:
                                    sf_dtype.append((col, float))
                                sf_dtype = np.dtype(sf_dtype)
                                data = np.array(list(map(tuple,
                                                         sf.coords.to_numpy(dtype=float))),
                                                dtype=sf_dtype)
                                name = ('_').join([frame.name, sf.name])
                                # in some classifications, there is a second set
                                # of points. Ignore these.
                                try:
                                    # write the array to the file
                                    out_file[helcats_name][frame.name][sf_type][front_label].create_dataset(name,
                                                                                                            data=data)
                                except:
                                    pass
                                # # add attributes to this storm front
                                for sf_attr in ['classification_id', 'subject_id',
                                                'user_id', 'user_ip', 'created_at',
                                                'points_lim', 'pixel_spacing',
                                                'kernel', 'bandwidth', 'thresh']:
                                    if sf_attr in sf.__dict__:
                                        out_file[helcats_name][frame.name][sf_type][front_label][name].attrs.create(sf_attr, 
                                                                                                                    str(getattr(sf, sf_attr)))
        out_file.close()


    def load_storm_front(self, helcats_name):
        hdf5 = os.path.join(data_loc(), self.project_savename,
                            ('-').join([self.project_savename,
                                        self.workflow_savename, 
                                        'all-cme-storm-fronts.hdf5']))
        in_file = h5py.File(hdf5, 'r')
        for f_name in list(in_file[helcats_name].keys()):
            f = Frame(helcats_name,
                      in_file[helcats_name][f_name].attrs['time_str'],
                      in_file[helcats_name][f_name].attrs['img_type'],
                      flip=self.is_flipped(helcats_name))
            for sf_type in list(in_file[helcats_name][f_name].keys()):
                for front_label in list(in_file[helcats_name][f_name][sf_type].keys()): 
                    for sf_n in list(in_file[helcats_name][f_name][sf_type][front_label].values()):
                        # convert coords back to pandas
                        coords_df = pd.DataFrame(data=sf_n[:], columns=sf_n.dtype.names)
                        if sf_type == 'raw_classifications':
                            sf = RawClassification(coords_df,
                                                   front_label,
                                                   sf_n.attrs['classification_id'],
                                                   sf_n.attrs['subject_id'],
                                                   sf_n.attrs['user_id'],
                                                   sf_n.attrs['user_ip'],
                                                   sf_n.attrs['created_at'])
                        elif sf_type == 'processed_classifications':
                            sf = ProcessedClassification(coords_df,
                                                         front_label,
                                                         int(sf_n.attrs['classification_id']),
                                                         int(sf_n.attrs['subject_id']),
                                                         int(sf_n.attrs['user_id']),
                                                         sf_n.attrs['user_ip'],
                                                         sf_n.attrs['created_at'],
                                                         eval(sf_n.attrs['points_lim']),
                                                         int(sf_n.attrs['pixel_spacing']))
                        else:
                            sf = ConsensusFront(coords_df,
                                                front_label,
                                                sf_n.attrs['kernel'],
                                                sf_n.attrs['bandwidth'],
                                                sf_n.attrs['thresh'])
                        f.add_storm_front(sf)
        return f


    def binary_search_pa(self, high, low, pa, sun_side, hi_map):
        # arr is 0 to 1023
        ylist = np.arange(0, 1024, 1)
        if sun_side == 'right':
            x = 1023
        else:
            x = 0
        # Check base case
        if high < low:
            # Element is not present in the array
            return np.NaN, np.NaN      
        # // gives int
        mid = (high + low) // 2
        # If element is present at the middle itself
        midel, midpa = hip.convert_pix_to_hpr(x * u.pix, ylist[mid] * u.pix,
                                              hi_map)
        midpa = midpa.value
        # find difference between midpa and pa
        diff = midpa - pa
        if abs(diff) < 1:
            return x, ylist[mid]
        # If element is smaller than mid, then can only be in left subarray
        # PAs go clockwise from N, so depends on which side of sun
        elif (diff > 1 and sun_side == 'left') or (diff < -1 and sun_side == 'right'):
            return self.binary_search_pa(mid - 1, low, pa, sun_side, hi_map)
        # Else the element can only be present in right subarray
        else:
            return self.binary_search_pa(high, mid + 1, pa, sun_side, hi_map)


    def binary_search_el(self, high, low, el, side, sun_side, hi_map):
        # arr is 0 to 1023
        xlist = np.arange(0, 1024, 1)
        # this is top row in hi_map. Note image will be flipped.
        if side == 'N':
            y = 1023
        else:
            y = 0
        # Check base case
        if high < low:
            # Element is not present in the array
            return np.NaN, np.NaN          
        # // gives int
        mid = (high + low) // 2
        # If element is present at the middle itself
        midel, midpa = hip.convert_pix_to_hpr(xlist[mid] * u.pix, y * u.pix,
                                              hi_map)
        midel, midpa = midel.value, midpa.value
        # find difference between midpa and pa
        diff = midel - el
        if abs(diff) < 1:
            # flip y for PIL
            return xlist[mid], y
        # If element is smaller than mid, then can only be in left subarray
        # el increases away from sun, so depends on sun side
        elif (diff > 1 and sun_side == 'left') or (diff < -1 and sun_side == 'right'):
            return self.binary_search_el(mid - 1, low, el, side, sun_side, hi_map)
        # Else the element can only be present in right subarray
        else:
            return self.binary_search_el(high, mid + 1, el, side, sun_side, hi_map)


    def get_corner_coords(self, side, sun_side):
        if sun_side == 'left':
            x = 0
        else:
            x = 1023
        if side == 'N':
            y = 1023
        else:
            y = 0
        return x, y
    
    
    def add_to_mask_points(self, extra_points, mask_points, side, sun_side):
        if sun_side == 'right':
            if side == 'N':
                return extra_points + mask_points
            else:
                return  mask_points + extra_points
        else:
            if side == 'N':
                return mask_points + extra_points
            else:
                return extra_points + mask_points          


    def get_mask_points(self, helcats_name, data_src=0, res=0.1):
        """finds coords of box of CME area.
        # CME names to test
        # STA, HELCATS says both off image, but SSW says top not:
            'HCME_A__20120701_01'
        # STB both off top/bottom: 'HCME_B__20120831_01'
        # STA / STB neither off: 'HCME_A__20081212_01', 'HCME_B__20081212_01'
        """
        import hi_processing.images as hip
        # (0, 0) at bottom left throughout
        frame = self.load_storm_front(helcats_name)
        if not isinstance(frame, Frame):
            raise ValueError("Code only designed for one frame per CME.")
        frame.get_hi_map(data_src=data_src)
        # TODO what if there are multiple consensus fronts defined?
        df = frame.consensus_fronts['front'][0].coords
        df = df.sort_values(by='pa')
        mask_points = list(zip(df['x'].values, df['y'].values))
        
        # need to specify two points along edge to start mask from
        if frame.sun_side == 'right':
            pas = {"N" : np.nanargmin(df['pa']), "S" : np.nanargmax(df['pa'])}
        else:
            pas = {"S" : np.nanargmin(df['pa']), "N" : np.nanargmax(df['pa'])}
        for side in ["N", "S"]:
            # if the CME is not off the edge of the image, trace PA to right
            if not isinstance(HELCATS.get_col_data("L:" + side, [helcats_name])[0], str):
                x, y = self.binary_search_pa(high=1023, low=0,
                                             pa=df['pa'][pas[side]],
                                             sun_side=frame.sun_side,
                                             hi_map=frame.hi_map)
                mask_points = self.add_to_mask_points([(x, y)], mask_points,
                                                      side, frame.sun_side)
            # if the CME IS off the edge of the image, track el to top 
            else:
                x, y = self.binary_search_el(high=1023, low=0,
                                             el=df['el'][pas[side]],
                                             side=side,
                                             sun_side=frame.sun_side,
                                             hi_map=frame.hi_map)
                # if el too near sun, go back to tracking right
                if np.isnan(x) or np.isnan(y):
                    x, y = self.binary_search_pa(high=1023, low=0,
                                                 pa=df['pa'][pas[side]],
                                                 sun_side=frame.sun_side,
                                                 hi_map=frame.hi_map)
                    mask_points = self.add_to_mask_points([(x, y)],
                                                          mask_points,
                                                          side, frame.sun_side)
                else:
                    mask_points = self.add_to_mask_points([(x, y)],
                                                          mask_points,
                                                          side, frame.sun_side)
                    xc, yc = self.get_corner_coords(side, frame.sun_side)
                    mask_points = self.add_to_mask_points([(xc, yc)],
                                                          mask_points,
                                                          side, frame.sun_side)
        # is image flipped?
        if self.is_flipped(helcats_name):
            mask_points = [(1023 - x, y) for (x, y) in mask_points]
        return mask_points


    @classmethod
    def load_masked_img(cls, helcats_name, img_type, invert=False,
                        data_src=0):
        """Loads the image for cme with helcats_name with mask layer applied.
        img_type: 'diff', 'norm', 'diff_be'
        """
        # load the image
        craft, time = HELCATS.get_cme_details(helcats_name)
        img = STEREOHI.load_img(helcats_name, craft, img_type, 'POPFSS')
        # define the mask
        workflow = POPFSSStormFrontWorkflow()
        out_path = os.path.join(workflow.root, 'all_cme_mask_points.csv')
        if os.path.exists(out_path):
            df = pd.read_csv(out_path, converters={'mask_points' : str})
            i_cme = df.index[df['helcats_name'] == helcats_name][0]
            mask_points = eval(df['mask_points'][i_cme], {'nan' : np.NaN})
        else:
            mask_points = workflow.get_mask_points(helcats_name, data_src=data_src)
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
    
    
    @staticmethod
    def plot_masked_img(helcats_name, img_type, invert=False, data_src=0):

        workflow = POPFSSStormFrontWorkflow()
        img = workflow.load_masked_img(helcats_name, img_type, invert=invert,
                                        data_src=data_src)
        plt.imshow(img, cmap='gray',  interpolation='nearest')     
    
    
    @classmethod
    def save_mask_points_all_cmes(cls, data_src=0):
        workflow = POPFSSStormFrontWorkflow()
        helcats_name_list = CMEComplexity.load('diff')['helcats_name'].values
        df = pd.DataFrame(columns=['helcats_name', 'mask_points'])
        for n, helcats_name in enumerate(helcats_name_list):
            print("%s %s of %s"%(helcats_name, n, len(helcats_name_list)))
            mask_points = workflow.get_mask_points(helcats_name,
                                                    data_src=data_src)
            df = df.append({'helcats_name' : helcats_name,
                            'mask_points' : mask_points}, ignore_index=True)
        out_path = os.path.join(workflow.root, 'all_cme_mask_points.csv')
        df.to_csv(out_path, index=False)
