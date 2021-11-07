import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import PIL.Image as Image
import scipy.interpolate as interpolate
import scipy.stats as sps
import pandas as pd
from matplotlib.ticker import PercentFormatter
sys.path.insert(0, os.path.join('..', 'useful_code'))
from misc import find_nearest
from config import data_loc
fig_loc = os.path.join('..', '..', 'Plots')


############################### Finding S and V ##############################
def find_img_v_list(img, L=2, mask=False, mask_val=255):
    """mask is 255 or 0
    """
    img_array = np.array(img)
    if mask:
        # NB images should be black and white!
        img_array = np.array(img)[:, :, 0]
        mask_array = np.array(img)[:, :, 1]
    M, N = np.shape(img_array)
    # 2. divide into boxes
    v_list = []
    coord_list = []
    Nb = 0
    while Nb + L - 1 < N:
        # loop over columns
        Mb = 0
        while Mb + L - 1 < M:
            # get list of pixels in the box
            pixels = []
            for i in range(Nb, Nb + L):
                for j in range(Mb, Mb + L):
                    if mask:
                        if mask_array[j, i] == mask_val:
                            pixels.append(img_array[j, i])
                    else:
                        pixels.append(img_array[j, i])
            # check there are the correct number of pixels (sorts edges)
            if len(pixels) == L**2:
                # 3. find mean grey level in the box
                mean = (1 / (L ** 2)) * sum(pixels)
                # 4. find sample variance in the box 
                v = (1 / (L ** 2 - 1)) * sum([(p - mean) ** 2 for p in pixels])
                v_list.append(v)
            else:
                v_list.append(np.NaN)
            coord_list.append((Mb, Nb))
            # move onto next box
            Mb = Mb + L
        Nb = Nb + L
    return v_list, coord_list


def v_list_to_v_map_vals(v_list, coord_list):
    """For pcolormesh.
    """
    df  = pd.DataFrame(data={'v' : v_list,
                             'x' : [x for (y, x) in coord_list],
                             'y' : [y for (y, x) in coord_list]})
    v_map_vals = []
    # for each y value there is a row of x values
    for y in np.unique(df.y):
        row = []
        # get all the x values in this row
        dfy = df[df.y == y]
        dfy = dfy.sort_values('x')
        for v in dfy.v:
            if np.isnan(v):
                row.append(np.NaN)
            # elif float(v) > float(0):
            else:
                row.append(v)
            # else:
            #     # as log scale v=0 will break it
            #     # replace v=0 with the lowest v in the image
            #     row.append(np.nanmin(df[df.v != v].v.values))
        v_map_vals.append(row)
    return v_map_vals


def find_img_av_v(v_list, img=None, L=2, include_zero=True):
    """ 5. average over the boxes
    # N, M = img.size
    # v = (L ** 2 / (N * M)) * sum(v_list)    
    """
    if not include_zero:
        v_list = [v for v in v_list if ~np.isnan(v)]
        v_list = [v for v in v_list if v > 0]
    v = np.nanmean(v_list)
    return v


def get_img_size_list(N, M, L=2, n=50, d=None):
    if ((N % L) != 0) or ((M % L) != 0):
        raise ValueError("Check inputs: L must be a factor of both N and M")
    if d != None:
        # check image resize number d is a factor of N
        if (N % d) != 0:
            raise ValueError(f'Check inputs: d must be a factor of N={str(N)}')
        N_list = []
        r = d
        while r < N + 1:
            N_list.append(r)
            r = r + d
    else:
        mins = np.log(L) / np.log(10)
        maxs = np.log(N) / np.log(10)
        s_list = np.logspace(mins, maxs, num=n)
        N_list = [int(L * N / s) for s in s_list]
        # round to nearest multiple of L
        N_list = np.unique([L * round(Nr / L) for Nr in N_list])
        # only keep ones for which both Nr and Mr are multiples of L
        M_list = [int(Nr / (N / M)) for Nr in N_list]
        N_list = [N_list[i] for i in range(len(N_list)) if M_list[i] % L == 0]
        # ensure both greater than 0
        fM_list = [M_list[i] for i in range(len(N_list)) if M_list[i] > 0]
        fN_list = [N_list[i] for i in range(len(N_list)) if M_list[i] > 0]
    return fN_list, fM_list


def check_N_list(N_list, N, M, L):
    # check all positive and greater than 0
    ok_N_list = [Nr for Nr in N_list if Nr > 0]
    if len(ok_N_list) != len(N_list):
        raise ValueError("Values in N_list must be greater than 0")    
    # check Ns are all multiples of L, and make sure they int!
    ok_N_list = [int(Nr) for Nr in N_list if Nr % L == 0]
    if len(ok_N_list) != len(N_list):
        raise ValueError("Values in N_list are not all multiples of L")
    # now generate M_list, and check them!
    M_list = [int(Nr / (N / M)) for Nr in N_list]
    ok_M_list = [Mr for Mr in M_list if Mr > 0]
    if len(ok_M_list) != len(N_list):
        raise ValueError("M values for N values in N_list must be greater than 0")   
    ok_M_list = [int(Mr) for Mr in M_list if Mr % L == 0]
    if len(ok_M_list) != len(M_list):
        raise ValueError("M values for N values in N_list are not all multiples of L")    
    return N_list, M_list


def find_s_and_v(img, N_list=[], n=50, L=2, resample=1, mask=False,
                 print_num=False, d=None, mask_val=255):
    """
    Parameters
    ----------
    img : PIL.Image instance
        The image to be used.
    N_list : list, optional
        A list of values to resize the image to. The default is [].
    n : int, optional
        To automatically generate N_list with n points
    L : int, optional
        The size of the box to find the variance over. The default is 2.
    resample : TYPE, optional
        The resampling technique to resize the image. See PIL.Image.
        The default is 1.
    mask : bool, optional
        If True, img must have an alpha channel, corresponding to a mask for 
        img. The code will skip pixels which correspond to 0 in the mask.
        The default is False.
    print_num : bool, optional
        If True, will print the code's progress. The default is False.

    Returns
    -------
    s_list : list
        A list of scale-sizes corresponding to the N values used.
    v_list : list
        A list of the average grey level variances over the images,
        corresponding to the scale-sizes in s_list.
    """
    N, M = img.size
    if N_list == []:
        # get list of resized image sizes
        N_list, M_list = get_img_size_list(N=N, M=M, L=L, n=n, d=d)
    else:
        N_list, M_list = check_N_list(N_list=N_list, N=N, M=M, L=L)
    v_list = []
    s_list = []
    # loop over each resized image
    for n, Nr in enumerate(N_list):
        # print code progress
        if print_num == True:
            print(f'image {str(n+1)} of {len(N_list)}')
        # resize the image
        Mr = M_list[n]
        # Mr = int(Nr / (N / M))
        img_r = img.resize((Nr, Mr), resample=resample)
        # calculate v for this resized image
        v_list_r, coord_list_r = find_img_v_list(img_r, L=L, mask=mask,
                                                 mask_val=mask_val)
        v = find_img_av_v(img=img_r, v_list=v_list_r, L=L)
        # calculate s for this resized image
        s = L * N / Nr
        v_list.append(v)
        s_list.append(s)
    # make sure lists are sorted by increasing s
    order = np.argsort(s_list)
    s_list = [s_list[i] for i in order]
    v_list = [v_list[i] for i in order]
    return s_list, v_list


############################ Calculate Q #####################################
def calc_q_centred_diff(s_list, v_list, size=1):
    # take logs
    s_list = np.log(s_list)
    v_list = np.log(v_list)
    inside_list = []
    s = [] 
    low_s = []
    high_s = []
    n = size    
    while n + size < len(s_list):
        x = np.array([s_list[n-size:n+size+1]])
        y = np.array([v_list[n-size:n+size+1]])
        slope, intercept, r_value, p_value, std_err = sps.linregress(x, y)
        inside_list.append(1 - (1/4)*(slope**2))
        s.append(s_list[n])
        low_s.append(s_list[n-size])
        high_s.append(s_list[n+size])
        n = n + size * 2
    # evaluate integral
    integral = 0
    for m in range(len(inside_list)):
        # check plus condition is met
        if inside_list[m] > 0:
            # integrate
            new_area = inside_list[m] * (high_s[m] - low_s[m])
            integral = integral + new_area
    Q = 1 / (np.max(high_s) - np.min(low_s)) * integral
    return Q


def calc_q_bspline(s_list, v_list):
    # take logs
    s_list = np.log(s_list)
    v_list = np.log(v_list)
    inside_integral = []
    new_s_list = []
    spl = interpolate.make_interp_spline(s_list, v_list, k=3)
    der = spl.derivative()
    for i in range(len(s_list)):
        d = der(s_list[i])
        inside = 1 - (1/4) * (d ** 2)
        if inside > 0:
            new_s_list.append(s_list[i])
            inside_integral.append(inside)
    new_spl = interpolate.make_interp_spline(new_s_list, inside_integral, k=3)
    integral = new_spl.integrate(a=np.min(s_list), b=np.max(s_list))
    Q = 1 / (np.max(s_list) - np.min(s_list)) * integral    
    return Q
    
    
def find_Q(s_list, v_list, size=1, method=0, plot=False, img=None):
    """
    method: 0 or 1, 0 for centred-differences, 1 for b-spline approx.
    """
    if method == 0:
        Q = calc_q_centred_diff(s_list, v_list, size=1)
    else:
        Q = calc_q_bspline(s_list, v_list)
    if plot == True:
        plt.figure()
        plt.imshow(img, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.title("Q: " + "{:.3f}".format(Q), fontsize=16)
    return Q


##################### Reshuffling images for testing #########################
def replace_pixels_with_ref_img(img, ref_img):
    ref_img = np.array(ref_img) 
    ref_list = np.reshape(ref_img,
                          (np.shape(ref_img)[0]*np.shape(ref_img)[1],))
    ref_sorted = np.sort(ref_list)
    img = np.array(img)
    img_list = np.reshape(img, (np.shape(img)[0]*np.shape(img)[1],))
    img_sorted = np.sort(img_list)
    # where is each element in the sorted array?
    sort_places = []
    for i in img_list:
        # note that searchsorted indexes from 1 not 0
        sort_places.append(np.searchsorted(img_sorted, i)-1)
    # swap values for ref values
    new_vals = []
    for i in sort_places:
        new_vals.append(ref_sorted[i])   
    out_img = Image.fromarray(np.reshape(new_vals, np.shape(img))).convert("L")
    return out_img


def make_random_img(img):
    img_arr = np.array(img)
    img_list = np.reshape(img_arr, (np.shape(img_arr)[0]*np.shape(img_arr)[1],))
    for i in range(1):
        random.shuffle(img_list)
        new_img_arr = np.reshape(img_list, np.shape(img_arr))
        new_img = Image.fromarray(new_img_arr).convert("L")
    return new_img      


############################## Plotting functions ############################
def plot_img(fig, ax, img):
    ax.imshow(img, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def plot_img_r(fig, ax, img, s, L=2, resample=1):
    N, M = img.size
    Nr = int(L * N / s)
    Mr = int(Nr / (N / M))
    img_r = img.resize((Nr, Mr), resample=resample)
    fig, ax = plot_img(fig, ax, img_r)
    ax.set_title("%s x %s pixels"%(Nr, Mr), fontsize=12)
    s = L * N / Nr
    ax.set_xlabel("S: %s"%("{:.2f}".format(s)), fontsize=12)
    return fig, ax


def plot_s_vs_v(fig, ax, xlabel=True, ylabel=True, aspect=False,
                xlim=[0.1, 1000], ylim=[0.1, 1000], fontsize=16):
    if xlabel:
        ax.set_xlabel('S', fontsize=fontsize)
    if ylabel:
        ax.set_ylabel('V', fontsize=fontsize)
    ax.set_xlim((xlim[0], xlim[1]))
    ax.set_ylim((ylim[0], ylim[1]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=fontsize)
    if aspect:
        ax.set_aspect(1)
    return fig, ax


def add_s_vs_v(fig, ax, s_list, v_list, label, color, marker, markersize=20):
    ax.scatter(s_list, v_list, zorder=1, label=label, color=color,
               marker=marker, s=markersize)
    return fig, ax


def get_v_map(img, s, L=2, resample=1, mask=False):
    # n is width, m is height
    N, M = img.size
    Nr = int(L * N / s)
    Mr = int(Nr / (N / M))    
    img_r = img.resize((Nr, Mr), resample=resample)
    v_list, coord_list = find_img_v_list(img_r, L=L, mask=mask)
    # plot v_map
    v_vals = v_list_to_v_map_vals(v_list, coord_list)
    if np.all(np.isnan(v_vals)):
        vmin = np.NaN
        vmax = np.NaN
    else:
        vmin = np.nanmin(v_vals)
        vmax = np.nanmax(v_vals)
    title = "%s x %s pixels, S: %s"%(Nr, Mr, "{:.2f}".format(s))
    return Nr, Mr, v_vals, vmin, vmax, title 

   
def plot_v_map(fig, ax, v_vals, vmin, vmax):
    # ignore cases where all values are NaNs
    if not np.all(np.isnan(v_vals)):
        plot = ax.pcolormesh(v_vals,
                             norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        ax.set_aspect(1)
        plot.axes.invert_yaxis()
        fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)   
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def plot_v_hist(fig, ax, v_vals, vmin, vmax, L=2,
                xlabel=True, ylabel=True, xlim=None, ylim=None):
    scales = [0.1, 1, 10, 100, 1000, 10000]
    bins = []
    for i in range(len(scales)-1):
        bins = np.concatenate([bins, np.arange(scales[i], scales[i+1],
                                               scales[i])])
    cmap = plt.cm.viridis
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    # convert into 1D list
    v_vals = np.array(v_vals).flatten()
    # drop NaNs, so not counted in total percentage
    v_vals = v_vals[~np.isnan(v_vals)]
    # weights argument to convert to percentage
    # does 1 / val for each val, so a weight for each val
    n, bins, patches = ax.hist(v_vals, bins=bins, rwidth=1,
                               weights=np.ones(len(v_vals)) / len(v_vals))
    for m, p in enumerate(patches):
        plt.setp(p, 'facecolor', cmap(norm(bins[m])))
    ax.axvline(np.nanmean(v_vals), ls='--', color='k')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    if xlabel:
        ax.set_xlabel("Sample Variance of Box (L: %s)"%(L), fontsize=12)
    if ylabel:
        ax.set_ylabel("% of Boxes", fontsize=12)
    else:
        ax.set_yticks([])
    ax.set_xlim((np.min(scales), np.max(scales)))
    if ylim != None:
        ax.set_ylim(ylim)
    ax.set_xscale('log')
    return fig, ax


def plot_resampling_methods_img(img, N):
    """N is the number of pixels in the resized image to be plotted.
    """
    M = int(N / (img._size[0] / img._size[1]))
    f, axes = plt.subplots(2, 3, figsize=(8, 5))
    i = 0
    j = 0
    names = ["NEAREST", "LANCZOS", "BILINEAR", "BICUBIC", "BOX", "HAMMING"]
    for n in range(0, 6):
        axes[j, i].imshow(img.resize((N, M), resample=n), cmap='gray', 
                          interpolation='nearest')
        axes[j, i].set_title(names[n])
        axes[j, i].set_xticks([])
        axes[j, i].set_yticks([])
        i = i + 1
        if i > 2:
            j = j + 1
            i = 0
    plt.suptitle("Size: %sx%s pixels"%(str(N), str(M)), fontsize=20)


def plot_resampling_methods_results(img, N_list=[], L=2, n=50):
    """Scatter plot showing s and v values calculated for the different
    resampling methods.
    """
    if N_list == []:
        # get list of resized image sizes
        N_list = get_img_size_list(img.size[0], L=L, n=n)
    names = ["NEAREST", "LANCZOS", "BILINEAR", "BICUBIC", "BOX", "HAMMING"]
    colours = ['k', '#0868ac', '#43a2ca', '#7bccc4', '#bae4bc', '#f0f9e8']
    markers = ['o', '^', '*', 'D', 's', 'x']
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    fig, ax = plot_s_vs_v(fig, ax, xlim=[1, 2000], ylim=[10, 10000])
    for n, name in enumerate(names):
        s_list, v_list = find_s_and_v(img, N_list=N_list, L=2, resample=n)
        fig, ax = add_s_vs_v(fig, ax, s_list, v_list, name, colours[n],
                             markers[n])
    plt.legend()


def plot_splines(img, s_list, v_list):
    """Plot showing different splines fitted to the data.
    """
    if len(s_list) < 6:
        raise ValueError("Must be at least 6 data points.")
    s_listr = np.log(s_list)
    v_listr = np.log(v_list)
    plt.figure(figsize=(9, 6))
    plt.scatter(s_listr, v_listr, label='data', color='k', marker='x',
                zorder=3)
    for nk in [1, 2, 3, 5]:
        x = np.linspace(np.min(s_listr), np.max(s_listr), 100)
        spl = interpolate.make_interp_spline(s_listr, v_listr, k=nk)
        plt.plot(x, spl(x), label="k=" + str(nk))
    plt.ylim((int(np.min(v_listr)), int(np.max(v_listr))+1))
    plt.xlim((0, int(np.max(s_listr))+1))
    plt.ylabel('V', fontsize=16)
    plt.xlabel('S', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()


################### Reproducing Zannete 2018 results #########################
def recreate_paper_plots():
    N_list = [360, 328, 300, 272, 248, 224, 204, 188, 168, 156, 140, 128, 116,
              108, 96, 88, 80, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 28, 24,
              20, 16, 12, 8]
    folder = os.path.join(data_loc(), 'Zanette')
    # figure 1, 3 different size check
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax = plot_s_vs_v(fig, ax, xlim=[1, 200], ylim=[1, 5000])
    img = Image.open(os.path.join(folder, 'chess06.bmp')).convert("L")
    s_list, v_list = find_s_and_v(img, N_list=N_list, L=2)
    ax = add_s_vs_v(ax, s_list, v_list, "Chess 6 pixels", 'teal', 'o')
    img =  Image.open(os.path.join(folder, 'chess18.bmp')).convert("L")
    s_list, v_list = find_s_and_v(img, N_list=N_list, L=2)
    ax = add_s_vs_v(ax, s_list, v_list, "Chess 18 pixels", 'darkgoldenrod', '^')
    img =  Image.open(os.path.join(folder, 'chess54.bmp')).convert("L")
    s_list, v_list = find_s_and_v(img, N_list=N_list, L=2)
    ax = add_s_vs_v(ax, s_list, v_list, "Chess 54 pixels", 'g', 's')
    ax.legend()
    
    # figure 2, mixed size checks, mona lisa
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax = plot_s_vs_v(fig, ax, xlim=[1, 200], ylim=[30, 5000])  
    img =  Image.open(os.path.join(folder, 'patches.bmp')).convert("L")
    s_list, v_list = find_s_and_v(img, N_list=N_list, L=2)
    ax = add_s_vs_v(ax, s_list, v_list, "Chess mix", 'b', 's')
    img =  Image.open(os.path.join(folder, 'g360.bmp')).convert("L")
    s_list, v_list = find_s_and_v(img, N_list=N_list, L=2)
    ax = add_s_vs_v(ax, s_list, v_list, "Mona Lisa", 'r', 'o')
    ax.legend()
    
    # figure 3, random and ordered mona lisa pixels
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax = plot_s_vs_v(fig, ax, xlim=[1, 200], ylim=[1, 5000])    
    img =  Image.open(os.path.join(folder, 'grandom.bmp')).convert("L")
    s_list, v_list = find_s_and_v(img, N_list=N_list, L=2, resample=1)
    ax = add_s_vs_v(ax, s_list, v_list, 'Random', 'purple', 's')
    img =  Image.open(os.path.join(folder, 'gord.bmp')).convert("L")
    s_list, v_list = find_s_and_v(img, N_list=N_list, L=2)
    ax = add_s_vs_v(ax, s_list, v_list, 'Ordered', 'darkorchid', 'o')
    img =  Image.open(os.path.join(folder, 'g360.bmp')).convert("L")
    s_list, v_list = find_s_and_v(img, N_list=N_list, L=2)
    ax = add_s_vs_v(ax, s_list, v_list, 'Mona Lisa', 'darkred', 'x')
    ax.legend()
    
    # figure 4
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
    names = ['Van Gough', 'New York', 'Fish', 'Woodland']
    imgs = ['vgoghg.bmp', 'cityg.bmp', 'pecesg.bmp',  'foliageg.bmp']
    ml =  Image.open(os.path.join(folder, 'g360.bmp')).convert("L")
    s_list_ml, v_list_ml = find_s_and_v(ml, N_list=N_list, L=2)
    i = 0
    j = 0
    for n in range(len(names)):
        axes[i, j] = plot_s_vs_v(fig, axes[i, j], xlim=[1, 200], ylim=[10, 2000])
        img = Image.open(os.path.join(folder, imgs[n])).convert("L")
        img = replace_pixels_with_ref_img(img=img, ref_img=ml)
        s_list, v_list = find_s_and_v(img, N_list=N_list, L=2)
        axes[i, j] = add_s_vs_v(axes[i, j], s_list, v_list, names[n], 'navy', 'o')  
        axes[i, j] = add_s_vs_v(axes[i, j], s_list_ml, v_list_ml, 'Mona Lisa',
                                'darkred', 'o')
        axes[i, j].legend()
        i = i + 1
        if i > 1:
            j = j + 1
            i = 0

    # table 1: Q-values
    for name in ['g360.bmp', 'grandom.bmp', 'gord.bmp']:
        img =  Image.open(os.path.join(folder, name)).convert("L")
        img = make_random_img(img)
        s_list, v_list = find_s_and_v(img, N_list=N_list)
        find_Q(img, s_list, v_list, method=1)
    for name in ['vgoghg.bmp', 'cityg.bmp', 'pecesg.bmp',  'foliageg.bmp']:
        img =  Image.open(os.path.join(folder, name)).convert("L")
        img = replace_pixels_with_ref_img(img=img, ref_img=ml)
        s_list, v_list = find_s_and_v(img, N_list=N_list)
        find_Q(img, s_list, v_list)
        