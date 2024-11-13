"""
September 2024: Modified by David Lafferty (rotation of group labels).
May 2024: Modified from https://uc-ebook.org/docs/html/A2_Jupyter_Notebooks.html

Original: https://github.com/calvinwhealton/SensitivityAnalysisPlots
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

from utils.constants import location_names
from utils.global_paths import project_code_path, project_data_path

sns.set_style('whitegrid', {'axes_linewidth': 0, 'axes.edgecolor': 'white'})

def is_significant(value, confidence_interval, threshold="conf"):
    if threshold == "conf":
        return value - abs(confidence_interval) > 0
    else:
        return value - abs(float(threshold)) > 0
        
def grouped_radial(SAresults_total, SAresults_2order, parameters, ax, radSc=2.0, scaling=1, widthSc=0.5, STthick=1, varNameMult=1.3, colors=None, groups=None, gpNameMult=1.5, threshold="conf"):
    # Derived from https://github.com/calvinwhealton/SensitivityAnalysisPlots
    color_map = {}

    # initialize parameters and colors
    if groups is None:

        if colors is None:
            colors = ["k"]

        for i, parameter in enumerate(parameters):
            color_map[parameter] = colors[i % len(colors)]
    else:
        if colors is None:
            colors = sns.color_palette("deep", max(3, len(groups)))

        for i, key in enumerate(groups.keys()):
            #parameters.extend(groups[key])

            for parameter in groups[key]:
                color_map[parameter] = colors[i % len(colors)]

    n = len(parameters)
    angles = radSc*math.pi*np.arange(0, n)/n
    x = radSc*np.cos(angles)
    y = radSc*np.sin(angles)

    # plot second-order indices
    for i, j in itertools.combinations(range(n), 2):
        key1 = parameters[i]
        key2 = parameters[j]

        # print
        
        try:
            bool_significant = is_significant(SAresults_2order.loc[key1].loc[key2]["S2"], SAresults_2order.loc[key1].loc[key2]["S2_conf"], threshold)
        except:
            bool_significant = is_significant(SAresults_2order.loc[key2].loc[key1]["S2"], SAresults_2order.loc[key2].loc[key1]["S2_conf"], threshold)
            
        if bool_significant:
            angle = math.atan((y[j]-y[i])/(x[j]-x[i]))

            if y[j]-y[i] < 0:
                angle += math.pi

            line_hw = scaling*(max(0, SAresults_2order.loc[key1].loc[key2]["S2"])**widthSc)/2

            coords = np.empty((4, 2))
            coords[0, 0] = x[i] - line_hw*math.sin(angle)
            coords[1, 0] = x[i] + line_hw*math.sin(angle)
            coords[2, 0] = x[j] + line_hw*math.sin(angle)
            coords[3, 0] = x[j] - line_hw*math.sin(angle)
            coords[0, 1] = y[i] + line_hw*math.cos(angle)
            coords[1, 1] = y[i] - line_hw*math.cos(angle)
            coords[2, 1] = y[j] - line_hw*math.cos(angle)
            coords[3, 1] = y[j] + line_hw*math.cos(angle)

            ax.add_artist(plt.Polygon(coords, color="0.75"))

    # plot total order indices
    for i, key in enumerate(parameters):
        if is_significant(SAresults_total.loc[key]["ST"], SAresults_total.loc[key]["ST_conf"], threshold):
            ax.add_artist(plt.Circle((x[i], y[i]), scaling*(SAresults_total.loc[key]["ST"]**widthSc)/2, color='w'))
            ax.add_artist(plt.Circle((x[i], y[i]), scaling*(SAresults_total.loc[key]["ST"]**widthSc)/2, lw=STthick, color='0.4', fill=False))

    # plot first-order indices
    for i, key in enumerate(parameters):
        if is_significant(SAresults_total.loc[key]["S1"], SAresults_total.loc[key]["S1_conf"], threshold):
            ax.add_artist(plt.Circle((x[i], y[i]), scaling*(SAresults_total["S1"][key]**widthSc)/2, color='0.4'))

    # add labels
    for i, key in enumerate(parameters):
        angle = angles[i]*360/(2*math.pi) - 90
        rotation = angle if angle < 90 else (angle-180)
        ax.text(varNameMult*x[i], varNameMult*y[i], key, ha='center', va='center',
                rotation=rotation,
                color=color_map[key])

    if groups is not None:
        for i, group in enumerate(groups.keys()):
            group_angle = np.mean([angles[j] for j in range(n) if parameters[j] in groups[group]])
            group_text_angle = group_angle*360/(2*math.pi) - 90
            roation = group_text_angle if group_text_angle < 180 else (group_text_angle-180)
            ax.text(gpNameMult*radSc*math.cos(group_angle), gpNameMult*radSc*math.sin(group_angle), group, ha='center', va='center',
                rotation=roation, fontweight='bold',
                color=colors[i % len(colors)])

    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('equal')
    ax.axis([-2*radSc, 2*radSc, -2*radSc, 2*radSc])


def read_second_order(experiment, N, obs_name):
    # Read total and first order
    df_total = pd.read_csv(f'{project_data_path}/WBM/SA/{experiment}_{obs_name}_{N}_noCC_res_total.csv').set_index(['metric', 'param'])

    # Read second order and add full complement (easier for plotting)
    df_2order = pd.read_csv(f'{project_data_path}/WBM/SA/{experiment}_{obs_name}_{N}_noCC_res_2order.csv').set_index(['param1', 'param2'])
    df_2order_swapped = pd.DataFrame(data = df_2order.values,
                                 index = [(y, x) for x, y in df_2order.index],
                                 columns = df_2order.columns)

    df_2order = pd.concat([df_2order, df_2order_swapped]).reset_index().rename(columns = {'level_0': 'param1', 'level_1': 'param2'}).set_index(['metric', 'param1', 'param2'])

    return df_total, df_2order


def plot_second_order_all(experiment, groups, N, savefig):
    """
    Makes plot showing all radial SA results for all metrics for given location.
    Only for noCC experiments.
    """
    df_smap_total, df_smap_2order = read_second_order(experiment, 'SMAP', N)
    df_nldas_total, df_nldas_2order = read_second_order(experiment, 'NLDAS', N)
    
    # Get params
    params = [groups[key] for key in groups.keys()]
    params = [x for y in params for x in y]
    
    colors = ['black', 'gray', 'orange', 'darkgreen', 'darkblue']
    
    plt.rcParams["font.size"] = 8
    
    ###### Plot
    fig, axs = plt.subplots(3,4, figsize=(14,12))
    
    # Top Row
    grouped_radial(df_smap_total.loc['mean'], df_smap_2order.loc['mean'], params, axs[0,0],
                   groups = groups,
                   colors = colors,
                   varNameMult=1.35, gpNameMult=1.55)
    axs[0,0].set_title('Mean \n (SMAP forcing)', fontsize=12, y=0.9)
    
    grouped_radial(df_smap_total.loc['sd'], df_smap_2order.loc['sd'], params, axs[0,1],
                   groups = groups,
                   colors = colors,
                   varNameMult=1.35, gpNameMult=1.55)
    axs[0,1].set_title('Standard Deviation \n (SMAP forcing)', fontsize=12, y=0.9)
    
    grouped_radial(df_nldas_total.loc['mean'], df_nldas_2order.loc['mean'], params, axs[0,2],
                   groups = groups,
                   colors = colors,
                   varNameMult=1.35, gpNameMult=1.55)
    axs[0,2].set_title('Mean \n (NLDAS forcing)', fontsize=12, y=0.9)
    
    grouped_radial(df_nldas_total.loc['sd'], df_nldas_2order.loc['sd'], params, axs[0,3],
                   groups = groups,
                   colors = colors,
                   varNameMult=1.35, gpNameMult=1.55)
    axs[0,3].set_title('Standard Deviation \n (NLDAS forcing)', fontsize=12, y=0.9)
    
    # Middle row
    axs[1,0].set_title('SMAP RMSE', fontsize=12, y=0.9)
    grouped_radial(df_smap_total.loc['rmse_SMAP'], df_smap_2order.loc['rmse_SMAP'], params, axs[1,0],
                   groups = groups,
                   colors = colors,
                   varNameMult=1.35, gpNameMult=1.55)
    
    for ido, obs_name in enumerate(['VIC', 'NOAH', 'MOSAIC']):
        axs[1,1+ido].set_title(f"{obs_name} RMSE", fontsize=12, y=0.9)
        grouped_radial(df_nldas_total.loc[f'rmse_{obs_name}'], df_nldas_2order.loc[f'rmse_{obs_name}'], params, axs[1,1+ido],
                   groups = groups,
                   colors = colors,
                   varNameMult=1.35, gpNameMult=1.55)
    
    # Bottom row
    axs[2,0].set_title('SMAP ubRMSE', fontsize=12, y=0.9)
    grouped_radial(df_smap_total.loc['ubrmse_SMAP'], df_smap_2order.loc['ubrmse_SMAP'], params, axs[2,0],
                   groups = groups,
                   colors = colors,
                   varNameMult=1.35, gpNameMult=1.55)
    
    for ido, obs_name in enumerate(['VIC', 'NOAH', 'MOSAIC']):
        axs[2,1+ido].set_title(f"{obs_name} ubRMSE", fontsize=12, y=0.9)
        grouped_radial(df_nldas_total.loc[f'ubrmse_{obs_name}'], df_nldas_2order.loc[f'ubrmse_{obs_name}'], params, axs[2,1+ido],
                   groups = groups,
                   colors = colors,
                   varNameMult=1.35, gpNameMult=1.55)

    
    fig.suptitle(location_names[experiment], fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{project_code_path}/figs/si/{experiment}_{N}_grouped_radial.png', dpi=600)
    else:
        plt.show()