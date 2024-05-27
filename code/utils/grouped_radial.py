"""
May 2024: Modified from https://uc-ebook.org/docs/html/A2_Jupyter_Notebooks.html

Original: https://github.com/calvinwhealton/SensitivityAnalysisPlots
"""

import numpy as np
import itertools
import seaborn as sns
import math
sns.set_style('whitegrid', {'axes_linewidth': 0, 'axes.edgecolor': 'white'})

def is_significant(value, confidence_interval, threshold="conf"):
    if threshold == "conf":
        return value - abs(confidence_interval) > 0
    else:
        return value - abs(float(threshold)) > 0
        
def grouped_radial(SAresults_total, SAresults_2order, parameters, radSc=2.0, scaling=1, widthSc=0.5, STthick=1, varNameMult=1.3, colors=None, groups=None, gpNameMult=1.5, threshold="conf"):
    # Derived from https://github.com/calvinwhealton/SensitivityAnalysisPlots
    fig, ax = plt.subplots(1, 1)
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
        ax.text(varNameMult*x[i], varNameMult*y[i], key, ha='center', va='center',
                rotation=angles[i]*360/(2*math.pi) - 90,
                color=color_map[key])

    if groups is not None:
        for i, group in enumerate(groups.keys()):
            print(group)
            group_angle = np.mean([angles[j] for j in range(n) if parameters[j] in groups[group]])

            ax.text(gpNameMult*radSc*math.cos(group_angle), gpNameMult*radSc*math.sin(group_angle), group, ha='center', va='center',
                rotation=group_angle*360/(2*math.pi) - 90,
                color=colors[i % len(colors)])

    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('equal')
    plt.axis([-2*radSc, 2*radSc, -2*radSc, 2*radSc])
    plt.show()