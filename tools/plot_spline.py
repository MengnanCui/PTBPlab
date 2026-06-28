import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import sys


def repulsive_potential(r, spline_sections, exp_coefficients):
    # Check if we're in the exponential region (before the first spline interval)
    if r < spline_sections[0][0]:
        a1, a2, a3 = exp_coefficients
        return np.exp(-a1 * r + a2) + a3

    # Iterate over the spline sections to find the correct interval
    for section in spline_sections:
        start, end, *coeffs = section
        if start <= r < end:
            # Evaluate the cubic spline for this interval
            return coeffs[0] + coeffs[1] * (r - start) + coeffs[2] * (r - start)**2 + coeffs[3] * (r - start)**3
        elif r == end:
            # Special case for the last point in the last interval (using a quintic polynomial)
            return coeffs[0] + coeffs[1] * (r - start) + coeffs[2] * (r - start)**2 + coeffs[3] * (r - start)**3 + coeffs[4] * (r - start)**4 + coeffs[5] * (r - start)**5

    # If r is beyond the last interval, return the potential at the cutoff distance
    last_coeffs = spline_sections[-1][2:]
    start = spline_sections[-1][0]
    return sum(c * (r - start)**n for n, c in enumerate(last_coeffs))

skf_infile = sys.argv[1]
png_outfile = skf_infile.replace('.skf', '.png')


# Open file and read Spline
with open(skf_infile, 'r') as f:
    spline_data = f.read()

spline_lines = spline_data.strip().split('\n')
spline_started = False
spline_sections = []
exp_coefficients = []

# Extract the spline sections and exponential coefficients
for line in spline_lines:
    if 'Spline' in line:
        spline_started = True  # Indicates the start of the spline data
    elif spline_started and not line.startswith("#"):  # Ignores comments
        spline_sections.append([float(x) for x in line.split()])


# The first line after "Spline" contains the number of intervals and the exponential coefficients
nInt, cutoff = spline_sections.pop(0)
print(f"nInt: {nInt}, cutoff: {cutoff}")

exp_coefficients = spline_sections.pop(0)


# Plot the spline sections
x = np.linspace(0.3, 7, 500)
y = [repulsive_potential(r, spline_sections, exp_coefficients) for r in x]

fig, ax = plt.subplots(1,1, figsize=(5,4))
ax.plot(x, y)
ax.set_xlabel('Distance (Å)')
ax.set_ylabel('Potential Energy (units)')
ax.set_title('Repulsive Potential vs. Distance')
fig.tight_layout()
fig.savefig(png_outfile, dpi=200)




colors = [
    "#2470a0",  #1 muted blue
    "#ca3e47",  #2 muted red
    "#f29c2b",  #3 muted orange
    '#1f640a',  #4 muted green
    "#2ca02c",  #5 cooked asparagus green
    "#9467bd",  #6 muted purple
    "#8c564b",  #7 chestnut brown
    "#e377c2",  #8 raspberry yogurt pink
    "#7f7f7f",  #9 middle gray
    "#bcbd22",  #10 curry yellow-green
    "#17becf",  # blue-teal
    "#005555",  # fhi-green
]

markers = [
'o',#0       circle 
'v',#1       triangle_down marker
's',#2       square
'*',#3       star
'p',#4       point marker
'P',#5       pentagon marker
',',#6       pixel marker
]
linestyles = [
'-',#       solid line style
'--',#      dashed line style
'-.',#      dash-dot line style
':',#       dotted line style
(0, (3, 5, 1, 5, 1, 5)),#   dashdotdotted '- .. - .. - .. '
 (0, (5, 5)),#  dashed '-  -  -  -  -  -'
]

# https://matplotlib.org/stable/tutorials/introductory/customizing.html
poster_rcParams = {"font.size": 15, 
                     "axes.labelsize": 13,
                     "axes.titlesize": 13,
                     "axes.linewidth": 1.5, # The width of the axes lines
                     "xtick.labelsize": 12,
                     "ytick.labelsize": 12,
                     "legend.fontsize": 12,
                     "lines.markersize": 20,
                     "lines.linewidth": 2.5,  # width of lines
                     }
plt.rcParams.update(poster_rcParams)