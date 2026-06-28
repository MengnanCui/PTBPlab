#!/work/software/anaconda3/bin/python
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sym
from ase.units import Bohr

# ============================================================================ #
# 1. Read in data
# ============================================================================ #
elem_sym, skf_name = sys.argv[1:]
#elem_sym = sys.argv[1]
#skf_name = elem_sym + '-' + elem_sym + '.skf'
with open(skf_name, "r") as f:
    lines = f.readlines()
gridist, ngridpoints = lines[0].strip().split()
gridist, ngridpoints = float(gridist.strip(',')), int(ngridpoints)
num = 3
non_line = 0
hotcent_array = open('band_term.txt', 'w')
hotcent_array.write("# rx Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1 Hsd0 Hsp0 Hss0 Sdd0 Sdd1 Sdd2 Spd0 Spd1 Spp0 Spp1 Ssd0 Ssp0 Sss0\n")
while num < int(ngridpoints) + 2:
    line = lines[num]
    if line.startswith('20*'):
        num += 1
        non_line +=1
        continue
    rmin = (non_line + 1)  * gridist # avoid 0.02, 520
    content_list = line.strip().split()
    # I want change [5*0.0] to [0.0, 0.0, 0.0, 0.0, 0.0], [1*0.0] to [0.0]
    new_list = []
    for content in content_list:
        cont = content.strip().split('*')
        if len(cont) == 1:
            new_list.append(float(cont[0]))
        elif len(cont) == 2:
            new_list.extend([float(cont[1])] * int(cont[0]))
        else:
            raise Exception("Abnormal content length: {}".format(len(cont)))
    rx = rmin + gridist * (num - non_line - 3)
    # print(num, rx, len(new_list))
    for i in range(len(new_list)):
        if i == 0:
            hotcent_array.write("{:5.3f}".format(round(rx,3)))
        hotcent_array.write("{:20.10e}".format(new_list[i]))
    hotcent_array.write('\n')
    num += 1
hotcent_array.close()
print("Make sure you input a right symbol: {}!\n".format(elem_sym))
print("Save hamitonian and overlap matrix from {} to band_term.txt!".format(skf_name))
# ============================================================================ #
# 2. Plot orbital hamitonian and overlap matrix
# ============================================================================ #
print("#==========================================================#")
band_term = np.loadtxt('band_term.txt', skiprows=1)
print("Read band_term.txt for ploting!")
band_pd = pd.DataFrame(band_term, columns=['rx', '$H_{dd}-\sigma$', '$H_{dd}-\pi$', '$H_{dd}-\delta$', '$H_{pd}-\sigma$', '$H_{pd}-\pi$', '$H_{pp}-\sigma$', '$H_{pp}-\pi$', '$H_{sd}-\sigma$', '$H_{sp}-\sigma$', '$H_{ss}-\sigma$', '$S_{dd}-\sigma$', '$S_{dd}-\pi$', '$S_{dd}-\delta$', '$_{Spd}-\sigma$', '$S_{pd}-\pi$', '$S_{pp}-\sigma$', '$S_{pp}-\pi$', '$S_{sd}-\sigma$', '$S_{sp}-\sigma$', '$S_{ss}-\sigma$'])

# strip zero columns
df = band_pd.loc[:, (band_pd != 0).any(axis=0)]
mid = int((len(df.keys()) - 1) /2)
hdf = df.iloc[:,1:mid+1]; sdf = df.iloc[:,mid+1:]
print("None zero {} Orbits: \n{}".format(len(df.keys())-1,df.keys()))
fs = 15
fig, ax = plt.subplots(2, 1, sharex=True,figsize=(4, 7), dpi=300)
x = df['rx'] * Bohr  # convert to angstrom
y00 = [0] * len(x)   # Zero line
xstart = 0.8 # Angstrom
nn = int((xstart/Bohr - rmin)/gridist)

# Plot Overlap matrix
ax[0].plot(x, y00, c='black', linestyle='--')
for label in sdf.keys():
    ax[0].plot(x, sdf[label], label=label)

ax[0].set_ylim(min(sdf[nn:].min())*1.5, max(sdf[nn:].max()*1.5))
ax[0].set_ylabel('$S_{\mu v}$', fontsize=fs)
ax[0].legend(loc='upper right', fontsize=fs-8)


# Plot Hamiltonian matrix
ax[1].plot(x, y00, c='black', linestyle='--')
for label in hdf.keys():
    ax[1].plot(x, hdf[label], label=label)

ax[1].set_xlim(xstart, max(x))
ax[1].set_ylim(min(hdf[nn:].min())*1.5, max(hdf[nn:].max()*1.5))
ax[1].set_xlabel('$r/\AA$', fontsize=fs)
ax[1].set_ylabel('$H_{\mu\ v}$', fontsize=fs)
ax[1].legend(loc='upper right', fontsize=fs-8)
fig.tight_layout()
fig_name = 'band_' + elem_sym + '-' + elem_sym + '.png'
plt.savefig(fig_name, dpi=300)
print("Save plot to", fig_name)
print("#==========================================================#")
