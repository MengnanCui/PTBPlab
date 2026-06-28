import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from utils.calculator import dist_min


matplotlib.use('Agg')

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

def plot_vs(symbol, parameters, dft, dft_AE_relative, dftb_AE_relative, bolt_f):
    """
    Plot the energy figure between DFT and Optimized DFTB

    Parameters:


    """
    r0, r0_dens, c_rep, mu = parameters
    n_moles = int(len(dft) / 11) ; print('n_moles:', n_moles)

    d_min = dist_min(dft)
    print("len:", len(d_min), len(dft_AE_relative), len(dftb_AE_relative))

    line_name  = ['Dimer', 'Graphite', 'Diamond', '$\\mathrm{{\\beta}}$-Sn', 'BCC', 'FCC']
    plt.rcParams["figure.figsize"] = (5,5)
    basic_ms = 20
    for i in range(n_moles):
        if bolt_f[i] >= 0.1:
            plt.scatter(d_min[i*11:i*11+11], dft_AE_relative[i*11:i*11+11], marker='x',  linewidths=2, s= 10 * basic_ms * bolt_f[i], color=colors[i], alpha=0.8)
            plt.plot(d_min[i*11:i*11+11], dftb_AE_relative[i*11:i*11+11], marker='o', color=colors[i],  linestyle=linestyles[i], markersize=1.2 * basic_ms * bolt_f[i], alpha=0.7,  label=line_name[i])
            

        else:
            # plt.plot(d_min[i*11:i*11+11], dftb_AE_relative[i*11:i*11+11],color='#568203', markersize=0.5, linestyle='-',  linewidth=2.0, alpha=0.01)
            continue


    plt.ylabel('Rel. Energy (eV/atom)')
    plt.xlabel('$d_{\mathrm{min}}$ ($\mathrm{\AA}$)')
    plt.title(f"$r_0$={r0:.2f},$r_0^{{\mathrm{{d}}}}$={r0_dens:.2f},$c_{{\mathrm{{rep}}}}$={c_rep:.2f}, $\mu$={mu:.2e}")

    # plt.suptitle("{}".format(symbol))
    plt.legend()
    plt.tight_layout()
    # plt.title('Boltzmann factor', fontsize=fs)
    plt.savefig('dmin.png', dpi=200)
    print("Save dmin.png!!") 
    plt.clf()

def plot_tradition(symbol, parameter_list, mu, dft_AE_relative, dftb_AE_relative, bolt_f):
    """
    Plot the energy figure between DFT and Optimized DFTB

    Parameters:
            symbol: chemical symbol, eg: 'C'
            parameter_list: the list of parameters, eg: [r_wave, r_dens, c_rep]
            mu: the loss function
            dft_AE_relative: the list of DFT relative energy per atom
            dftb_AE_relative: the list of Optimized DFTB relative energy per atom
            bolt_f: the list of Boltzmann factor of each configuration

    """

    n_moles = int(len(dft_AE_relative) / 11)
    r_wave, r_dens, c_rep, sigma = parameter_list

    large_font=10
    small_font=7
    X       = np.arange(80, 120.1, 4)
    config_list = ['dimer', 'graphite', 'diamond', 'beta-Sn', 'bcc', 'fcc']
    fig     = plt.figure(figsize=(6,6))
    plt.suptitle('Para_dftb:$r_w$={:0.3f} $r_d$={:0.3f} $c_{{rep}}$={:0.3f} loss={:.3e}'.format(r_wave, r_dens, c_rep, mu), fontsize=large_font)
    for p in range(0,n_moles):
        ax     = fig.add_subplot(math.ceil(n_moles/2), 2, p+1)
        Y_dftb = dftb_AE_relative[11*p : 11*(p+1)]
        Y_dft  = dft_AE_relative[11*p  : 11*(p+1)]
        if bolt_f[p] < 0.10:
            alpha=0.3
        else:
            alpha=1.0
        
        ax.scatter(X, Y_dft , marker='x', s=40, linewidths=1, color = '#0d3f67', \
            alpha=alpha, label = 'DFT_'  + symbol)
        ax.plot(X, Y_dftb, marker='o', markersize=5, linestyle='-', color = '#ca3e47',\
                linewidth=0.8, alpha=alpha, label = 'DFTB_' + symbol)


        # Add annotation arrow to show min volume
        down = min(min(Y_dft, Y_dftb))
        up   = max(max(Y_dft, Y_dftb))
        len_of_arrow = (up - down) / 5
        end_coor, start_coor = (X[np.argmin(Y_dftb)], min(Y_dftb) + 0.5 * len_of_arrow), (X[np.argmin(Y_dftb)], min(Y_dftb) + 1.2 * len_of_arrow)
        plt.annotate('',xy=end_coor,xytext=start_coor,arrowprops=dict(arrowstyle="wedge",\
                        facecolor='#ca3e47'))

        ax.set_title(config_list[p] + ', bolt:' + str(round(bolt_f[p], 2)), fontsize=small_font, alpha=alpha)
        ax.tick_params(axis='both', labelsize=small_font)
        if bolt_f[p] < 0.10:
            plt.tick_params(axis='x',colors='grey')
            plt.tick_params(axis='y',colors='grey')
        if p > 3:
            ax.set_xlabel('Volume(%)',  alpha=alpha, fontsize=small_font)
            ax.set_xticks(np.arange(80, 120.1, 10))
        if p % 2 ==0:
            ax.set_ylabel('$\Delta$E/atom(eV)',  alpha=alpha, fontsize=small_font)
        if p == 0:
            ax.legend(loc='best',fontsize=small_font-1)
        else:
            continue
    plt.tight_layout()
    plt.savefig('dftb_ener.png', dpi=300)
    plt.close(fig)

    
