from hotcent.atomic_dft import AtomicDFT

from parameters_set import get_PTBP, get_QNplusRep, get_PRIOR
import pylibxc
import re
from hotcent.confinement import PowerConfinement


def get_orbit(valence_shell:str) -> tuple:
            """
            Parser valence_shell string occupation, valances, orbits

            Parameters:
            -----------
                valence_shell(str): valence shell of an element
                eg. '4d8 5s1 5p0' for 'Rh' 

            Return:
                eg. (occupations,valences,orbits,valence_electron_numbers) = ({'4d': 8, '5s': 1, '5p': 0}, ['4d', '5s', '5p'],['d', 's', 'p'], 9)
            """

            info, occupations, valences, orbits = [], {}, [], []


            [info.append(i) for i in re.split('\s+', valence_shell.strip())]

            valence_electron_numbers = 0  # number of valence electrons

            for orb in info:
                # maches numbers from the end
                num_elec = re.findall(r'\d+$', orb)
                occupations[orb[:2]] = int(num_elec[0])
                valence_electron_numbers += int(num_elec[0])
                # append valences to [valences] and
                orbits.append(orb[1])
                valences.append(orb[0:2])      # [orbits]
            return(occupations, valences, orbits, valence_electron_numbers)  


def get_AtomicDFT(symbol, parameter_info) -> str:
       
    # Get the parameters:

    occupations, valences, orbits, valence_electron_numbers = get_orbit(parameter_info['valence_shell'])
    conf_wf = {}
    r0_x = [parameter_info['r0_w'], parameter_info['r0_d']]

    for val in valences:  # set confinement for each valence
        conf_wf[val] = PowerConfinement(r0=(r0_x[0]), s=2.0)

    conf_dens = PowerConfinement(r0=r0_x[1], s=2.0)

    atom = AtomicDFT(symbol,
                xc='GGA_X_PBE+GGA_C_PBE',
                configuration=parameter_info['configuration'], #'[Kr] 4d10 5s2 5p2',
                perturbative_confinement=False,
                valence=valences,       #['5s', '5p', '4d'],
                scalarrel=True,
                confinement=conf_dens,
                wf_confinement=conf_wf,
                nodegpts=150,
                mix=0.2,
                txt='-',
                timing=True,
                )

    atom.run()
    atom.plot_density()
    # atom.plot_Rnl()
    # atom.fit_sto('5s', 5, 4, filename='Sn_5s_STO.png')
    # atom.fit_sto('5p', 5, 4, filename='Sn_5p_STO.png')
    # atom.fit_sto('4d', 5, 4, filename='Sn_4d_STO.png')
    atom.write_hsd(filename=f'{symbol}_wfc.hsd')



def main():

    # symbol_list = ['In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Ra', 'Th']
    # Create the AtomicDFT object:
        # Get the parameters:
    known_parameters_list = [get_PTBP(), get_QNplusRep(), get_PRIOR()]
    known_parameters = get_PTBP()

    symbol_list = known_parameters.keys()
    print(symbol_list)

    name_list = []
    for symbol in known_parameters:

        if symbol in symbol_list:
            parameter_info = known_parameters[symbol]  
            get_AtomicDFT(symbol, parameter_info)

        
    name_list.append(f"{symbol}_wfc.hsd")


    full_name = ' '.join(name_list)
    print(full_name)
main()

