import os
from logging import raiseExceptions
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.units import Bohr, Hartree

from scipy import interpolate

import xml.etree.ElementTree as ET
from utils.parameters_set import get_PTBP, get_QNplusRep, get_PRIOR
from xml.dom import minidom

class physrep():
    def __init__(self,elements=[1,6,8],
                 rmin=0.5,
                 rmax=20.0,
                 npoints=750,
                 scale_cov=[1.0,1.0,1.0],
                 k_xc=[0.1,0.1,0.1],
                 filename='',
                 Ecut=1e-8,
                 Cons_zeff=[1.,1.]):
        
        self.elements  = elements
        self.rmin      = rmin
        self.rmax      = rmax
        self.npoints   = npoints
        self.Cons_zeff = Cons_zeff
        self.scale_cov = {} #scale_cov
        self.k_xc      = {} #kxc
        for i,Zi in enumerate(self.elements):
            self.scale_cov[Zi] = scale_cov[i]
            self.k_xc[Zi]      = k_xc[i]
        
        self.filename  = filename
        self.Ecut      = Ecut
        
        self.r         = np.linspace(rmin,rmax,npoints)
        self.atsize    = {Zi:(covalent_radii[Zi]/Bohr)*self.scale_cov[Zi] for Zi in self.elements}
        
        self.Vreps     = {}
        self.rcuts     = {}
        #self.Vsplines = {}
        
        # mncui

        #compute all potentials
        for i in range(len(self.elements)):
            # i in [1,6,8]
            for j in range(i+1): 
                key1 = str(self.elements[i])+'_'+str(self.elements[j])   # 1_6, 6_1, 6_6 ...
                key2 = str(self.elements[j])+'_'+str(self.elements[i])
                Vrep,rcut = self.calc_Vrep(self.elements[i],self.elements[j])  # This step do not distinguish C_H and H_C
                Vspline   = self.get_splines(Vrep,rcut)  # This step sldo do not distinguish C_H and H_C
                self.Vreps[key1] = Vspline
                self.rcuts[key1] = rcut
                self.Vreps[key2] = Vspline
                self.rcuts[key2] = rcut


    def get_energy(self, mole):

        # def Compute_Forces(pos,acc,ene_pot,epsilon,BoxSize,DIM,N):
    # Compute forces on positions using the Lennard-Jones potential
    # Uses double nested loop which is slow O(N^2) time unsuitable for large systems
        # pos = mole.get_positions() / Bohr
        pos = mole.get_scaled_positions()
        ANs = mole.get_atomic_numbers()
        DIM = 3
        N = len(pos)
        BoxSize = mole.cell.cellpar()[:3] / Bohr

        Sij = np.zeros(DIM) # Box scaled units
        Rij = np.zeros(DIM) # Real space units
        
        #Set all variables to zero
        # ene_pot = ene_pot*0.0
        # acc = acc*0.0
        ene_pot = np.zeros(len(ANs))
        acc = np.zeros((len(ANs),3))  # mncui
        virial=0.0
        
        # Loop over all pairs of particles(except one particle inside box)
        for i in range(N-1):
            for j in range(i+1,N): #i+1 to N ensures we do not double count
                key1 = str(ANs[i])+'_'+str(ANs[j]) # C_C, C_H, H-H...
                # print('Evaluate:\n', key1, '\n', self.rcuts[key1], '\n', 'Total:\n', np.sum(self.Vreps[key1].c) * Hartree / 8)
                Rcutoff = self.rcuts[key1]
                Sij = pos[i,:]-pos[j,:] # Distance in box scaled units ( already / Bohr)
                for l in range(DIM): # Periodic interactions
                    if (np.abs(Sij[l])>0.5):
                        Sij[l] = Sij[l] - np.copysign(1.0,Sij[l]) # If distance is greater than 0.5  (scaled units) then subtract 0.5 to find periodic interaction distance.
                
                Rij = BoxSize*Sij # Scale the box to the real units in this case reduced LJ units (already / Bohr)
                Rsqij = np.dot(Rij,Rij) # Calculate the square of the distance
                
                if(Rsqij < Rcutoff**2):
                    # Calculate LJ potential inside cutoff
                    # We calculate parts of the LJ potential at a time to improve the efficieny of the computation (most important for compiled code)
                    # rm2 = 1.0/Rsqij # 1/r^2
                    # rm6 = rm2**3.0 # 1/r^6
                    # rm12 = rm6**2.0 # 1/r^12
                    # phi = epsilon*(4.0*(rm12-rm6)-phicutoff) # 4[1/r^12 - 1/r^6] - phi(Rc) - we are using the shifted LJ potential
                    # # The following is dphi = -(1/r)(dV/dr)
                    # dphi = epsilon*24.0*rm2*(2.0*rm12-rm6) # 24[2/r^14 - 1/r^8]
                    phi = self.Vreps[key1](Rsqij)
                    ene_pot[i] = ene_pot[i]+0.5*phi # Accumulate energy
                    ene_pot[j] = ene_pot[j]+0.5*phi # Accumulate energy

                    dphi = (self.Vreps[key1](Rsqij,1)/Rsqij)*Sij
                    virial = virial + dphi*np.sqrt(Rsqij) # Virial is needed to calculate the pressure
                    acc[i,:] = acc[i,:]+dphi*Sij # Accumulate forces
                    acc[j,:] = acc[j,:]-dphi*Sij # (Fji=-Fij)
        return acc, np.sum(ene_pot)*Hartree, -virial/DIM # return the acceleration vector, potential energy and virial coefficient
    
    def evaluate_pbc(self, mol):
        ANs = mol.get_atomic_numbers()
        e   = 0.0
        f   = np.zeros((len(ANs),3))

# ============================================================================ #
# Edit it for bulk structures
        n_atom= mol.get_global_number_of_atoms() # total number of atoms
        i_cell= np.min(mol.cell.cellpar()[:3])   # The smallest value from a/b/c
        rcut  = np.max([rcut for rcut in self.rcuts.values()])  # The largest rcut value
        num   = ceil(rcut / i_cell)
        n_sup = num * 2 + 1                      # The size of supercell
        s_cell= mol * (n_sup, n_sup, n_sup)      # Create supercell
        s_pos = s_cell.get_positions() / Bohr    # Get positions of atoms in supercell(Bohr)
        s_ANs = s_cell.get_atomic_numbers()      # Atomic numbers list [6,6,6...]

        # The most confused part of the index of the first atom in the central cell
        # Part 1: Atoms of the (num) neighboring cells 
        # Part 2: Atoms of central neighboring cells
        # index = n_atom * n_sup ** 2 * num  +  n_atom * num * (n_sup + 1) 
        index = n_atom * num * (n_sup ** 2 + n_sup + 1) # Consised way

        p_cell= s_cell[index: index + n_atom]
        p_pos = p_cell.get_positions() / Bohr
        p_ANs = p_cell.get_atomic_numbers()

        for i in range(len(p_cell)):
            for j in range(len(s_cell)):
                key1= str(p_ANs[i]) + '_' + str(s_ANs[j])
                vij = p_pos[i] - s_pos[j]
                rij = np.linalg.norm(vij)
                # print(dists)
                arg_previous = i % n_atom
                arg_supercell = j % n_atom
                # if arg_previous == arg_supercell:
                #     continue

                if rij < self.rcuts[key1] and rij > 0.01:
                    e += self.Vreps[key1](rij) / 2
                    f[i,:] += -(self.Vreps[key1](rij,1)/rij)*vij

        return e*Hartree, f*Hartree/Bohr


    def evaluate_bulk(self, mol):
        # print("Evaluate bulk not stable for forces!!!!!!!!!!!!!!")
        ANs = mol.get_atomic_numbers()
        e   = 0.0
        f   = np.zeros((len(ANs),3))        

# ============================================================================ #
# Edit it for bulk structures
        n_atom= mol.get_global_number_of_atoms()
        i_cell= np.min(mol.cell.cellpar()[:3])   # The smallest constant value 
        rcut  = np.max([rcut for rcut in self.rcuts.values()])  # The largest rcut value
        num   = ceil(rcut / i_cell) 
        n_sup = num * 2 + 1                                     # How big the supercell
        s_cell= mol * (n_sup, n_sup, n_sup)
        s_pos = s_cell.get_positions() / Bohr
        s_ANs = s_cell.get_atomic_numbers()

        index = n_atom * n_sup ** 2 * num + n_atom * (n_sup * num + num)
        p_cell= s_cell[index: index + n_atom]
        p_pos = p_cell.get_positions() / Bohr
        p_ANs = p_cell.get_atomic_numbers()

        for i in range(len(p_cell)):
            for j in range(len(s_cell)):
                key1= str(p_ANs[i]) + '_' + str(s_ANs[j])
                vij = p_pos[i] - s_pos[j]
                rij = np.linalg.norm(vij)
                # print(dists)
                if rij < self.rcuts[key1] and rij > 0.1:
                    e += self.Vreps[key1](rij) / 2  # Now Vreps is a function (Cubic Spline) in respect to rcut
                    f[i,:] += -(self.Vreps[key1](rij,1)/rij)*vij # difference
                    if j > index -1  and j < index + n_atom:
                        f[j - index,:] +=  (self.Vreps[key1](rij,1)/rij)*vij# The sign represent the front and back forces.

        # # To minus the double count forces / energies.
        # pos = mol.get_positions() / Bohr        # convert unit to Bohr
        # for i in range(len(ANs)):
        #     for j in range(i):                  # avoid atomic couple repeatting
        #         key1 = str(ANs[i])+'_'+str(ANs[j]) # C_C, C_H, H-H...
        #         vij = pos[i] - pos[j]      
        #         rij = np.linalg.norm(vij)       # distance between atoms
        #         # print('Evaluate:\n', key1, '\n', self.rcuts[key1], '\n', 'Total:\n', np.sum(self.Vreps[key1].c) * Hartree / 8)
        #         if rij < self.rcuts[key1]:
        #             # e -= self.Vreps[key1](rij)  # Now Vreps is a function (Cubic Spline) in respect to rcut
        #             f[i,:] -= -(self.Vreps[key1](rij,1)/rij)*vij  # difference
        #             f[j,:] -=  (self.Vreps[key1](rij,1)/rij)*vij  # The sign represent the front and back forces.
        t1,t2 = e*Hartree, f*Hartree/Bohr
        # return e*Hartree,f*Hartree/Bohr             
        # print('original:',t1, t2)
        # print('edit:', t1, np.zeros_like(t2))
        return t1, np.zeros_like(t2)


    def evaluate_mole(self,mol):
        ANs = mol.get_atomic_numbers()
        pos = mol.get_positions() / Bohr        # convert unit to Bohr
        e   = 0.0
        f   = np.zeros((len(ANs),3))
        for i in range(len(ANs)):
            for j in range(i):                  # avoid atomic couple repeatting
                key1 = str(ANs[i])+'_'+str(ANs[j]) # C_C, C_H, H-H...
                vij = pos[i] - pos[j]      
                rij = np.linalg.norm(vij)       # distance between atoms
                # print('Evaluate:\n', key1, '\n', self.rcuts[key1], '\n', 'Total:\n', np.sum(self.Vreps[key1].c) * Hartree / 8)
                if rij < self.rcuts[key1]:
                    e += self.Vreps[key1](rij)  # Now Vreps is a function (Cubic Spline) in respect to rcut
                    f[i,:] += -(self.Vreps[key1](rij,1)/rij)*vij  # difference
                    f[j,:] +=  (self.Vreps[key1](rij,1)/rij)*vij  # The sign represent the front and back forces.
        return e*Hartree,f*Hartree/Bohr                
                
    def plot_potentials(self):
        for i in range(len(self.elements)):
            for j in range(i+1): 
                key1 = str(self.elements[i])+'_'+str(self.elements[j])
                
                rij = np.linspace(1.0,self.rcuts[key1],100)
                plt.plot(rij,self.Vreps[key1](rij))
                plt.xlabel('$r_{ij}$ / $a_{0}$',fontsize=14)
                plt.ylabel('$V_{rep}$ / $E_{H}$',fontsize=14)
                plt.title(chemical_symbols[self.elements[i]]+chemical_symbols[self.elements[j]])
                plt.savefig('rep_pot.png', dpi = 300)
                plt.clf()
                
    def calc_Hartree(self,ga_ij,Rij):
        from scipy.special import erf
        if Rij> 0.0:
            return erf(ga_ij*Rij)/Rij
        else: 
            return 2.0*ga_ij/np.sqrt(np.pi)

    def calc_Gamma(self,a1,a2):
        return 1.0/np.sqrt(a1**2+a2**2)

    def calc_Overlap(self,atsize1,atsize2,Rij):
        a1  = 1./atsize1**2
        a2  = 1./atsize2**2
        k = 4.*(a1*a2)
        k /= (a1+a2)**2
        k = np.power(k,3/4)
        return k*np.exp(((-a1*a2)/(a1+a2))*Rij**2)

    def calc_Vrep(self,Z1,Z2):
        k_mean      = (self.k_xc[Z1]+self.k_xc[Z2])/2.                                            # the place that k_xc used
        ga_ij       = self.calc_Gamma(self.atsize[Z1],self.atsize[Z2])                            # Gamma_ij
        Hartree     = -Z1*Z2*np.array([self.calc_Hartree(ga_ij,Rij) for Rij in self.r])           # Hartree potential
        Core        =  Z1*Z2*np.array([1./Rij for Rij in self.r])                                 # static interaction between Atomic Core
        xc          = -k_mean*Z1*Z2*np.array([self.calc_Overlap(self.atsize[Z1],self.atsize[Z2],Rij) for Rij in self.r])

        # exchange and correlation interaction

        Vrep        = Hartree.reshape(self.npoints,) + Core.reshape(self.npoints,) + xc.reshape(self.npoints,)   # np.ndarray type, 750x1
        Npts        = Vrep.shape[0]     # 750
        iszero      = True
# mncui
        rcut        = self.r[Npts-1] + 0.1
        for i in range(Npts):
            ind = Npts -1 -i
            if abs(Vrep[ind]) > self.Ecut:   # Continue until Vrep > 1e-8, save the largest distance in [0.5, 20, 750]
                iszero = False         
                rcut = self.r[ind] + 0.1
            if not iszero:
                break
        # print(Z1,'_', Z2,'rcut:', rcut, 'type', type(rcut))
        return Vrep, rcut
        

    def get_splines(self,V,rcut):
        Vcut = []
        rscut = []
        for i,Rij in enumerate(self.r):
            if Rij > rcut:
                break
            else:
                rscut.append(Rij) # Add cut off value  
                Vcut.append(V[i]) # V = Vrep
        Vcut = np.array(Vcut)
        rscut = np.array(rscut)
        
        spli = interpolate.CubicSpline(rscut, Vcut,bc_type='natural')
        return spli
    
    def exp_func(self,x, a1, a2, a3):
        return np.exp(-a1*x+a2) + a3

    def write0_skf(self):
        for Z1 in self.elements:
            for Z2 in self.elements:
                key1 = str(Z1)+'_'+str(Z2)
                spli = self.Vreps[key1]
                # print('Write_skf:\n',key1,'\n', 'Total:\n', np.sum(spli.c),'\n', spli.c)
                fname = self.filename + chemical_symbols[Z1] + '-' + chemical_symbols[Z2] + '.skf'
# mncui
                print('-----*.skf file existed--backup--Write in new *.skf files-----')
                os.rename(fname, fname.lower())

                # copy Hamiltonian part
                skf_f = open(fname,'w')  
                for l in open(fname.lower(),'r').readlines():
                    if l != 'Spline \n':
                        skf_f.write(l)
                    else:
                        break
                skf_f.close()
                
                with open(fname,"a+") as f:
                    f.write("Spline \n")
                    nint = spli.c.shape[1] #len(rscut)-1
                    coef = spli.c
                    rscut = spli.x
                    line = str(nint) + " " + str(rscut[-1]) + "\n"
                    f.write(line)
                    line = "1.0 0.0 " + str(0.0) + " \n"
                    f.write(line)
                    for i in range(nint-1):
                        line = (str(rscut[i]) + " " +
                                str(rscut[i+1]) + " " +
                                str(0.0) + " " +
                                str(0.0) + " " +
                                str(0.0) + " " +
                                str(0.0) + " \n")
                        f.write(line)

                    line = (str(rscut[nint-1]) + " " +
                            str(rscut[nint]) + " " +
                            str(0.0) + " " +
                            str(0.0) + " " +
                            str(0.0) + " " +
                            str(0.0) + " " +
                            str(0.0) + " " +
                            str(0.0) + " \n")
                    f.write(line)

    def write_info(self, fname):

        """
        Write the parameters info into skf file for elementary
        """


    def get_info(self, symbol, known_parameters):
        """
        symbol: str, e.g. 'O'
        known_parameters: str, 'PTBP', 'QNplusRep', 'PRIOR'
        """
        if known_parameters.lower() == 'ptbp':
            parameter_info = get_PTBP()[symbol]
        elif known_parameters.lower() == 'qnplusrep':
            parameter_info = get_QNplusRep()[symbol]
        elif known_parameters.lower() == 'prior':
            parameter_info = get_PRIOR()[symbol]

        info = {
        'General': {
            'Identifier': f'{symbol}-{symbol}.skf',
            'Author': 'Mengnan Cui, Margraf T. Johannes et al.',
            'Creation': '2024, June. 12',
            'Element': 'O',
            },
        'SK_table': {
            'Code': 'hotcent-2.0',
            'Functional': 'Exchange: PBE, Correlation: PBE',
            'Superposition': 'dense',
            'Basis': {
                'atom': '1',
                'Shells': f"{parameter_info['valence_shell']}",
                'Relativistic': 'yes',
                'Power': f"{parameter_info['r0_d']}",
                'Density': f"{parameter_info['r0_d']}",
                'Wavefunction': ' '.join([str(parameter_info['r0_w']) for i in range(len(parameter_info['valence_shell']))]),
                },
            },
        'E_rep': {
            'Storange': 'spline',
            'SCC': 'yes',
            'Ab_initio': 'VASP, DFT(PBE)',
            'Fit_system': 'Elementary solid',
            },
        }

        return info
    

    def create_xml_document(self, symbol, known_parameters):

        info = self.get_info(symbol, known_parameters)

        # Set xml file
        root = ET.Element("Documentation")


        # Add General element
        general = ET.SubElement(root, "General")
        for key, value in info['General'].items():
            sub_elem = ET.SubElement(general, key)
            sub_elem.text = value


        sk_table = ET.SubElement(root, "SK_table")
        for key, value in info['SK_table'].items():
            sub_elem = ET.SubElement(sk_table, key)
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    sub_sub_elem = ET.SubElement(sub_elem, sub_key)
                    sub_sub_elem.text = sub_value
            else:
                sub_elem.text = value
        sk_table = ET.SubElement(root, "E_rep")


        for key, value in info['E_rep'].items():
            sub_elem = ET.SubElement(sk_table, key)
            sub_elem.text = value

        # tree = ET.ElementTree(root)
        xml_string = ET.tostring(root, encoding='utf8')
        pretty_xml_as_string = minidom.parseString(xml_string).toprettyxml(indent="    ")
        pretty_xml_as_string = '\n'.join(pretty_xml_as_string.split('\n')[1:])
        return pretty_xml_as_string


    def add_parameter_info(self, fname, known_parameters):
        sym_a = fname.split('-')[0]
        print(f"Symbol:{sym_a}")

        xml_string = self.create_xml_document(sym_a, known_parameters)

        with open(fname, 'a', encoding='utf-8') as f:
            f.write(xml_string)

        print(f"Write parameter info to {fname}")  

                    
    def write_skf(self, known_parameters):
        for Z1 in self.elements:
            for Z2 in self.elements:
                key1 = str(Z1)+'_'+str(Z2)
                spli = self.Vreps[key1]
                # print('Write_skf:\n',key1,'\n', 'Total:\n', np.sum(spli.c),'\n', spli.c)
                fname = self.filename + chemical_symbols[Z1] + '-' + chemical_symbols[Z2] + '.skf'
# mncui
                if not os.path.exists(fname):
                    raiseExceptions('-----*.skf file not existed-----')

                else: # to avoid write too many times 'Spline' content
                    # print('-----*.skf file existed--backup--Write in new *.skf files-----')
                    # Use a tmp suffix instead of fname.lower() so the rename is a
                    # real rename on case-insensitive filesystems (macOS APFS).
                    # On case-insensitive fs, fname and fname.lower() share an
                    # inode, so the subsequent open(fname, 'w') would truncate
                    # the same file we are trying to read from.
                    tmp = fname + '.tmp'
                    os.rename(fname, tmp)
                    # copy Hamiltonian part
                    skf_f = open(fname,'w')
                    for l in open(tmp,'r').readlines():
                        if l != 'Spline \n':
                            skf_f.write(l)
                        else:
                            break
                    skf_f.close()

                    os.system(f"rm {tmp}")
                    
                    with open(fname,"a+") as f:
                        f.write("Spline \n")
                        nint = spli.c.shape[1] #len(rscut)-1
                        coef = spli.c
                        rscut = spli.x
                        line = str(nint) + " " + str(rscut[-1]) + "\n"
                        f.write(line)
                        line = "1.0 0.0 " + str(coef[3,0]-self.exp_func(rscut[0],1.0,0.0,0.0)) + " \n"
                        f.write(line)
                        for i in range(nint-1):
                            line = (str(rscut[i]) + " " +
                                    str(rscut[i+1]) + " " +
                                    str(coef[3,i]) + " " +
                                    str(coef[2,i]) + " " +
                                    str(coef[1,i]) + " " +
                                    str(coef[0,i]) + " \n")
                            f.write(line)
  
                        line = (str(rscut[nint-1]) + " " +
                                str(rscut[nint]) + " " +
                                str(coef[3,nint-1]) + " " +
                                str(coef[2,nint-1]) + " " +
                                str(coef[1,nint-1]) + " " +
                                str(coef[0,nint-1]) + " " +
                                str(0.0) + " " +
                                str(0.0) + " \n")
                        f.write(line)
                    if Z1 == Z2:
                        self.add_parameter_info(fname, known_parameters)
                    




    
    def write_specific_skf(self, except_elem, known_parameters):

        element_num = [i for i in self.elements if i not in except_elem]

        for Z1 in element_num:
            Z2 = except_elem[0]
            key_list = [str(Z1)+'_'+str(Z2), str(Z2)+'_'+str(Z1), str(Z1)+'_'+str(Z1), str(Z2)+'_'+str(Z2)]
            spline_list = [self.Vreps[key] for key in key_list]
            fname_list = [self.filename + chemical_symbols[Z1] + '-' + chemical_symbols[Z2] + '.skf', \
                          self.filename + chemical_symbols[Z2] + '-' + chemical_symbols[Z1] + '.skf', \
                          self.filename + chemical_symbols[Z1] + '-' + chemical_symbols[Z1] + '.skf', \
                          self.filename + chemical_symbols[Z2] + '-' + chemical_symbols[Z2] + '.skf']
            
            for n, fname in enumerate(fname_list):
                
                print("fname: ", fname, "\nkey: ", key_list[n])
                key = key_list[n]
                spli = spline_list[n]

                if not os.path.exists(fname):
                    raiseExceptions('-----*.skf file not existed-----')

                else: # to avoid write too many times 'Spline' content
                    print('-----*.skf file existed--backup--Write in new *.skf files-----')
                    # See write_skf above: fname.lower() collides with fname on
                    # case-insensitive filesystems. Use a real tmp suffix.
                    tmp = fname + '.tmp'
                    os.rename(fname, tmp)

                    # copy Hamiltonian part
                    skf_f = open(fname,'w')
                    for l in open(tmp,'r').readlines():
                        if l != 'Spline \n':
                            skf_f.write(l)
                        else:
                            break
                    skf_f.close()
                    os.remove(tmp)
                    
                    with open(fname,"a+") as f:
                        f.write("Spline \n")
                        nint = spli.c.shape[1] #len(rscut)-1
                        coef = spli.c
                        rscut = spli.x
                        line = str(nint) + " " + str(rscut[-1]) + "\n"
                        f.write(line)
                        line = "1.0 0.0 " + str(coef[3,0]-self.exp_func(rscut[0],1.0,0.0,0.0)) + " \n"
                        f.write(line)
                        for i in range(nint-1):
                            line = (str(rscut[i]) + " " +
                                    str(rscut[i+1]) + " " +
                                    str(coef[3,i]) + " " +
                                    str(coef[2,i]) + " " +
                                    str(coef[1,i]) + " " +
                                    str(coef[0,i]) + " \n")
                            f.write(line)

                        line = (str(rscut[nint-1]) + " " +
                                str(rscut[nint]) + " " +
                                str(coef[3,nint-1]) + " " +
                                str(coef[2,nint-1]) + " " +
                                str(coef[1,nint-1]) + " " +
                                str(coef[0,nint-1]) + " " +
                                str(0.0) + " " +
                                str(0.0) + " \n")
                        f.write(line)

                    if Z1 == Z2:
                        self.add_parameter_info(fname, known_parameters)
        
        
