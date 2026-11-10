#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
from lxml import etree, objectify
import numpy as np
import _hotcent
from hotcent.interpolation import build_interpolator, CubicSplineFunction
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.radial_grid import RadialGrid
from hotcent.separable_pseudopotential import SeparablePP
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    from pylibxc.util import xc_functional_get_name
    has_pylibxc = True
except ImportError:
    print('Warning -- could not load pylibxc')
    has_pylibxc = False


class KleinmanBylanderPP(SeparablePP):
    """
    Class representing a Kleinman-Bylander type pseudopotential
    (i.e. a norm-conserving pseudopotential in fully separable form;
    Kleinman and Bylander, Phys. Rev. Lett. (1982),
    doi:10.1103/PhysRevLett.48.1425).

    Parameters
    ----------
    filename : str
        Name of a pseudopotential file in PSF or PSML format.
    local_component : str, optional
        Controls the choice of local potential:
        * 'siesta': use a similar exponential distribution
           as in the Siesta code (as described in the main text
           of doi:10.1088/0953-8984/14/11/302). Note that the
           'large core' modification (footnote 9 in the above paper)
           is not applied here. This is the default for PSF
           pseudopotentials.
        * 'vanderbilt': apply the same analytical function
           as used by default in the APE code
           (see https://tddft.org/programs/APE/input/#llocal).
        * 'from_pp_file': uses the local potential supplied by
           the pseudopotential file (PSML format only).
           This is the default for PSML pseudopotentials.
        * 's', 'p', 'd', ...: use the pseudopotential of this
           angular momentum as the local potential.

    Other Parameters
    ----------------
    txt : str or file handle, optional
        See SeparablePP.__init__().
    rcore : float, optional
        Option for controlling the shape of the local potential
        (see self.initialize_local_potential())
    lmax : int, optional
        Option controlling which KB projectors to include
        (see self.filter_angular_momenta()).
    valence : list of str, optional
        Option controlling which KB projectors to include
        (see self.filter_angular_momenta()).
    with_polarization : bool, optional
        Option controlling which KB projectors to include
        (see self.filter_angular_momenta()).
    """
    def __init__(self, filename, local_component=None, txt='-',
                 rcore=None, lmax=None, valence=None, with_polarization=None):
        SeparablePP.__init__(self, txt=txt)

        self.angular_momenta = []
        self.occupations = {}
        self.subshells = {}
        self.cosines = {}
        self.energies = {}
        self.projectors = {}
        self.rcore = {}
        self.Vl = {}
        self.Vl_fct = {}
        self.Vloc = []
        self.Vloc_fct = None
        self.rgrid = []
        self.rho_core = []
        self.rho_core_fct = None
        self.rho_val = []
        self.rho_val_fct = None
        self.rho_loc = None
        self.rho_loc_fct = None

        self.initialize_from_file(filename)

        if local_component is None:
            if self.source == 'psf':
                self.local_component = 'siesta'
            elif self.source == 'psml':
                self.local_component = 'from_pp_file'
        elif local_component in ANGULAR_MOMENTUM:
            self.local_component = ANGULAR_MOMENTUM[local_component]
        else:
            self.local_component = local_component

        for l in self.angular_momenta:
            self.Vl_fct[l] = CubicSplineFunction(self.rgrid, self.Vl[l])

        self.initialize_local_potential(rcore=rcore)

        self.filter_angular_momenta(lmax=lmax, valence=valence,
                                    with_polarization=with_polarization)
        print('PP: included angular momenta for {0}: {1}'.format(self.symbol,
              self.angular_momenta), file=self.txt)

        assert len(self.angular_momenta) == len(set(self.angular_momenta)), \
               'List of included angular momenta should not contain duplicates'

        assert max(self.angular_momenta) <= 3, \
               'Can only handle projectors with angular momenta up to f'

    def initialize_from_file(self, filename):
        print('PP: using pseudopotential file {0}'.format(filename),
              file=self.txt)

        self.source = filename.split('.')[-1]

        if self.source == 'psf':
            self.initialize_from_psf(filename)
        elif self.source == 'psml':
            self.initialize_from_psml(filename)
        else:
            msg = 'Only PSF and PSML files are supported at the moment.'
            raise NotImplementedError(msg)

    def initialize_from_psf(self, filename):
        """ Read in the given pseudopotential file in '.psf' format.

        Attributes that are initialized in this way include the chemical
        symbol (self.symbol), the radial grid (self.rgrid), the l-dependent
        potentials (self.Vl), the core and valence charge densities
        (self.rho_core, self.rho_val) and the self.has_nonzero_rho_core flag.
        """
        with open(filename, 'r') as f:
            # First line
            items = f.readline().split()

            if len(items) == 3:
                self.symbol, self.relativistic, self.pptype = items
            elif len(items) == 4:
                self.symbol, self.xcname, self.relativistic, self.pptype = items
            else:
                msg = 'The 1st line in the psf file is expected to contain ' + \
                      'either 3 or 4 entries, not {0}'
                raise ValueError(msg.format(len(items)))

            # Second line
            f.readline()

            # Third line
            for item in f.readline().split('/'):
                if len(item.strip()) == 0:
                    continue

                try:
                    nl, occ, _, rc = item.split()
                except ValueError:
                    nlocc, _, rc = item.split()
                    nl = nlocc[:2]
                    occ = nlocc[2:]

                l = ANGULAR_MOMENTUM[nl[1]]
                self.angular_momenta.append(l)
                self.occupations[nl] = float(occ)
                self.rcore[l] = float(rc)
                self.subshells[l] = nl

            # Fourth line
            items = f.readline().split()
            Ndown, Nup, self.N, b, a, self.Zval = items
            Ndown = int(Ndown)
            if int(Nup) > 0:
                print('PP: warning: "up" pseudopotential(s) are present',
                      'but these will not be used', file=self.txt)
            self.N = int(self.N)
            self.Zval = float(self.Zval)
            assert np.isclose(self.Zval, sum(self.occupations.values()))

            # Next lines contain 1D arrays
            label_dict = {
                'radial grid': 'rgrid',
                'down pseudopotential': 'Vl',
                'core charge': 'rho_core',
                'valence charge': 'rho_val',
            }

            new_section = True
            need_l_read = False
            counter = 0

            for line in f:
                if new_section:
                    assert 'follows' in line
                    label = ' '.join(line.lower().split()[:2])
                    assert label in label_dict
                    key = label_dict[label]
                    need_l_read = key == 'Vl'
                    new_section = False
                    array = []
                    counter = 0

                elif need_l_read:
                    l = int(line)
                    need_l_read = False

                else:
                    items = list(map(float, line.split()))
                    array.extend(items)
                    counter += len(items)
                    assert counter <= self.N

                    if counter == self.N:
                        # Reached the end of the section
                        if key == 'Vl':
                            array = np.array(array) / self.rgrid / 2.  # in Ha
                            self.Vl[l] = array
                        else:
                            setattr(self, key, np.array(array))

                        new_section = True
                        l = None

        self.grid = RadialGrid(self.rgrid)

        # Normalize the valence charge density
        norm = self.grid.integrate(self.rho_val)
        assert abs(norm - self.Zval) < 1e-3, (norm, self.Zval)
        self.rho_val /= norm / self.Zval
        self.rho_val /= 4. * np.pi * self.rgrid ** 2

        # Normalize the (model) core charge density
        self.rho_core /= 4. * np.pi * self.rgrid ** 2
        self.has_nonzero_rho_core = not np.allclose(self.rho_core, 0.)

        # Final checks
        assert len(self.Vl) == Ndown, \
               'One or more semilocal potential(s) are missing'

        for key, val in label_dict.items():
            assert hasattr(self, val)

    def initialize_from_psml(self, filename):
        """ Read in the given pseudopotential file in PSML format.

        See e.g. https://siesta-project.github.io/psml-docs/page/index.html

        Attributes that are initialized in this way include the chemical
        symbol (self.symbol), the radial grid (self.rgrid), the l-dependent
        potentials (self.Vl), the core and valence charge densities
        (self.rho_core, self.rho_val) and the self.has_nonzero_rho_core flag,
        as well as the KB projectors and the local potential.
        """
        error_msg_grids = 'Only PSML pseudopotentials with a ' + \
                          'single (top-level) radial grid are supported'

        def check_attribute(obj, key, expected):
            value = obj.get(key)
            msg = f'{filename}: only {expected} value(s) are supported '
            msg += f'for "{key}" and "{value}" is not one of them'
            if value not in expected:
                raise NotImplementedError(msg)
            return value

        def to_float_array(obj, key='data', ext='zero'):
            data = getattr(obj, key)

            try:
                npts = int(data.get('npts'))
            except TypeError:
                npts = int(obj.get('npts'))
            assert npts <= self.N

            if hasattr(obj, 'grid'):
                raise NotImplementedError(error_msg_grids)

            array = np.zeros(self.N)
            array[:npts] = np.genfromtxt([str(data)])

            if ext == 'zero':
                pass
            elif ext == '-Zval/r':
                array[npts:] = -self.Zval / self.rgrid[npts:]
            else:
                raise NotImplementedError(ext)

            return array

        doc = etree.parse(filename)
        for tag in doc.iter():
            if '-' in tag.tag:
                tag.tag = tag.tag.replace('-', '_')
        obj = objectify.fromstring(etree.tostring(doc))

        check_attribute(obj, 'energy_unit', ['hartree'])
        check_attribute(obj, 'length_unit', ['bohr'])

        self.symbol = obj.pseudo_atom_spec.get('atomic-label')
        self.relativistic = check_attribute(
            obj.pseudo_atom_spec, 'relativity', ['no', 'scalar']
        )

        check_attribute(obj.pseudo_atom_spec, 'meta-gga', ['no', None])
        check_attribute(obj.pseudo_atom_spec, 'spin-dft', ['no'])

        if has_pylibxc:
            # Note: optional 'weights' are not taken into account here
            libxc_info = obj.pseudo_atom_spec.exchange_correlation.libxc_info
            self.xcname = '+'.join([
                xc_functional_get_name(int(xc.get('id')))
                for xc in libxc_info.functional
            ])
        else:
            self.xcname = None

        for shell in obj.pseudo_atom_spec.valence_configuration.shell:
            nl = shell.get('n') + shell.get('l')
            self.occupations[nl] = float(shell.get('occupation'))

        self.Zval = float(obj.pseudo_atom_spec.get('z-pseudo'))
        assert np.isclose(self.Zval, sum(self.occupations.values()))

        if not hasattr(obj, 'grid'):
            raise NotImplementedError(error_msg_grids)

        self.N = int(obj.grid.get('npts'))
        self.rgrid = to_float_array(obj.grid, key='grid_data')
        self.grid = RadialGrid(self.rgrid)

        if abs(self.rgrid[0]) < 1e-12:
            # For radial grids that start at r=0,
            # let them start at 'epsilon' to avoid zero-divisions
            assert abs(self.rgrid[1]) > 1e-5, \
                   'For radial grids starting at r=0, ' + \
                   'the 2nd grid point should be > 1e-5'
            self.rgrid[0] = 1e-9

        for slps in obj.semilocal_potentials.slps:
            l = ANGULAR_MOMENTUM[slps.get('l')]
            self.angular_momenta.append(l)
            self.rcore[l] = float(slps.get('rc'))
            self.subshells[l] = slps.get('n') + slps.get('l')
            self.Vl[l] = to_float_array(slps.radfunc, ext='-Zval/r')

        try:
            self.rho_core = to_float_array(obj.pseudocore_charge.radfunc)
        except AttributeError:
            self.rho_core = np.zeros_like(self.rgrid)

        self.rho_val = to_float_array(obj.valence_charge.radfunc)
        if obj.valence_charge.get('rescaled-to-z-pseudo') in ['no', None]:
            norm = self.grid.integrate(self.rho_val, use_dV=True)
            self.rho_val *= self.Zval / norm

        self.Vloc = to_float_array(obj.local_potential.radfunc, ext='-Zval/r')

        try:
            self.rho_loc = to_float_array(obj.local_potential.local_charge)
            self.rho_loc /= 4 * np.pi
        except AttributeError:
            self.rho_loc = None

        self.has_nonzero_rho_core = \
            obj.pseudo_atom_spec.get('core-corrections') == 'yes'

        if self.has_nonzero_rho_core:
            self.rho_core = to_float_array(obj.pseudocore_charge.radfunc)
            self.rho_core /= 4 * np.pi
            assert not np.allclose(self.rho_core, 0)

        self.pptype = 'pcec' if self.has_nonzero_rho_core else 'nc'

        if hasattr(obj.nonlocal_projectors, 'proj'):
            self.energies = {}
            self.projectors = {}

            for proj in obj.nonlocal_projectors.proj:
                l = ANGULAR_MOMENTUM[proj.get('l')]
                nl = proj.get('seq') + proj.get('l')
                self.energies[nl] = float(proj.get('ekb'))
                self.projectors[nl] = build_interpolator(
                    self.rgrid, to_float_array(proj.radfunc),
                    bc_type='clamped')
                self.cosines[nl] = None
                self.overlap_onsite[(nl, nl)] = None

    def get_occupations(self):
        """ Returns the (valence) electronic configuration for which
        the pseudopotential has been generated.

        Returns
        -------
        occupations : dict
            Dictionary with occupation for every (valence) subshell.
        """
        occupations = self.occupations.copy()
        return occupations

    def all_zero_onsite_overlaps(self):
        """ Returns whether all on-site overlaps of the (valence)
        states with the PP projectors are zero.
        """
        return False

    def filter_angular_momenta(self, lmax=None, valence=None,
                               with_polarization=None):
        """ Filters out angular momenta (and associated properties).

        Parameters
        ----------
        lmax : see filter_angular_momenta_from_lmax().
        valence : see filter_angular_momenta_from_basis().
        with_polarization : see filter_angular_momenta_from_basis().
        """
        if lmax is not None:
            self.filter_angular_momenta_from_lmax(lmax)

        if valence is not None:
            self.filter_angular_momenta_from_basis(valence, with_polarization)
        return

    def filter_angular_momenta_from_lmax(self, lmax):
        """ Filters out angular momenta (and associated properties)
        higher than the given one.

        Parameters
        ----------
        lmax : int
            Maximum angular momentum for the KB projectors.
        """
        for l in self.angular_momenta.copy():
            if l > lmax:
                self.remove_angular_momentum(l)
        return

    def filter_angular_momenta_from_basis(self, valence, with_polarization):
        """ Filters out pangular momenta (and associated properties)
        based on which angular momenta are included in the main basis set.

        Angular momenta higher than "one plus the highest angular momentum
        in the given main basis set" will be discarded.

        Parameters
        ----------
        valence : list of str
            Set of nl values defining the set of minimal valence subshells.
        with_polarization : bool
            Whether polarization functions will be included in the complete
            basis set.
        """
        assert with_polarization is not None, \
               'with_polarization must either be True or False'

        lval = [ANGULAR_MOMENTUM[nl[1]] for nl in valence]

        if with_polarization:
            for l in range(max(lval)+2):
                if l not in lval:
                    lval.append(l)
                    break

        lmax = max(lval) + 1
        self.filter_angular_momenta_from_lmax(lmax)
        return

    def remove_angular_momentum(self, l):
        """ Removes the attributes associated with the given
        angular momentum channel.

        Parameters
        ----------
        l : int
            Angular momentum to remove.
        """
        self.angular_momenta.remove(l)
        self.rcore.pop(l)
        self.subshells.pop(l)
        self.Vl.pop(l)
        self.Vl_fct.pop(l)
        # Note: self.occupations should remain as it is
        return

    def get_subshells(self):
        return sorted(self.subshells.values())

    def initialize_local_potential(self, rcore=None):
        """ Builds the local potential self.Vloc and (if applicable)
        the associated charge distribution self.rho_loc.

        Parameters
        ----------
        rcore : float, optional
            Radius determining the parameters of the 'siesta'- and
            'vanderbilt'-type local potentials. By default the largest
            semilocal potential core radius among the included angular
            momenta is used.
        """
        if rcore is None:
            rcore = self.get_max_rcore()

        if self.local_component == 'siesta':
            a, b = 1.82 / rcore, 1.
            self.rho_loc = np.exp(-(np.sinh(a*b*self.rgrid) / np.sinh(b))**2)
            norm = self.grid.integrate(self.rho_loc, use_dV=True)
            self.rho_loc /= -norm / self.Zval

            dV = self.grid.get_dvolumes()
            r, r0 = self.rgrid, self.grid.get_r0grid()
            N = len(self.rho_loc)
            n0 = 0.5 * (self.rho_loc[1:] + self.rho_loc[:-1])
            n0 *= -self.Zval / np.sum(n0 * dV)
            self.Vloc = _hotcent.hartree(n0, dV, r, r0, N)

        elif self.local_component == 'vanderbilt':
            v0 = -self.Zval * np.exp(11. / 12) / rcore
            v1 = -1.5 / rcore**2
            v2 = 0.75 / rcore**4
            v3 = -1. / (6 * rcore**6)
            poly = v1*self.rgrid**2 + v2*self.rgrid**4 + v3*self.rgrid**6
            self.Vloc = v0 * np.exp(poly)
            index = np.argmax(self.rgrid > rcore)
            self.Vloc[index:] = -self.Zval / self.rgrid[index:]

            self.rho_loc = -2 * self.rgrid**2 * self.Vloc \
                * (18 * self.rgrid**10 * v3**2 + 24 * self.rgrid**8 * v2 * v3 \
                   + 12 * self.rgrid**6 * v1 * v3 + 8 * self.rgrid**6 * v2**2 \
                   + 8 * self.rgrid**4 * v1 * v2 + 21 * self.rgrid**4 * v3 \
                   + 2 * self.rgrid**2 * v1**2 + 10 * self.rgrid**2 * v2 \
                   + 3 * v1)
            self.rho_loc /= 4 * np.pi * self.rgrid**2
            self.rho_loc[index:] = 0.

            norm = self.grid.integrate(self.rho_loc, use_dV=True)
            assert np.isclose(norm, -self.Zval, atol=1e-2), \
                   f'The integrated local charge density ({norm}) ' + \
                   f'is not sufficiently close to -Zval ({-self.Zval})'

        elif self.local_component == 'from_pp_file':
            assert len(self.Vloc) != 0, \
                   'The pseudopotential file did not provide a local potential'

            self.Vloc_fct = CubicSplineFunction(self.rgrid, self.Vloc,
                                                bc_type='clamped')

            if self.rho_loc is None:
                # Solving the Poisson equaltion to get rho_loc from Vloc
                # d/dr (r**2 * dV/dr) = 2 * r * dV/dr + r**2 * d2V/dr2
                dvdr = self.local_potential(self.rgrid, der=1)
                d2vdr2 = self.local_potential(self.rgrid, der=2)
                self.rho_loc = -dvdr / (2 * np.pi * self.rgrid) \
                               - d2vdr2 / (4 * np.pi)
                self.rho_loc[0] = self.rho_loc[1]

                # Determine rcore based on Vloc closeness to -Zval/r
                for i in range(len(self.rgrid), 0, -1):
                    diff = self.Vloc[i-1] + self.Zval / self.rgrid[i-1]
                    if abs(diff) > 1e-5:
                        break
                self.rho_loc[i:] = 0.

            norm = self.grid.integrate(self.rho_loc, use_dV=True)
            assert np.isclose(norm, -self.Zval, atol=1e-2), \
                   f'The integrated local charge density ({norm}) ' + \
                   f'is not sufficiently close to -Zval ({-self.Zval})'

        elif isinstance(self.local_component, int):
            assert self.local_component in self.angular_momenta, \
                   f'Pseudopotential has no l={self.local_component} component'
            self.Vloc = np.copy(self.Vl[self.local_component])
            self.remove_angular_momentum(self.local_component)
            self.rho_loc = None

        else:
            msg = f'Unknown type of local component: {self.local_component}'
            raise ValueError(msg)
        return

    def get_core_density(self, r, der=0):
        """ Evaluates the core electron density at the given radius. """
        if self.rho_core_fct is None:
            self.rho_core_fct = CubicSplineFunction(self.rgrid, self.rho_core)

        if self.has_nonzero_rho_core:
            return self.rho_core_fct(r, der=der)
        else:
            return 0.

    def get_local_density(self, r, der=0):
        """ Evaluates the local electron density at the given radius. """
        assert self.has_local_density, 'Local electron density is not available'
        if self.rho_loc_fct is None:
            self.rho_loc_fct = CubicSplineFunction(self.rgrid, self.rho_loc)
        return self.rho_loc_fct(r, der=der)

    def get_valence_density(self, r, der=0):
        """ Evaluates the valence electron density at the given radius. """
        if self.rho_val_fct is None:
            self.rho_val_fct = CubicSplineFunction(self.rgrid, self.rho_val)
        return self.rho_val_fct(r, der=der)

    def has_local_density(self):
        """ Returns whether a 'local' electron density is available
        from which the local potential can be constructed by the
        Hartree equation.
        """
        return self.rho_loc is not None

    def get_max_rcore(self):
        """ Returns the largest of the different core radii. """
        return max(self.rcore.values())

    def get_cutoff_radius(self, threshold=1e-6):
        """ Returns the radius beyond which all semi-local potentials
        (and hence KB projectors) are considered to be negligible.

        Parameters
        ----------
        threshold : float, optional
            Threshold for the maximum absolute value of the
            different semi-local potentials.

        Returns
        -------
        rc : float
            The cutoff radius.
        """
        rcore_max = self.get_max_rcore()
        i = np.argmax(self.rgrid > rcore_max)

        diff = np.inf
        while diff > threshold:
            rc = self.rgrid[i]
            diff = max([abs(self.semilocal_potential(rc, l))
                        for l in self.angular_momenta])
            i = i + 1
        return rc

    def get_self_energy(self):
        """ Returns the 'self energy' as defined in
        Soler et al., J. Phys.: Condens. Matter (2002),
        doi:10.1088/0953-8984/14/11/302.
        """
        e_self = self.grid.integrate(self.Vloc * self.rho_loc, use_dV=True) / 2
        return e_self

    def local_potential(self, r, der=0):
        """ Returns the local part of the pseudopotential at r. """
        if self.Vloc_fct is None:
            self.Vloc_fct = CubicSplineFunction(self.rgrid, self.Vloc)
        return self.Vloc_fct(r, der=der)

    def semilocal_potential(self, r, l, der=0):
        """ Returns the chosen l-dependent semilocal potential at r. """
        if self.local_component == l:
            return np.zeros_like(r)
        else:
            return self.Vl_fct[l](r, der=der) - self.local_potential(r, der=der)

    def plot_potentials(self, filename=None):
        """ Plots the l-dependent and local potentials. """
        assert plt is not None, 'Matplotlib could not be imported!'

        vmin = np.inf
        imin = np.argmax(self.rgrid > 0.2)

        for l in self.angular_momenta:
            plt.plot(self.rgrid, self.Vl[l], label=r'$\ell$='+str(l))
            vmin = min(vmin, self.Vl[l][imin])

        plt.plot(self.rgrid, self.Vloc, color='k', label='local')
        vmin = min(vmin, self.Vloc[imin])

        Vz = -self.Zval/self.rgrid
        imin = np.argmax(Vz > vmin)
        plt.plot(self.rgrid[imin:], Vz[imin:], ls='--', color='gray',
                 label=r'-Z$_{val}$/r', zorder=0)

        plt.xlim([0., self.get_cutoff_radius()])
        plt.grid()
        plt.xlabel(r'r [a$_0$]')
        plt.ylabel(r'$V$ [Ha]')
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.clf()

    def plot_density(self, valence=True, logscale=True, filename=None):
        assert plt is not None, 'Matplotlib could not be imported!'

        plot = plt.semilogy if logscale else plt.plot
        rho = self.rho_val if valence else self.rho_core
        plot(self.rgrid, rho)

        plt.xlim([0., self.get_cutoff_radius()])
        plt.ylim([1e-5, None])
        plt.grid()
        plt.xlabel(r'r [a$_0$]')
        label = r'\rho_{val}' if valence else r'\rho_{core}'
        plt.ylabel(r'$%s$ [a.u.]' % label)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.clf()

    def plot_valence_density(self, logscale=True, filename=None):
        """ Plots the valence electron density. """
        self.plot_density(valence=True, logscale=logscale, filename=filename)

    def plot_core_density(self, logscale=True, filename=None):
        """ Plots the core electron density. """
        self.plot_density(valence=False, logscale=logscale, filename=filename)

    def build_projectors(self, e3, force=False):
        """ Build the pseudopotential projectors and the
        associated properties (e.g. KB energies).

        Note: currently only one projector per angular momentum
              is considered.

        Parameters
        ----------
        e3: AtomicBase-like object
            Object with atomic properties (esp. basis functions).
        force : bool, optional
            Whether to rebuild even if the projectors and associated
            properties are already available (for example when they
            have been extracted from the pseudopotential file).
            Default: False.
        """
        assert e3.get_symbol() == self.symbol

        if not force:
            if len(self.energies) > 0 and len(self.projectors) > 0 and \
               len(self.cosines) > 0 and len(self.overlap_onsite) > 0:
                # Quick return
                return

        self.energies = {}
        self.projectors = {}
        rc = self.get_cutoff_radius()

        for l in self.angular_momenta:
            nl = self.subshells[l]
            Rnl_free = e3.Rnl_free(e3.rgrid, nl)
            dVl = self.semilocal_potential(e3.rgrid, l)

            projector = Rnl_free * dVl
            integrand = projector**2 * e3.rgrid**2
            projector /= np.sqrt(e3.grid.integrate(integrand, use_dV=False))
            self.projectors[nl] = build_interpolator(e3.rgrid, projector, rc)
            integrand = Rnl_free**2 * dVl**2 * e3.rgrid**2
            self.energies[nl] = e3.grid.integrate(integrand, use_dV=False)
            integrand = Rnl_free**2 * dVl * e3.rgrid**2
            self.energies[nl] /= e3.grid.integrate(integrand, use_dV=False)

            integrand = Rnl_free * projector * e3.rgrid**2
            self.cosines[nl] = e3.grid.integrate(integrand, use_dV=False)

            energy = self.energies[nl]
            print('PP: E_KB for %s_%s: %.6f' % (self.symbol, nl, energy),
                  file=self.txt)
            cosine = self.cosines[nl]
            print('PP: KB_cos for %s_%s: %.6f' % (self.symbol, nl, cosine),
                  file=self.txt)

            # Calculate onsite overlaps
            for valence in e3.basis_sets:
                for nl2 in valence:
                    if ANGULAR_MOMENTUM[nl2[1]] != l:
                        continue

                    integrand = projector * e3.rgrid**2 * e3.Rnl(e3.rgrid, nl2)
                    self.overlap_onsite[(nl, nl2)] = \
                        e3.grid.integrate(integrand, use_dV=False)


if __name__ == '__main__':
    import argparse
    import os

    description = 'Tool for plotting (semi)local potentials and densities.'
    usage = """
    python kleinman_bylander.py --help

    python kleinman_bylander.py C.psf --valence=2s,2p --local-component=siesta

    python kleinman_bylander.py Au.psf --valence=5d,6s --with-polarization
                                       --local-component=f
    """
    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument('filename', type=str)
    parser.add_argument('--lmax', type=int, default=None)
    parser.add_argument('--local-component', type=str, default=None)
    parser.add_argument('--rcore', type=float, default=None)
    parser.add_argument('--valence', type=str, default=None)
    parser.add_argument('--with-polarization', action='store_true')
    args = parser.parse_args()

    valence = None if args.valence is None else args.valence.split(',')
    pp = KleinmanBylanderPP(args.filename, valence=valence,
                            with_polarization=args.with_polarization,
                            local_component=args.local_component,
                            lmax=args.lmax, rcore=args.rcore, txt='-')

    prefix, _ = os.path.splitext(os.path.basename(args.filename))
    pp.plot_potentials(filename='{0}_potentials.png'.format(prefix))
    pp.plot_valence_density(filename='{0}_rhovalence.png'.format(prefix))
    if pp.has_nonzero_rho_core:
        pp.plot_core_density(filename='{0}_rhocore.png'.format(prefix))
