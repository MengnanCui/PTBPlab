#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2026 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.interpolate import CubicSpline
try:
    from pylibxc import LibXCFunctional
    from pylibxc.version import __version__ as pylibxc_version
    has_pylibxc = True
    assert int(pylibxc_version[0]) >= 5, \
           'pylibxc >= v5 is needed (found {0})'.format(pylibxc_version)
except ImportError:
    print('Warning -- could not load pylibxc')
    has_pylibxc = False


class LibXC:
    def __init__(self, xcname, spin_polarized=False):
        """
        Interface to pylibxc.

        Parameters
        ----------
        xcname : str
            Combination of Libxc functional names,
            e.g. 'GGA_X_PBE+GGA_C_PBE'.
        spin_polarized : bool, optional
            Whether to select the spin-polarized version
            of the functionals (default: False).
        """
        assert has_pylibxc, 'Using XC other than LDA requires pylibxc!'

        self.xcname = xcname
        self.names = self.xcname.split('+')
        self.functionals = []
        self.types = []
        self.add_gradient_corrections = False

        for name in self.names:
            try:
                spin = 'polarized' if spin_polarized else 'unpolarized'
                func = LibXCFunctional(name, spin)
                self.functionals.append(func)
            except KeyError as err:
                print('KeyError:', err)
                print('>>> Bad XC name. For valid Libxc functional names, see')
                print('>>> https://www.tddft.org/programs/libxc/functionals/')
                raise

            if 'mgga' in name.lower():
                raise ValueError('Meta-GGA functionals not allowed:', name)
            if 'lda' in name.lower():
                self.types.append('LDA')
            elif 'gga' in name.lower():
                self.types.append('GGA')
                self.add_gradient_corrections = True
            else:
                raise ValueError('XC func %s is not LDA or GGA' % name)

    def compute_exc(self, rho, sigma=None):
        """
        Returns the exchange-correlation energy density.

        Parameters
        ----------
        rho : np.ndarray
            Electron density.
        sigma : np.ndarray, optional
            Norm of the electron density gradient
            (only needed if GGA functionals are involved).

        Returns
        -------
        exc: np.ndarray
            Exchange-correlation energy density.
        """
        inp = {'rho': rho, 'sigma': sigma}
        exc = np.zeros_like(rho)

        for i, func in enumerate(self.functionals):
            out = func.compute(inp, do_exc=True, do_vxc=False)
            exc += out['zk'][:, 0]

        return exc

    def compute_vxc(self, rho, sigma=None, fxc=False):
        """
        Returns a dictionary with the arrays needed to compute
        the XC potential.

        Parameters
        ----------
        rho : np.ndarray
            Electron density.
        sigma : np.ndarray, optional
            Norm of the electron density gradient
            (only needed if GGA functionals are involved).
        fxc : bool, optional
            Whether to also compute selected second derivatives
            with respect to the electron density and its gradient.

        Returns
        -------
        results : dict of np.ndarray
            Dictionary with 'vrho' and (if fxc=True) 'v2rho2'.
            If GGA functionals are involved, also 'vsigma' and
            (if fxc=True) 'v2rhosigma' and 'v2sigma2' are included.
        """
        inp = {'rho': rho, 'sigma': sigma}

        includes = ['vrho']
        if fxc:
            includes += ['v2rho2']

        if self.add_gradient_corrections:
            includes += ['vsigma']
            if fxc:
                includes += ['v2rhosigma', 'v2sigma2']

        N = len(rho)
        results = {}
        for key in includes:
            results[key] = np.zeros(N)

        for i, func in enumerate(self.functionals):
            is_gga = self.types[i] == 'GGA'
            out = func.compute(inp, do_exc=False, do_vxc=True, do_fxc=fxc)

            for key in includes:
                if 'sigma' in key and not is_gga:
                    continue

                results[key] += out[key][:, 0]

        return results

    def compute_vxc_polarized(self, rho_up, rho_down, sigma_up=None,
                              sigma_updown=None, sigma_down=None, fxc=False):
        """
        Returns a dictionary with the arrays needed to compute
        the spin-resolved XC potentials.

        Note: so far this function has only been applied with equal
        up and down densities.

        Parameters
        ----------
        rho_up : np.ndarray
            Electron density for the 'up' spin channel.
        rho_down : np.ndarray
            Electron density for the 'down' spin channel.
        sigma_up : np.ndarray, optional
            Norm of the electron density gradient for the 'up' spin
            channel (only needed if GGA functionals are involved).
        sigma_updown : np.ndarray, optional
            Dot product of the density gradients for the 'up'
            and 'down' spin channels (only needed if GGA functionals
            are involved).
        sigma_down : np.ndarray, optional
            Norm of the electron density gradient for the 'down' spin
            channel (only needed if GGA functionals are involved).
        fxc : bool, optional
            Whether to also compute selected second derivatives
            with respect to the electron density and its gradient.

        Returns
        -------
        results : dict of np.ndarray
            Dictionary with 'vrho' and (if fxc=True) 'v2rho2'.
            If GGA functionals are involved, also 'vsigma' and
            (if fxc=True) 'v2rhosigma' and 'v2sigma2' are included.
        """
        rho = np.array([rho_up, rho_down])
        sigma = np.array([sigma_up, sigma_updown, sigma_down])
        inp = {
            'rho': np.ascontiguousarray(rho.T),
            'sigma': np.ascontiguousarray(sigma.T),
        }

        suffices = {
            'vrho': ['_up', '_down'],
            'v2rho2': ['_up', '_updown', '_down'],
            'vsigma': ['_up', '_updown', '_down'],
            'v2rhosigma': ['_up_up', '_up_updown', '_up_down',
                           '_down_up', '_down_updown', '_down_down'],
            'v2sigma2': ['_up_up', '_up_updown', '_up_down', '_updown_updown',
                         '_updown_down', '_down_down'],
        }
        includes = ['vrho']
        if fxc:
            includes += ['v2rho2']

        if self.add_gradient_corrections:
            includes += ['vsigma']
            if fxc:
                includes += ['v2rhosigma', 'v2sigma2']

        N = len(rho_up)
        results = {}
        for key in includes:
            for suffix in suffices[key]:
                results[key + suffix] = np.zeros(N)

        for i, func in enumerate(self.functionals):
            is_gga = self.types[i] == 'GGA'
            out = func.compute(inp, do_exc=False, do_vxc=True, do_fxc=fxc)

            for key in includes:
                if 'sigma' in key and not is_gga:
                    continue

                for j, suffix in enumerate(suffices[key]):
                    results[key + suffix] += out[key][:, j]

        return results

    def compute_all(self, rho, sigma=None):
        """
        Returns the results of self.compute_vxc() together with
        a 'zk' entry with the result from self.compute_exc().
        """
        results = self.compute_vxc(rho, sigma=sigma)
        results['zk'] = self.compute_exc(rho, sigma=sigma)
        return results

    def evaluate(self, rho, gd):
        """
        Returns the XC energy density and the XC potential.

        Parameters
        ----------
        rho : np.ndarray
            Electron density.
        gd : Grid
            An object that can carry out gradient and divergence
            operations on a grid-based array.

        Returns
        -------
        exc : np.ndarray
            XC energy density.
        vxc : np.ndarray
            XC potential.
        """
        grad = gd.gradient(rho)
        sigma = grad ** 2
        out = self.compute_all(rho, sigma)
        exc = out['zk']
        vxc = out['vrho']
        if self.add_gradient_corrections:
            assert out['vsigma'] is not None
            vxc += -2 * gd.divergence(out['vsigma'] * grad)
        return exc, vxc

    def evaluate_polarized(self, rho_up, rho_down, gd):
        """
        Returns the XC potential for the 'up' spin channel.

        Note: so far this function has only been applied with equal
        up and down densities.

        Parameters
        ----------
        rho_up : np.ndarray
            Electron density for the 'up' spin channel.
        rho_down : np.ndarray
            Electron density for the 'down' spin channel.
        gd : Grid
            An object that can carry out gradient and divergence
            operations on a grid-based array.

        Returns
        -------
        vxc_up : np.ndarray
            XC potential for spin 'up' electrons.
        """
        grad_up = gd.gradient(rho_up)
        grad_down = gd.gradient(rho_down)

        sigma_up = grad_up**2
        sigma_down = grad_down**2
        sigma_updown = grad_up * grad_down

        out = self.compute_vxc_polarized(rho_up, rho_down, sigma_up=sigma_up,
                                         sigma_updown=sigma_updown,
                                         sigma_down=sigma_down)
        vxc_up = out['vrho_up']
        if self.add_gradient_corrections:
            vxc_up -= 2. * gd.divergence(out['vsigma_up'] * grad_up)
            vxc_up -= gd.divergence(out['vsigma_updown'] * grad_down)
        return vxc_up

    def evaluate_fxc(self, rho, gd, f1, f2):
        r"""
        Returns a matrix element of the XC kernel:
        \int \int f_1(r)
                  \frac{\partial^2 E_xc}{\partial \rho(r) \partial \rho(r')}
                  f_2(r') dr dr'

        Parameters
        ----------
        rho : np.ndarray
            Electron density.
        gd : Grid
            An object that can carry out gradient and divergence
            operations on a grid-based array.
        f1 : np.ndarray
            Values of function f_1.
        f2 : np.ndarray
            Values of function f_2.

        Returns
        -------
        fxc : float
            XC kernel matrix element
        """
        grad = gd.gradient(rho)
        sigma = grad**2

        out = self.compute_vxc(rho, sigma=sigma, fxc=True)

        integrand = out['v2rho2'] * f1 * f2

        if self.add_gradient_corrections:
            integrand += 2. * out['v2rhosigma'] \
                         * (gd.gradient(f1) * gd.gradient(rho) * f2 \
                            + f1 * gd.gradient(f2) * gd.gradient(rho))
            integrand += 4. * out['v2sigma2'] \
                         * gd.gradient(f1) * gd.gradient(rho) \
                         * gd.gradient(f2) * gd.gradient(rho)
            integrand += 2. * out['vsigma'] \
                         * gd.gradient(f1) *  gd.gradient(f2)

        fxc = gd.integrate(integrand, use_dV=True)
        return fxc

    def evaluate_fxc_polarized(self, rho_up, rho_down, gd, f1, f2):
        r"""
        Returns a matrix element of the spin-polarized XC kernel:
        \int \int f_1(r)
                  \frac{\partial^2 E_xc}{\partial \mu(r) \partial \mu(r')}
                  f_2(r') dr dr'

        Note: so far this function has only been applied with equal
        up and down densities.

        Parameters
        ----------
        rho_up : np.ndarray
            Electron density for the 'up' spin channel.
        rho_down : np.ndarray
            Electron density for the 'up' spin channel.
        gd : Grid
            An object that can carry out gradient and divergence
            operations on a grid-based array.
        f1 : np.ndarray
            Values of function f_1.
        f2 : np.ndarray
            Values of function f_2.

        Returns
        -------
        fxc : float
            Spin-polarized XC kernel matrix element
        """
        grad_up = gd.gradient(rho_up)
        grad_down = gd.gradient(rho_down)

        sigma_up = grad_up**2
        sigma_down = grad_down**2
        sigma_updown = grad_up * grad_down

        out = self.compute_vxc_polarized(rho_up, rho_down, sigma_up=sigma_up,
                                         sigma_updown=sigma_updown,
                                         sigma_down=sigma_down, fxc=True)

        integrand = (out['v2rho2_up'] - out['v2rho2_updown']) * f1 * f2

        if self.add_gradient_corrections:
            grad_f1_grad_f2 = gd.gradient(f1) * gd.gradient(f2)
            grad_f1_grad_rho_up = gd.gradient(f1) * gd.gradient(rho_up)
            grad_f1_grad_rho_down = gd.gradient(f1) * gd.gradient(rho_down)
            grad_f2_grad_rho_up = gd.gradient(f2) * gd.gradient(rho_up)
            grad_f2_grad_rho_down = gd.gradient(f2) * gd.gradient(rho_down)

            # up-up contribution
            integrand += 2. * out['v2rhosigma_up_up'] \
                         * (grad_f1_grad_rho_up * f2 \
                            + f1 * grad_f2_grad_rho_up)

            integrand += 4. * out['v2sigma2_up_up'] \
                         * grad_f1_grad_rho_up * grad_f2_grad_rho_up
            integrand += 2. * out['v2sigma2_up_updown'] \
                         * grad_f1_grad_rho_down * grad_f2_grad_rho_up
            integrand += 2. * out['v2sigma2_up_updown'] \
                         * grad_f1_grad_rho_up * grad_f2_grad_rho_down
            integrand += 1. * out['v2sigma2_updown_updown'] \
                         * grad_f1_grad_rho_down * grad_f2_grad_rho_down
            integrand += 2. * out['vsigma_up'] * grad_f1_grad_f2

            # up-down contribution
            integrand -= 2. * out['v2rhosigma_up_down'] \
                         * (grad_f1_grad_rho_down * f2 \
                            + f1 * grad_f2_grad_rho_down)

            integrand -= 4. * out['v2sigma2_up_down'] \
                         * grad_f1_grad_rho_down * grad_f2_grad_rho_up
            integrand -= 2. * out['v2sigma2_up_updown'] \
                         * grad_f1_grad_rho_down * grad_f2_grad_rho_up
            integrand -= 2. * out['v2sigma2_updown_down'] \
                         * grad_f1_grad_rho_down * grad_f2_grad_rho_down
            integrand -= 1. * out['v2sigma2_updown_updown'] \
                         * grad_f1_grad_rho_up * grad_f2_grad_rho_down
            integrand -= 1. * out['vsigma_updown'] * grad_f1_grad_f2

        fxc = gd.integrate(integrand, use_dV=True) / 2.
        return fxc


class XC_PW92:
    def __init__(self):
        """ The Perdew-Wang 1992 LDA exchange-correlation functional. """
        self.small = 1e-90
        self.a1 = 0.21370
        self.c0 = 0.031091
        self.c1 = 0.046644
        self.b1 = 1.0 / 2.0 / self.c0 * np.exp(-self.c1 / 2.0 / self.c0)
        self.b2 = 2 * self.c0 * self.b1 ** 2
        self.b3 = 1.6382
        self.b4 = 0.49294
        self.add_gradient_corrections = False

    def exc(self, n, der=0):
        """ Exchange-correlation with electron density n. """
        is_scalar = type(n) == np.float64
        n = np.array([n]) if is_scalar else n
        indices = n < self.small
        n[indices] = self.small
        e = self.e_x(n, der=der) + self.e_corr(n, der=der)
        e[indices] = 0.
        if is_scalar:
            e = e[0]
        return e

    def e_x(self, n, der=0):
        """ Exchange. """
        if der == 0:
            return -3. / 4 * (3 * n / np.pi) ** (1. / 3)
        elif der == 1:
            return -3. / (4 * np.pi) * (3 * n / np.pi) ** (-2. / 3)

    def e_corr(self, n, der=0):
        """ Correlation energy. """
        rs = (3. / (4 * np.pi * n)) ** (1. / 3)
        aux = 2 * self.c0
        aux *= self.b1 * np.sqrt(rs) + self.b2 * rs + self.b3 * rs ** (3. / 2) + self.b4 * rs ** 2
        if der == 0:
            return -2 * self.c0 * (1 + self.a1 * rs) * np.log(1 + aux ** -1)
        elif der == 1:
            return (-2 * self.c0 * self.a1 * np.log(1 + aux ** -1) \
                    -2 * self.c0 * (1 + self.a1 * rs) * (1 + aux ** -1) ** -1 * (-aux ** -2) \
                   * 2 * self.c0 * (self.b1 / (2 * np.sqrt(rs)) + self.b2 + 3 * self.b3 * np.sqrt(rs) / 2 \
                   + 2 * self.b4 * rs)) * (-(4 * np.pi * n ** 2 * rs ** 2) ** -1)

    def vxc(self, n):
        """ Exchange-correlation potential (functional derivative of exc). """
        indices = n < self.small
        n[indices] = self.small
        v = self.exc(n) + n * self.exc(n, der=1)
        v[indices] = 0.
        return v

    def evaluate(self, n, *args, **kwargs):
        """
        Returns the XC energy density and potential.

        Parameters
        ----------
        n : array-like,
            The electron density.
        """
        return self.exc(n), self.vxc(n)


class LDA_Spline(CubicSpline):
    """
    Class for representing LDA functionals using a (cubic) spline.

    Parameters
    ----------
    func : function
        Function for evaluating the functional when constructing
        the spline.
    rho_min, rho_max : float, optional
        Lower and upper bounds for the electron density. Zero will
        be returned for electron densities outside these bounds.
    num : int, optional
        The number of grid points to use in the spline.
    """
    def __init__(self, func, rho_min=1e-12, rho_max=1e12, num=1000):
        self.rho_min = rho_min
        self.rho_max = rho_max
        x = np.exp(np.linspace(np.log(rho_min), np.log(rho_max), num=num,
                               endpoint=True))
        y = func(x)
        CubicSpline.__init__(self, x, y, bc_type=('clamped', 'natural'),
                             extrapolate=False)
        self.add_gradient_corrections = False

    def __call__(self, rho, **kwargs):
        """
        Evaluates the exchange-correlation functional.

        Parameters
        ----------
        rho : float or np.ndarray
            The electron density.

        Returns
        -------
        y : float or np.ndarray
            The interpolated function value.
        """
        y = CubicSpline.__call__(self, rho, **kwargs)
        np.nan_to_num(y, copy=False)
        return y


class EXC_PW92_Spline(LDA_Spline):
    """
    The Perdew-Wang 1992 LDA exchange-correlation energy density
    represented by a cubic spline.
    """
    def __init__(self, **kwargs):
        LDA_Spline.__init__(self, XC_PW92().exc, **kwargs)


class VXC_PW92_Spline(LDA_Spline):
    """
    The Perdew-Wang 1992 LDA exchange-correlation potential
    represented by a cubic spline.
    """
    def __init__(self, **kwargs):
        LDA_Spline.__init__(self, XC_PW92().vxc, **kwargs)
