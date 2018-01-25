import numpy as np
import numpy.linalg as la

from dipy.core import geometry as geo
from dipy.data import default_sphere
from dipy.reconst import shm
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst import csdeconv as csd
from scipy import optimize

from ..utils.optpkg import optional_package

cvxpy, have_cvxpy, _ = optional_package("cvxpy")

sh_const = .5 / np.sqrt(np.pi)

def multi_tissue_basis(gtab, sh_order, iso_comp):
    """Builds a basis for multi-shell CSD model"""
    if iso_comp < 1:
        msg = ("Multi-tissue CSD requires at least 2 tissue compartments")
        raise ValueError(msg)
    r, theta, phi = geo.cart2sphere(*gtab.gradients.T)
    m, n = shm.sph_harm_ind_list(sh_order)
    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    B[np.ix_(gtab.b0s_mask, n > 0)] = 0.

    iso = np.empty([B.shape[0], iso_comp])
    iso[:] = sh_const

    B = np.concatenate([iso, B], axis=1)
    return B, m, n


class MultiShellResponse(object):

    def __init__(self, response, sh_order, shells):
        self.response = response
        self.sh_order = sh_order
        self.n = np.arange(0, sh_order + 1, 2)
        self.m = np.zeros_like(self.n)
        self.shells = shells
        if self.iso < 1:
            raise ValueError("sh_order and shape of response do not agree")

    @property
    def iso(self):
        return self.response.shape[1] - (self.sh_order // 2) - 1


def closest(haystack, needle):
    diff = abs(haystack[:, None] - needle)
    return diff.argmin(axis=0)


def _inflate_response(response, gtab, n, delta):
    if any((n % 2) != 0) or (n.max() // 2) >= response.sh_order:
        raise ValueError("Response and n do not match")

    iso = response.iso
    n_idx = np.empty(len(n) + iso, dtype=int)
    n_idx[:iso] = np.arange(0, iso)
    n_idx[iso:] = n // 2 + iso

    b_idx = closest(response.shells, gtab.bvals)
    kernal = response.response / delta

    return kernal[np.ix_(b_idx, n_idx)]


def _basic_delta(iso, m, n, theta, phi):
    """Simple delta function"""
    wm_d = shm.gen_dirac(m, n, theta, phi)
    iso_d = [sh_const] * iso
    return np.concatenate([iso_d, wm_d])


def _pos_constrained_delta(iso, m, n, theta, phi, reg_sphere=default_sphere):
    """Delta function optimized to avoid negative lobes."""

    x, y, z = geo.sphere2cart(1., theta, phi)

    # Realign reg_sphere so that the first vertex is aligned with delta
    # orientation (theta, phi).
    M = geo.vec2vec_rotmat(reg_sphere.vertices[0], [x, y, z])
    new_vertices = np.dot(reg_sphere.vertices, M.T)
    _, t, p = geo.cart2sphere(*new_vertices.T)

    B = shm.real_sph_harm(m, n, t[:, None], p[:, None])
    G = B[:, n != 0]
    # c samples the delta function at the delta orientation.
    c = G[0]
    a, b = G.shape

    C = -c
    x = cvxpy.Variable(C.shape[0])
    objective = cvxpy.Minimize(C.T * x)
    constraints = [-G * x <= np.full((a, 1), sh_const**2)]
    p = cvxpy.Problem(objective, constraints)
    p.solve(solver="SCS")

    out = np.zeros(B.shape[1])
    out[n == 0] = sh_const
    out[n != 0] = np.asarray(x.value).squeeze()

    iso_d = [sh_const] * iso
    return np.concatenate([iso_d, out])

delta_functions = {"basic":_basic_delta,
                   "positivity_constrained":_pos_constrained_delta}


class MultiShellDeconvModel(shm.SphHarmModel):

    def __init__(self, gtab, response, reg_sphere=default_sphere, iso=2,
                 delta_form='basic'):
        """
        """
        sh_order = response.sh_order
        super(MultiShellDeconvModel, self).__init__(gtab)
        B, m, n = multi_tissue_basis(gtab, sh_order, iso)

        delta_f = delta_functions[delta_form]
        delta = delta_f(response.iso, response.m, response.n, 0., 0.)
        self.delta = delta
        multiplier_matrix = _inflate_response(response, gtab, n, delta)

        r, theta, phi = geo.cart2sphere(*reg_sphere.vertices.T)
        odf_reg, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)
        reg = np.zeros([i + iso for i in odf_reg.shape])
        reg[:iso, :iso] = np.eye(iso)
        reg[iso:, iso:] = odf_reg

        X = B * multiplier_matrix

        self.fitter = QpFitter(X, reg)
        self.sh_order = sh_order
        self._X = X
        self.sphere = reg_sphere
        self.response = response
        self.B_dwi = B
        self.m = m
        self.n = n

    def predict(self, params, gtab=None, S0=None):
        if gtab is None:
            X = self._X
        else:
            iso = self.response.iso
            B, m, n = multi_tissue_basis(gtab, self.sh_order, iso)
            multiplier_matrix = _inflate_response(self.response, gtab, n,
                                                  self.delta)
            X = B * multiplier_matrix
        return np.dot(params, X.T)

    @multi_voxel_fit
    def fit(self, data):
        coeff = self.fitter(data)
        return MSDeconvFit(self, coeff, None)


class MSDeconvFit(shm.SphHarmFit):

    def __init__(self, model, coeff, mask):
        self._shm_coef = coeff
        self.mask = mask
        self.model = model

    @property
    def shm_coeff(self):
        return self._shm_coef[..., self.model.response.iso:]

    @property
    def volume_fractions(self):
        tissue_classes = self.model.response.iso + 1
        return self._shm_coef[..., :tissue_classes] / sh_const


def _rank(A, tol=1e-8):
    s = la.svd(A, False, False)
    threshold = (s[0] * tol)
    rnk = (s > threshold).sum()
    return rnk


class QpFitter(object):
    def _lstsq_initial(self, z):
        fodf_sh = csd._solve_cholesky(self._P, z)
        s = np.dot(self._reg, fodf_sh)
        init = {'x': fodf_sh,
                's': s.clip(1e-10)}
        return init

    def __init__(self, X, reg):
        self._P = P = np.dot(X.T, X)
        self._X = X

        # No super res for now.
        assert _rank(P) == P.shape[0]

        self._reg = reg
        # self._P_init = np.dot(X[:, :N].T, X[:, :N])

        # Make cvxopt matrix types for later re-use.
        self._P_py = P
        self._reg_py = -reg
        self._h_py = np.full((reg.shape[0], 1), 0.)

    def __call__(self, signal):
        z = np.dot(self._X.T, signal)
        init = self._lstsq_initial(z)

        x0 = init['x']

        def loss(x, sign=1.):
            return sign * (0.5 * np.dot(x.T, np.dot(self._P_py, x)) + np.dot(-z.T, x))

        def jac(x, sign=1.):
            return sign * (np.dot(x.T, self._P_py) + -z.T)

        cons = {'type': 'ineq',
                'fun': lambda x: 0.1 - np.dot(self._reg_py, x),
                'jac': lambda x: -self._reg_py}

        opt = {'disp': False}
        import time
        i = time.time()
        res_cons = optimize.minimize(loss, x0, jac=jac, constraints=cons,
                                     method='SLSQP', options=opt)
        o = time.time()
        print(o-i)

        # x = cvxpy.Variable(self._P_py.shape[0])
        # x.value = init['x']
        # objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, self._P_py) - z.T * x)
        # constraints = [self._reg_py * x <= 0.1]
        # p = cvxpy.Problem(objective, constraints)
        # import dccp
        # p.solve(method='dccp', verbose=True)
        return np.asarray(res_cons['x']).squeeze()