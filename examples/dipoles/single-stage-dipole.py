# Taking Rithik's banana coils code and adapting it into dipole coils
# Important note: banana coils are still the same variable name, I will rename later
# Main differences: fixed coil geometry, only optimize currents; added a current penalty term

import os
import io
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize

# SIMSOPT imports
from simsopt._core.optimizable import Optimizable
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, CurveLength
from simsopt.geo.surfaceobjectives import Volume, BoozerResidual, Iotas, NonQuasiSymmetricRatio, SurfaceSurfaceDistance
#from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.field import BiotSavart, Coil, Current
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt._core.optimizable import load, save
from simsopt.field.coil import ScaledCurrent
import matplotlib.pyplot as plt
from simsopt._core.derivative import derivative_dec
from simsopt.mhd.vmec import Vmec
from simsopt._core.derivative import Derivative
from helper_functions import coil_currents_on_theta_phi_grid, plot_coil_currents_on_theta_phi_grid
"""
class CurrentCap(Optimizable):
    #Soft cap on |I| for a set of current DOFs:
      #Jcp = sum_i max(|I_i| - threshold, 0)^2

    def __init__(self, current_dofs, threshold):
        # 'current_dofs' must be a list of Current objects whose DOFs are unfixed
        super().__init__(depends_on=list(current_dofs))
        self.threshold = float(threshold)
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            I = self.x  # vector of currents (same order as children)
            excess = np.maximum(np.abs(I) - self.threshold, 0.0)
            self._J = float(np.dot(excess, excess))
        return self._J

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            I = self.x
            diff = np.abs(I) - self.threshold
            mask = diff > 0
            # d/dI of (max(|I|-T,0))^2 = 2*max(|I|-T,0) * sign(I)
            grad = np.where(I > 0.0, 2.0 * (I - self.threshold),
                            2.0 * (I + self.threshold))
            full_grad = (grad * mask).astype(float)

            partials = {}
            idx = 0
            for dep in self.parents:
                # Get the number of DOFs for the current dependency
                num_dofs = dep.dof_size
                # Assign the corresponding slice of the full gradient
                partials[dep] = full_grad[idx:idx + num_dofs]
                idx += num_dofs

            self._dJ = Derivative(partials)
        return self._dJ
"""


class CurrentCap(Optimizable):
    """
    Soft cap on |I_physical| for a set of current DOFs:
    Jcp = sum_i max(|I_i| - threshold, 0)^2
    """

    def __init__(self, current_dofs, threshold):
        # 'current_dofs' should be the list of Current or ScaledCurrent objects
        super().__init__(depends_on=list(current_dofs))
        self.threshold = float(threshold)
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            # key fix: get physical currents (Amperes) using get_value()
            # This handles both Current (returns value) and ScaledCurrent (returns scale*base)
            I_vals = np.array([c.get_value() for c in self.parents])
            excess = np.maximum(np.abs(I_vals) - self.threshold, 0.0)
            self._J = float(np.dot(excess, excess))
        return self._J

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            partials = {}

            for c in self.parents:
                val = c.get_value()  # Physical current

                # 1. Calculate derivative of Penalty w.r.t Physical Current
                # d(Penalty)/d(I_physical)
                if abs(val) > self.threshold:
                    dJ_dval = 2.0 * (abs(val) - self.threshold) * np.sign(val)
                else:
                    dJ_dval = 0.0

                # 2. Calculate derivative of Physical Current w.r.t the Optimization Variable (DOF)
                # Chain rule: d(Penalty)/d(DOF) = d(Penalty)/d(I_phys) * d(I_phys)/d(DOF)

                # For ScaledCurrent: I_phys = scale * base. d(I_phys)/d(scale) = base.
                if hasattr(c, 'current_to_scale'):
                    grad_factor = c.current_to_scale.get_value()
                # For regular Current: I_phys = x. d(I_phys)/d(x) = 1.
                else:
                    grad_factor = 1.0

                # Store the derivative (Simsopt expects an array)
                partials[c] = np.array([dJ_dval * grad_factor])

            self._dJ = Derivative(partials)
        return self._dJ

class BoozerResidualExact(Optimizable):
    r"""
    This term returns the Boozer residual penalty term

    .. math::
       J = \int_0^{1/n_{\text{fp}}} \int_0^1 \| \mathbf r \|^2 ~d\theta ~d\varphi + w (\text{label.J()-boozer_surface.constraint_weight})^2.

    where

    .. math::
        \mathbf r = \frac{1}{\|\mathbf B\|}[G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)]

    """

    def __init__(self, boozer_surface, bs):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface

        # same number of points as on the solved surface
        nphis = in_surface.quadpoints_phi.size
        phis = np.linspace(0, 1. / in_surface.nfp, nphis * 4, endpoint=False)
        nthetas = in_surface.quadpoints_theta.size
        thetas = np.linspace(0, 1, nthetas * 4, endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym,
                                    nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        # self.constraint_weight = boozer_surface.constraint_weight
        print("warning: constraint weight set to 0")
        self.constraint_weight = 0.0
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """

        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to the coil degrees of freedom.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))

        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=True)
        rtil = np.concatenate((r / np.sqrt(num_points), [np.sqrt(self.constraint_weight) * (
                    self.boozer_surface.label.J() - self.boozer_surface.targetlabel)]))
        self._J = 0.5 * np.sum(rtil ** 2)

        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = booz_surf.res['vjp']

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[:dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate((J / np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T @ rtil

        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """

        surface = self.surface
        res = self.boozer_surface.res
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'],
                                             self.biotsavart, derivatives=0, weight_inv_modB=True)

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)

        dJ_by_dB = r[:, None] * r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB

def initialize_boozer_surface(surf_prev, mpol, bs, vol_target, constraint_weight, iota, G0):
    # This initializes the boozer surface, using either the boozer "exact" algorithm, or the boozer "least squares" algorithm
    # surf_prev: Any instance of simsopt.geo.Surface. This is the initial guess for the boozer surface solver
    # mpol: SurfaceXYZTensorFourier resolution (both toroidal and poloidal)
    # bs: simsopt.field.BiotSavart instance
    # vol_target: target volume to be enclosed by the boozer surface
    # constraint_weight: Set to 1.0 to use Boozer least square, None to use Boozer exact
    # iota: initial guess for iota value on the surface
    # G0: Value of net current going through the torus hole
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=mpol, nfp=surf_prev.nfp, stellsym=True,
        quadpoints_theta=surf_prev.quadpoints_theta,
        quadpoints_phi=surf_prev.quadpoints_phi
    )
    surf.least_squares_fit(surf_prev.gamma())
    # surf.plot()
    # plt.show()

    if constraint_weight:
        # Boozer least square approach
        print("Generating Boozer least squares surface...")
        vol = Volume(surf)
        boozer_surface = BoozerSurface(bs, surf, vol, vol_target, constraint_weight, options={'verbose': True})
    else:
        # Boozer exact approach
        print("Generating Boozer exact surface...")
        surf_exact = SurfaceXYZTensorFourier(
            mpol=mpol, ntor=mpol, nfp=surf_prev.nfp, stellsym=True,
            quadpoints_theta=np.linspace(0, 1, 2 * mpol + 1, endpoint=False),
            quadpoints_phi=np.linspace(0, 1. / surf.nfp, 2 * mpol + 1, endpoint=False),
            dofs=surf.dofs
        )

        vol = Volume(surf_exact)
        boozer_surface = BoozerSurface(bs, surf_exact, vol, vol_target, None, options={'verbose': True})

    # Run boozer surface algorithm
    res = boozer_surface.run_code(iota, G0)
    print(f"G0 from solve: {res['G']}")
    print(f"iota from solve: {res['iota']}")

    # Check if boozer algo is successful
    success1 = res['success']  # True if the boozer surface algo converged
    success2 = not boozer_surface.surface.is_self_intersecting()  # True if surface is not self intersecting
    # print(success1, success2)
    if not (success1 and success2):
        raise RuntimeError("Something went wrong with the Boozer solve...")

    return boozer_surface


def normPlot(surf, bs, filename):
    """Plot |B·n|/|B| on the plasma surface."""
    theta = surf.quadpoints_theta
    phi = surf.quadpoints_phi
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n * (1. / absn)[:, :, None]
    sqrt_area = np.sqrt(absn.reshape((-1, 1)) / float(absn.size))
    surf_area = sqrt_area ** 2
    bs.set_points(surf.gamma().reshape((-1, 3)))
    B = bs.B().reshape(n.shape)
    Bnorm = np.sum(B * unitn, axis=2)[:, :, None]
    modB = np.sqrt(np.sum(B ** 2, axis=2))[:, :, None]
    relBnorm = Bnorm / modB
    abs_relBnorm_dA = np.abs(relBnorm.reshape((-1, 1))) * surf_area
    mean_abs_relBnorm = np.sum(abs_relBnorm_dA) / np.sum(surf_area)
    max_rnorm = np.max(np.abs(relBnorm))

    fig, ax = plt.subplots()
    contour = ax.contourf(phi, theta, np.squeeze(relBnorm).T,
                          levels=50, cmap='seismic',
                          vmin=-max_rnorm, vmax=max_rnorm)
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=18, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=18, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$\mathbf{B}\cdot\mathbf{n}/|\mathbf{B}|$',
                       fontsize=16, fontweight='bold')
    cbar.ax.tick_params(axis='y', labelsize=14)
    ax.set_title(f'Surface-avg |Bn|/|B| = {mean_abs_relBnorm:.4e}',
                 fontsize=18, fontweight='bold')
    plt.savefig(f"{filename}.png")
    plt.close()

# surf_coils is just VV in my case
# dipole_curve (like rithik's banana_curve) is an unneeded parameter, I am keeping it just to not cause an error
# also keeping surf_coils and vv both of them to check if same (should be)
def crossSectionPlot(surf_coils, surf, dipole_curve, filename):
    # plots cross section of plasma at a few toroidal locations and relevant HBT cross sections
    plt.figure(figsize=(7, 6))
    cs2 = surf_coils.cross_section(0)
    rs2 = np.sqrt(cs2[:, 0] ** 2 + cs2[:, 1] ** 2);
    rs2 = np.append(rs2, rs2[0])
    zs2 = cs2[:, 2];
    zs2 = np.append(zs2, zs2[0])
    #plt.plot(rs2, zs2, label='Dipole Surface')
    cs3 = hbt.cross_section(0)
    rs3 = np.sqrt(cs3[:, 0] ** 2 + cs3[:, 1] ** 2);
    rs3 = np.append(rs3, rs3[0])
    zs3 = cs3[:, 2];
    zs3 = np.append(zs3, zs3[0])
    hbt_poly = Polygon(zip(rs3, zs3))
    #plt.plot(rs3, zs3, label='HBT LCFS')
    cs_vv = VV.cross_section(0)
    rs_vv = np.sqrt(cs_vv[:, 0] ** 2 + cs_vv[:, 1] ** 2);
    zs_vv = cs_vv[:, 2]
    rs_vv = np.append(rs_vv, rs_vv[0]);
    zs_vv = np.append(zs_vv, zs_vv[0])
    plt.plot(rs_vv, zs_vv, label='Vacuum Vessel')
    phi_array = np.linspace(0, 1 / surf_coils.nfp, 5)  # scaled from 0 to 1
    for phi_slice in phi_array:
        cs = surf.cross_section(phi_slice * 2 * np.pi)
        rs = np.sqrt(cs[:, 0] ** 2 + cs[:, 1] ** 2)
        rs = np.append(rs, rs[0])
        zs = cs[:, 2];
        zs = np.append(zs, zs[0])
        '''
        plasma_poly = Polygon(zip(rs, zs))
        if not plasma_poly.within(hbt_poly):
            plt.close()
            print("Plasma surface not within HBT boundary — skipping plot.")
            return False
        '''
        plt.plot(rs, zs, label=f'Φ={phi_slice * 2:0.2f}π')
    plt.xlabel('R [m]', fontsize=18, fontweight='bold')
    plt.ylabel('Z [m]', fontsize=18, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.minorticks_on()
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.close()
    return True

def plotHistory(array, y_label, filename):
    OUT_DIR_ITER_PLOTS = OUT_DIR_ITER + "/iteration-histories"
    os.makedirs(OUT_DIR_ITER_PLOTS, exist_ok=True)
    plt.figure()

    # x-axis as integer iteration numbers
    x_vals = np.arange(len(array))

    # replace zeros with small positive value to avoid log(0) error
    y_vals = np.array(array)
    y_vals = np.where(y_vals == 0, 1e-12, y_vals)  # adjust small value if needed

    plt.plot(x_vals, y_vals, marker='o')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel(y_label.replace('_', ' '), fontsize=14)
    plt.title(f"{y_label.replace('_', ' ')} vs Iteration", fontsize=14)
    plt.grid(True, which='both')

    # Set log scale for y-axis
    plt.yscale('log')

    # Set x-axis ticks to integers only
    plt.xticks(x_vals)

    plt.tight_layout()
    plt.savefig(OUT_DIR_ITER_PLOTS + f"/{filename}.png")
    plt.close()

def fun(x):
    dx = np.linalg.norm(x - run_dict['x_prev'])
    run_dict['x_prev'] = x.copy()
    print(f"Step size: {dx:.2e}")

    run_dict['lscount'] += 1

    # initialize to last accepted surface values
    boozer_surface.surface.x = run_dict['sdofs']
    boozer_surface.res['iota'] = run_dict['iota']
    boozer_surface.res['G'] = run_dict['G']

    # Set new coil dofs
    JF.x = x

    # Run boozer surface
    res = boozer_surface.run_code(run_dict['iota'], run_dict['G'])

    # Check success
    try:
        success1 = boozer_surface.res['success']
        success2 = not boozer_surface.surface.is_self_intersecting()
    except Exception as e:
        print("Surface check failed:", e)
        success2 = False
    success = success1 and success2

    if success:
        J = JF.J()
        dJ = JF.dJ()

        print(f"Volume: {boozer_surface.surface.volume()}")
        print(f"Iota: {Iotas(boozer_surface).J()}")

    else:
        print("/!\\ /!\\ Boozer surface rejected /!\\ /!\\")
        if not success1:
            print("Boozer solver failed")
        if not success2:
            print("Surface is self-intersecting")

        J = run_dict['J']
        dJ = -run_dict['dJ']
        boozer_surface.surface.x = run_dict['sdofs']
        boozer_surface.res['iota'] = run_dict['iota']
        boozer_surface.res['G'] = run_dict['G']

    return J, dJ


def callback(x):
    # Update count for tracking
    run_dict['lscount'] = 0

    # Store last accepted state
    run_dict['sdofs'] = boozer_surface.surface.x.copy()
    run_dict['iota'] = boozer_surface.res['iota']
    run_dict['dipole_current'] = sum(c.current.get_value() for c in dipole_coils)
    run_dict['G'] = boozer_surface.res['G']
    run_dict['J'] = JF.J()
    run_dict['dJ'] = JF.dJ().copy()

    # Evaluate diagnostics
    J = run_dict['J']
    grad = run_dict['dJ']

    J_QS = JnonQSRatio.J()
    dJ_QS = np.linalg.norm(JnonQSRatio.dJ())
    J_Boozer = JBoozerResidual.J()
    dJ_Boozer = np.linalg.norm(JBoozerResidual.dJ())
    J_iota = Jiotamax.J()
    dJ_iota = np.linalg.norm(Jiotamax.dJ())
    """ also removed for dipoles
    J_len = JCurveLength.J()
    dJ_len = np.linalg.norm(JCurveLength.dJ())
    J_cc = JCurveCurve.J()
    dJ_cc = np.linalg.norm(JCurveCurve.dJ())
    J_cs = JCurveSurface.J()
    dJ_cs = np.linalg.norm(JCurveSurface.dJ())
    """
    J_surf = JSurfSurf.J()
    dJ_surf = np.linalg.norm(JSurfSurf.dJ())
    J_cap = JCurrentCap.J()
    dJ_cap = np.linalg.norm(JCurrentCap.dJ())

    iotas_list = [iota.J() for iota in iotas]
    iota_str = ", ".join([f"{val:.4f}" for val in iotas_list])

    # Curve.gamma(): returns a (n_theta, 3) array containing n quadrature points, i.e. a list of XYZ coordinates along the curve.
    max_r = np.max(np.sqrt(dipole_curve.gamma()[:, 1] ** 2 + dipole_curve.gamma()[:, 2] ** 2))
    max_z = np.max(np.abs(dipole_curve.gamma()[:, 0]))
    """ also removed for dipoles
    length = curvelength.J()
    curvecurve_min = JCurveCurve.shortest_distance()
    curvesurf_min = JCurveSurface.shortest_distance()
    """

    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * boozer_surface.surface.unitnormal(), axis=2)))
    intersecting = boozer_surface.surface.is_self_intersecting()

    # new GPT addition for dipole currents
    dipole_currents = [abs(c.current.get_value()) for c in dipole_coils]
    '''
    total_dipole = sum(dipole_currents)
    repeat_factor = int(boozer_surface.surface.nfp) * 2
    unique_len = max(1, len(dipole_currents) // max(1, repeat_factor))
    scaled_total = repeat_factor * sum(dipole_currents[:unique_len])
    '''
    max_individual = max((abs(i) for i in dipole_currents))
    run_dict['dipole_currents'] = dipole_currents
    #run_dict['dipole_current_total'] = total_dipole
    #run_dict['dipole_current_scaled'] = scaled_total
    run_dict['dipole_current_max'] = max_individual


    width = 35
    buffer = io.StringIO()
    print("=" * 70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    print(f"{'nonQS ratio':{width}} = {J_QS:.6e} (dJ = {dJ_QS:.6e})", file=buffer)
    print(f"{'Boozer Residual':{width}} = {J_Boozer:.6e} (dJ = {dJ_Boozer:.6e})", file=buffer)
    print(f"{'ι Penalty':{width}} = {J_iota:.6e} (dJ = {dJ_iota:.6e})", file=buffer)
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    """ also removed for dipoles
    print(f"{'Curve Length Penalty':{width}} = {J_len:.6e} (dJ = {dJ_len:.6e})", file=buffer)
    print(f"{'Curve-Curve Penalty':{width}} = {J_cc:.6e} (min={curvecurve_min:.3e}) (dJ = {dJ_cc:.6e})", file=buffer)
    print(f"{'Curve-Surface Penalty':{width}} = {J_cs:.6e} (min={curvesurf_min:.3e}) (dJ = {dJ_cs:.6e})", file=buffer)
    print(f"{'Curve Length':{width}} = {length:.6e}", file=buffer)
    """
    print(f"{'Surf-Vessel Penalty':{width}} = {J_surf:.6e} (dJ = {dJ_surf:.6e})", file=buffer)
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)
    print(f"{'Intersecting':{width}} = {intersecting}", file=buffer)
    print(f"{'Max Curve R':{width}} = {max_r:.6e}", file=buffer)
    print(f"{'Max Curve Z':{width}} = {max_z:.6e}", file=buffer)
    print(f"{'Current Cap':{width}} = {J_cap:.6e} (dJ = {dJ_cap:.6e})", file=buffer)
    #print(f"{'Dipole currents':{width}} = {dipole_currents}", file=buffer)
    # GPT addition for dipole currents
    '''
    label = "Total Dipole Current (all)"
    print(f"{label:{width}} = {total_dipole:.6e}", file=buffer)
    label = f"Total Dipole Current (scaled {repeat_factor}x)"
    print(f"{label:{width}} = {scaled_total:.6e}", file=buffer)
    '''

    label = "Max dipole current (abs)"
    print(f"{label:{width}} = {max_individual:.6e}", file=buffer)

    print("=" * 70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()

    print(output_str)

    filename = OUT_DIR_ITER + "/log.txt"
    with open(filename, "a") as f:
        f.write(output_str + "\n")

    # Advance iteration counter
    run_dict['it'] += 1


# ------------------------
# Constants and Parameters
# ------------------------
#dipole_surf_radius = 0.22
#dipole_surf_nfp = 2  # Field periods for coil surface - number of times the field pattern repeats toroidally
nphi = 128  # Toroidal resolution
ntheta = 64  # Poloidal resolution
mpol = 5  # Initial mpol for surface
vol_target = 0.3  # Target plasma volume - from the results.json
CONSTRAINT_WEIGHT = 1.0  # Use least squares Boozer solver
MAXITER = 300  # Max optimizer iterations
num_tf_coils = 4  # Number of TF coils in BiotSavart file
# maybe lower the tolerances
ftol_by_mpol = {5: 5e-7, 6: 1e-7, 7: 5e-8, 8: 1e-8, 9: 5e-9, 10: 1e-10}
gtol_by_mpol = {5: 5e-4, 6: 1e-4, 7: 5e-5, 8: 1e-6, 9: 1e-7, 10: 1e-8}

# Output directory setup
# yes, I accidentally put the vol 0.45 results in this folder
OUT_DIR = f"./scans/current200kA_tfunfixed_iota0.1_vol0.4"
os.makedirs(OUT_DIR, exist_ok=True)

# Determine whether using least squares or exact Boozer solver
boozer_type = {'initial': 'least_squares', 'final': 'exact'}
stage = 'initial'

# ------------------------
# Define Vacuum Vessel and HBT Boundary for Plotting/Constraints
# ------------------------
VV = SurfaceRZFourier(nfp=2, stellsym=True) #move that later
# these are the fourier coefficients for the Vacuum Vessel
# set_rc and set_zs are (m, n, value) where m is poloidal mode number, n is toroidal mode number
# keep first two parameters in each the same, get third one from results.json of run_optimize_scan
VV.set_rc(0, 0, 1.037468882271737) #VVR0 - constant term
VV.set_rc(1, 0, 0.2655263272670667) #VVA
VV.set_zs(1, 0, 0.3012169999944218) #VVB

# HBT LCFS definition with conservative plasma margin
hbt = SurfaceRZFourier(nfp=5, stellsym=True)
hbt.set_rc(0, 0, 0.9115)
hbt.set_rc(1, 0, 0.1685)
hbt.set_zs(1, 0, 0.152)

# Dipole coil winding surface - how many field periods?
# winding surface - surface on which the coils lie
# in my case, it is the same as VV
surf_coils = SurfaceRZFourier(nfp=2, stellsym=True)
surf_coils.set_rc(0, 0, 1.037468882271737)
surf_coils.set_rc(1, 0, 0.2655263272670667)
surf_coils.set_zs(1, 0, 0.3012169999944218)

# ------------------------
# Load Magnetic Field and Initial Surface
# ------------------------
## the wout file nc is the surface, the biot_savart_opt.json is in the outputs folder for that surface
# for the filename I should include the full path to the wout file from run optimize on that surface
#filename = 'wout_nfp22ginsburg_000_000281.nc'
#bs = load('..outputs/20250314_unfixed_TFs/wout_nfp22ginsburg_000_000281.nc/03_ntf4_diprad_0.045_VVa_0.2355263272670667_VV_R0_1.037468882271737_ellipticalVV/bs_opt.json')
filename = 'wout_nfp22ginsburg_000_000281.nc'
# taken from simsoppt/examples/outputs/20250303_08_iota_unfixed_TFs
bs = load('scan42_bs_opt.json')
#surf = SurfaceRZFourier.from_wout(filename, range="half period", nphi=255, ntheta=64, s=0.24)
# s is normalized toroidal flux, I can see it in Jakes' run_optimize_scan.py
surf = SurfaceRZFourier.from_wout(filename, range="half period", nphi=128, ntheta=64)
#surf.set_dofs(surf.get_dofs() / surf.major_radius())  # scale to desired major radius

coils = bs.coils
curves = [c.curve for c in coils]
tf_coils = coils[:num_tf_coils*4] # do not hardcode, surf_nfp is 2 in this specific case
tf_curves = [c.curve for c in tf_coils]
dipole_coils = coils[num_tf_coils*4:]
dipole_curves = [c.curve for c in dipole_coils]
dipole_curve = dipole_curves[0]
for c in coils:
    c.curve.fix_all()
for c in tf_coils:
    # I will try unfixing them and see what happens, it was fixed before
    c.current.unfix_all()
for c in dipole_coils:
    c.current.unfix_all()

n_tf = len(tf_coils)
n_dip = len(dipole_coils)
n_total = len(coils)

print(f"TF coils: {n_tf}")
print(f"Dipole coils: {n_dip}")
print(f"Total coils: {n_total}")

mu0 = 4.0 * np.pi * 1e-7
# I gotta check whether to use abs or not, because with abs it fails
currents = [(c.current.get_value()) for c in tf_coils]
current_sum = sum(currents)
#G0 = mu0 * current_sum  # signed-net-current guess
#print(f'G0 (net-current) = {G0:.6e}')
# just trying to see if it will work
current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
G0 = - 2 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
#print(f"coil currents:{[c.current.get_value() for c in tf_coils]}")
vmec = Vmec(filename)
#iota_target = vmec.iota_edge()
iota_target = 0.1
# iota 0.11 failed
print(f'iota target: {iota_target}')
#vol_target = vmec.volume()
vol_target = 0.45
print(f'volume initial: {vmec.volume():.12e}')
#vol_target = 0.45 # failed with fixed tf current, succeeded with unfixed tf current
print(f'volume target: {vol_target:.12e}')
surf_volume = float(surf.volume())
print(f'initial surface volume: {surf_volume:.12e}')
'''
# Export loaded dipole coils into vtk for visualization
wp_currents = [c.current.get_value() for c in dipole_coils]
curves_to_vtk(curves=[c.curve for c in coils],
              filename=os.path.join(OUT_DIR, "wp_coils_new"),
              close=True)
print("initial coilset saved to vtk")
'''
# constants for the plotting of dipole currents:
dpi = 200
axisfontsize = 14
titlefontsize = 16
cbarfontsize = 12
ticklabelfontsize = 10
# ------------------------
# Adaptive Optimization Loop Over mpol
# ------------------------

for mpol in range(5,7):
    print(f"\n===== Starting adaptive-resolution optimization for mpol = {mpol} =====")

    OUT_DIR_ITER = OUT_DIR + f"/mpol={mpol}"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    if mpol > 5:
        LOAD_DIR = OUT_DIR + f"/mpol={mpol - 1}"
        surf = load(LOAD_DIR + "/surf_opt.json")
        bs = load(LOAD_DIR + "/biot_savart_opt.json")

    # Initialize Boozer surface (least squares or exact)
    boozer_surface = initialize_boozer_surface(surf, mpol, bs, vol_target, CONSTRAINT_WEIGHT, iota_target, G0)

    # Save initial geometry
    #curves_to_vtk(curves, OUT_DIR_ITER + f"/curves_init", close=True)
    curves_to_vtk(curves, OUT_DIR_ITER + f"/curves_init", close=True, I = [c.current.get_value() for c in coils])
    bs.save(OUT_DIR_ITER + f"/biot_savart_init.json")

    pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
                                 boozer_surface.surface.unitnormal(), axis=2)[:, :, None] /
                          np.sqrt(np.sum(bs.B().reshape((nphi, ntheta, 3)) ** 2, axis=2))[:, :, None]}
    boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_init", extra_data=pointData)
    boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_init.json")

    # Print diagnostics for initial surface
    print(f"Volume: {boozer_surface.surface.volume()}")
    normPlot(boozer_surface.surface, bs, OUT_DIR_ITER + "/NormPlotInitial")
    crossSectionPlot(surf_coils, boozer_surface.surface, dipole_curve, OUT_DIR_ITER + "/CrossSectionInitial")
    # currents plot optimized
    '''
    for i, wp in enumerate(dipole_coils):
        wp.curve.unfix_all()
        print(wp.curve.dof_names[i])
    '''
    wp_currents_phis_thetas = coil_currents_on_theta_phi_grid(dipole_coils, surf_coils)

    plot_coil_currents_on_theta_phi_grid(
        wp_currents_phis_thetas,
        OUT_DIR_ITER,
        axisfontsize,
        titlefontsize,
        cbarfontsize,
        ticklabelfontsize,
        dpi
    )
    _initial_src = os.path.join(OUT_DIR_ITER, 'wp_coil_currents.png')
    _initial_dst = os.path.join(OUT_DIR_ITER, 'wp_coil_currents_initial.png')
    try:
        if os.path.exists(_initial_src):
            os.replace(_initial_src, _initial_dst)
    except Exception:
        pass
    # ----------------------------------------
    # DEFINE OBJECTIVE FUNCTION
    # ----------------------------------------
    bs_obj = BiotSavart(coils)
    nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs_obj)]
    brs = [BoozerResidual(boozer_surface, bs_obj)] if boozer_type[stage] == 'least_squares' else [
        BoozerResidualExact(boozer_surface, bs_obj)]

    # Define constraint weights
    LENGTH_WEIGHT = 10
    RES_WEIGHT = 1e5
    IOTAS_WEIGHT = 5e5
    CC_WEIGHT = 10
    CC_DIST = 0.05
    CS_WEIGHT = 10
    CS_DIST = 0.02
    SURF_DIST_WEIGHT = 1e3
    SS_DIST = 0.04
    phi_list = np.linspace(0, 1 / boozer_surface.surface.nfp, 5)
    #NEWLY ADDED - I have to ask for threshold and weight
    CURRENT_THRESHOLD = 2e5
    CURRENT_WEIGHT = 1e4

    iotas = [Iotas(boozer_surface)]
    #curvelength = CurveLength(banana_curves[0]) commented out for dipoles
    # length_target = curvelength.J() commented out for dipoles
    # Construct full objective function JF
    Jiotamax = sum([QuadraticPenalty(iota, iota_target) for iota in iotas])
    JnonQSRatio = sum(nonQSs)
    JBoozerResidual = sum(brs)
    # JCurveLength = QuadraticPenalty(curvelength, length_target, 'max')
    # JCurveCurve = CurveCurveDistance(curves, CC_DIST)
    # JCurveSurface = CurveSurfaceDistance(curves, boozer_surface.surface, CS_DIST)
    JSurfSurf = SurfaceSurfaceDistance(boozer_surface.surface, VV, SS_DIST)
    dipole_current_dofs = [c.current for c in dipole_coils]
    JCurrentCap = CurrentCap(dipole_current_dofs, threshold=CURRENT_THRESHOLD)
    # Modified for dipoles
    JF = JnonQSRatio + RES_WEIGHT * JBoozerResidual + IOTAS_WEIGHT * Jiotamax \
         + SURF_DIST_WEIGHT * JSurfSurf + CURRENT_WEIGHT * JCurrentCap
    # + LENGTH_WEIGHT * JCurveLength + CC_WEIGHT * JCurveCurve \
    # + CS_WEIGHT * JCurveSurface

    dofs = JF.x
    # Construct current penalty term

    # ----------------------
    # Set Initial Run State
    # ----------------------
    run_dict = {
        'sdofs': boozer_surface.surface.x.copy(),
        'iota': boozer_surface.res['iota'],
        'dipole_current': sum(c.get_value() for c in dipole_current_dofs),
        'G': boozer_surface.res['G'],
        'J': JF.J(),
        'dJ': JF.dJ().copy(),
        'it': 1,
        'lscount': 0,
        'x_prev': dofs.copy()
    }
    #commented out for dipoles


    # ----------------------
    # Perform Optimization
    # ----------------------
    ftol = ftol_by_mpol.get(mpol)
    gtol = gtol_by_mpol.get(mpol)
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', callback=callback,
                   options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': ftol, 'gtol': gtol})
    print(res.message)

    # ----------------------
    # Save Final Results
    # ----------------------
    #curves_to_vtk(curves, OUT_DIR_ITER + "/curves_opt", close=True)
    curves_to_vtk(curves, OUT_DIR_ITER + "/curves_opt", close=True, I=[c.current.get_value() for c in coils])
    bs.save(OUT_DIR_ITER + "/biot_savart_opt.json")
    pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
                                 boozer_surface.surface.unitnormal(), axis=2)[:, :, None] /
                          np.sqrt(np.sum(bs.B().reshape((nphi, ntheta, 3)) ** 2, axis=2))[:, :, None]}
    boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_opt", extra_data=pointData)
    boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_opt.json")
    print(f"Volume: {boozer_surface.surface.volume()}")
    print(f"Iota: {Iotas(boozer_surface).J()}")
    normPlot(boozer_surface.surface, bs, OUT_DIR_ITER + "/NormPlotOptimized")
    crossSectionPlot(surf_coils, boozer_surface.surface, dipole_curve, OUT_DIR_ITER + "/CrossSectionOptimized")
    # currents plot optimized
    wp_currents_phis_thetas = coil_currents_on_theta_phi_grid(dipole_coils, surf_coils)
    plot_coil_currents_on_theta_phi_grid(
        wp_currents_phis_thetas,
        OUT_DIR_ITER,
        axisfontsize,
        titlefontsize,
        cbarfontsize,
        ticklabelfontsize,
        dpi
    )
    _opt_src = os.path.join(OUT_DIR_ITER, 'wp_coil_currents.png')
    _opt_dst = os.path.join(OUT_DIR_ITER, 'wp_coil_currents_optimized.png')
    try:
        if os.path.exists(_opt_src):
            os.replace(_opt_src, _opt_dst)
    except Exception:
        pass
    # that is the check I used to determine G0 initial guesses, if I want it, I should insert it after unfix_all and delete the mu0.. block
    '''
    # python
    def check_and_rescale_bs(bs, surf, out_dir=".", tol=1e-3, do_rescale=False):

        # compute mean major radius of surface
        surf_pts = surf.gamma().reshape((-1, 3))
        surf_R = np.sqrt(surf_pts[:, 0] ** 2 + surf_pts[:, 1] ** 2)
        surf_R0 = float(np.mean(surf_R))

        # gather coil points for mean major radius
        coil_pts_list = []
        for c in bs.coils:
            pts = c.curve.gamma()
            coil_pts_list.append(pts.reshape((-1, 3)))
        coil_pts = np.vstack(coil_pts_list)
        coil_R = np.sqrt(coil_pts[:, 0] ** 2 + coil_pts[:, 1] ** 2)
        coil_R0 = float(np.mean(coil_R))

        if coil_R0 == 0:
            raise RuntimeError("Invalid coil geometry: zero coil radius")

        scale = surf_R0 / coil_R0
        print(f"[scale check] surf_R0 = {surf_R0:.6e}, coils_R0 = {coil_R0:.6e}, scale = {scale:.6e}")

        if abs(scale - 1.0) <= tol:
            print("[scale check] Length units match within tolerance.")
            return scale

        print("[scale check] Length-unit mismatch detected.")
        if not do_rescale:
            print("[scale check] Call with do_rescale=True or reload a matching `bs`.")
            return scale

        # Attempt to rescale each coil curve using several supported APIs.
        for idx, c in enumerate(bs.coils):
            pts = c.curve.gamma().reshape((-1, 3))
            pts_scaled = (pts * scale).reshape(pts.shape)
            success = False

            # Preferred API: set_gamma (accepts shaped points)
            if hasattr(c.curve, "set_gamma"):
                try:
                    c.curve.set_gamma(pts_scaled)
                    success = True
                except Exception:
                    success = False

            # Alternative: set_points
            if not success and hasattr(c.curve, "set_points"):
                try:
                    c.curve.set_points(pts_scaled)
                    success = True
                except Exception:
                    success = False

            # Alternative: get_dofs / set_dofs (scale dofs if they represent flattened points)
            if not success and hasattr(c.curve, "get_dofs") and hasattr(c.curve, "set_dofs"):
                try:
                    dofs = np.asarray(c.curve.get_dofs(), dtype=float)
                    if dofs.size == pts_scaled.size:
                        scaled = (dofs.reshape(pts.shape) * scale).reshape(-1)
                        c.curve.set_dofs(scaled)
                        success = True
                    else:
                        success = False
                except Exception:
                    success = False

            # Fallback: direct .dofs attribute mutation if it exists and is numeric
            if not success and hasattr(c.curve, "dofs"):
                try:
                    dofs_attr = getattr(c.curve, "dofs")
                    arr = np.asarray(dofs_attr, dtype=float)
                    if arr.size == pts_scaled.size:
                        new = (arr.reshape(pts.shape) * scale).reshape(-1)
                        try:
                            setattr(c.curve, "dofs", new)
                        except Exception:
                            # try elementwise mutation for list-like dofs_attr
                            if isinstance(dofs_attr, (list, tuple)):
                                mutable = list(dofs_attr)
                                for i in range(len(mutable)):
                                    mutable[i] = float(new[i])
                                setattr(c.curve, "dofs", mutable)
                        success = True
                    else:
                        success = False
                except Exception:
                    success = False

            if not success:
                raise RuntimeError(f"Unable to rescale coil {idx}: no supported setter found. Reload `bs` at matching units instead.")

        # save rescaled bs
        outpath = os.path.join(out_dir, "biot_savart_rescaled.json")
        try:
            if hasattr(bs, "save"):
                bs.save(outpath)
            else:
                save(bs, outpath)
            print(f"[scale check] Rescaled BiotSavart saved to {outpath}")
        except Exception as e:
            print("[scale check] Warning: failed to save rescaled BiotSavart:", e)

        return scale


    # Compute VMEC targets and robust G0 candidates, then try Boozer init with fallbacks
    mu0 = 4.0 * np.pi * 1e-7

    # load vmec and targets (once)
    vmec = Vmec(filename)
    iota_target = vmec.iota_edge()
    vol_target = float(vmec.volume())
    print(f'iota from vmec: {iota_target}')
    print(f'volume from vmec: {vol_target:.12e}')
    print(f'initial surface volume: {float(surf.volume()):.12e}')

    # Attempt to ensure bs uses same length units as surf
    try:
        check_and_rescale_bs(bs, surf, out_dir=OUT_DIR, tol=1e-3, do_rescale=True)
    except Exception as e:
        print("[warning] Coil rescale attempt failed:", e)
        print("[warning] Prefer re-generating `bs` in the same units as the wout file.")

    # Compute currents and multiple G0 candidates
    currents = [float(c.current.get_value()) for c in coils] if len(coils) > 0 else [0.0]
    current_sum_signed = float(np.sum(currents))
    current_sum_abs = float(np.sum(np.abs(currents)))
    G0_signed = mu0 * current_sum_signed
    G0_abs = mu0 * current_sum_abs
    G0_small = mu0 * max(1.0, current_sum_abs) * 1e-4
    G0_candidates = [G0_signed, G0_abs, G0_small]
    print(f"[G0 candidates] signed = {G0_signed:.6e}, abs = {G0_abs:.6e}, small = {G0_small:.6e}")

    # Build unique candidate list and sort: non-zero by descending abs, zero last
    raw_candidates = G0_candidates
    candidates = []
    tol_unique = 1e-15
    for g in raw_candidates:
        if not any(abs(g - gg) < tol_unique for gg in candidates):
            candidates.append(g)
    # Sort so that zeros are last and others by descending magnitude
    candidates = sorted(candidates, key=lambda g: (abs(g) == 0.0, -abs(g)))
    print(f"[G0 candidates ordered] {['{:.6e}'.format(g) for g in candidates]}")

    # Try initialize with each candidate G0 until it succeeds
    boozer_surface = None
    last_exception = None
    G0 = None
    for g_try in candidates:
        try:
            print(f"[boozer init] attempting initialize_boozer_surface with initial G0 = {g_try:.6e} and iota = {iota_target:.6e}")
            boozer_surface = initialize_boozer_surface(surf, mpol, bs, vol_target, CONSTRAINT_WEIGHT, iota_target, g_try)
            # prefer the solver-returned G value for consistency
            try:
                G0 = float(boozer_surface.res.get('G', g_try))
            except Exception:
                # fallback if res is not a dict-like
                G0 = float(getattr(boozer_surface, "res", {}).get('G', g_try) if hasattr(boozer_surface, "res") else g_try)
            print(f"[boozer init] succeeded; solver returned G0 = {G0:.6e}")
            break
        except Exception as e:
            print(f"[boozer init] attempt with initial G0 = {g_try:.6e} failed: {e}")
            last_exception = e

    if boozer_surface is None:
        # final fallback: try exact solver with a small nonzero G0
        try:
            print("[boozer init] final fallback: trying small initial G0 and CONSTRAINT_WEIGHT=None (exact solver)")
            boozer_surface = initialize_boozer_surface(surf, mpol, bs, vol_target, None, iota_target, G0_small)
            try:
                G0 = float(boozer_surface.res.get('G', G0_small))
            except Exception:
                G0 = float(getattr(boozer_surface, "res", {}).get('G', G0_small) if hasattr(boozer_surface, "res") else G0_small)
            print(f"[boozer init] fallback succeeded; solver returned G0 = {G0:.6e}")
        except Exception as e:
            print("[boozer init] fallback also failed:", e)
            raise RuntimeError("Boozer initialization failed with all candidate G0 values.") from (last_exception or e)

    # boozer_surface and G0 are now set and can be used in the rest of the script
    '''

    ''' Replaced with section above by chatgpt, to remove, delete the things between unfix and that line
    # Compute G0 from TF coil currents (ampere-turns)
    # or maybe try with all currents
    current_sum = sum(abs(c.current.get_value()) for c in coils)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))  # µ_0 * total current
    print(f'G0 from all coils: {G0}')
    vmec = Vmec(filename)
    iota_target = vmec.iota_edge()
    print(f'iota from vmec: {iota_target}')
    vol_target = vmec.volume()
    print(f'volume from vmec: {vol_target}')
    surf_volume = surf.volume()
    print(f'initial surface volume: {surf_volume}')
    # Export loaded dipole coils into vtk for visualization
    wp_currents = [c.current.get_value() for c in dipole_coils]
    curves_to_vtk(
            curves = [c.curve for c in coils], filename=os.path.join(OUT_DIR, "wp_coils_new "), close=True
            #I = wp_currents
        )
    print("initial coilset saved to vtk")
    '''

    ''' this one messed up the code and gave errors, I replaced it
    # Create current cap class
    class CurrentCap(Optimizable):
        """
        Soft cap on |I| for a set of current DOFs:
          Jcp = sum_i max(|I_i| - threshold, 0)^2
        """
        def __init__(self, current_dofs, threshold):
            # 'current_dofs' must be a list of Current objects whose DOFs are unfixed
            super().__init__(depends_on=list(current_dofs))
            self.threshold = float(threshold)
            self._J = None
            self._dJ = None

        def J(self):
            if self._J is None:
                I = self.x  # vector of currents (same order as children)
                excess = np.maximum(np.abs(I) - self.threshold, 0.0)
                self._J = float(np.dot(excess, excess))
            return self._J

        def recompute_bell(self, parent=None):
            self._J = None
            self._dJ = None

        @derivative_dec
        def dJ(self):
            if self._dJ is None:
                I = self.x
                diff = np.abs(I) - self.threshold
                mask = diff > 0
                # d/dI of (max(|I|-T,0))^2 = 2*max(|I|-T,0) * sign(I)
                grad = np.where(I > 0.0, 2.0 * (I - self.threshold),
                                          2.0 * (I + self.threshold))
                self._dJ = (grad * mask).astype(float)
            return self._dJ
    '''