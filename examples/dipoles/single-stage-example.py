import os
import io
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize

# SIMSOPT imports
from simsopt._core.optimizable import Optimizable
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, CurveLength
from simsopt.geo.surfaceobjectives import Volume, BoozerResidual, Iotas, NonQuasiSymmetricRatio, SurfaceSurfaceDistance
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.field import BiotSavart, Coil, Current
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt._core.optimizable import load, save
from simsopt.field.coil import ScaledCurrent
import matplotlib.pyplot as plt
from simsopt._core.derivative import derivative_dec


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
        phis = np.linspace(0,1./in_surface.nfp,nphis*4,endpoint=False)
        nthetas = in_surface.quadpoints_theta.size
        thetas = np.linspace(0,1,nthetas*4,endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        #self.constraint_weight = boozer_surface.constraint_weight
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
        rtil = np.concatenate((r/np.sqrt(num_points), [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)]))
        self._J = 0.5*np.sum(rtil**2)
        
        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = booz_surf.res['vjp']

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[:dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate((J/np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T@rtil
        
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
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0, weight_inv_modB=True)

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)
        
        dJ_by_dB = r[:, None]*r_dB
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
          mpol=mpol,ntor=mpol,nfp=5,stellsym=True,
          quadpoints_theta=surf_prev.quadpoints_theta,
          quadpoints_phi=surf_prev.quadpoints_phi
          )
    surf.least_squares_fit(surf_prev.gamma())
    #surf.plot()
    #plt.show()

    if constraint_weight:
        # Boozer least square approach
        print("Generating Boozer least squares surface...")
        vol = Volume(surf)
        boozer_surface = BoozerSurface(bs, surf, vol, vol_target, constraint_weight, options={'verbose':True})
    else:
        # Boozer exact approach
        print("Generating Boozer exact surface...")
        surf_exact = SurfaceXYZTensorFourier(
              mpol=mpol,ntor=mpol,nfp=5,stellsym=True,
              quadpoints_theta=np.linspace(0,1,2*mpol+1,endpoint=False),
              quadpoints_phi=np.linspace(0,1./surf.nfp,2*mpol+1,endpoint=False),
              dofs=surf.dofs
              )
    
        vol = Volume(surf_exact)
        boozer_surface = BoozerSurface(bs, surf_exact, vol, vol_target, None, options={'verbose':True})

    # Run boozer surface algorithm
    res = boozer_surface.run_code(iota, G0)
    print(f"G0 from solve: {res['G']}")
    print(f"iota from solve: {res['iota']}")

    # Check if boozer algo is successful
    success1 = res['success'] # True if the boozer surface algo converged
    success2 = not boozer_surface.surface.is_self_intersecting() # True if surface is not self intersecting
    #print(success1, success2)
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

def crossSectionPlot(surf_coils, surf, banana_curve, filename):
    # plots cross section of plasma at a few toroidal locations and relevant HBT cross sections
    plt.figure(figsize=(7,6))
    cs2 = surf_coils.cross_section(0)
    rs2 = np.sqrt(cs2[:,0]**2 + cs2[:,1]**2); rs2 = np.append(rs2, rs2[0])
    zs2 = cs2[:,2]; zs2 = np.append(zs2, zs2[0])    
    plt.plot(rs2, zs2, label='Banana Surface')
    cs3 = hbt.cross_section(0)
    rs3 = np.sqrt(cs3[:,0]**2 + cs3[:,1]**2); rs3 = np.append(rs3, rs3[0])
    zs3 = cs3[:,2]; zs3 = np.append(zs3, zs3[0])
    hbt_poly = Polygon(zip(rs3, zs3))
    plt.plot(rs3, zs3, label='HBT LCFS')
    cs_vv = VV.cross_section(0)
    rs_vv = np.sqrt(cs_vv[:, 0]**2 + cs_vv[:, 1]**2); zs_vv = cs_vv[:, 2]
    rs_vv = np.append(rs_vv, rs_vv[0]); zs_vv = np.append(zs_vv, zs_vv[0])
    plt.plot(rs_vv, zs_vv, label='Vacuum Vessel')
    phi_array = np.linspace(0, 1 / surf_coils.nfp, 5) # scaled from 0 to 1
    for phi_slice in phi_array:
        cs = surf.cross_section(phi_slice * 2 * np.pi)
        rs = np.sqrt(cs[:,0]**2 + cs[:,1]**2); rs = np.append(rs, rs[0])
        zs = cs[:,2]; zs = np.append(zs, zs[0])
        plasma_poly = Polygon(zip(rs, zs))
        if not plasma_poly.within(hbt_poly):
            plt.close()
            print("Plasma surface not within HBT boundary — skipping plot.")
            return False
        plt.plot(rs, zs, label=f'Φ={phi_slice * 2:0.2f}π')
    plt.xlabel('R [m]', fontsize=18, fontweight='bold')
    plt.ylabel('Z [m]', fontsize=18, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.minorticks_on()
    plt.grid(True)
    #plt.tight_layout()
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

    run_dict['lscount']+=1

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
    J_len = JCurveLength.J()
    dJ_len = np.linalg.norm(JCurveLength.dJ())
    J_cc = JCurveCurve.J()
    dJ_cc = np.linalg.norm(JCurveCurve.dJ())
    J_cs = JCurveSurface.J()
    dJ_cs = np.linalg.norm(JCurveSurface.dJ())
    J_surf = JSurfSurf.J()
    dJ_surf = np.linalg.norm(JSurfSurf.dJ())

    iotas_list = [iota.J() for iota in iotas]
    iota_str = ", ".join([f"{val:.4f}" for val in iotas_list])

    max_r = np.max(np.sqrt(banana_curve.gamma()[:,1]**2 + banana_curve.gamma()[:,2]**2))
    max_z = np.max(np.abs(banana_curve.gamma()[:,0]))
    length = curvelength.J()
    curvecurve_min = JCurveCurve.shortest_distance()
    curvesurf_min = JCurveSurface.shortest_distance()

    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * boozer_surface.surface.unitnormal(), axis=2)))
    intersecting = boozer_surface.surface.is_self_intersecting()

    width = 35
    buffer = io.StringIO()
    print("="*70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    print(f"{'nonQS ratio':{width}} = {J_QS:.6e} (dJ = {dJ_QS:.6e})", file=buffer)
    print(f"{'Boozer Residual':{width}} = {J_Boozer:.6e} (dJ = {dJ_Boozer:.6e})", file=buffer)
    print(f"{'ι Penalty':{width}} = {J_iota:.6e} (dJ = {dJ_iota:.6e})", file=buffer)
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    print(f"{'Curve Length Penalty':{width}} = {J_len:.6e} (dJ = {dJ_len:.6e})", file=buffer)
    print(f"{'Curve-Curve Penalty':{width}} = {J_cc:.6e} (min={curvecurve_min:.3e}) (dJ = {dJ_cc:.6e})", file=buffer)
    print(f"{'Curve-Surface Penalty':{width}} = {J_cs:.6e} (min={curvesurf_min:.3e}) (dJ = {dJ_cs:.6e})", file=buffer)
    print(f"{'Surf-Vessel Penalty':{width}} = {J_surf:.6e} (dJ = {dJ_surf:.6e})", file=buffer) 
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)
    print(f"{'Intersecting':{width}} = {intersecting}", file=buffer)
    print(f"{'Max Curve R':{width}} = {max_r:.6e}", file=buffer)
    print(f"{'Max Curve Z':{width}} = {max_z:.6e}", file=buffer)
    print(f"{'Curve Length':{width}} = {length:.6e}", file=buffer)
    print("="*70, file=buffer)

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
banana_surf_radius = 0.22  # Radius of coil surface
banana_surf_nfp = 5        # Field periods for coil surface
nphi = 255                 # Toroidal resolution
ntheta = 64                # Poloidal resolution
mpol = 5                   # Initial mpol for surface
vol_target = 0.10          # Target plasma volume
CONSTRAINT_WEIGHT = 1.0    # Use least squares Boozer solver
MAXITER = 300              # Max optimizer iterations
iota_target = 0.15         # Desired iota value
num_tf_coils = 20          # Number of TF coils in BiotSavart file
ftol_by_mpol = {5: 5e-7, 6: 1e-7, 7: 5e-8, 8: 1e-8, 9: 5e-9, 10: 1e-10}
gtol_by_mpol = {5: 5e-4, 6: 1e-4, 7: 5e-5, 8: 1e-6, 9: 1e-7, 10: 1e-8}

# Output directory setup
OUT_DIR = f"./scans/adaptive"
os.makedirs(OUT_DIR, exist_ok=True)

# Determine whether using least squares or exact Boozer solver
boozer_type = {'initial': 'least_squares', 'final': 'exact'}
stage = 'initial'

# ------------------------
# Define Vacuum Vessel and HBT Boundary for Plotting/Constraints
# ------------------------
VV = SurfaceRZFourier(nfp=5, stellsym=True)
VV.set_rc(0, 0, 1.0)
VV.set_rc(1, 0, 0.25)
VV.set_zs(1, 0, 0.25)

# HBT LCFS definition with conservative plasma margin
hbt = SurfaceRZFourier(nfp=5, stellsym=True)
hbt.set_rc(0, 0, 0.9115)
hbt.set_rc(1, 0, 0.1685)
hbt.set_zs(1, 0, 0.16)

# Banana coil winding surface (1 coil per field period)
surf_coils = SurfaceRZFourier(nfp=banana_surf_nfp, stellsym=True)
surf_coils.set_rc(0, 0, 0.985)
surf_coils.set_rc(1, 0, banana_surf_radius)
surf_coils.set_zs(1, 0, banana_surf_radius)

# ------------------------
# Load Magnetic Field and Initial Surface
# ------------------------
filename = 'wout_nfp22ginsburg_000_014417_iota15.nc'
bs = load(f'outputs-{filename}/MR=0.925-TF=0.24-LW=0.0005-CCW=100-CW=0.0001-SR=0.22-Order=2/biot_savart_opt.json')
surf = SurfaceRZFourier.from_wout(filename, range="half period", nphi=255, ntheta=64, s=0.24)
surf.set_dofs(surf.get_dofs()*0.925/surf.major_radius())  # scale to desired major radius

coils = bs.coils
curves = [c.curve for c in coils]
tf_coils = coils[:num_tf_coils]
tf_curves = [c.curve for c in tf_coils]
banana_coils = coils[num_tf_coils:]
banana_curves = [c.curve for c in banana_coils]
banana_curve = banana_curves[0]

# Compute G0 from TF coil currents (ampere-turns)
current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))  # µ_0 * total current

# ------------------------
# Adaptive Optimization Loop Over mpol
# ------------------------
for mpol in range(5, 11):
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
    curves_to_vtk(curves, OUT_DIR_ITER + f"/curves_init", close=True)
    bs.save(OUT_DIR_ITER + f"/biot_savart_init.json")
    
    pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
        boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((nphi, ntheta, 3))**2, axis=2))[:, :, None]}
    boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_init", extra_data=pointData)
    boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_init.json")

    # Print diagnostics for initial surface
    print(f"Volume: {boozer_surface.surface.volume()}")
    normPlot(boozer_surface.surface, bs, OUT_DIR_ITER + "/NormPlotInitial")
    crossSectionPlot(surf_coils, boozer_surface.surface, banana_curve, OUT_DIR_ITER + "/CrossSectionInitial")

    # ----------------------------------------
    # DEFINE OBJECTIVE FUNCTION
    # ----------------------------------------
    bs_obj = BiotSavart(coils)
    nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs_obj)]
    brs = [BoozerResidual(boozer_surface, bs_obj)] if boozer_type[stage]=='least_squares' else [BoozerResidualExact(boozer_surface, bs_obj)]

    # Define constraint weights
    LENGTH_WEIGHT = 10
    RES_WEIGHT = 1e5
    IOTAS_WEIGHT = 1e3
    CC_WEIGHT = 10
    CC_DIST = 0.05
    CS_WEIGHT = 10
    CS_DIST = 0.02
    SURF_DIST_WEIGHT = 1e3
    SS_DIST = 0.04
    phi_list = np.linspace(0, 1 / boozer_surface.surface.nfp, 5)

    iotas = [Iotas(boozer_surface)]
    curvelength = CurveLength(banana_curves[0])
    length_target = curvelength.J()

    # Construct full objective function JF
    Jiotamax = sum([QuadraticPenalty(iota, iota_target) for iota in iotas])
    JnonQSRatio = sum(nonQSs)
    JBoozerResidual = sum(brs)
    JCurveLength = QuadraticPenalty(curvelength, length_target, 'max')
    JCurveCurve = CurveCurveDistance(curves, CC_DIST)
    JCurveSurface = CurveSurfaceDistance(curves, boozer_surface.surface, CS_DIST)
    JSurfSurf = SurfaceSurfaceDistance(boozer_surface.surface, VV, SS_DIST)

    JF = JnonQSRatio + RES_WEIGHT * JBoozerResidual + IOTAS_WEIGHT * Jiotamax \
        + LENGTH_WEIGHT * JCurveLength + CC_WEIGHT * JCurveCurve \
        + CS_WEIGHT * JCurveSurface + SURF_DIST_WEIGHT * JSurfSurf

    dofs = JF.x

    # ----------------------
    # Set Initial Run State
    # ----------------------
    run_dict = {
        'sdofs': boozer_surface.surface.x.copy(),
        'iota': boozer_surface.res['iota'],
        'G': boozer_surface.res['G'],
        'J': JF.J(),
        'dJ': JF.dJ().copy(),
        'it': 1,
        'lscount': 0,
        'x_prev': dofs.copy()
    }

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
    curves_to_vtk(curves, OUT_DIR_ITER + "/curves_opt", close=True)
    bs.save(OUT_DIR_ITER + "/biot_savart_opt.json")
    pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
        boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((nphi, ntheta, 3))**2, axis=2))[:, :, None]}
    boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_opt", extra_data=pointData)
    boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_opt.json")
    print(f"Volume: {boozer_surface.surface.volume()}")
    print(f"Iota: {Iotas(boozer_surface).J()}")
    normPlot(boozer_surface.surface, bs, OUT_DIR_ITER + "/NormPlotOptimized")
    crossSectionPlot(surf_coils, boozer_surface.surface, banana_curve, OUT_DIR_ITER + "/CrossSectionOptimized")
