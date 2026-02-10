"""
File: windowpane_tf_opt_iteration.py
Author: Jake Halpern
Date: 02/10/2025
Description: This script shows how to iterate between TF/windowpane coil optimization, as well as removing
             low current coils to promote sparsity in between iterations. 
"""
from scipy.interpolate import RegularGridInterpolator
from simsopt.geo.curveplanarfourier import CurvePlanarFourier
from simsopt.field.coil import Coil, Current, ScaledCurrent
import numpy as np
from scipy.integrate import simpson
from simsopt.field import apply_symmetries_to_curves, apply_symmetries_to_currents, BiotSavart, Coil, Current
from simsopt.objectives import SquaredFlux
from scipy.optimize import minimize
from simsopt.field import coils_via_symmetries
from simsopt.geo import CurveLength, CurveCurveDistance, MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

def compute_quaternion(normal):
    """
    Compute the quaternion to rotate the upward direction [0, 0, 1] to the given unit normal vector.
    Parameters:
        normal (array-like): A 3-element array representing the unit normal vector [n_x, n_y, n_z].
    Returns:
        tuple: Quaternion as (q0, qi, qj, qk).
    """
    # Ensure the input is a numpy array
    normal = np.asarray(normal)
    # Check if the input is already normalized
    if not np.isclose(np.linalg.norm(normal), 1.0):
        raise ValueError("The input vector must be a unit vector.")
    upward = np.array([0.0, 0.0, 1.0]) # curve is initialized with +z normal
    # Compute the cosine and axis of rotation
    cos_theta = np.dot(upward, normal)
    axis = np.cross(upward, normal)
    # Handle the edge case where the vectors are parallel
    if np.allclose(axis, 0):
        if np.allclose(normal, upward): # 0 degree rotation
            return (1.0, 0.0, 0.0, 0.0)
        else: # 180 degree rotation
            return (0.0, 1.0, 0.0, 0.0)
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    # Compute the quaternion components
    theta_half = np.arccos(cos_theta) / 2
    q0 = np.cos(theta_half)
    sin_theta_half = np.sin(theta_half)
    qi, qj, qk = axis * sin_theta_half
    return (q0, qi, qj, qk)

def generate_windowpane_array(winding_surface, nwps_poloidal, nwps_toroidal, radius, numquadpoints=32, order=0):
    """
    Initialize an array of nwps_poloidal x nwps_toroidal planar windowpane coils on a winding surface
    Coils are initialized with a current of 1, that can then be scaled using ScaledCurrent
    Parameters:
        winding_surface: surface upon which to place the coils, with coil plane locally tangent to the surface normal
        nwps_poloidal: number of windowpane coils in the poloidal direction
        nwps_toroidal: number of windowpane coils per half field period in the toroidal direction
        radius: radius of the circular coils
        numquadpoints: number of points representing each coil (see CurvePlanarFourier documentation)
        order: number of Fourier moments for the planar coil representation, 0 = circle (see CurvePlanarFourier documentation)
    Returns:
        base_wp_curves: list of initialized curves (half field period)
        base_wp_currents: list of initialized currents (half field period)
    """    
    # Identify locations of windowpanes
    theta_locs = np.linspace(0, 1, nwps_poloidal, endpoint=False)
    dphi = 1/winding_surface.nfp/2/nwps_toroidal/2
    phi_locs   = np.linspace(0, 1/2/winding_surface.nfp, nwps_toroidal, endpoint=False) + dphi
    # Interpolate unit normal and gamma vectors of winding surface at location of windowpane centers
    unitn_interpolators = [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.unitnormal()[..., i], method='linear') for i in range(3)]
    gamma_interpolators = [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.gamma()[..., i], method='linear') for i in range(3)]
    # Initialize curves
    base_wp_curves = []
    for ii in range(nwps_poloidal):
        for jj in range(nwps_toroidal):
            unitn_interp = np.stack([interp((phi_locs[jj], theta_locs[ii])) for interp in unitn_interpolators], axis=-1)
            gamma_interp = np.stack([interp((phi_locs[jj], theta_locs[ii])) for interp in gamma_interpolators], axis=-1)
            curve = CurvePlanarFourier(numquadpoints, order, winding_surface.nfp, stellsym=True)
            # do it this way because number of dofs changes with ordering
            # dofs stored as: [r0, higher order curve terms, q_0, q_i, q_j, q_k, x0, y0, z0]
            curve.set('x0', radius)
            # align the coil normal with the surface normal
            # renormalize the vector because interpolation can slightly modify its norm
            quaternion = compute_quaternion(unitn_interp/np.linalg.norm(unitn_interp))
            curve.set(f'x{curve.dof_size-7}', quaternion[0])
            curve.set(f'x{curve.dof_size-6}', quaternion[1])
            curve.set(f'x{curve.dof_size-5}', quaternion[2])
            curve.set(f'x{curve.dof_size-4}', quaternion[3])
            # align the coil center with the winding surface gamma
            curve.set(f"x{curve.dof_size-3}", gamma_interp[0])
            curve.set(f"x{curve.dof_size-2}", gamma_interp[1])
            curve.set(f"x{curve.dof_size-1}", gamma_interp[2])
            base_wp_curves.append(curve)

    # raise error if coils are overlapping (approximate by less than 1 cm apart)
    Jccdist = CurveCurveDistance(base_wp_curves, 0.01)
    if Jccdist.J() != 0:
        raise ValueError('WP coils are less than 1 cm apart! Check for overlaps!')
    # now make the curves into coils
    base_wp_currents = [Current(1) for c in base_wp_curves]
    return base_wp_curves, base_wp_currents

# this should be a surface integral, int(f dS) = int(f|n| dtheta dphi)
def surf_int(f, n_norm, theta, phi):
    return simpson(simpson(f * n_norm, theta), phi)
# not 100% sure if I need this, but does surface integral with 3D array f
def surf_int_3D(f, n_norm, theta, phi): 
    return simpson(simpson(f * n_norm[None,:,:], theta), phi)

# differentiates between dipole and TF currents
def classify_current(item, threshold):
    # Extract the number after 'Current' and before ':'
    current_num = int(item.split(':')[0].replace('Current', ''))
    # Return 0 if the number is less than threshold, otherwise return 1
    return 0 if current_num < threshold else 1

# defines derivative of current penalty objective
def derivativeJcp(current, CURRENT_THRESHOLD):
    diff = np.abs(current) - CURRENT_THRESHOLD
    mask = diff > 0
    grad = np.where(current > 0, 2 * (current - CURRENT_THRESHOLD), 2 * (current + CURRENT_THRESHOLD))
    return grad * mask

def optimize_windowpane_currents(base_wp_coils, base_tf_coils, surf_plasma, definition='local', current_threshold=1e12, current_weight=1, num_fixed=1):
    """
    Optimize the currents in a set of windowpane coils given optimized tf_coils and a plasma surface
    Parameters:
        base_wp_coils: list of windowpane coils over half field period
        base_tf_coils: list of optimized tf_coils over half field period
        surf_plasma: half field period of plasma surface to optimize normal field on
        definition: definition of quadratic flux (https://simsopt.readthedocs.io/en/latest/simsopt.objectives.html#simsopt.objectives.fluxobjective.SquaredFlux)
        current_threshold: threshold to begin penalizing currents in the coils
        current_weight: weight on the current threshold penalty
        num_fixed: number of TF coils per half field period to fix current for, the rest will be optimized with the wp coils
    Returns:
        bs: optimized BiotSavart object
    """
    if definition!='local' and definition!='normalized' and definition!='quadratic flux':
        raise ValueError('definition must either be "local", "normalized", or "quadratic flux"')
    
    base_coils = base_tf_coils + base_wp_coils
    
    # make sure the geometric dofs are fixed for the coils, and wp currents are unfixed
    for c in base_wp_coils:
        c.curve.fix_all()
        c.current.unfix_all()
    for c in base_tf_coils:
        c.curve.fix_all()
    # fix the currents in some TF coils (so that we avoid the trivial solution), choose either 1 or ntf
    for i in range(num_fixed):
        base_tf_coils[i].current.fix_all()
    
    # I have to do this because the dofs are out of order and sorted like current18, current19, 2, 20... 29, 3, 30 etc., 
    # and I want them like 1, 2, 3 etc., this should fix them
    # This might get messed up if I start naming currents - don't do that
    # shoutout chatGPT for this section
    ndofs = len(base_coils) - num_fixed
    # Create the array of numbers in numerical order
    numbers = list(range(1+num_fixed, ndofs + num_fixed + 1))
    # Convert each number to a string
    string_numbers = [str(num) for num in numbers]
    # Sort the list of strings lexicographically
    sorted_string_numbers = sorted(string_numbers)
    # Convert back to integers
    sorted_indices = [int(num) for num in sorted_string_numbers]

    surf_plasma_nphi = len(surf_plasma.quadpoints_phi)
    surf_plasma_ntheta = len(surf_plasma.quadpoints_theta)

    # Initialize arrays for Squared Flux calc 
    BdotNcoil = np.zeros((ndofs, surf_plasma_nphi, surf_plasma_ntheta))
    BdotNcoil_fixed = np.zeros((num_fixed, surf_plasma_nphi, surf_plasma_ntheta))
    # these two are only used if local or normalized
    Bcoil = np.zeros((ndofs, surf_plasma_nphi, surf_plasma_ntheta, 3))
    Bcoil_fixed = np.zeros((num_fixed, surf_plasma_nphi, surf_plasma_ntheta, 3))

    # now we will apply symmetries to the coils one at a time and calculate their respective BdotN contribution
    coils = []
    # need to add the fixed coils to the BdotN calc even though it isn't a dof
    for ii, c  in enumerate(base_coils[0:num_fixed]):
        paired_curves_fixed = apply_symmetries_to_curves(base_curves=[c.curve], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym)
        paired_currents_fixed = apply_symmetries_to_currents(base_currents=[c.current], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym)
        paired_coils_fixed = [Coil(curve, current) for curve, current in zip(paired_curves_fixed, paired_currents_fixed)]
        bs_fixed = BiotSavart(paired_coils_fixed)
        bs_fixed.set_points(surf_plasma.gamma().reshape((-1,3)))
        BdotNcoil_fixed[ii, :, :] = np.sum(bs_fixed.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3)) * surf_plasma.unitnormal(), axis = 2) # this is BdotN for the unfixed coil
        if (definition=='local' or definition=='normalized'):
            Bcoil_fixed[ii, :, :, :] = bs_fixed.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3))
        coils += paired_coils_fixed
    # precompute normal field from coils at initial current, which just gets scaled up/down linearly during optization
    for ii, c  in enumerate(base_coils[num_fixed:]):
        paired_curves = apply_symmetries_to_curves(base_curves=[c.curve], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym)
        paired_currents = apply_symmetries_to_currents(base_currents=[c.current], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym)
        paired_coils = [Coil(curve, current) for curve, current in zip(paired_curves, paired_currents)]
        bs_coil = BiotSavart(paired_coils)
        bs_coil.set_points(surf_plasma.gamma().reshape((-1,3)))
        i = sorted_indices.index(ii+1+num_fixed)
        BdotNcoil[i, :, :] = np.sum(bs_coil.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3)) * surf_plasma.unitnormal(), axis = 2) # this is BdotN for each coil
        if (definition=='local' or definition=='normalized'):
            Bcoil[i, :, :, :] = bs_coil.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3))
        coils += paired_coils

    bs = BiotSavart(coils)
    Jf = SquaredFlux(surf_plasma, bs, definition=definition) # we do this through the above now, just using this to get dofs
    dofs = Jf.x

    dofs_init = dofs.copy() # need to use to normalize dofs when in iteration and initial values aren't 1

    #use this to only set dJ for dipole current dofs
    dJscale = [classify_current(dof_name, len(base_tf_coils)+1) for dof_name in Jf.dof_names]

    def fun(dofs):
        n_norm = np.linalg.norm(surf_plasma.normal(), axis=2)
        phi = surf_plasma.quadpoints_phi
        theta = surf_plasma.quadpoints_theta
        BdotN = np.sum((dofs/dofs_init)[:,None,None]*BdotNcoil, axis=0) + np.sum(BdotNcoil_fixed, axis=0)
        if definition=='local' or definition=='normalized':
            B = np.sum((dofs/dofs_init)[:,None,None,None]*Bcoil, axis=0) + np.sum(Bcoil_fixed, axis=0)
            modB = np.linalg.norm(B, axis=2)
            BcoildotB = np.sum(Bcoil * B, axis = 3)
        # calculate the squared flux
        if definition=='local':
            SF = 0.5 * surf_int((BdotN / modB)**2, n_norm, theta, phi)
            gradSF = surf_int_3D((modB[None, :, :]**2 * BdotNcoil * BdotN[None,:,:] - BcoildotB * BdotN[None,:,:]**2) / modB[None, :, :]**4, n_norm, theta, phi)
        elif definition=='normalized':
            SF = 0.5 * surf_int(BdotN**2, n_norm, theta, phi) /  surf_int(modB**2, n_norm, theta, phi)
            gradSF = (surf_int_3D(modB[None, :, :]**2, n_norm, theta, phi) * surf_int_3D(BdotNcoil * BdotN[None,:,:], n_norm, theta, phi) - surf_int_3D(BcoildotB, n_norm, theta, phi) * surf_int_3D(BdotN[None,:,:]**2, n_norm, theta, phi)) / surf_int_3D(modB[None, :, :]**2 , n_norm, theta, phi)**2
        else: # regular quadratic flux definition
            SF = 0.5 * surf_int(BdotN**2, n_norm, theta, phi) 
            gradSF = surf_int_3D(BdotNcoil * BdotN[None,:,:],n_norm, theta, phi)
        # this is a hardcoded current threshold penalty
        Jcp = np.sum(dJscale*np.array([np.maximum(np.abs(dofs[i]/dofs_init[i]*base_coils[i].current.get_value()) - current_threshold, 0)**2 for i in range(len(dofs))]))
        dJcp = dJscale * np.array([derivativeJcp(dofs[i]/dofs_init[i]*base_coils[i].current.get_value(), current_threshold) for i in range(len(dofs))])
        if Jcp > 0:
            raise ValueError("Current Threshold Penalty Currently Not Working For Iterative Optimization!")
        J = current_weight * Jcp + SF
        grad = current_weight * dJcp + gradSF
        outstr = f"J={J:.4e}, Squared Flux={SF:.4e}, Jcp ={Jcp:.4e}, |dJ|={np.linalg.norm(grad):.4e}"
        print(outstr)
        return J, grad

    # optimize currents
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 10, 'maxcor': 300}, tol=1e-12)
    # update optimized currents in the coils
    Jf.x = res.x
    return bs

def optimize_tf_coils(base_tf_coils, base_wp_coils, surf_plasma):
    base_tf_curves = [c.curve for c in base_tf_coils]
    tf_coils = coils_via_symmetries(base_tf_curves, [c.current for c in base_tf_coils], surf_plasma.nfp, surf_plasma.stellsym)
    wp_coils = coils_via_symmetries([c.curve for c in base_wp_coils], [c.current for c in base_wp_coils], surf_plasma.nfp, surf_plasma.stellsym)
    # fix all dofs for wp coils
    for c in wp_coils:
        c.curve.fix_all()
        c.current.fix_all()
    # just do a TF current optimization for the test
    for c in tf_coils:
        c.curve.unfix_all()
        c.current.unfix_all()
    # Define objective function
    coils = tf_coils + wp_coils
    bs = BiotSavart( coils )
    LENGTH_WEIGHT = Weight(1e-5)
    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = 0.1
    CC_WEIGHT = 1000
    # Threshold and weight for the coil-to-surface distance penalty in the objective function:
    CS_THRESHOLD = 0.1
    CS_WEIGHT = 10
    # Threshold and weight for the curvature penalty in the objective function:
    CURVATURE_THRESHOLD = 5.
    CURVATURE_WEIGHT = 1e-6
    # Threshold and weight for the mean squared curvature penalty in the objective function:
    MSC_THRESHOLD = 5.
    MSC_WEIGHT = 1e-6
    Jf = SquaredFlux(surf_plasma, bs, definition='local')
    Jls = [CurveLength(c) for c in base_tf_curves]
    Jccdist = CurveCurveDistance(base_tf_curves, CC_THRESHOLD, num_basecurves=len(base_tf_curves))
    Jcsdist = CurveSurfaceDistance(base_tf_curves, surf_plasma, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_tf_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_tf_curves]
    JF = Jf \
        + LENGTH_WEIGHT * sum(Jls) \
        + CC_WEIGHT * Jccdist \
        + CS_WEIGHT * Jcsdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()

        nphi,ntheta,_ = surf_plasma.gamma().shape
        
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf_plasma.unitnormal(), axis=2)))
        outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_tf_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        #outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad
    
    dofs = JF.x
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 15, 'maxcor': 300}, tol=1e-10)
    
def pop_windowpane_array(base_wp_coils, current_threshold):
    """
    Delete windowpane coils with currents less than a threshold from an array
    Parameters:
        base_wp_coils: list of windowpane coils over half field period
        current_threshold: wp coils with currents below this are deleted
    Returns:
        new_base_wp_coils: new list of windopane coils
    """
    for c in base_wp_coils:
        if np.abs(c.current.get_value()) < current_threshold:
            base_wp_coils.remove(c)
    return base_wp_coils

def run_tests():
    from simsopt.geo import curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier

    # location of the VMEC wout file
    surf_loc = 'equilibria/wout_nfp22ginsburg_000_000281.nc'
    
    # Initialize plasma and axisymmetric surface to place windowpanes on
    surf_plasma = SurfaceRZFourier.from_wout(surf_loc, range='half period')
    surf_wf = SurfaceRZFourier(nfp=surf_plasma.nfp)
    surf_wf.set_rc(0,0,surf_plasma.major_radius())
    surf_wf.set_rc(1,0,surf_plasma.minor_radius()*1.5)
    surf_wf.set_zs(1,0,surf_plasma.minor_radius()*1.5)
    surf_plasma.to_vtk('surf_plasma')

    # Initialize TF curves
    ntf = 5
    base_tf_curves = create_equally_spaced_curves(ncurves=ntf, nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym, R0=surf_plasma.major_radius(), R1=surf_plasma.minor_radius()*2.5, order=4, numquadpoints=32)
    base_tf_current = [Current(1) for c in base_tf_curves]
    mu0 = 4.0 * np.pi * 1e-7
    field_on_axis=1.0
    poloidal_current = -2.0*np.pi*surf_plasma.get_rc(0,0)*field_on_axis/mu0
    scale_factor = -poloidal_current/(2*ntf*surf_plasma.nfp)

    # Initialize WP curves
    base_wp_curves, base_wp_currents = generate_windowpane_array(surf_wf, 10, 10, 0.05, numquadpoints=32, order=0)

    # Create coils from the curves/currents
    base_tf_coils = [Coil(curve, ScaledCurrent(current, scale_factor)) for curve, current in zip(base_tf_curves, base_tf_current)]
    wp_init_current = 100 # initialize to 1A to keep the initialized normal field small
    base_wp_coils = [Coil(curve, ScaledCurrent(current, wp_init_current)) for curve, current in zip(base_wp_curves, base_wp_currents)]

    for i in range(3):
        optimize_tf_coils(base_tf_coils, base_wp_coils, surf_plasma)
        _ = optimize_windowpane_currents(base_wp_coils, base_tf_coils, surf_plasma, definition='local', current_threshold=1e10, current_weight=1e-5, num_fixed=ntf)
        base_wp_coils = pop_windowpane_array(base_wp_coils, current_threshold=1000)

    base_wp_curves = [c.curve for c in base_wp_coils]

    curves_to_vtk(base_wp_curves, "wp_coils_test")
    curves_to_vtk(base_tf_curves, "tf_coils_test")

if __name__=='__main__':
    run_tests()
