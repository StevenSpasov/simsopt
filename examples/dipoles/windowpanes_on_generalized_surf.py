"""
File: windowpane_on_generalized_surf.py
Author: Jake Halpern
Last Edit Date: 02/2025
Description: This script shows how to set up planar coils tangent to a general winding surface. An attempt
             is made to space them equally by setting the ellipse axis lengths based on the arc length
             between equally spaced angles - this process is far from optimized packing, but is slightly 
             better than equally spaced angles and constant width
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import simpson as simps #simps is the outdated name - Steven changed this Sep11 7:22pm
from simsopt.geo import CurvePlanarFourier
import scipy.integrate as spi

def quaternion_from_axis_angle(axis, theta):
    """Compute a quaternion from a rotation axis and angle."""
    axis = axis / np.linalg.norm(axis)
    q0 = np.cos(theta / 2)
    q_vec = axis * np.sin(theta / 2)
    return np.array([q0, *q_vec])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_vector(v, q):
    """Rotate a vector v using quaternion q."""
    q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])  # q* (conjugate)
    v_quat = np.array([0, *v])  # Convert v to quaternion form
    v_rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conjugate)
    return v_rotated[1:]  # Return the vector part

def compute_quaternion(normal, tangent):
    """
    Compute the quaternion to rotate the upward direction [0, 0, 1] to the given unit normal vector.
    Then, compute the change needed to align the x direction [1, 0, 0] to the desired tangent after the first rotation.
    
    Parameters:
        normal (array-like): A 3-element array representing the unit normal vector [n_x, n_y, n_z].
        tangent (array-like): A 3-element array representing the unit tangent vector [t_x, t_y, t_z].
    
    Returns:
        np.array: Quaternion as [q0, qi, qj, qk].
    """
    # Ensure input vectors are numpy arrays and normalized
    normal = np.asarray(normal)
    tangent = np.asarray(tangent)

    if not np.isclose(np.linalg.norm(normal), 1.0):
        raise ValueError("The input normal must be a unit vector.")
    if not np.isclose(np.linalg.norm(tangent), 1.0):
        raise ValueError("The input tangent must be a unit vector.")

    # Step 1: Rotate +z (upward) to normal
    upward = np.array([0.0, 0.0, 1.0])
    cos_theta1 = np.dot(upward, normal)
    axis1 = np.cross(upward, normal)

    if np.allclose(axis1, 0):
        q1 = np.array([1.0, 0.0, 0.0, 0.0]) if np.allclose(normal, upward) else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        axis1 /= np.linalg.norm(axis1)
        theta1 = np.arccos(np.clip(cos_theta1, -1.0, 1.0))
        q1 = quaternion_from_axis_angle(axis1, theta1)

    # Step 2: Rotate initial x-direction using q1
    initial_x = np.array([1.0, 0.0, 0.0])
    rotated_x = rotate_vector(initial_x, q1)

    # Step 3: Rotate rotated_x to match desired tangent
    axis2 = np.cross(rotated_x, tangent)
    if np.linalg.norm(axis2) < 1e-8:
        q2 = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation needed
    else:
        axis2 /= np.linalg.norm(axis2)
        theta2 = np.arccos(np.clip(np.dot(rotated_x, tangent), -1.0, 1.0))
        q2 = quaternion_from_axis_angle(axis2, theta2)

    # Final quaternion (q2 * q1)
    q_final = quaternion_multiply(q2, q1)
    return q_final

def rho(theta, a, b, n):
    return ( 1/(abs(np.cos(theta)/a)**(n) + abs(np.sin(theta)/b)**(n) ) ** (1/(n)))
# Define Fourier coefficient integrals
def a_m(m, a, b, n):
    integrand = lambda theta: rho(theta, a, b, n) * np.cos(m * theta)
    if m==0:
        return (1 / (2 * np.pi)) * spi.quad(integrand, 0, 2 * np.pi)[0]
    else:
        return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]

def b_m(m, a, b, n):
    integrand = lambda theta: rho(theta, a, b, n) * np.sin(m * theta)
    return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]

# Compute Fourier coefficients up to a given order
def compute_fourier_coeffs(max_order, a, b, n):
    coeffs = {'a_m': [], 'b_m': []}
    for m in range(max_order + 1):
        coeffs['a_m'].append(a_m(m, a, b, n))
        coeffs['b_m'].append(b_m(m, a, b, n))
    return coeffs

# Reconstruct the Fourier series approximation
def rho_fourier(theta, coeffs, max_order):
    rho_approx = coeffs['a_m'][0]
    for m in range(1, max_order + 1):
        rho_approx += coeffs['a_m'][m] * np.cos(m * theta) + coeffs['b_m'][m] * np.sin(m * theta)
    return rho_approx

def compute_toroidal_arc_length(winding_surface, poloidal_angle, phi1, phi2):
    """
    Compute the total arc length in the toroidal direction between phi1 and phi2 at a given poloidal angle.
    
    Parameters:
    - winding_surface: winding surface to compute arc length for
    - poloidal_angle: The specific poloidal angle at which to compute the arc length.
    - phi1, phi2: The toroidal angle range for integration.

    Returns:
    - Total arc length in the toroidal direction from phi1 to phi2 at the given poloidal angle.
    """
    # Ensure phi values are within [0, 1]
    phi1 = phi1 % 1
    phi2 = phi2 % 1
    dgamma_dphi = winding_surface.dgammadphi()
    dgamma_dphi_mag = np.linalg.norm(dgamma_dphi, axis=2)
    phi_vals = np.array(winding_surface.get_phi_quadpoints(nphi=dgamma_dphi_mag.shape[0]))
    theta_vals = np.array(winding_surface.get_theta_quadpoints(ntheta=dgamma_dphi_mag.shape[1]))
    interpolator = RegularGridInterpolator((phi_vals, theta_vals), dgamma_dphi_mag, bounds_error=False, fill_value=None)
    # Interpolate dgamma/dphi at the requested poloidal angle for all phi values
    dgamma_dphi_interp = np.array([interpolator((phi, poloidal_angle)) for phi in phi_vals])
    # Case 1: If the range does *not* cross zero, integrate normally
    if phi1 <= phi2:
        mask = (phi_vals >= phi1) & (phi_vals <= phi2)
        if not np.any(mask):  # Check if the mask is empty
            raise ValueError("No valid points found in integration range.")
        arc_length = simps(dgamma_dphi_interp[mask], phi_vals[mask])
    # Case 2: If the range *does* cross zero, split integration into two parts
    else:
        mask1 = (phi_vals >= phi1)
        mask2 = (phi_vals <= phi2)
        if not np.any(mask1) and not np.any(mask2):
            raise ValueError("No valid points found in integration range.")
        arc_length1 = simps(dgamma_dphi_interp[mask1], phi_vals[mask1]) if np.any(mask1) else 0
        arc_length2 = simps(dgamma_dphi_interp[mask2], phi_vals[mask2]) if np.any(mask2) else 0
        arc_length = arc_length1 + arc_length2
    return 2 * np.pi * arc_length

def compute_poloidal_arc_length(winding_surface, toroidal_angle, theta1, theta2):
    """
    Compute the total arc length in the poloidal direction between theta1 and theta2 at a given poloidal angle.
    
    Parameters:
    - winding_surface: winding surface to compute arc length for
    - toroidal_angle: The specific toroidal angle at which to compute the arc length.
    - theta1, theta2: The poloidal angle range for integration.

    Returns:
    - Total arc length in the poloidal direction from theta1 to theta2 at the given toroidal angle.
    """
    # Ensure theta values are within [0, 1]
    theta1 = theta1 % 1
    theta2 = theta2 % 1
    dgamma_dtheta = winding_surface.dgammadtheta()
    dgamma_dtheta_mag = np.linalg.norm(dgamma_dtheta, axis=2)
    phi_vals = np.array(winding_surface.get_phi_quadpoints(nphi=dgamma_dtheta_mag.shape[0]))
    theta_vals = np.array(winding_surface.get_theta_quadpoints(ntheta=dgamma_dtheta_mag.shape[1]))
    interpolator = RegularGridInterpolator((phi_vals, theta_vals), dgamma_dtheta_mag, bounds_error=False, fill_value=None)
    # Interpolate dgamma/dphi at the requested poloidal angle for all phi values
    dgamma_dtheta_interp = np.array([interpolator((theta, toroidal_angle)) for theta in theta_vals])
    # Case 1: If the range does *not* cross zero, integrate normally
    if theta1 <= theta2:
        mask = (theta_vals >= theta1) & (theta_vals <= theta2)
        if not np.any(mask):  # Check if the mask is empty
            raise ValueError("No valid points found in integration range.")
        arc_length = simps(dgamma_dtheta_interp[mask], theta_vals[mask])
    # Case 2: If the range *does* cross zero, split integration into two parts
    else:
        mask1 = (theta_vals >= theta1)
        mask2 = (theta_vals <= theta2)
        if not np.any(mask1) and not np.any(mask2):
            raise ValueError("No valid points found in integration range.")
        arc_length1 = simps(dgamma_dtheta_interp[mask1], theta_vals[mask1]) if np.any(mask1) else 0
        arc_length2 = simps(dgamma_dtheta_interp[mask2], theta_vals[mask2]) if np.any(mask2) else 0
        arc_length = arc_length1 + arc_length2
    return 2 * np.pi * arc_length

def generate_windowpane_array(winding_surface, nwps_poloidal, nwps_toroidal, wp_fil_spacing, wp_n, numquadpoints=32, order=12):
    """
    Initialize an array of nwps_poloidal x nwps_toroidal planar windowpane coils on a winding surface
    Coils are initialized with a current of 1, that can then be scaled using ScaledCurrent
    Parameters:
        winding_surface: surface upon which to place the coils, with coil plane locally tangent to the surface normal
        nwps_poloidal: number of windowpane coils in the poloidal direction
        nwps_toroidal: number of windowpane coils per half field period in the toroidal direction
        wp_spacing: approximate spacing between wp filaments
        wp_n: value of n for superellipse, see https://en.wikipedia.org/wiki/Superellipse
        numquadpoints: number of points representing each coil (see CurvePlanarFourier documentation)
        order: number of Fourier moments for the planar coil representation, 0 = circle (see CurvePlanarFourier documentation), more for ellipse approximation
    Returns:
        base_wp_curves: list of initialized curves (half field period)
        base_wp_currents: list of initialized currents (half field period)
    """    
    # Identify locations of windowpanes
    theta_locs = np.linspace(0, 1, nwps_poloidal, endpoint=False)
    dphi = 1/winding_surface.nfp/2/nwps_toroidal/2
    phi_locs   = np.linspace(0, 1/2/winding_surface.nfp, nwps_toroidal, endpoint=False) + dphi
    # Interpolate unit normal and gamma vectors of winding surface at location of windowpane centers
    unitn_interpolators =        [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.unitnormal()[..., i], method='linear') for i in range(3)]
    gamma_interpolators =        [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.gamma()[..., i], method='linear') for i in range(3)]
    dgammadtheta_interpolators = [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.dgammadtheta()[..., i], method='linear') for i in range(3)]
    theta_diff = theta_locs[1] - theta_locs[0]
    phi_diff = phi_locs[1] - phi_locs[0]
    # Initialize curves
    base_wp_curves = []
    for ii in range(nwps_poloidal):
        for jj in range(nwps_toroidal):
            # right now, tries to vary radii to account for varying arc length
            # could instead precompute total arc length, equally spacing in terms of arc length and solve for dipole theta/phi
            toroidal_arc_length = compute_toroidal_arc_length(winding_surface, theta_locs[ii], phi_locs[jj]-phi_diff/2, phi_locs[jj]+phi_diff/2)
            poloidal_arc_length = compute_poloidal_arc_length(winding_surface, phi_locs[jj], theta_locs[ii]-theta_diff/2, theta_locs[ii]+theta_diff/2)
            dip_a = (poloidal_arc_length - wp_fil_spacing) / 2
            dip_b = (toroidal_arc_length - wp_fil_spacing) / 2
            # Compute fourier coefficients for given dipole size
            coeffs = compute_fourier_coeffs(order, dip_a, dip_b, wp_n)
            unitn_interp =        np.stack([interp((phi_locs[jj], theta_locs[ii])) for interp in unitn_interpolators], axis=-1)
            gamma_interp =        np.stack([interp((phi_locs[jj], theta_locs[ii])) for interp in gamma_interpolators], axis=-1)
            dgammadtheta_interp = np.stack([interp((phi_locs[jj], theta_locs[ii]))for interp in dgammadtheta_interpolators], axis=-1)
            curve = CurvePlanarFourier(numquadpoints, order, winding_surface.nfp, stellsym=True)
            # dofs stored as: [r0, higher order curve terms, q_0, q_i, q_j, q_k, x0, y0, z0]
            for m in range(order+1):
                curve.set(f'x{m}', coeffs['a_m'][m])
                if m!=0:
                    curve.set(f'x{m+order+1}', coeffs['b_m'][m])
            # align the coil normal with the surface normal
            # renormalize the vector because interpolation can slightly modify its norm
            quaternion = compute_quaternion(unitn_interp/np.linalg.norm(unitn_interp), dgammadtheta_interp/np.linalg.norm(dgammadtheta_interp))
            curve.set(f'x{curve.dof_size-7}', quaternion[0])
            curve.set(f'x{curve.dof_size-6}', quaternion[1])
            curve.set(f'x{curve.dof_size-5}', quaternion[2])
            curve.set(f'x{curve.dof_size-4}', quaternion[3])
            # align the coil center with the winding surface gamma
            curve.set(f"x{curve.dof_size-3}", gamma_interp[0])
            curve.set(f"x{curve.dof_size-2}", gamma_interp[1])
            curve.set(f"x{curve.dof_size-1}", gamma_interp[2])
            base_wp_curves.append(curve)

    # now make the curves into coils
    base_wp_currents = [] #passing an empty array for the test, but could make this any current
    return base_wp_curves, base_wp_currents

def run_tests():
    from simsopt.geo import SurfaceRZFourier
    from simsopt.geo import curves_to_vtk
    # set this to be decently high! Want high resolution for arc length integration
    winding_surf = SurfaceRZFourier.from_nphi_ntheta(nfp=2, nphi = 512, ntheta=512, mpol=2, ntor=2, range='half period')
    # just making this random to test
    winding_surf.set_rc(0,0,1)
    winding_surf.set_rc(1,0,0.35)
    winding_surf.set_zs(1,0,0.25)
    winding_surf.set_rc(2, 0, -0.1)
    winding_surf.set_rc(0, 2, 0.1)
    base_wp_curves, _ = generate_windowpane_array(winding_surf, 20, 15, 0.02, 2, 32, 12)
    curves_to_vtk(base_wp_curves, 'test_wp')
    winding_surf.to_vtk('test_winding_surf')

if __name__=='__main__':
    run_tests()
