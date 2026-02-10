from simsopt.geo import create_equally_spaced_curves, curves_to_vtk, SurfaceRZFourier
from simsopt.field import coils_via_symmetries, apply_symmetries_to_curves, apply_symmetries_to_currents, Current, ScaledCurrent, Coil, BiotSavart
from simsopt.geo.orientedcurve import OrientedCurveRTPFourier
from simsopt.objectives import SquaredFlux
from simsopt.mhd import VirtualCasing
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from create_surface import *
from helper_functions import *

########################################################################
########################## Input Parameters ############################
########################################################################
# Dipole parameters
fil_distance = FIL_DIST
half_per_distance = HALF_PER_DIST
dipole_radius = DIP_RAD

# Vacuum vessel parameters
VV_a = VVA_VAL
VV_b = VVB_VAL
VV_R0 = VVR0_VAL

# Plasma surface parameters
plas_nPhi = SURF_NPHI_VAL
plas_nTheta = SURF_NTHETA_VAL
surf_s = SURF_S_VAL
surf_dof_scale = SURF_DOF_SCALE_VAL
eq_dir = EQ_DIR
eq_name = EQ_NAME_VAL

# TF coil parameters
ntf = NTF_VAL
num_fixed = NUM_FIXED_VAL
field_on_axis = BT_VAL

# Optimization parameters
definition = "DEF_VAL"
MAXITER = MAX_ITER_VAL
CURRENT_THRESHOLD = CUR_THRESH_VAL
CURRENT_WEIGHT = CUR_WEIGHT_VAL

# Figure parameters
dpi = DPI_VAL
titlefontsize = TITLE_SIZE_VAL
axisfontsize = AXIS_SIZE_VAL
legendfontsize = LEGEND_SIZE_VAL
ticklabelfontsize = TICK_SIZE_VAL
cbarfontsize = CBAR_SIZE_VAL
###############################

# Create a surface representing the vacuum vessel that dipoles will be placed on
VV = SurfaceRZFourier()
VV.set_rc(0,0,VV_R0)
VV.set_rc(1,0,VV_a)
VV.set_zs(1,0,VV_b)
VV_phi = VV.quadpoints_phi
VV_theta = VV.quadpoints_theta

# Create the plasma surface
eq_name_full = os.path.join(eq_dir, eq_name + '.nc')
surf = SurfaceRZFourier.from_wout(eq_name_full, surf_s, range='half period', nphi=plas_nPhi, ntheta=plas_nTheta)
surf.set_rc(0,0,VV_R0)
theta = surf.quadpoints_theta
phi = surf.quadpoints_phi

# Geometric properties of the target plasma boundary
n = surf.normal()
absn = np.linalg.norm(n, axis=2)
unitn = n * (1./absn)[:,:,None]
sqrt_area = np.sqrt(absn.reshape((-1,1))/float(absn.size))
surf_area = sqrt_area**2
major_radius = surf.major_radius()
minor_radius = surf.minor_radius()
aspect_ratio = surf.aspect_ratio()
volume = surf.volume()

print('Initalizing Plasma Surface')
print(f'     Number of field periods = {surf.nfp}')
print(f'     Major radius = {major_radius:.2f}')
print(f'     Minor radius = {minor_radius:.2f}')
print(f'     Aspect ratio = {aspect_ratio:.2f}')
print(f'     Volume       = {volume:.2f}')

## Plot cross sections
plt.figure(figsize=(8,5))
phi_array = np.arange(0, 1.01, 0.2)
for phi_slice in phi_array:
    cs = surf.cross_section(phi_slice*np.pi)
    rs = np.sqrt(cs[:,0]**2 + cs[:,1]**2); rs = np.append(rs, rs[0])
    zs = cs[:,2]; zs = np.append(zs, zs[0])
    cs2 = VV.cross_section(phi_slice*np.pi)
    rs2 = np.sqrt(cs2[:,0]**2 + cs2[:,1]**2); rs2 = np.append(rs2, rs2[0])
    zs2 = cs2[:,2]; zs2 = np.append(zs2, zs2[0])    
    plt.plot(rs, zs, label=fr'$\phi$ = {phi_slice:.2f}π')
    plt.plot(rs2, zs2, 'k')
    plt.plot(np.mean(rs), np.mean(zs), 'kx')
plt.xlabel('R [m]', fontsize=axisfontsize, fontweight='bold')
plt.ylabel('Z [m]', fontsize=axisfontsize, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=legendfontsize)
plt.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('x_section.png', dpi = dpi)
plt.clf()  

# Run virtual casing calculation on equilibrium
vc_src_nphi = 80
vc = VirtualCasing.from_vmec(eq_name_full, src_nphi=vc_src_nphi, trgt_nphi=plas_nPhi, trgt_ntheta=plas_nTheta)
target = vc.B_external_normal # for the squared flux calculation

########################################################################
######################### Coil Initialization ##########################
########################################################################

# Initialize TF Coils
print('Initalizing Coils')
# order is the number of fourier moments to include - we don't need a large value
# 2 should be fine for elliptical
numquadpoints=32 # number of points per coil - this should be fine for both types of coils

base_toroidal_curves = create_equally_spaced_curves(
    ncurves=ntf,
    nfp=surf.nfp,
    stellsym=surf.stellsym,
    R0=VV_R0,
    R1=VV_b*1.6, # HBT is 0.4m/0.25m = 1.6, so keep this for now? 
    order=2, 
    numquadpoints=numquadpoints
)

print(f'     TF Coil Radius: {1.6*VV_b} [m]')
print(f'     Vessel Radius : {VV_R0} [m]')
# We define the currents with 1A, so that our dof is order unity. We
# then scale the current by a scale_factor of 1E6.
mu0 = 4.0 * np.pi * 1e-7
# toroidal solenoid approximation - B_T = mu0 * I * N / L = mu0 * I * N / L
# therefore, TF current = B_T * 2 * pi * R0 / mu0 / (2 * nfp * n_tf), n_tf = # per half field period!
poloidal_current = -2.0*np.pi*surf.get_rc(0,0)*field_on_axis/mu0
scale_factor = -poloidal_current/(2*ntf*surf.nfp)
base_toroidal_current = [ScaledCurrent(Current(1), scale_factor) for c in base_toroidal_curves]
base_toroidal_coils = [Coil(curve, current) for curve, current in zip(base_toroidal_curves, base_toroidal_current)]

# get initial field parameters
tf_curves_temp = apply_symmetries_to_curves(base_curves=base_toroidal_curves, nfp=surf.nfp, stellsym=surf.stellsym)
tf_currents_temp = apply_symmetries_to_currents(base_currents=base_toroidal_current, nfp=surf.nfp, stellsym=surf.stellsym)
tf_coils_temp = [Coil(curve, current) for curve, current in zip(tf_curves_temp, tf_currents_temp)]
bs_tf = BiotSavart(tf_coils_temp)
bs_tf.set_points(surf.gamma().reshape((-1, 3)))
Bfinal = bs_tf.B().reshape(n.shape)
modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))
_, _, _ = plot_relBfinal_norm_modB(
    bs_tf, surf, unitn, surf_area, n, phi, theta, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi, 'Initial')
delta = (np.max(modBfinal, axis=0) - np.min(modBfinal, axis=0)) / (np.max(modBfinal, axis=0) + np.min(modBfinal, axis=0))
print(f'     TF-only delta: {np.max(delta):.4f}')
plt.figure(figsize=(10, 6))
plt.plot(theta, delta, marker='o')
plt.xlabel(r'$\theta/2\pi$', fontsize=axisfontsize, fontweight='bold')
plt.ylabel(r'$\delta \approx \left.\frac{B_{\mathrm{max}} - B_{\mathrm{min}}}{B_{\mathrm{max}} + B_{\mathrm{min}}}\right|_\phi$', fontsize=axisfontsize, fontweight='bold')
plt.title(r'Initial LCFS-averaged $\delta$: {:.2e}'.format(np.mean(delta)), fontsize=titlefontsize, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)
plt.tight_layout()
plt.grid(True)
plt.savefig('delta_initial.png')
plt.clf()  

# Initialize dipoles 
# this section creates a dipole array packed into a hexagonal formation
# they will be circular at the inboard midplane, and will increase in toroidal
# length toward the outboard side due to the R dependence of the Jacobian

order=1 # number of Fourier modes - 1 is good for elliptical coils

# Figure out how many dipoles will fit in poloidal/toroidal directions
_, total_arc_length = generate_even_arc_angles(VV_a, VV_b, 1) # get total arc length
npol = int(total_arc_length / (2 * dipole_radius + fil_distance)) # figure out how many poloidal dipoles can fit for target radius
Rpol = total_arc_length / 2 / npol - fil_distance / 2 # adjust the poloidal length based off npol to fix filament distance
thetas, _ = generate_even_arc_angles(VV_b, VV_a, npol) # TODO: why do a and b need to be switched here? Def a bug, but works? 
dtheta = np.diff(np.append(thetas, 2*np.pi)) / 2 
# Due to spacing in between half field periods, equation governing toroidal dipoles will be:
# D + 2Rtor(theta) + sqrt(3)(ntor - 1)(Rtor(theta) + d/2) = pi/nfp*(R0+r(theta)cos(theta))
# where D is half period spacing of dipoles, d is the distance between filaments, ntor is number of toroidal dipoles, and Rtor is the toroidal radius
# toroidal distance between centers in hexagonal packing = sqrt(3) * (Rdip + fil_distance/2) (i.e. 30-60-90 triangle)
# Set ntor based on the number of circular dipoles that can fit on the inboard side
ntor = int((np.pi/surf.nfp*(VV_R0-VV_a) - half_per_distance - dipole_radius)/np.sqrt(3)/(dipole_radius+fil_distance/2)) + 1

print(f'     Number of Toroidal Dipoles: {ntor}')
print(f'     Number of Poloidal Dipoles: {npol}')

base_wp_curves = []
for ii in range(ntor):
    for jj in range(npol):
        # handle the poloidal angle offsets needed for hexagonal packing
        if (ii % 2 == 1):
            theta_coil = thetas[jj] + dtheta[jj]
        else:
            theta_coil = thetas[jj]
        # see orientedcurve.py for where r/x_int come from/how they're used
        r = VV_a*VV_b / np.sqrt((VV_b*np.cos(theta_coil))**2 + (VV_a*np.sin(theta_coil))**2)
        x_int = r * np.cos(theta_coil) * (1 - VV_b**2/VV_a**2)
        # increase toroidal length to keep filament distance constant in toroidal geometry
        # NOTE: Rtor is the physical width of the dipole, scale is a correction due to implementation specifics and is not physical
        # scale offsets the R dependence of the Jacobian that messes with toroidal width in the cartesian->toroidal transformation
        Rtor = (np.pi/surf.nfp*(VV_R0 + r * np.cos(theta_coil)) - half_per_distance - np.sqrt(3) * (ntor-1) * fil_distance/2) / (2 + np.sqrt(3) * (ntor - 1))
        scale = (VV_R0 + x_int + np.sqrt((r*np.cos(theta_coil) - x_int)**2 + (r*np.sin(theta_coil))**2)) / (VV_R0 + r*np.cos(theta_coil))
        # calculate toroidal angle of center of coil
        dphi = (half_per_distance/2 + Rtor) / (VV_R0 + r * np.cos(theta_coil)) # need to add buffer in phi for gaps in panels
        phi_coil = dphi + ii * (np.sqrt(3) * (Rtor + fil_distance/2)) / (VV_R0 + r * np.cos(theta_coil))
        c = OrientedCurveRTPFourier( numquadpoints, order)
        c.set('yc(1)',Rtor*scale) # toroidal direction
        c.set('zs(1)',Rpol) # poloidal direction
        c.set('R0', VV_R0)
        c.set('a', VV_a)
        c.set('b', VV_b)
        c.set('phi', phi_coil)
        c.set('theta', theta_coil)
        base_wp_curves.append( c )

nwptot = len(base_wp_curves * 2 * surf.nfp)
base_wp_current = [Current(1) for c in base_wp_curves]
wp_scale_factor = 1E5 # Initialize coils at 100kA (reasonable guess for 1T field)
base_wp_coils = [Coil(curve, ScaledCurrent(current, wp_scale_factor)) for curve, current in zip(base_wp_curves, base_wp_current)]
base_coils = base_toroidal_coils + base_wp_coils

########################################################################
######################### Current Optimization #########################
########################################################################

print('Beginning Coil Optimization')
# we aren't shape optimizing - fix the coil geometry
for c in base_coils:
    c.curve.fix_all()

# fix the currents in some TF coils (so that we avoid just finding B=0), choose either 1 or ntf
num_fixed = 1
for i in range(num_fixed):
    base_coils[i].current.fix_all()

# I have to do this because the dofs are out of order and sorted like current18, current19, 2, 20... 29, 3, 30 etc.
# This creates a mapping from lexicographic to numerical order, thanks chatGPT
# This might get messed up if I start naming currents - don't do that
ndofs = len(base_coils) - num_fixed
numbers = list(range(1+num_fixed, ndofs + num_fixed + 1))
string_numbers = [str(num) for num in numbers]
sorted_string_numbers = sorted(string_numbers)
sorted_indices = [int(num) for num in sorted_string_numbers]

print('     Precomputing Biot Savart from coils')
# Precompute normal field components from BiotSavart for each coil, that way we only vary linearly via current in optimization
BdotNcoil = np.zeros((ndofs, plas_nPhi, plas_nTheta))
BdotNcoil_fixed = np.zeros((num_fixed, plas_nPhi, plas_nTheta))
# only use these two if definition is local or normalized
Bcoil = np.zeros((ndofs, plas_nPhi, plas_nTheta, 3))
Bcoil_fixed = np.zeros((num_fixed, plas_nPhi, plas_nTheta, 3))

# now we will apply symmetries to the coils one at a time and calculate their respective BdotN contribution
coils = []
# need to add the fixed coil to the BdotN calc even though it isn't a dof
for ii, c  in enumerate(base_coils[0:num_fixed]):
    paired_curves_fixed = apply_symmetries_to_curves(base_curves=[c.curve], nfp=surf.nfp, stellsym=surf.stellsym)
    paired_currents_fixed = apply_symmetries_to_currents(base_currents=[c.current], nfp=surf.nfp, stellsym=surf.stellsym)
    paired_coils_fixed = [Coil(curve, current) for curve, current in zip(paired_curves_fixed, paired_currents_fixed)]
    bs_fixed = BiotSavart(paired_coils_fixed)
    bs_fixed.set_points(surf.gamma().reshape((-1,3)))
    BdotNcoil_fixed[ii, :, :] = np.sum(bs_fixed.B().reshape((plas_nPhi, plas_nTheta, 3)) * surf.unitnormal(), axis = 2) # this is BdotN for the unfixed coil
    if definition=='local' or definition=='normalized':
        Bcoil_fixed[ii, :, :, :] = bs_fixed.B().reshape((plas_nPhi, plas_nTheta, 3))
    coils += paired_coils_fixed

# precompute normal field from coils at initial current, which just gets scaled up/down during optization
for ii, c  in enumerate(base_coils[num_fixed:]):
    paired_curves = apply_symmetries_to_curves(base_curves=[c.curve], nfp=surf.nfp, stellsym=surf.stellsym)
    paired_currents = apply_symmetries_to_currents(base_currents=[c.current], nfp=surf.nfp, stellsym=surf.stellsym)
    paired_coils = [Coil(curve, current) for curve, current in zip(paired_curves, paired_currents)]
    bs_coil = BiotSavart(paired_coils)
    bs_coil.set_points(surf.gamma().reshape((-1,3)))
    i = sorted_indices.index(ii+1+num_fixed)
    BdotNcoil[i, :, :] = np.sum(bs_coil.B().reshape((plas_nPhi, plas_nTheta, 3)) * surf.unitnormal(), axis = 2) # this is BdotN for each coil
    if definition=='local' or definition=='normalized':
        Bcoil[i, :, :, :] = bs_coil.B().reshape((plas_nPhi, plas_nTheta, 3))
    coils += paired_coils

# Name the coils
for ii in range(0,ntf*2*surf.nfp,2*surf.nfp):
    coils[ii].name = f'TF_{int(ii/(2 * surf.nfp))+1}'
for ii in range(ntf*2*surf.nfp, len(coils), 2*surf.nfp):
    coils[ii].name = f'Dipole_{int(ii/(2 * surf.nfp))+1-ntf}'

bs = BiotSavart(coils)
Jf = SquaredFlux(surf, bs) # we do this through BdotN now, just using this to get dofs
dofs = Jf.x
dofs_init = dofs.copy() # need to use to normalize dofs when in iteration and initial values aren't 1

#use this to only set dJ for dipole current dofs
dJscale = [classify_current(dof_name, ntf+1) for dof_name in Jf.dof_names]

def fun(dofs):
        n_norm = np.linalg.norm(surf.normal(), axis=2)
        phi = surf.quadpoints_phi
        theta = surf.quadpoints_theta
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
        Jcp = np.sum(dJscale*np.array([np.maximum(np.abs(dofs[i]/dofs_init[i]*base_wp_coils[i].current.get_value()) - CURRENT_THRESHOLD, 0)**2 for i in range(len(dofs))]))
        dJcp = dJscale * np.array([derivativeJcp(dofs[i]/dofs_init[i]*base_wp_coils[i].current.get_value(), CURRENT_THRESHOLD) for i in range(len(dofs))])

        J = CURRENT_WEIGHT * Jcp + SF
        grad = CURRENT_WEIGHT * dJcp + gradSF
        outstr = f"J={J:.4e}, Squared Flux={SF:.4e}, Jcp ={Jcp:.4e}, |dJ|={np.linalg.norm(grad):.4e}"
        print(outstr)
        return J, grad

print('     Running gradient descent for coil currents')
# Perform the optimization on the coil currents
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-8)
# this tells you why the optimization stopped
message = res.message
print('     Reason for optimization end: ', message)

########################################################################
########################### Post-Processing ############################
########################################################################

print('Beginning data post-processing')
#update currents in the coils
Jf.x = res.x

# get final Bnormal
relBfinal_norm, final_mean_abs_relBfinal_norm, final_relBfinal_norm_max = plot_relBfinal_norm_modB(
    bs, surf, unitn, surf_area, n, phi, theta, axisfontsize, titlefontsize, cbarfontsize, ticklabelfontsize, dpi, 'Final')

# plot coil ripple parameter delta
bs.set_points(surf.gamma().reshape((-1, 3)))
Bfinal = bs.B().reshape(n.shape)
modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))
delta = (np.max(modBfinal, axis=0) - np.min(modBfinal, axis=0)) / (np.max(modBfinal, axis=0) + np.min(modBfinal, axis=0))
print(f'     Dipole-corrected delta: {np.max(delta):.4f}')
plt.figure(figsize=(10, 6))
plt.plot(theta, delta, marker='o')
plt.xlabel(r'$\theta/2\pi$', fontsize=axisfontsize, fontweight='bold')
plt.ylabel(r'$\delta \approx \left.\frac{B_{\mathrm{max}} - B_{\mathrm{min}}}{B_{\mathrm{max}} + B_{\mathrm{min}}}\right|_\phi$', fontsize=axisfontsize, fontweight='bold')
plt.title(r'Final LCFS-averaged $\delta$: {:.2e}'.format(np.mean(delta)), fontsize=titlefontsize, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)
plt.tight_layout()
plt.grid(True)
plt.savefig('delta_final.png')
plt.clf()  

with open("coilcurrents.txt", "w") as file0:
    for ii, c in enumerate(coils[0:-1:2*surf.nfp]):
        file0.write(f"{c.name} has current = {c.current.get_value():.2E} A\n")

# Prep coil data
curves = [c.curve for c in coils]
base_curves = [c.curve for c in base_coils]
currents = [c.current.get_value() for c in coils]
base_currents = [c.current.get_value() for c in coils]
# B field of an circle = mu0 I / 2 R
B_fields = [mu0 * base_wp_coils[i].current.get_value() / 2 / Rpol for i in range(len(base_wp_coils))]
max_It = np.max([coils[i].current.get_value() for i in range(ntf*2*surf.nfp)])
max_Iwp = np.max([coils[i].current.get_value() for i in range(ntf*2*surf.nfp, len(coils))])
mean_It = np.mean([np.abs(coils[i].current.get_value()) for i in range(ntf*2*surf.nfp)])
mean_Iwp = np.mean([np.abs(coils[i].current.get_value()) for i in range(ntf*2*surf.nfp, len(coils))])
print(f"     Total number of dipole coils = {nwptot}")
print(f"     Maximum dipole current       = {max_Iwp:.0f} A")
print(f"     Peak dipole B field          = {np.max(B_fields):.2f} T")
print(f"     Max TF/dipole current        = {max_It/max_Iwp:.2f}")
print(f"     Mean TF/dipole current       = {mean_It/mean_Iwp:.2f}")

# Save various files
surf.to_vtk('surf_opt', extra_data={"B_N": relBfinal_norm})
VV.to_vtk('vacuum_vessel')
curves_to_vtk(base_curves, 'base_coils', close=True, extra_data={'currents':np.array([current for current in base_currents for _ in range(numquadpoints+1)])})
curves_to_vtk(curves, 'coils', close=True, extra_data={'currents':np.array([current for current in currents for _ in range(numquadpoints+1)])})
bs.save('bs_opt.json');

surf_full = SurfaceRZFourier.from_wout(eq_name_full, surf_s, range='full torus', nphi=2*surf.nfp*plas_nPhi, ntheta=plas_nTheta)
surf_full.set_rc(0,0,VV_R0)
bs.set_points(surf_full.gamma().reshape(-1,3))
Bdotn = np.sum(bs.B().reshape((2*surf.nfp*plas_nPhi, plas_nTheta, 3)) * surf_full.unitnormal(), axis=2)
modB = bs.AbsB().reshape((2*surf.nfp*plas_nPhi,plas_nTheta))
BdotN_norm = Bdotn / modB
surf_full.to_vtk('surf_full', extra_data={"B_N": BdotN_norm[:, :, None]})