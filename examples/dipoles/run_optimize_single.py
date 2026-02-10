"""
File: run_optimize_scan.py
Author: Jake Halpern
Last Edit Date: 02/2025
Description: This script sends off a single optimization run of windowpane coil currents on 
             an axisymmetric surface for a specified plasma equilibrium
"""
import os
import shutil
import numpy as np
from optimize import optimize
from simsopt.geo import SurfaceRZFourier

script_dir = os.path.dirname(os.path.abspath(__file__))
###### set this to wherever you'd like all the outputs from this script to go
run_dir = os.path.join(script_dir, '../outputs/20250314_unfixed_TFs')

### Simulation parameters ###
# Dipole parameters
fil_distance = 0.05 # distance between dipole filaments for finite coil winding pack [m]
half_per_distance = 0.05 # distance between dipole panels between half field periods of the device [m]
dipole_radius = 0.045 # radius of dipoles (poloidally constant, will vary toroidally so this is at inboard midplane) [m]
numquadpoints = 64 # number of quad points for coils
# Plasma Surface
plas_nPhi = 128; plas_nTheta=64    # plasma surface quad points
surf_s = 1                  # value of s to cut the surface at (if already HBT sized, these will both be 1)
surf_dof_scale = 1      # used to scale the dofs of the surface
eq_name = 'wout_nfp2ginsburg_000_003186'  # name of the wout file from vmec
eq_dir = os.path.join(script_dir, 'equilibria') # equilibria should be in this folder
# Vacuum Vessel
VV_plas_dist = 0.10
# Extract the minimum axisymmetric VV size
eq_name_full = os.path.join(eq_dir, eq_name + ".nc")
surf = SurfaceRZFourier.from_wout(
    eq_name_full, s=surf_s, range="full torus", nphi=plas_nPhi, ntheta=plas_nTheta
)
surf.set_dofs(surf_dof_scale * surf.get_dofs())
R = np.sqrt(surf.gamma()[:, :, 0]**2 + surf.gamma()[:, :, 1]**2)
Rmin = np.min(R)
Rmax = np.max(R)
Zmin = np.min(surf.gamma()[:, :, 2])
Zmax = np.max(surf.gamma()[:, :, 2])
VV_R0 = (Rmin + Rmax) / 2
VV_amin = (Rmax - Rmin) / 2
VV_bmin = (Zmax - Zmin) / 2
VV_a = VV_amin + VV_plas_dist               # minor radius of vacuum vessel (horizontal)
VV_b = VV_bmin + VV_plas_dist               # minor radius of vacuum vessel (vertical)
# TF coils parameters (radius current set as 1.6 * VV_b)
n_tf = 4                       # number of TF coils per half field period
num_fixed = n_tf                  # number of TF coil currents to fix during combined optimization
field_on_axis = 1.0            # on-axis magnetic field (Tesla)
TF_R0 = VV_R0
TF_a = 0.40
TF_b = TF_a * VV_b / VV_a
fixed_geo_TFs = True
CC_THRESHOLD = 0.1
CC_WEIGHT = 100
CS_THRESHOLD = 0.1
CS_WEIGHT = 100
if fixed_geo_TFs:
    CC_THRESHOLD = None
    CC_WEIGHT = None 
    CS_THRESHOLD = None
    CS_WEIGHT = None
# Optimization parameters
definition = "local"         # definition of squared flux, either local, normalized, or quadratic flux
precomputed = True           # if true, will use precomputed Biot-Savart with scaled currents during optimization
MAXITER = 2500               # Number of iterations to perform:
CURRENT_THRESHOLD = 1E6      # Current penality threshold and weight
CURRENT_WEIGHT = 1E-12       # make sure weight is appropriate for the current threshold
verbose=True
# Figure parameters
dpi = 100; titlefontsize = 18; axisfontsize = 16; legendfontsize = 14; ticklabelfontsize = 14; cbarfontsize = 18

extra = ""
# only add these parameters to output file if they are unusual runs, can add more as needed
if VV_a != 0.25:
    extra = extra + f"_VVa_{VV_a}"
if VV_R0 != 1.0:
    extra = extra + f"_VV_R0_{VV_R0}"
if VV_a != VV_b:
    extra = extra + f"_ellipticalVV"
if dpi != 100:
    extra = extra + f"_for_poster"

# Name the file the next optimization number
num = str(input("Run number for directory naming: "))
if num == "":
    if not os.path.exists(os.path.join(run_dir, f'{eq_name}')) or not os.listdir(os.path.join(run_dir, f'{eq_name}')):
        num = 1
    else:
        num = max(int(f.split('_')[0]) for f in os.listdir(os.path.join(run_dir, f'{eq_name}')) if f.split('_')[0].isdigit()) + 1
else:
    num = int(num)
extra = extra + str(input("Anything extra to add to pathname? If none, just press enter: "))

run_dir = os.path.join(run_dir, f'{eq_name}/{num:02}_ntf{n_tf}_diprad_{dipole_radius}{extra}/')

###### Change directory
os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)

# Copy all helpful files to document the optimization here
shutil.copy(os.path.join(script_dir, "optimize.py"), os.path.join(run_dir, "optimize.py"), follow_symlinks=True)
shutil.copy(os.path.join(script_dir, 'helper_functions.py'), os.path.join(run_dir, 'helper_functions.py'), follow_symlinks=True)

optimize(fil_distance=fil_distance, half_per_distance=half_per_distance, dipole_radius=dipole_radius, numquadpoints=numquadpoints, # dipole parameters
            VV_a=VV_a, VV_b=VV_b, VV_R0=VV_R0,  # vessel parameters
            plas_nPhi=plas_nPhi, plas_nTheta=plas_nTheta, surf_s=surf_s, surf_dof_scale=surf_dof_scale, eq_dir=eq_dir, eq_name=eq_name,  # equilibrium parameters
            ntf=n_tf, num_fixed=num_fixed, field_on_axis=field_on_axis, TF_R0=TF_R0, TF_a=TF_a, TF_b=TF_b, fixed_geo_TFs=fixed_geo_TFs,
            CC_THRESHOLD=CC_THRESHOLD, CC_WEIGHT=CC_WEIGHT, CS_THRESHOLD=CS_THRESHOLD, CS_WEIGHT=CS_WEIGHT, # TF parameters
            definition=definition, precomputed=precomputed, MAXITER=MAXITER, CURRENT_THRESHOLD=CURRENT_THRESHOLD, CURRENT_WEIGHT=CURRENT_WEIGHT, 
            dpi=dpi, titlefontsize=titlefontsize, axisfontsize=axisfontsize, legendfontsize=legendfontsize, ticklabelfontsize=ticklabelfontsize, cbarfontsize=cbarfontsize,
            output_dir=run_dir, verbose=verbose)