"""
File: run_optimize_scan.py
Author: Jake Halpern
Last Edit Date: 02/10/2025
Description: This script sends off an optimization scan over various geometric parameters for
             windowpane coil currents on an axisymmetric surface for a specified plasma equilibrium
"""

import os
import shutil
import numpy as np
from datetime import datetime
from optimize import *
from simsopt.geo import SurfaceRZFourier

# Set script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define output parent folder
output_subfolder = "20250303_08_iota_unfixed_TFs"  # Change this as needed
parent_run_dir = os.path.join(script_dir, f"../outputs/{output_subfolder}")

# Define ranges for dipole_radius and VV_a
dipole_radius_range = np.linspace(0.035, 0.055, 5)
VV_plas_dist_range = np.linspace(0.07, 0.13, 7)
ntf_range = [4]

### Base Simulation Parameters ###
# Dipole parameters
fil_distance = (
    0.05  # distance between dipole filaments for finite coil winding pack [m]
)
half_per_distance = (
    0.05  # distance between dipole panels between half field periods of the device [m]
)
numquadpoints = 64  # number of quad points for coils
# Plasma Surface
plas_nPhi = 128
plas_nTheta = 64  # plasma surface quad points
surf_s = (
    1  # value of s to cut the surface at (if already HBT sized, these will both be 1)
)
surf_dof_scale = 1  # used to scale the dofs of the surface
eq_name = "wout_nfp22ginsburg_000_000281"  # name of the wout file from vmec
eq_dir = os.path.join(script_dir, "equilibria")  # equilibria should be in this folder

# Optimization parameters
definition = (
    "local"  # definition of squared flux, either local, normalized, or quadratic flux
)
precomputed = True  # if true, will use precomputed Biot-Savart with scaled currents during optimization
MAXITER = 2500  # Number of iterations to perform:
CURRENT_THRESHOLD = 5e5  # Current penality threshold and weight
CURRENT_WEIGHT = 1e-12  # make sure weight is appropriate for the current threshold
verbose = False  # keep printing to a minimum
# Figure parameters
dpi = 100
titlefontsize = 18
axisfontsize = 16
legendfontsize = 14
ticklabelfontsize = 14
cbarfontsize = 18

# Extract the minimum axisymmetric VV size
eq_name_full = os.path.join(eq_dir, eq_name + ".nc")
surf = SurfaceRZFourier.from_wout(
    eq_name_full, s=surf_s, range="full torus", nphi=plas_nPhi, ntheta=plas_nTheta
)
surf.set_dofs(surf_dof_scale * surf.get_dofs())
gamma = surf.gamma()
X = gamma[:, :, 0]
Y = gamma[:, :, 1]
Z = gamma[:, :, 2]
R = np.sqrt(X**2 + Y**2)
Rmin = np.min(R)
Rmax = np.max(R)
Zmin = np.min(Z)
Zmax = np.max(Z)
VV_R0 = (Rmin + Rmax) / 2
VV_a = (Rmax - Rmin) / 2
VV_b = (Zmax - Zmin) / 2
print(f"VV_R0 = {VV_R0:.2f}, VV_a_min = {VV_a:.2f}, VV_b_min = {VV_b:.2f}")

# TF coils parameters
# n_tf = 10                       # number of TF coils per half field period
# num_fixed = n_tf                  # number of TF coil currents to fix during combined optimization
field_on_axis = 0.5  # on-axis magnetic field (Tesla)
TF_R0 = VV_R0
TF_a = 0.4  # this is what HBT is - keep for now, since if this varies so will ripple and thus BdotN
TF_b = (
    TF_a * VV_b / VV_a
)  # keep ellipticity of initial VV/plasma? Shouldn't matter as long as its fixed
fixed_geo_TFs = False
CC_THRESHOLD = 0.1
CC_WEIGHT = 100
CS_THRESHOLD = 0.1
CS_WEIGHT = 100
if fixed_geo_TFs:
    CC_THRESHOLD = None
    CC_WEIGHT = None
    CS_THRESHOLD = None
    CS_WEIGHT = None

# Ensure the parent output folder exists
os.makedirs(parent_run_dir, exist_ok=True)

# Get the current date for directory naming
current_date = datetime.now().strftime("%Y%m%d")

# Determine the starting run number
existing_runs = [d for d in os.listdir(parent_run_dir) if d.split("_")[0].isdigit()]
if existing_runs:
    next_run_number = max(int(d.split("_")[0]) for d in existing_runs) + 1
else:
    next_run_number = 1

# Loop over dipole_radius and VV-plasma distance and number of tf coils
for dipole_radius in dipole_radius_range:
    for VV_plas_dist in VV_plas_dist_range:
        for ntf in ntf_range:
            ntf = int(ntf)
            num_fixed = int(ntf)
            print(
                f"WP radius = {dipole_radius}, VV-plasma Distance = {VV_plas_dist}, ntf = {ntf}"
            )
            # Create a unique directory name
            run_dir_name = f"{next_run_number:02}_{current_date}_diprad_{dipole_radius:.3f}_VV_plas_dist_{VV_plas_dist:.3f}"
            run_dir = os.path.join(parent_run_dir, run_dir_name)
            # Create directory and copy files into it
            os.makedirs(run_dir, exist_ok=True)
            shutil.copy(
                os.path.join(script_dir, "optimize.py"),
                os.path.join(run_dir, "optimize.py"),
            )
            shutil.copy(
                os.path.join(script_dir, "helper_functions.py"),
                os.path.join(run_dir, "helper_functions.py"),
            )
            # Run the optimization
            optimize(
                fil_distance=fil_distance,
                half_per_distance=half_per_distance,
                dipole_radius=dipole_radius,
                numquadpoints=numquadpoints,  # dipole parameters
                VV_a=VV_a + VV_plas_dist,
                VV_b=VV_b + VV_plas_dist,
                VV_R0=VV_R0,  # vessel parameters
                plas_nPhi=plas_nPhi,
                plas_nTheta=plas_nTheta,
                surf_s=surf_s,
                surf_dof_scale=surf_dof_scale,
                eq_dir=eq_dir,
                eq_name=eq_name,  # equilibrium parameters
                ntf=ntf,
                num_fixed=num_fixed,
                field_on_axis=field_on_axis,
                TF_R0=TF_R0,
                TF_a=TF_a,
                TF_b=TF_b,
                fixed_geo_TFs=fixed_geo_TFs,
                CC_THRESHOLD=CC_THRESHOLD,
                CC_WEIGHT=CC_WEIGHT,
                CS_THRESHOLD=CS_THRESHOLD,
                CS_WEIGHT=CS_WEIGHT,  # TF parameters
                definition=definition,
                precomputed=precomputed,
                MAXITER=MAXITER,
                CURRENT_THRESHOLD=CURRENT_THRESHOLD,
                CURRENT_WEIGHT=CURRENT_WEIGHT,
                dpi=dpi,
                titlefontsize=titlefontsize,
                axisfontsize=axisfontsize,
                legendfontsize=legendfontsize,
                ticklabelfontsize=ticklabelfontsize,
                cbarfontsize=cbarfontsize,
                output_dir=run_dir,
                verbose=verbose,
            )

            next_run_number += 1
