## About this repository
This repository is set up with the essential scripts necessary for optimizing windowpane (dipole) coils. 
The majority of optimization scripts are directly related to the coil optimization for Columbia's
on campus hybrid experiment, (i.e. axisymmetric shaping coil sets with no TF geometry optimization); 
however, I included an example as used in the windowpane study for Antoine Baillod's port optimization
paper - see windowpanes_on_generalized_surf.py. 

## How to run the code
A lot of the base functionality of these scripts can be fun from the master simsopt branch (commenting out the functionality that isn't available), but to get the most use out of them the following sections of code must be available (as it is in this branch): (an ongoing list)

- pointwise/net forces and torques (currently in a [pull request](https://github.com/hiddenSymmetries/simsopt/pull/509) from Alan Kaptanoglu as of 9/10/25)
- dgammadtheta and dgammadphi functions (available in the main windowpane branch of simsopt)
- CurvePlanarEllipticalCylindrical Class, used for TFs that pivot about a specific point (see curveplanarellipticalcylindrical.py)

The main focus of these scripts is how to use the geometric dofs, i.e. the coil center and rotation quaternion, to orient planar coils on a given winding surface and then optimize their currents.

I have been having difficulties getting simsopt to compile with the standard script [here](https://github.com/hiddenSymmetries/simsopt/wiki/Mac-Conda-script-installation), so I have included a conda environment yml file in this folder. The basic form of the install instructions for just simsopt would then look like
- create the conda environment directly from environment.yml
- clone my fork of simsopt and checkout the [windowpane branch](https://github.com/jhalpern30/simsopt/tree/windowpane)
- run 'pip install -e .'

## Notes
As of today (9/10/25), this was a quick reorganization of adding my other repository containing these scripts into this branch, so a lot of the directory/file IO management in the scripts will need to be adjusted, but for now, this process should get the code running.

This branch is not up to date with the simsopt master branch (since it is a bit of a mess of wireframe + coil force things that haven't been integrated). For now this should be fine to run, but once the coil forces from the above-mentioned PR are added to Simsopt, this can either be merged or it might be simpler to check out a new branch from master and either merge with or directly add in the dgammadtheta/phi calculations in `surfacerzfourier.py` and add in the curveplanarellipticalcylindrical class file (which definitely needs a better name).