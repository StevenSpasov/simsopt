from simsopt._core.optimizable import load
from simsopt.field import InterpolatedField
from simsopt.field import SurfaceClassifier, \
    compute_fieldlines, LevelsetStoppingCriterion#, plot_poincare_data
import numpy as np
from simsopt.field import MaxZStoppingCriterion, MinZStoppingCriterion, MaxRStoppingCriterion, MinRStoppingCriterion
from simsopt.geo import SurfaceRZFourier
from matplotlib.lines import Line2D


def plot_poincare_data(fieldlines_phi_hits, phis, filename, mark_lost=False, aspect='equal', dpi=600, xlims=None,
                       ylims=None, surf=None, surf1=None, s=2, marker='o'):
    """
    Create a poincare plot. Usage:

    .. code-block::

        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')

    Requires matplotlib to be installed.

    """
    import matplotlib.pyplot as plt
    from math import ceil, sqrt
    nrowcol = ceil(sqrt(len(phis)))
    plt.figure()
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    for ax in axs.ravel():
        ax.set_aspect(aspect)
    color = None
    for i in range(len(phis)):
        row = i//nrowcol
        col = i % nrowcol
        if i != len(phis) - 1:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.2f}\\pi$ ", loc='left', y=0.0)
        else:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.2f}\\pi$ ", loc='right', y=0.0)
        if row == nrowcol - 1:
            axs[row, col].set_xlabel("$r$")
        if col == 0:
            axs[row, col].set_ylabel("$z$")
        if col == 1:
            axs[row, col].set_yticklabels([])
        if xlims is not None:
            axs[row, col].set_xlim(xlims)
        if ylims is not None:
            axs[row, col].set_ylim(ylims)
        for j in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            if mark_lost:
                color = 'r' if lost else 'g'
            data_this_phi = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == i)[0], :]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2)
            axs[row, col].scatter(r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color)

        plt.rc('axes', axisbelow=True)
        axs[row, col].grid(True, linewidth=0.5)

        # if passed a surface, plot the plasma surface outline
        if surf is not None:
            #phi_new = phis[i] * 1 / (2 * np.pi)
            phi_new = phis[i]
            cross_section = surf.cross_section(phi=phi_new)
            phi_new = phis[i]
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c='k')
        if surf1 is not None:
            #phi_new = phis[i] * 1 / (2 * np.pi)
            phi_new = phis[i]
            cross_section = surf1.cross_section(phi=phi_new)
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c='r')

    # Create dummy lines for legend
    '''legend_lines = [
        Line2D([0], [0], color='k', linewidth=1, label='Initialized Boozer Surface')#,
        Line2D([0], [0], color='r', linewidth=1, label='Stage 1 Surface')
    ]
    #fig.legend(handles=legend_lines, loc='upper left', ncol=2, fontsize=10)
    fig.legend(handles=legend_lines, loc='upper left', ncol=1, fontsize=10)'''

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


nfieldlines = 50 # Number of field lines for integration 
tmax_fl = 7000 # Maximum toroidal angle for integration
tol = 1e-7 # Tolerance for field line integration
interpolate = True # If True, then the BiotSavart magnetic field is interpolated 
                   # on a grid for the magnetic field evaluation
nr = 20 # Number of radial points for interpolation
nphi = 10 # Number of toroidal angle points for interpolation
nz = 10 # Number of vertical points for interpolation
degree = 3 # Degree for interpolation

#bs = load(f'outputs-wout_nfp22ginsburg_000_014417_iota15.nc/MR=0.925-TF=0.24-LW=0.0005-CCW=100-CW=0.0001-SR=0.22-Order=2/biot_savart_opt.json')
#bs = load(f'outputs-nfp=4/MR=0.925-TF=0.24-LW=1e-05-CCW=100-CW=0.0001-SR=0.22-Order=2/biot_savart_opt.json')

#OUT_DIR = f'scans2/adaptive/mpol=10'
#OUT_DIR = f'scans/adaptive/mpol-ntor_convergence-iota15-vol115/mpol=16-ntor=6'
OUT_DIR = f'./scans/current200kA_tfunfixed_iota0.1_vol0.4/mpol=5'
bs = load(OUT_DIR + f'/biot_savart_init.json')

#surf = load('test/surf_init.json')
surf = load(OUT_DIR + f'/surf_init.json')
# Extend surface, since we want to look at field lines beyond it
#surf_extended = load('test/surf_init.json')
surf_extended = load(OUT_DIR + f'/surf_init.json')
surf_extended.extend_via_normal(0.05)

# Use extended surface to determine initial conditions
gamma = surf_extended.gamma()
R = np.sqrt(gamma[:,:,0]**2 + gamma[:,:,1]**2)
Z = gamma[:,:,2]

nfp =  surf.nfp

Rmin = np.min(R)
Rmax = np.max(R)
Zmax = np.max(Z)

# The parameter h sets the grid size for the classifier, 
# and p is the order. These parameters are not too critical
# to the Poincare calculation. 
#sc_fieldline = SurfaceClassifier(surf_extended, h=0.02, p=2)
stop_crit = [MaxZStoppingCriterion(Zmax*1.05), MinZStoppingCriterion(-Zmax*1.05), MinRStoppingCriterion(Rmin*0.95), MaxRStoppingCriterion(Rmax*1.05)]



#filename = 'wout_nfp22ginsburg_000_014417_iota15.nc'
#filename = 'wout_nfp22ginsburg_000_000281.nc'
#surf_new = SurfaceRZFourier.from_wout(filename, range="half period", nphi=128, ntheta=64)


# The parameter h sets the grid size for the classifier, 
# and p is the order. These parameters are not too critical
# to the Poincare calculation. 
#sc_fieldline = SurfaceClassifier(surf_extended, h=0.02, p=2)
#stop_crit_new = [MaxZStoppingCriterion(Zmax*1.05), MinZStoppingCriterion(-Zmax*1.05), MinRStoppingCriterion(Rmin*0.95), MaxRStoppingCriterion(Rmax*1.05)]


def trace_fieldlines(bfield):
    # Set up initial conditions 
    R0 = np.linspace(Rmin, Rmax, nfieldlines)
    Z0 = np.zeros((nfieldlines,))
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=tol,
        phis=phis, stopping_criteria=stop_crit)
    '''fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=tol,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])'''
    plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'/PoincarePlot.png', dpi=600, surf=surf, mark_lost=False)
    return fieldlines_phi_hits

rrange = (Rmin, Rmax, nr)
phirange = (0, 2*np.pi/nfp, nphi)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, Zmax, nz)

if interpolate:
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
    )

    bsh.set_points(surf.gamma().reshape((-1, 3)))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    print("Maximum field interpolation error: ", np.max(np.abs(B-Bh)))
else:
    bsh = bs

hits = trace_fieldlines(bsh)


