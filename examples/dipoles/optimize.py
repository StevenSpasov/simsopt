from simsopt.geo import curves_to_vtk, SurfaceRZFourier
from simsopt.field import (
    Current,
    ScaledCurrent,
    Coil,
    apply_symmetries_to_curves,
    apply_symmetries_to_currents,
)
from simsopt.field.force import coil_net_forces, coil_net_torques, coil_force
from simsopt.field.selffield import regularization_circ
from simsopt.objectives import SquaredFlux
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from helper_functions import *


def optimize(
    fil_distance,
    half_per_distance,
    dipole_radius,
    numquadpoints,  # dipole parameters
    VV_a,
    VV_b,
    VV_R0,  # vessel parameters
    plas_nPhi,
    plas_nTheta,
    surf_s,
    surf_dof_scale,
    eq_dir,
    eq_name,  # equilibrium parameters
    ntf,
    num_fixed,
    field_on_axis,
    TF_R0,
    TF_a,
    TF_b,
    fixed_geo_TFs,
    CC_THRESHOLD,
    CC_WEIGHT,
    CS_THRESHOLD,
    CS_WEIGHT,  # TF parameters
    definition,
    precomputed,
    MAXITER,
    CURRENT_THRESHOLD,
    CURRENT_WEIGHT,
    dpi,
    titlefontsize,
    axisfontsize,
    legendfontsize,
    ticklabelfontsize,
    cbarfontsize,
    output_dir,
    verbose=False,
):

    # Create the plasma surface
    eq_name_full = os.path.join(eq_dir, eq_name + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_name_full, s=surf_s, range="half period", nphi=plas_nPhi, ntheta=plas_nTheta
    )
    surf.set_dofs(surf_dof_scale * surf.get_dofs())
    # Create a surface representing the vacuum vessel that dipoles will be placed on
    VV = SurfaceRZFourier(nfp=surf.nfp)
    VV.set_rc(0, 0, VV_R0)
    VV.set_rc(1, 0, VV_a)
    VV.set_zs(1, 0, VV_b)
    plot_cross_section(
        surf, VV, output_dir, axisfontsize, legendfontsize, ticklabelfontsize, dpi
    )

    # Initialize TF Coils
    base_tf_curves, base_tf_currents = generate_tf_array(
        winding_surface=VV,
        ntf=ntf,
        TF_R0=TF_R0,
        TF_a=TF_a,
        TF_b=TF_b,
        fixed_geo_tfs=fixed_geo_TFs,
        numquadpoints=numquadpoints,
    )
    # We define the currents with 1A, so that our dof is order unity. We
    # then scale the current by a scale_factor
    # from toroidal solenoid approximation, I = B_T * 2 * pi * R0 / mu0 / (2 * nfp * n_tf)
    mu0 = 4.0 * np.pi * 1e-7
    scale_factor = (
        2.0 * np.pi * surf.get_rc(0, 0) * field_on_axis / mu0 / (2 * ntf * surf.nfp)
    )
    base_tf_coils = [
        Coil(curve, ScaledCurrent(current, scale_factor))
        for curve, current in zip(base_tf_curves, base_tf_currents)
    ]
    tf_coils = coils_via_symmetries(
    [c.curve for c in base_tf_coils], 
    [c.current for c in base_tf_coils], 
    surf.nfp, True
    )
    bs_tf = BiotSavart(tf_coils)
    _, _, _ = (
        plot_relBfinal_norm_modB(
            bs_tf,
            surf,
            output_dir,
            axisfontsize,
            titlefontsize,
            cbarfontsize,
            ticklabelfontsize,
            dpi,
            "Pre TF Optimization",
        )
    )
    if not fixed_geo_TFs:
        optimize_tfs(
            base_tf_coils=base_tf_coils,
            surf_plasma=surf,
            winding_surface=VV,
            CC_THRESHOLD=CC_THRESHOLD,
            CC_WEIGHT=CC_WEIGHT,
            CS_THRESHOLD=CS_THRESHOLD,
            CS_WEIGHT=CS_WEIGHT,
            num_fixed=num_fixed,
            definition=definition,
            maxiter=MAXITER,
            verbose=verbose,
        )

    _, _, _ = (
        plot_relBfinal_norm_modB(
            bs_tf,
            surf,
            output_dir,
            axisfontsize,
            titlefontsize,
            cbarfontsize,
            ticklabelfontsize,
            dpi,
            "Post TF Optimization",
        )
    )
        
    # Initialize dipoles
    base_wp_curves, base_wp_currents = generate_windowpane_array(
        winding_surface=VV,
        inboard_radius=dipole_radius,
        wp_fil_spacing=fil_distance,
        half_per_spacing=half_per_distance,
        wp_n=4,
        numquadpoints=numquadpoints,
        order=12,
        verbose=verbose,
    )
    nwptot = len(base_wp_curves * 2 * surf.nfp)
    wp_scale_factor = 1e2  # Initialize coils at 100kA (reasonable guess for 1T field)
    base_wp_coils = [
        Coil(curve, ScaledCurrent(current, wp_scale_factor))
        for curve, current in zip(base_wp_curves, base_wp_currents)
    ]

    # Optimize currents
    res, bs = optimize_windowpane_currents(
        base_wp_coils=base_wp_coils,
        base_tf_coils=base_tf_coils,
        surf_plasma=surf,
        definition=definition,
        precomputed=precomputed,
        current_threshold=CURRENT_THRESHOLD,
        current_weight=CURRENT_WEIGHT,
        maxiter=MAXITER,
        num_fixed=num_fixed,
        verbose=verbose,
    )

    # Post-processing
    # get final Bnormal
    relBfinal_norm, final_mean_abs_relBfinal_norm, final_relBfinal_norm_max = (
        plot_relBfinal_norm_modB(
            bs,
            surf,
            output_dir,
            axisfontsize,
            titlefontsize,
            cbarfontsize,
            ticklabelfontsize,
            dpi,
            "Final",
        )
    )
    Jf = SquaredFlux(surf, bs, definition=definition)
    # plots currents on surface
    wp_currents_phis_thetas = coil_currents_on_theta_phi_grid(base_wp_coils, VV)
    plot_coil_currents_on_theta_phi_grid(
        wp_currents_phis_thetas,
        output_dir,
        axisfontsize,
        titlefontsize,
        cbarfontsize,
        ticklabelfontsize,
        dpi,
    )

    # Prep coil data
    tf_coils = coils_via_symmetries(
        [c.curve for c in base_tf_coils], 
        [c.current for c in base_tf_coils], 
        surf.nfp, True
    )
    wp_coils = coils_via_symmetries(
        [c.curve for c in base_wp_coils], 
        [c.current for c in base_wp_coils], 
        surf.nfp, True
    )
    coils = tf_coils + wp_coils

    tf_currents = [c.current.get_value() for c in tf_coils]
    wp_currents = [c.current.get_value() for c in wp_coils]
    
    # Save various files
    VV.to_vtk(os.path.join(output_dir, "vacuum_vessel"))
    a = 0.05 # TF coil filament radius
    a_list = regularization_circ(a) * np.ones(len(tf_coils))
    # note: coil_force gives force per unit length, dF/dl - extract the max, min, and RMS for each coil
    max_tf_forces = [np.max(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1)) for c in tf_coils]
    min_tf_forces = [np.min(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1)) for c in tf_coils]
    RMS_tf_forces = [np.sqrt(np.mean(np.square(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1)))) for c in tf_coils]
    max_tf_torques = [np.max(np.linalg.norm(coil_torque(c, coils, regularization_circ(a)), axis=1)) for c in tf_coils]
    min_tf_torques = [np.min(np.linalg.norm(coil_torque(c, coils, regularization_circ(a)), axis=1)) for c in tf_coils]
    RMS_tf_torques = [np.sqrt(np.mean(np.square(np.linalg.norm(coil_torque(c, coils, regularization_circ(a)), axis=1)))) for c in tf_coils]
    curves_to_vtk(
        curves = [c.curve for c in tf_coils], filename=os.path.join(output_dir, "tf_coils"), close=True,
        I = tf_currents,
        extra_point_data=pointData_forces_torques(tf_coils, a),
        NetForces=coil_net_forces(tf_coils, coils, a_list),
        NetTorques=coil_net_torques(tf_coils, coils, a_list)
    )
    a = 0.025 # WP coil filament radius, gives 5cm spacing in between coils
    a_list = regularization_circ(a) * np.ones(len(wp_coils))
    max_wp_forces = [np.max(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1)) for c in wp_coils]
    min_wp_forces = [np.min(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1)) for c in wp_coils]
    RMS_wp_forces = [np.sqrt(np.mean(np.square(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1)))) for c in wp_coils]
    max_wp_torques = [np.max(np.linalg.norm(coil_torque(c, coils, regularization_circ(a)), axis=1)) for c in wp_coils]
    min_wp_torques = [np.min(np.linalg.norm(coil_torque(c, coils, regularization_circ(a)), axis=1)) for c in wp_coils]
    RMS_wp_torques = [np.sqrt(np.mean(np.square(np.linalg.norm(coil_torque(c, coils, regularization_circ(a)), axis=1)))) for c in wp_coils]
    curves_to_vtk(
        curves = [c.curve for c in wp_coils], filename=os.path.join(output_dir, "wp_coils"), close=True,
        I = wp_currents,
        extra_point_data=pointData_forces_torques(wp_coils, a),
        NetForces=coil_net_forces(wp_coils, coils, a_list),
        NetTorques=coil_net_torques(wp_coils, coils, a_list)
    )
    bs.save(os.path.join(output_dir, "bs_opt.json"))
    # Save the BdotN on the full torus surface
    surf_full = SurfaceRZFourier.from_wout(
        eq_name_full,
        s=surf_s,
        range="full torus",
        nphi=2 * surf.nfp * plas_nPhi,
        ntheta=plas_nTheta,
    )
    bs.set_points(surf_full.gamma().reshape(-1, 3))
    Bdotn = np.sum(
        bs.B().reshape(surf_full.unitnormal().shape) * surf_full.unitnormal(), axis=2
    )
    modB = bs.AbsB().reshape((2 * surf.nfp * plas_nPhi, plas_nTheta))
    BdotN_norm = Bdotn / modB
    surf_full.to_vtk(
        os.path.join(output_dir, "surf_full"),
        extra_data={"B_N": BdotN_norm[:, :, None]},
    )
    bs.set_points(
        surf.gamma().reshape(-1, 3)
    )  # have to set points back for Jf.J in results section

    # Extract TF optimized geometric dofs
    if not fixed_geo_TFs:
        R0s = np.zeros_like(base_tf_curves)
        r_rotations = np.zeros_like(base_tf_curves)
        for i, c in enumerate(base_tf_curves):
            R0s[i] = c.get("R0")
            r_rotations[i] = c.get("r_rotation")
    else:
        R0s = None
        r_rotations = None

    results = {
        # input parameters
        "filament_distance": fil_distance,
        "half_period_distance": half_per_distance,
        "inboard_radius": dipole_radius,
        "numquadpoints": numquadpoints,
        "VV_a": VV_a,
        "VV_b": VV_b,
        "VV_R0": VV_R0,
        "plas_nPhi": plas_nPhi,
        "plas_nTheta": plas_nTheta,
        "surf_s": surf_s,
        "surf_dof_scale": surf_dof_scale,
        "eq_dir": eq_dir,
        "eq_name": eq_name,
        "ntf": ntf,
        "num_fixed": num_fixed,
        "TF_R0": TF_R0,
        "TF_a": TF_a,
        "TF_b": TF_b,
        "fixed_geo_TFs": fixed_geo_TFs,
        "CC_THRESHOLD": CC_THRESHOLD,
        "CC_WEIGHT": CC_WEIGHT,
        "CS_THRESHOLD": CS_THRESHOLD,
        "CS_WEIGHT": CS_WEIGHT,
        "field_on_axis": field_on_axis,
        "squared_flux_def": definition,
        "max_iterations": MAXITER,
        "current_threshold": CURRENT_THRESHOLD,
        "current_weight": CURRENT_WEIGHT,
        # derived quantities
        "surf_nfp": surf.nfp,
        "surf_major_radius": surf.major_radius(),
        "surf_minor_radius": surf.minor_radius(),
        "surf_aspect_ratio": surf.aspect_ratio(),
        "surf_volume": surf.volume(),
        "initial_tf_current": scale_factor,
        "initial_wp_current": wp_scale_factor,
        "num_wps": nwptot,
        "ntoroidal": int((np.pi/surf.nfp*(VV_R0 - VV_a) - half_per_distance + fil_distance) / (2 * dipole_radius + fil_distance)),
        "npoloidal": int(len(base_wp_curves) / int((np.pi/surf.nfp*(VV_R0 - VV_a) - half_per_distance + fil_distance) / (2 * dipole_radius + fil_distance))),
        # optimization results
        "message":                  res.message,
        "success":                  res.success,
        "iterations":               res.nit,
        "function_evaluations":     res.nfev,
        "max_tf_current": np.max(np.abs(np.array(tf_currents))),
        "min_tf_current": np.min(np.abs(np.array(tf_currents))),
        "max_wp_current": np.max(np.abs(np.array(wp_currents))),
        "min_wp_current": np.min(np.abs(np.array(wp_currents))),
        "tf_max_max_force":            max(float(f) for f in max_tf_forces),
        "tf_min_min_force":            min(float(f) for f in min_tf_forces),
        "tf_mean_RMS_force":            float(np.mean([f for f in RMS_tf_forces])),
        "wp_max_max_force":            max(float(f) for f in max_wp_forces),
        "wp_min_min_force":            min(float(f) for f in min_wp_forces),
        "wp_mean_RMS_force":            float(np.mean([f for f in RMS_wp_forces])),
        "tf_max_max_torque":            max(float(f) for f in max_tf_torques),
        "tf_min_min_torque":            min(float(f) for f in min_tf_torques),
        "tf_mean_RMS_torque":            float(np.mean([f for f in RMS_tf_torques])),
        "wp_max_max_torque":            max(float(f) for f in max_wp_torques),
        "wp_min_min_torque":            min(float(f) for f in min_wp_torques),
        "wp_mean_RMS_torque":            float(np.mean([f for f in RMS_wp_torques])),
        "final_squared_flux": Jf.J(),
        "avg_Bnormal": final_mean_abs_relBfinal_norm,
        "max_Bnormal": final_relBfinal_norm_max,
        "peak_wp_field": np.max(np.abs(np.array(wp_currents))) * mu0 / 2 / dipole_radius,
        "MA_meters": get_total_amp_meters(base_tf_coils, base_wp_coils, VV) / 1e6,
        "maxR0": np.max(R0s) if R0s is not None else None,
        "minR0": np.min(R0s) if R0s is not None else None,
        "avgR0": np.mean(R0s) if R0s is not None else None,
        "maxtilt": np.max(r_rotations) if r_rotations is not None else None,
        "mintilt": np.min(r_rotations) if r_rotations is not None else None,
        "avgtilt": np.mean(np.abs(r_rotations)) if r_rotations is not None else None,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)
