from simsopt.geo.surfacerzfourier import SurfaceRZFourier

# these functions can act as shorthand for equilibrium imports

def create_surface_from_fit(eq_name_full, surf_range, plas_nPhi, plas_nTheta, surf_s, dof_scale, R0):
    surf_nfp1 = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range=surf_range, nphi=plas_nPhi, ntheta=plas_nTheta)
    surf_plas = SurfaceRZFourier(mpol=surf_nfp1.mpol,ntor=surf_nfp1.ntor,nfp=2,stellsym=True,
                                    quadpoints_theta=surf_nfp1.quadpoints_theta,
                                    quadpoints_phi=surf_nfp1.quadpoints_phi)
    surf_plas.least_squares_fit(surf_nfp1.gamma())
    
    surf_plas.set_dofs(dof_scale*surf_plas.get_dofs())
    #surf_plas.set_rc(0,0,R0)
    return surf_plas

def create_surface(eq_name_full, surf_range, plas_nPhi, plas_nTheta, surf_s, dof_scale):
    surf_plas = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range=surf_range, nphi=plas_nPhi, ntheta=plas_nTheta)
    surf_plas.set_dofs(dof_scale*surf_plas.get_dofs())
    return surf_plas

def create_axisym_wf_surface(quad_phi, quad_theta, nfp, VV_R0, VV_a):
    surf_wf = SurfaceRZFourier(quadpoints_phi=quad_phi, quadpoints_theta=quad_theta, nfp = nfp)
    surf_wf.set_rc(0,0,VV_R0)
    surf_wf.set_rc(1,0,VV_a)
    surf_wf.set_zs(1,0,VV_a)
    return surf_wf
