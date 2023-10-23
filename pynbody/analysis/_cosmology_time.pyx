cimport numpy as np

import numpy as np

cimport cython
from libc.math cimport sqrt


cdef double dadtau(double aexp_tau,double O_mat_0,double O_vac_0,double O_k_0):
    cdef double aexp_tau3 = aexp_tau * aexp_tau * aexp_tau
    return sqrt( aexp_tau3 * (O_mat_0 + O_vac_0*aexp_tau3 + O_k_0*aexp_tau) )

@cython.cdivision(True)
cdef double dadt(double aexp_t,double O_mat_0,double O_vac_0,double O_k_0):
    cdef double aexp_t3 = aexp_t * aexp_t * aexp_t
    return sqrt( (1./aexp_t)*(O_mat_0 + O_vac_0*aexp_t3 + O_k_0*aexp_t) )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef step_cosmo(double alpha,double tau,double aexp_tau,double t,double aexp_t,double O_mat_0,double O_vac_0,double O_k_0):
    cdef double dtau,aexp_tau_pre,dt,aexp_t_pre

    cdef double dadtau_tmp = dadtau(aexp_tau,O_mat_0,O_vac_0,O_k_0)

    dtau = alpha * aexp_tau / dadtau_tmp
    aexp_tau_pre = aexp_tau - dadtau_tmp*dtau/2.0
    aexp_tau = aexp_tau - dadtau(aexp_tau_pre,O_mat_0,O_vac_0,O_k_0)*dtau
    tau = tau - dtau

    cdef double dadt_tmp = dadt(aexp_t,O_mat_0,O_vac_0,O_k_0)

    dt = alpha * aexp_t / dadt_tmp
    aexp_t_pre = aexp_t - dadt_tmp*dt/2.0
    aexp_t = aexp_t - dadt(aexp_t_pre,O_mat_0,O_vac_0,O_k_0)*dt
    t = t - dt


    return tau,aexp_tau,t,aexp_t


cpdef friedman(double O_mat_0,double O_vac_0,double O_k_0):
    cdef double alpha=1.e-6,aexp_min=1.e-3,aexp_tau=1.,aexp_t=1.,tau=0.,t=0.
    cdef int nstep=0,ntable=1000,n_out
    cdef np.ndarray[double,mode='c'] t_out=np.zeros([ntable+1])
    cdef np.ndarray[double,mode='c'] tau_out=np.zeros([ntable+1])
    cdef np.ndarray[double,mode='c'] axp_out=np.zeros([ntable+1])
    cdef np.ndarray[double,mode='c'] hexp_out=np.zeros([ntable+1])
    cdef double age_tot,delta_tau,next_tau

    while aexp_tau >= aexp_min or aexp_t >= aexp_min:
       nstep = nstep + 1
       tau,aexp_tau,t,aexp_t = step_cosmo(alpha,tau,aexp_tau,t,aexp_t,O_mat_0,O_vac_0,O_k_0)

    age_tot=-t
    if nstep < ntable :
        ntable = nstep
        alpha = alpha / 2.

    cdef int nskip = nstep // ntable

    aexp_tau = 1.
    aexp_t = 1.
    tau = 0.
    t = 0.

    n_out = 0
    t_out[n_out] = t
    tau_out[n_out] = tau
    axp_out[n_out] = aexp_tau
    hexp_out[n_out] = dadtau(aexp_tau,O_mat_0,O_vac_0,O_k_0) / aexp_tau

    nstep = 0
    while aexp_tau >= aexp_min or aexp_t >= aexp_min:
       nstep = nstep + 1
       tau,aexp_tau,t,aexp_t = step_cosmo(alpha,tau,aexp_tau,t,aexp_t,O_mat_0,O_vac_0,O_k_0)

       if (nstep % nskip) == 0:
            n_out += 1
            t_out[n_out] = t
            tau_out[n_out] = tau
            axp_out[n_out] = aexp_t
            hexp_out[n_out] = dadt(aexp_t,O_mat_0,O_vac_0,O_k_0) / aexp_tau


    n_out = ntable
    t_out[n_out] = t
    tau_out[n_out] = tau
    axp_out[n_out] = aexp_tau
    hexp_out[n_out] = dadtau(aexp_tau,O_mat_0,O_vac_0,O_k_0) / aexp_tau

    return tau_out,t_out,axp_out,hexp_out,ntable,age_tot
