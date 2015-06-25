# cython: profile=False

"""
optimized cython pair counters.  These are called by "rect_cuboid_pairs" module as the 
engine to actually calculate the pair-wise distances and do the binning.  These functions 
should be used with care as there are no 'checks' preformed to ensure the arguments are 
of the correct format.
"""

from __future__ import print_function, division
import sys
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmin

__all__ = ['xy_z_wnpairs_no_pbc']
__author__=['Duncan Campbell']



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_wnpairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                        np.ndarray[np.float64_t, ndim=1] y_icell1,
                        np.ndarray[np.float64_t, ndim=1] z_icell1,
                        np.ndarray[np.float64_t, ndim=1] x_icell2,
                        np.ndarray[np.float64_t, ndim=1] y_icell2,
                        np.ndarray[np.float64_t, ndim=1] z_icell2,
                        np.ndarray[np.float64_t, ndim=1] w_icell1,
                        np.ndarray[np.float64_t, ndim=1] w_icell2,
                        np.ndarray[np.float64_t, ndim=1] rp_bins,
                        np.ndarray[np.float64_t, ndim=1] pi_bins):
    """
    2+1D pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with perpendicular separations less than or equal 
    to rp_bins[i], and parallel separations less than or equal to pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins) -1
    cdef int npi_bins_minus_one = len(pi_bins) -1
    cdef np.ndarray[np.float64_t, ndim=2] counts =\
        np.zeros((nrp_bins, npi_bins), dtype=np.float64)
    cdef double d_perp, d_para
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    cdef double sx, sy, sz, lx, ly, lz, s2, l2, sl, sl2, l
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
            
            sx = (x_icell1[i] - x_icell2[j])
            sy = (y_icell1[i] - y_icell2[j])
            sz = (z_icell1[i] - z_icell2[j])
            lx = 0.5*(x_icell1[i] + x_icell2[j])
            ly = 0.5*(y_icell1[i] + y_icell2[j])
            lz = 0.5*(z_icell1[i] + z_icell2[j])
            s2 = sx*sx + sy*sy + sz*sz
            l2 = lx*lx + ly*ly + lz*lz
            sl = sx*lx + sy*ly + sz*lz
            sl2 = sl*sl
            
            if sl>0:
                with cython.cdivision(True):
                    d_para = (sl2)/l2
                    d_perp = s2 - (sl2)/l2
            else :
                d_para = 0.0
                d_perp = s2
                        
            #calculate counts in bins
            xy_z_wbinning(<np.float64_t*>counts.data,\
                          <np.float64_t*>rp_bins.data,\
                          <np.float64_t*>pi_bins.data,\
                          d_perp, d_para, nrp_bins_minus_one, npi_bins_minus_one,\
                          w_icell1[i], w_icell2[j])
    
    return counts


cdef inline xy_z_wbinning(np.float64_t* counts, np.float64_t* rp_bins,\
                          np.float64_t* pi_bins, np.float64_t d_perp,\
                          np.float64_t d_para, np.int_t k,\
                          np.int_t npi_bins_minus_one, np.float64_t w1, np.float64_t w2):
    """
    2D+1 weighted binning function
    """
    cdef int g
    cdef int max_k = npi_bins_minus_one+1
    
    while d_perp<=rp_bins[k]:
        g = npi_bins_minus_one
        while d_para<=pi_bins[g]:
            #counts[k,g] += w1*w2
            counts[k*max_k+g] += w1*w2
            g=g-1
            if g<0: break
        k=k-1
        if k<0: break
        
        
