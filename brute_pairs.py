# -*- coding: utf-8 -*-

"""
Brute force weighted parallel and perpendicular pair counter.

Duncan Campbell, Yale June 2015
"""

from __future__ import print_function, division
import numpy as np
from cpairs import *
from time import time
import sys
import multiprocessing
from astropy.io import ascii
from functools import partial
try: 
    from mpi4py import MPI
    mpi4py_installed=True
except ImportError:
    print("mpi4py module not available.  MPI functionality will not work.")
    mpi4py_installed=False


__all__=['pp_wnpairs']
__author__=['Duncan Campbell']

def main():
    """
    main function can be run from the command line, e.g.
    mpirun -np 4 python mpipairs.py output.dat input1.dat input2.dat, rpbins.dat, pibins.dat
    """
    
    if mpi4py_installed==True:
        comm = MPI.COMM_WORLD
        rank = comm.rank
    else: 
        comm = None
        rank = 0
    
    if len(sys.argv)==4:
        savename = sys.argv[1]
        filename_1 = sys.argv[2]
        filename_2 = sys.argv[3]
        if rank==0:
            print("Running code with user supplied files. Saving output as:", savename)
    elif rank==0:
        raise ValueError("User needs to provide input.")
        sys.exit()
    else:
        sys.exit()
    
    #read in data
    names = ['x','y','z','w','j']
    data1 = np.loadtxt(filename_1, dtype=None)
    if filename_1==filename_2: data2=data1
    else: data2 = np.loadtxt(filename_2, dtype=None)
    
    weights1 = data1[:,3]
    weights2 = data2[:,3]
    jweights1 = data1[:,4]
    jweights2 = data2[:,4]
    data1 = data1[:,:3]
    data2 = data2[:,:3]
    
    data1 = data1[0:100000]
    data2 = data2[0:100000]
    weights1 = weights1[0:100000]
    weights2 = weights2[0:100000]
    
    rp_bins = np.logspace(-1,1,10)
    pi_bins = np.linspace(0,50,10)
    
    counts = pp_wnpairs(data1, data2, rp_bins, pi_bins, weights1, weights2,\
                        verbose=False, comm=comm)
    
    #save result
    if rank==0:
        print(counts)
        #np.save(savename,counts)
    

def pp_wnpairs(data1, data2, rp_bins, pi_bins, weights1=None, weights2=None,\
               verbose=False, comm=None):
    """
    weighted perp/para pair counter.
    
    Count the weighted number of pairs (x1,x2) that can be formed, with x1 drawn from 
    data1 and x2 drawn from data2, and where distance(x1, x2) <= rp_bins & pi_bins. 
    Weighted counts are calculated as w1*w2.
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rp_bins: array_like
        numpy array of boundaries defining the radial projected bins in which pairs are 
        counted.
    
    pi_bins: array_like
        numpy array of boundaries defining the parallel bins in which pairs are counted. 
    
    weights1: array_like, optional
        length N1 array containing weights used for weighted pair counts
        
    weights2: array_like, optional
        length N2 array containing weights used for weighted pair counts.
    
    verbose: Boolean, optional
        If True, print out information and progress.
    
    comm: mpi Intracommunicator object, optional
        
    Returns
    -------
    N_pairs : array of length len(rp_bins)xlen(pi_bins)
        number counts of pairs
    """
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    rp_bins = np.array(rp_bins)
    pi_bins = np.array(pi_bins)
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (N,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (N,3)")
    if rp_bins.ndim != 1:
        raise ValueError("rp_bins must be a 1D array")
    if pi_bins.ndim != 1:
        raise ValueError("pi_bins must be a 1D array")
    
    #Process weights1 entry and check for consistency.
    if weights1 is None:
            weights1 = np.array([1.0]*np.shape(data1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(data1)[0]:
            raise ValueError("weights1 should have same len as data1")
    #Process weights2 entry and check for consistency.
    if weights2 is None:
            weights2 = np.array([1.0]*np.shape(data2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(data2)[0]:
            raise ValueError("weights2 should have same len as data2")
    
    #get the data into the right shape
    x1 = np.ascontiguousarray(data1[:,0],dtype=np.float64)
    y1 = np.ascontiguousarray(data1[:,1],dtype=np.float64)
    z1 = np.ascontiguousarray(data1[:,2],dtype=np.float64)
    x2 = np.ascontiguousarray(data2[:,0],dtype=np.float64)
    y2 = np.ascontiguousarray(data2[:,1],dtype=np.float64)
    z2 = np.ascontiguousarray(data2[:,2],dtype=np.float64)
    
    #sort the weights arrays
    weights1 = np.ascontiguousarray(weights1,dtype=np.float64)
    weights2 = np.ascontiguousarray(weights2,dtype=np.float64)
    
    #square radial bins to make distance calculation cheaper
    rp_bins = rp_bins**2.0
    pi_bins = pi_bins**2.0
    
    #define the indices
    N1 = len(data1)
    N2 = len(data2)
    inds1 = np.arange(0,N1)
    inds2 = np.arange(0,N2)
    
    #split up indices for each subprocess
    sendbuf_1=[] #need these as place holders till each process get its list
    sendbuf_2=[]
    if rank==0:
        chunks = np.array_split(inds1,size) #evenly split up the indices
        sendbuf_1 = chunks
    
    if comm!=None:
        #send out lists of indices for each subprocess to use
        inds1=comm.scatter(sendbuf_1,root=0)
    
    #do the counting
    #create a function to call with only one argument
    engine = partial(_xy_z_wnpairs_engine, x1, y1, z1, x2, y2, z2, weights1, weights2,\
                     rp_bins, pi_bins)
    counts = engine(inds1)
    
    if comm==None:
        return counts
    else:
        #gather results from each subprocess
        counts = comm.gather(counts,root=0)
        
    if (rank==0) & (comm!=None):
        #combine counts from subprocesses
        counts=np.sum(counts, axis=0)
    
    #receive result from rank 0
    counts = comm.bcast(counts, root=0)
    
    return counts


def _xy_z_wnpairs_engine(x1,y1,z1,x2,y2,z2, weights1, weights2, rp_bins, pi_bins, chunk):
    
    counts = np.zeros((len(rp_bins),len(pi_bins)))
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (x1[chunk], y1[chunk], z1[chunk])
        
    #extract the weights in the cell
    w_icell1 = weights1[chunk]
        
    counts = xy_z_wnpairs_no_pbc(x_icell1, y_icell1, z_icell1,\
                                 x2, y2, z2, w_icell1, weights2,\
                                 rp_bins, pi_bins)
    
    return counts


if __name__ == '__main__':
    main()
