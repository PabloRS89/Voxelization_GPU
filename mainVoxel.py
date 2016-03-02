#!/usr/bin/python

import os
import numpy as np
import ctypes
from ctypes import *	

class gadget:
    def __init__(self, file_in):

        #--- Open Gadget file

        file = open(file_in,'rb')

        #--- Read header
        dummy         = file.read(4)
        self.npart         = np.fromfile(file, dtype='i', count=6)
        self.massarr       = np.fromfile(file, dtype='d', count=6)
        self.time          = (np.fromfile(file, dtype='d', count=1))[0]
        self.redshift      = (np.fromfile(file, dtype='d', count=1))[0]
        self.flag_sfr      = (np.fromfile(file, dtype='i', count=1))[0]
        self.flag_feedback = (np.fromfile(file, dtype='i', count=1))[0]
        self.nparttotal    = np.fromfile(file, dtype='i', count=6)
        self.flag_cooling  = (np.fromfile(file, dtype='i', count=1))[0]
        self.NumFiles      = (np.fromfile(file, dtype='i', count=1))[0]
        self.BoxSize       = (np.fromfile(file, dtype='d', count=1))[0]
        self.Omega0        = (np.fromfile(file, dtype='d', count=1))[0]
        self.OmegaLambda   = (np.fromfile(file, dtype='d', count=1))[0]
        self.HubbleParam   = (np.fromfile(file, dtype='d', count=1))[0]
        self.header        = file.read(256-6*4 - 6*8 - 8 - 8 - 2*4-6*4 -4 -4 -4*8)
        dummy         = file.read(4)

        #--- Read positions
        dummy =file.read(4)
        self.pos = np.fromfile(file, dtype='f', count=self.npart[1]*3)
        dummy =file.read(4)
        file.close()        
        self.pos = self.pos.reshape((self.npart[1],3))              
s = gadget('run_600')
#s = gadget('32Mpc_051')
#s = gadget('32Mpc_050.0256.fvol')
#print s.pos

os.system("./build.sh")
# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_sum():
    dll = ctypes.CDLL('libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_sum
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_sum = get_cuda_sum()

def get_test_vox():
    dll = ctypes.CDLL('libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.test_voxelization
    return func

__vox = get_test_vox()

def get_metrics():
    dll = ctypes.CDLL('libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.calc_metrics
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__metrics = get_metrics()


def get_mediana():
    dll = ctypes.CDLL('libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.calc_mediana
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__mediana = get_mediana()

def get_desv():
    dll = ctypes.CDLL('libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.calc_desvEst
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__desv = get_desv()

def get_maxmin():
    dll = ctypes.CDLL('libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.calc_MaxMin
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__maxmin = get_maxmin()
# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones 
def cuda_sum(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __cuda_sum(size, pos)

def voxelization_test():
    __vox()

def cuda_metrics(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __metrics(size, pos)

def cuda_mediana(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __mediana(size, pos)

def cuda_desv(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __desv(size, pos)

def cuda_MaxMin(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __maxmin(size, pos)

# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':
    #size=int(1024*1024)
    size=len(s.pos)*3
    #print size
    #cuda_sum(size, s.pos)
    cuda_metrics(size, s.pos)
    cuda_mediana(size, s.pos)
    cuda_desv(size, s.pos)
    cuda_MaxMin(size, s.pos)    

    #print c[:]    
    #print s.pos.shape
    #print s.pos
    #voxelization_test()