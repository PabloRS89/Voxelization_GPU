#!/usr/bin/python
import os
import numpy as np
import ctypes
from ctypes import *	
import sys
import scipy.misc
import matplotlib.pyplot as plt
from struct import unpack

file = str(sys.argv[1])
class gadget:
    def __init__(self, file_in):
        #--- Open Gadget file
        file = open(file_in,'rb')
        #--- Read header
        dummy = file.read(4)                
        self.npart                     = np.fromfile(file, dtype='i', count=6)
        self.massarr                = np.fromfile(file, dtype='d', count=6)
        self.time                      = (np.fromfile(file, dtype='d', count=1))[0]
        self.redshift                 = (np.fromfile(file, dtype='d', count=1))[0]
        self.flag_sfr                  = (np.fromfile(file, dtype='i', count=1))[0]
        self.flag_feedback       = (np.fromfile(file, dtype='i', count=1))[0]
        self.nparttotal              = np.fromfile(file, dtype='i', count=6)
        self.flag_cooling          = (np.fromfile(file, dtype='i', count=1))[0]
        self.NumFiles               = (np.fromfile(file, dtype='i', count=1))[0]
        self.BoxSize                  = (np.fromfile(file, dtype='d', count=1))[0]
        self.Omega0                 = (np.fromfile(file, dtype='d', count=1))[0]
        self.OmegaLambda      = (np.fromfile(file, dtype='d', count=1))[0]
        self.HubbleParam         = (np.fromfile(file, dtype='d', count=1))[0]
        self.header                    = file.read(256-6*4 - 6*8 - 8 - 8 - 2*4-6*4 -4 -4 -4*8)
        dummy = file.read(4)
        #--- Read positions
        c = (self.npart[0]+self.npart[1]+self.npart[2]+self.npart[3]+self.npart[4]+self.npart[5])
        print "c: " + str(c)
        dummy = file.read(4)

        #self.pos = np.fromfile(file, dtype='f', count=self.npart[0]*3)
        self.pos = np.fromfile(file, dtype='f', count=c*3)        
        
        file.close()

        #self.pos = self.pos.reshape((self.npart[0],3))
        self.pos = self.pos.reshape((c,3))        
       
s = gadget(file)
os.chdir(".")
os.system("make")

def get_avg():
    dll = ctypes.CDLL('./libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.calc_avg
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__average = get_avg()

def get_medium():
    dll = ctypes.CDLL('./libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.calc_medium
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__medium = get_medium()

def get_stdev():
    dll = ctypes.CDLL('./libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.calc_StDev
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__sd = get_stdev()

def get_maxmin():
    dll = ctypes.CDLL('./libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.calc_MaxMin
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__maxmin = get_maxmin()

def get_fft():
    dll = ctypes.CDLL('./libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.calc_FFT
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__fft = get_fft()

def test_voxelization():
    dll = ctypes.CDLL('./libr3d.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.test_voxelization    
    return func

__vox = test_voxelization()

# convenient python wrapper
# it does all job with types convertation
# from python ones to C++ ones 
def voxelization_test():
    __vox()

def cuda_average(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __average(size, pos)

def cuda_medium(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __medium(size, pos)

def cuda_stdev(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __sd(size, pos)

def cuda_MaxMin(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __maxmin(size, pos)

def cuda_FFT(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __fft(size, pos)

if __name__ == '__main__':    
    size=len(s.pos)
    '''
    cuda_average(size, s.pos)
    cuda_medium(size, s.pos)
    cuda_stdev(size, s.pos)    
    cuda_MaxMin(size, s.pos)    
    cuda_FFT(size, s.pos)

    print "\n"
    print s.npart
    print "\n"
    print s.pos
    '''
    voxelization_test()