#import pylab as pl
import numpy as np

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
print s.npart
print s.HubbleParam
print s.pos[:,0:3]


#########################################
#A modificar
#npart[1] * 3
#count=npart[1]*3
#self.pos = self.pos.reshape((npart,3))
#128^3