import numpy as np
import pynbody
import pynbody.plot as pp
import pynbody.plot.sph
import pynbody.filt as filt
import pynbody.units as units
import pynbody.analysis.profile as profile
import sys, os, glob, pickle

simname = sys.argv[1]   # use first argument as simulation name to analyze
pp.plt.ion()            # plot in interactive mode to show all plots as 
                        #     they are made

s = pynbody.load(simname) # load file into pynbody
h = s.halos()             # find and load halos (using AHF if in path)
diskf = filt.Disc('40 kpc','2 kpc')  # filter for particles in disk
notdiskf = filt.Not(filt.Disc('40 kpc','3 kpc')) # particles outside of disk
i=1
if (len(sys.argv) > 2):
    # if there's a second argument, it can be a photogenic file
    # and analysis will follow the halo that contains those particles
    photiords = np.genfromtxt(sys.argv[2],dtype='i8')
    frac = np.float(len(np.where(np.in1d(photiords,h[i]['iord']))[0]))/len(photiords)
    print 'i: %d frac: %.2f'%(i,frac)
    while(((frac) < 0.5) & (i<100)): 
        i=i+1
        frac = np.float(len(np.where(np.in1d(photiords,h[i]['iord']))[0]))/len(photiords)
        print 'i: %d frac: %.2f'%(i,frac)
else:
    # otherwise follow largest halo with at least 2 stars
    while len(h[i].star) <2: i=i+1

if (i==100): sys.exit()
pynbody.analysis.angmom.faceon(h[i])  # align halo faceon
s.physical_units()   # change to physical units
