import numpy as np
import pynbody
import pynbody.plot as pp
import pynbody.plot.sph
import pynbody.filt as filt
import pynbody.units as units
import pynbody.analysis.profile as profile
import sys, os, glob, pickle

simname = sys.argv[1]
pp.plt.ion()

s = pynbody.load(simname)
h = s.halos()
diskf = filt.Disc('40 kpc','2 kpc')
notdiskf = filt.Not(filt.Disc('40 kpc','3 kpc'))
i=1
if (len(sys.argv) > 2):
    photiords = np.genfromtxt(sys.argv[2],dtype='i8')
    frac = np.float(len(np.where(np.in1d(photiords,h[i]['iord']))[0]))/len(photiords)
    print 'i: %d frac: %.2f'%(i,frac)
    while(((frac) < 0.5) & (i<100)): 
        i=i+1
        frac = np.float(len(np.where(np.in1d(photiords,h[i]['iord']))[0]))/len(photiords)
        print 'i: %d frac: %.2f'%(i,frac)
else:
    while len(h[i].star) <2: i=i+1

if (i==100): sys.exit()
pynbody.analysis.angmom.faceon(h[i])
s.physical_units()
Jtot = np.sqrt(((np.multiply(h[i]['j'].transpose(),h[i]['mass']).sum(axis=1))**2).sum())
W = np.sum(h[i]['phi']*h[i]['mass'])
K = np.sum(h[i]['ke']*h[i]['mass'])
absE = np.fabs(W+K)
mvir=np.sum(h[i]['mass'].in_units('Msol'))
rvir=pynbody.array.SimArray(np.max(h[i]['r'].in_units('kpc')),'kpc')
# 3D density profile
rhoprof = profile.Profile(h[i],dim=3,type='log')
# Rotation curve
rcpro = profile.Profile(h[i], type='equaln', nbins = 50, max = '40 kpc')
# surface brightness profile
diskstars = h[i].star[filt.Disc('20 kpc','3 kpc')]
sbprof = profile.Profile(diskstars,type='equaln')
# Kinematic decomposition
#decompprof = pynbody.analysis.decomp(h[i])
#dec = h[i].star['decomp']

### Save important numbers using pickle.  
pickle.dump({'z':s.properties['z'],
             'time':s.properties['time'].in_units('Gyr'),
             'rvir':rvir,
             'mvir':mvir,
             'mgas': np.sum(h[i].gas['mass'].in_units('Msol')),
             'mstar': np.sum(h[i].star['mass'].in_units('Msol')),
#             'mdisk': np.sum(h[i].star[np.where(dec == 1)]['mass'].in_units('Msol')),
#             'msphere': np.sum(h[i].star[np.where(dec == 2)]['mass'].in_units('Msol')),
#             'mbulge': np.sum(h[i].star[np.where(dec == 3)]['mass'].in_units('Msol')),
#             'mthick': np.sum(h[i].star[np.where(dec == 4)]['mass'].in_units('Msol')),
#             'mpseudob': np.sum(h[i].star[np.where(dec == 5)]['mass'].in_units('Msol')),
             'mgashot': np.sum(h[i].gas[filt.HighPass('temp',1e5)]['mass'].in_units('Msol')),
             'mgascool': np.sum(h[i].gas[filt.LowPass('temp',1e5)]['mass'].in_units('Msol')),
             'Jtot':Jtot,'lambda':(Jtot / np.sqrt(2*np.power(mvir,3)*rvir*units.G)).in_units('1'),
             'denprof':{'r':rhoprof['rbins'].in_units('kpc'), 
                        'den':rhoprof['density']},
             'rotcur':{'r':rcpro['rbins'].in_units('kpc'), 
                       'vc':rcpro['rotation_curve_spherical'].in_units('km s^-1'),
                       'fourier':rcpro['fourier']},
             'sb':{'r':sbprof['rbins'].in_units('kpc'), 
                   'sb':sbprof['sb,i']}
             },
            open(simname+'.data','w'))#, pickle.HIGHEST_PROTOCOL)

