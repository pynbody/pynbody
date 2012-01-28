import numpy as np
import pynbody
import pynbody.plot as pp
import pynbody.plot.sph
import pynbody.filt as filt
import pynbody.units as units
import pynbody.analysis.profile as profile
import sys, os, glob, pickle

tfile = sys.argv[1]
sim,step = tfile.split('.')
if not os.path.exists(step):   os.mkdir(step)
simname = step+'/'+sim+'.'+step
if not os.path.exists(simname): os.system('ln -s ../'+tfile+' '+simname)
if not os.path.exists(simname+'.OxMassFrac'): 
    os.system('ln -s ../'+tfile+'.OxMassFrac '+simname+'.OxMassFrac')
if not os.path.exists(simname+'.FeMassFrac'): 
    os.system('ln -s ../'+tfile+'.FeMassFrac '+simname+'.FeMassFrac')
if not os.path.exists(simname+'.HI'): 
    os.system('ln -s ../'+tfile+'.HI '+simname+'.HI')

s = pynbody.load(simname)
h = s.halos()
pynbody.analysis.angmom.faceon(h[1])
s.physical_units()
Jtot = np.sum(((h[1]['j']**2).sum(axis=1))**(1,2) * h[1]['mass'])
W = np.sum(h[1]['phi']*h[1]['mass'])
K = np.sum(h[1]['ke']*h[1]['mass'])
absE = np.fabs(W+K)
mvir=np.sum(h[1]['mass'].in_units('Msol'))
rvir=np.max(h[1]['r'].in_units('kpc'))
# 3D density profile
rhoprof = profile.Profile(h[1],dim=3,type='log')
# Rotation curve
rcpro = profile.Profile(h[1], type='equaln', nbins = 50, max = '40 kpc')
# surface brightness profile
diskstars = h[1].star[filt.Disc('20 kpc','3 kpc')]
sbprof = profile.Profile(diskstars,type='equaln')
# Kinematic decomposition
#decompprof = pynbody.analysis.decomp(h[1])
#dec = h[1].star['decomp']

### Save important numbers using pickle.  Currently not working for SimArrays
pickle.dump({'z':s.properties['z'],
             'time':s.properties['time'].in_units('Gyr'),
             'rvir':rvir,
             'mvir':mvir,
             'mgas': np.sum(h[1].gas['mass'].in_units('Msol')),
             'mstar': np.sum(h[1].star['mass'].in_units('Msol')),
             #'mdisk': np.sum(h[1].star[np.where(dec == 1)]['mass'].in_units('Msol')),
             #'msphere': np.sum(h[1].star[np.where(dec == 2)]['mass'].in_units('Msol')),
             #'mbulge': np.sum(h[1].star[np.where(dec == 3)]['mass'].in_units('Msol')),
             #'mthick': np.sum(h[1].star[np.where(dec == 4)]['mass'].in_units('Msol')),
             #'mpseudob': np.sum(h[1].star[np.where(dec == 5)]['mass'].in_units('Msol')),
             'mgashot': np.sum(h[1].gas[filt.HighPass('temp',1e5)]['mass'].in_units('Msol')),
             'mgascool': np.sum(h[1].gas[filt.LowPass('temp',1e5)]['mass'].in_units('Msol')),
             'Jtot':Jtot,'lambda':Jtot / np.sqrt(5.0/3.0*mvir**3 * rvir),
             'denprof':{'r':rhoprof['rbins'].in_units('kpc'), 
                        'den':rhoprof['density']},
             'rotcur':{'r':rcpro['rbins'].in_units('kpc'), 
                       'vc':rcpro['rotation_curve_spherical'].in_units('km s^-1'),
                       'fourier':rcpro['fourier']},
             'sb':{'r':sbprof['rbins'].in_units('kpc'), 
                   'sb':sbprof['sb,i']}
             },
            open(simname+'.data','w'))#, pickle.HIGHEST_PROTOCOL)

### Make plots
try:
    pp.sbprofile(h[1],filename=simname+'.sbprof.png',center=False)
    pp.sfh(h[1],filename=simname+'.sfh.png',nbins=500)
    pp.rotation_curve(h[1],filename=simname+'.rc.png',quick=True,
                      max='40 kpc',center=False)
    pp.rotation_curve(h[1],filename=simname+'.rcparts.png',quick=True,
                      parts=True, legend=True, max='40 kpc',center=False)
    pp.rho_T(h[1],filename=simname+'.phase.png')
    pp.ofefeh(h[1].stars, filename=simname+'.ofefeh.png',
              x_range=[-3,0.3],y_range=[-0.5,1.0])
    pp.mdf(h[1].stars,filename=simname+'.mdf.png', range=[-4,0.5])
    pp.density_profile(h[1].dark,filename=simname+'.dmprof.png',center=False)
    pp.guo(h,baryfrac=True,filename=simname+'.guo.png')
    pp.schmidtlaw(h[1],filename=simname+'.schmidt.png',center=False)
    pp.satlf(h[1],filename=simname+'.satlf.png')
except:
    pass

### Make pictures: not working past first one
try:
    pp.sph.sideon_image(h[1].gas,filename=simname+'.sidegas.png',width=30)
    pp.sph.image(h[1].star,filename=simname+'.sidestar.png',width=30)
    pp.sph.faceon_image(h[1].gas,filename=simname+'.facegas.png',width=30)
    pp.sph.image(h[1].star,filename=simname+'.facestar.png',width=30)
except:
    pass

try:
    s.gas['oxden'] = s.gas['rho']*s.gas['OxMassFrac']
    s.gas['oxden'].units = s.gas['rho'].units
    pynbody.plot.sideon_image(s.gas,qty='oxden',units='16 m_p cm^-2',width=500,center=False,filename=simname+'.ox500kpc.png',vmin=12,vmax=18)
    pp.sph.image(h[1].gas,qty='temp',filename=simname+'.tempgasside.png',
                 width=320,vmin=3,vmax=7)
    pynbody.plot.sideon_image(s.gas,qty='temp',width=500,center=False,filename=simname+'.temp500kpc.png',vmin=3,vmax=7)
    s.gas['hiden'] = s.gas['rho']*s.gas['HI']
    s.gas['hiden'].units = s.gas['rho'].units
    pynbody.plot.sideon_image(s.gas,qty='hiden',units='m_p cm^-2',width=1000,center=False,filename=simname+'.hi500kpc.png',vmin=14,vmax=22)
    pynbody.plot.sideon_image(s.gas,qty='hiden',units='m_p cm^-2',width=500,center=False,filename=simname+'.hi250kpc.png',vmin=14,vmax=22)
except:
    pass

os.system('rm '+simname)
os.system('rm '+simname+'.OxMassFrac')
os.system('rm '+simname+'.FeMassFrac')
os.system('rm '+simname+'.HI')
if os.path.exists(simname+'.log'): 
    os.system('rm '+simname+'.log')
