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
rvir=np.max(h[i]['r'].in_units('kpc'))
rvir.units=units.Unit('kpc')
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

### Save important numbers using pickle.  Currently not working for SimArrays
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

### Make plots
try:
	pp.sbprofile(h[i],filename=simname+'.sbprof.png',center=False)
except:
	pass
try:
	pp.sfh(h[i],filename=simname+'.sfh.png',nbins=500)
except:
	pass
try:
	pp.rotation_curve(h[i],filename=simname+'.rc.png',quick=True,
					  max='40 kpc',center=False)
except:
	pass
try:
	pp.rotation_curve(h[i],filename=simname+'.rcparts.png',quick=True,
					  parts=True, legend=True, max='40 kpc',center=False)
except:
	pass
try:
	pp.rho_T(h[i],filename=simname+'.phase.png')
except:
	pass
try:
	pp.ofefeh(h[i].stars, filename=simname+'.ofefeh.png',
			  weights=h[i].stars['mass'].in_units('Msol'), scalemin=1e3,
			  scalemax=1e6, x_range=[-3,0.3],y_range=[-0.5,1.0])
except:
	pass
try:
	pp.mdf(h[i].stars,filename=simname+'.mdf.png', range=[-4,0.5])
except:
	pass
try:
	pp.density_profile(h[i].dark,filename=simname+'.dmprof.png',center=False)
except:
	pass
try:
	pp.guo(h,baryfrac=True,filename=simname+'.guo.png')
except:
	pass
try:
	pp.schmidtlaw(h[i],filename=simname+'.schmidt.png',center=False)
except:
	pass
try:
	pp.satlf(h[i],filename=simname+'.satlf.png')
except:
	pass

diskgas=s.gas[diskf]
### Make pictures: not working past first one
try:
    pp.sph.image(h[i].gas,filename=simname+'.facegas.png',width=30)
except:
	pass
try:
    pp.sph.image(h[i].star,filename=simname+'.facestar.png',width=30)
except:
	pass
try:
    pp.sph.image(diskgas,qty='temp',filename=simname+'.tempgasdiskface.png',
                 width=30,vmin=3,vmax=7)
except:
	pass
try:
    s.gas['hiden'] = s.gas['rho']*s.gas['HI']
    s.gas['hiden'].units = s.gas['rho'].units
    pynbody.plot.image(s.gas,qty='hiden',units='m_p cm^-2',width=1000,
                       center=False,filename=simname+'.hi500kpc.png',
                       vmin=14,vmax=22)
except:
	pass
try:
    pynbody.plot.image(s.gas,qty='hiden',units='m_p cm^-2',width=500,
                       center=False,filename=simname+'.hi250kpc.png',
                       vmin=14,vmax=22)
except:
    pass

try:
    oviif = pynbody.analysis.ionfrac.calculate(s.gas)
    s.gas['oviden'] = s.gas['rho']*s.gas['OxMassFrac']*oviif
    s.gas['oviden'].units = s.gas['rho'].units
    soviim = pynbody.plot.image(s.gas[notdiskf],qty='oviden',
                                units='16 m_p cm^-2', width=1000,
                                filename=simname+'.ovi500kpc.png',
                                vmin=12,vmax=17)
except:
	pass
try:
    s.gas['oxden'] = s.gas['rho']*s.gas['OxMassFrac']
    s.gas['oxden'].units = s.gas['rho'].units
    pynbody.plot.image(s.gas,qty='oxden',units='16 m_p cm^-2',
                       width=500,center=False,
                       filename=simname+'.ox500kpc.png',vmin=12,vmax=18)
except:
	pass
try:
    pynbody.analysis.angmom.sideon(h[i])
    pp.sph.image(h[i].gas,filename=simname+'.sidegas.png',width=30)
except:
	pass
try:
    pp.sph.image(h[i].star,filename=simname+'.sidestar.png',width=30)
except:
	pass
try:
    pp.sph.image(s.gas,qty='temp',filename=simname+'.tempgasside.png',
                 width=320,vmin=3,vmax=7)
except:
	pass
try:
    pp.sph.image(s.gas,qty='temp',filename=simname+'.tempgasdiskside.png',
                 width=30,vmin=3,vmax=7)
except:
	pass
try:
    pynbody.plot.image(s.gas,qty='temp',width=500,center=False,
                       filename=simname+'.temp500kpc.png',vmin=3,vmax=7)
except:
    pass
