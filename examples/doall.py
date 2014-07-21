import numpy as np
import pynbody
import pynbody.plot as pp
import pynbody.plot.sph
import pynbody.filt as filt
import pynbody.units as units
import pynbody.analysis.profile as profile
import sys, os, glob, pickle, pylab as plt, struct

def find_sfh(h,bins=100):
    trange = [h.star['tform'].in_units("Gyr").min(),h.star['tform'].in_units("Gyr").max()]
    binnorm = 1e-9*bins / (trange[1] - trange[0])
    tforms = h.star['tform'].in_units('Gyr')
    try:
        weight = h.star['massform'].in_units('Msol') * binnorm
    except:
        weight = h.star['mass'].in_units('Msol') * binnorm
    sfh,sfhbines = np.histogram(tforms, weights=weight, bins=bins)
    sfhtimes = 0.5*(sfhbines[1:]+sfhbines[:-1])
    return sfh,sfhtimes

def mdf(s,quant='oxh',bins=100,totmass=False,range=(-5,0.5)):
    if not totmass: totmass = np.sum(s['mass'].in_units('Msol'))
    hist, binedges = np.histogram(s[quant],
                                  weights=s['mass'].in_units('Msol')/totmass,
                                  bins=bins, range=range)
    bins = 0.5*(binedges[:-1]+binedges[1:])
    return hist,bins

def write(s,filename='dumbfile.out'):
    f = pynbody.util.open_(filename,'wb')
    f.write(struct.pack(">i", len(s)))
    s.tofile(f)
    f.close()

simname = sys.argv[1]
#pp.plt.ion()

s = pynbody.load(simname)
h = s.halos()
diskf = filt.Disc('40 kpc','3 kpc')
fifmyrf = filt.LowPass('age','15 Myr')
twokpcf = filt.Sphere('2 kpc')
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
try:
    s.g['tconv'] = 4.0 * s.g['smooth'] / np.sqrt(s.g['u']+s.g['uNoncool'])
    s.g['tconv'].units = s.s['tform'].units
except:
    pass
s.physical_units()
#diskf = filt.Disc(np.max(h[i].s[diskf]['r']),np.max(h[i].s[diskf]['z']))
notdiskf = filt.Not(diskf)
notcooldenf = filt.And(filt.LowPass('rho','0.1 m_p cm^-3'),
                       filt.HighPass('temp',3e4))
hot=filt.HighPass('temp',3e4)
cold=filt.LowPass('temp',3e4)
Jtot = np.sqrt(((np.multiply(h[i]['j'].transpose(),h[i]['mass']).sum(axis=1))**2).sum())
W = np.sum(h[i]['phi']*h[i]['mass'])
K = np.sum(h[i]['ke']*h[i]['mass'])
absE = np.fabs(W+K)
mvir=np.sum(h[i]['mass'].in_units('Msol'))
r = s['r']
rvir=pynbody.array.SimArray(np.max(h[i]['r'].in_units('kpc')),'kpc')
mdiskgas=np.sum([h[i][diskf].g['mass'].in_units('Msol')]),
mtwokpcgas=np.sum([h[i][twokpcf].g['mass'].in_units('Msol')]),
mtwokpcstar=np.sum([h[i][twokpcf].s['mass'].in_units('Msol')]),
mhalogas=np.sum([h[i][notdiskf].g['mass'].in_units('Msol')]),
# 3D density profile
rhoprof = profile.Profile(h[i].dm,ndim=3,type='log')
gashaloprof = profile.Profile(h[i].g,ndim=3,type='log')
spheregasprof = profile.Profile(s.g,ndim=3,type='equaln',max=2*rvir,bins=200)
spherecoldgasprof = profile.Profile(s.g[cold],ndim=3,type='equaln',max=2*rvir,bins=200)
spherehotgasprof = profile.Profile(s.g[hot],ndim=3,type='equaln',max=2*rvir,bins=200)
spherestarprof = profile.Profile(s.s,ndim=3,type='equaln',max=2*rvir,bins=200)
spheredarkprof = profile.Profile(s.dm,ndim=3,type='equaln',max=2*rvir,bins=200)
# Rotation curve
rcpro = profile.Profile(h[i], type='equaln', nbins = 200, max = '40 kpc')
# surface brightness profile
diskstars = h[i].star[diskf]
sbprof = profile.Profile(diskstars,type='equaln',bins=200)
surfcoldgasprof = profile.Profile(h[i].g[diskf][cold],type='equaln',bins=200)
surfhotgasprof = profile.Profile(h[i].g[diskf][hot],type='equaln',bins=200)
surfgasprof = profile.Profile(h[i].g[diskf],type='equaln',bins=200)
# Kinematic decomposition
#decompprof = pynbody.analysis.decomp(h[i])
#dec = h[i].star['decomp']
sfh,sfhtimes = find_sfh(h[i],bins=100)
hrsfh,hrsfhtimes = find_sfh(h[i],bins=600)
sfr = np.sum(h[i].star[fifmyrf]['mass'].in_units('Msol')) / 1.5e7

# where did halo come from?  SN or virial shocks
# look at all particles inside 2.5 r_vir
twopfivef = pynbody.filt.Sphere(2.5*rvir)
stfrvir = s[twopfivef]
halogas = stfrvir[notdiskf].g[notcooldenf]
mtfhalogas = np.sum(halogas['mass'].in_units('Msol'))
iposcoolontime = halogas['coolontime'].in_units('yr')>0
igasords = np.in1d(halogas['iord'],stfrvir.s['igasorder'])
halooxhist, halooxbins = mdf(halogas,totmass=mtfhalogas)
cohalooxhist, cohalooxbins = mdf(halogas[iposcoolontime],totmass=mtfhalogas)

### Save important numbers using pickle.  Currently not working for SimArrays
pickle.dump({'z':s.properties['z'],
             'time':s.properties['time'].in_units('Gyr'),
             'rvir':rvir,
             'mvir':mvir,
             'mgas': np.sum(h[i].gas['mass'].in_units('Msol')),
             'mstar': np.sum(h[i].star['mass'].in_units('Msol')),
             'sfr': sfr,
#             'mdisk': np.sum(h[i].star[np.where(dec == 1)]['mass'].in_units('Msol')),
#             'msphere': np.sum(h[i].star[np.where(dec == 2)]['mass'].in_units('Msol')),
#             'mbulge': np.sum(h[i].star[np.where(dec == 3)]['mass'].in_units('Msol')),
#             'mthick': np.sum(h[i].star[np.where(dec == 4)]['mass'].in_units('Msol')),
#             'mpseudob': np.sum(h[i].star[np.where(dec == 5)]['mass'].in_units('Msol')),
             'mgashot': np.sum(h[i].gas[hot]['mass'].in_units('Msol')),
             'mgascool': np.sum(h[i].gas[cool]['mass'].in_units('Msol')),
             'mdiskbary': np.sum(h[i][diskf].s['mass'].in_units('Msol')) +
                          np.sum(h[i][diskf].g['mass'].in_units('Msol')),
             'mdiskgas': mdiskgas,
             'diskmeanmet': np.sum(h[i][diskf].g['mass'].in_units('Msol')*
                                   h[i][diskf].g['metals'])/mdiskgas,
             'mdiskstar': np.sum([h[i][diskf].s['mass'].in_units('Msol')]),
             'mdiskcoolgas': np.sum([h[i][diskf].g[filt.LowPass('temp',1e5)]['mass'].in_units('Msol')]),
             'mtwokpcgas': mtwokpcgas,
             'mtwokpcstar': mtwokpcstar,
             'mtwokpcbary': mtwokpcgas+mtwokpcstar,
             'twokpcmeanmet': np.sum(h[i][twokpcf].g['mass'].in_units('Msol')*
                                   h[i][twokpcf].g['metals'])/mtwokpcgas,
             'mtwokpccoolgas': np.sum([h[i][twokpcf].g[filt.LowPass('temp',1e5)]['mass'].in_units('Msol')]),
             'mhalogas': mhalogas,
             'mhalohotgas': np.sum([h[i][notdiskf].g[filt.HighPass('temp',1e5)]['mass'].in_units('Msol')]),
             'halomeanmet': np.sum(h[i][notdiskf].g['mass'].in_units('Msol')*
                                   h[i][notdiskf].g['metals'])/mhalogas,
             'mhalogasformedstar':np.sum(halogas[igasords]['mass'].in_units('Msol')),
             'mhalogascoolwasoff':np.sum(halogas[iposcoolontime]['mass'].in_units('Msol')),
             'rmaxcooloffgas':np.max(halogas[iposcoolontime]['r'].in_units('kpc')),
             'halooxmdf':{'bins':halooxbins,'hist':halooxhist},
             'cohalooxmdf':{'bins':cohalooxbins,'hist':cohalooxhist},
#             'Jtot':Jtot,'lambda':(Jtot / np.sqrt(2*np.power(mvir,3)*rvir*units.G)).in_units('1'),
             'denprof':{'r':rhoprof['rbins'].in_units('kpc'), 
                        'den':rhoprof['density']},
             'gashaloprof':{'r':gashaloprof['rbins'].in_units('kpc'), 
                            'den':gashaloprof['density'].in_units('m_p cm^-3'),
                            'temp':gashaloprof['temp']},
             'rotcur':{'r':rcpro['rbins'].in_units('kpc'), 
                       'vc':rcpro['rotation_curve_spherical'].in_units('km s^-1'),
                       'fourier':rcpro['fourier']},
             'surfstar':{'r':sbprof['rbins'].in_units('kpc'), 
                   'sb':sbprof['sb,b'],'sv':sbprof['sb,v'],'si':sbprof['sb,i'],
                   'den':sbprof['density'],'mass':sbprof['mass']},
             'surfcoldgas':{'r':surfcoldgasprof['rbins'].in_units('kpc'), 
                        'den':surfcoldgasprof['density'],
                        'mass':surfcoldgasprof['mass']},
             'surfhotgas':{'r':surfhotgasprof['rbins'].in_units('kpc'), 
                       'den':surfhotgasprof['density'],
                       'mass':surfhotgasprof['mass']},
             'surfgas':{'r':surfgasprof['rbins'].in_units('kpc'), 
                       'den':surfgasprof['density'],
                       'mass':surfgasprof['mass']},
             'spherestar':{'r':spherestarprof['rbins'].in_units('kpc'), 
                   'den':spherestarprof['density'],'mass':spherestarprof['mass']},
             'spherecoldgas':{'r':spherecoldgasprof['rbins'].in_units('kpc'), 
                        'den':spherecoldgasprof['density'],
                        'mass':spherecoldgasprof['mass']},
             'spherehotgas':{'r':spherehotgasprof['rbins'].in_units('kpc'), 
                       'den':spherehotgasprof['density'],
                       'mass':spherehotgasprof['mass']},
             'spheregas':{'r':spheregasprof['rbins'].in_units('kpc'), 
                       'den':spheregasprof['density'],
                       'mass':spheregasprof['mass']},
             'sfh':{'sfh':sfh,'t':sfhtimes},
             'hrsfh':{'sfh':hrsfh,'t':hrsfhtimes}
             },
            open(simname+'.data','w'))#, pickle.HIGHEST_PROTOCOL)
write(h[i][diskf].g['iord'],filename=simname+'.disk.iord')
write(s.g[filt.LowPass('temp',30000)][filt.HighPass('rho','0.1 m_p cm^-3')]['iord'],filename=simname+'.cooldense.iord')
write(halogas['iord'],filename=simname+'.halo.iord')

### Make plots
try:
    s.g['effTemp'] = (s.g['u']+s.g['uNoncool']) / s.g['u'] * s.g['temp']
#    pynbody.plot.generic.hist2d(s.g['rho'].in_units('m_p cm^-3'),s.g['u']+s.g['uNoncool'],xlogrange=True,ylogrange=True)
    pynbody.plot.generic.hist2d(s.g['rho'].in_units('m_p cm^-3'),s.g['effTemp'],xlogrange=True,ylogrange=True)
    pp.plt.xlabel('log(n [cm$^{-3}$])')
    pp.plt.ylabel('log(T$_{eff}$ [K])')
#    pp.plt.ylabel('log(u [sys. units])')
    pp.plt.savefig(simname+'.uncoolphase.png')
    pp.plt.clf()
    unc = s.g[(s.g['uNoncool'] / (s.g['uNoncool']+s.g['u'])) > 0.1]
#    pynbody.plot.generic.hist2d(s.g['rho'].in_units('m_p cm^-3'),s.g['tconv'].in_units('yr'),xlogrange=True,ylogrange=True)
    pynbody.plot.generic.hist2d(unc['smooth'].in_units('pc'),unc['tconv'].in_units('yr'),xlogrange=True,ylogrange=True)
    pp.plt.xlabel('log(h [pc])')
    pp.plt.ylabel('log(t$_{conv}$ [yr])')
#    pp.plt.ylabel('log(u [sys. units])')
    pp.plt.savefig(simname+'.tconv.png')
    pp.plt.clf()
    unc['uDotFB'][unc['uDotFB'] < 1e-4] = 1e-4
    pp.plt.loglog(unc['tconv'].in_units('yr'),unc['uDotFB'],'.')
#    pynbody.plot.generic.hist2d(unc['tconv'].in_units('yr'),unc['uDotFB'],xlogrange=True,ylogrange=True)
    pp.plt.ylabel(r'log($\dot{u}_{FB}$ [system units])')
    pp.plt.xlabel('log(t$_{conv}$ [yr])')
    pp.plt.savefig(simname+'.uDotFBtconv.png')
    pp.plt.clf()
    pynbody.plot.generic.hist2d(unc['effTemp'],unc['uDotFB'],xlogrange=True,ylogrange=True)
    pp.plt.ylabel(r'log($\dot{u}_{FB}$ [system units])')
    pp.plt.xlabel('log(T$_{eff}$ [K])')
    pp.plt.savefig(simname+'.uDotFBeffTemp.png')
    pp.plt.clf()
except:
    pass

try:
    pp.sbprofile(h[i],filename=simname+'.sbprof.png',center=False)
    pp.sfh(h[i],filename=simname+'.sfh.png',nbins=500,clear=True)
    pp.rotation_curve(h[i],filename=simname+'.rc.png',quick=True,
                      max='40 kpc',center=False)
    pp.rotation_curve(h[i],filename=simname+'.rcparts.png',quick=True,
                      parts=True, legend=True, max='40 kpc',center=False)
    pp.rho_T(h[i].gas,filename=simname+'.phase.png',rho_units='m_p cm^-3',
             x_range=[-5,2], y_range=[3,8])
    pp.ofefeh(h[i].stars, filename=simname+'.ofefeh.png',
              weights=h[i].stars['mass'].in_units('Msol'), scalemin=1e3,
              scalemax=1e9, x_range=[-3,0.3],y_range=[-0.5,1.0])
    pp.mdf(h[i].stars,filename=simname+'.mdf.png', range=[-4,0.5])
    pp.density_profile(h[i].dark,filename=simname+'.dmprof.png',center=False)
    plt.clf()
    plt.loglog(gashaloprof['rbins'].in_units('kpc'),
               gashaloprof['density'].in_units('m_p cm^-3'))
    plt.xlabel('r [kpc]')
    plt.ylabel(r'$\rho$ [m$_p$ cm$^{-3}$]')
    plt.ylim(1e-4,1e3)
    plt.xlim(0.5,500)
    plt.savefig(simname+'.gasdenprof.png')
    plt.clf()
    plt.loglog(gashaloprof['rbins'].in_units('kpc'),
               gashaloprof['temp'])
    plt.ylim(1e5,1e8)
    plt.xlim(0.5,500)
    plt.xlabel('r [kpc]')
    plt.ylabel('T [K]')
    plt.savefig(simname+'.gastempprof.png')
    plt.clf()
    pp.guo(h,baryfrac=True,filename=simname+'.guo.png')
    pp.schmidtlaw(h[i],filename=simname+'.schmidt.png',center=False)
    pp.satlf(h[i],filename=simname+'.satlf.png')
except:
    pass


diskgas=s.gas[diskf]
### Make pictures: not working past first one
try:
    pp.sph.image(h[i].gas,filename=simname+'.facegas.png',width=30,
                 units='m_p cm^-3')
    pp.stars.render(h[i].star,file=simname+'.facestar.png')
    pp.sph.image(diskgas,qty='temp',filename=simname+'.tempgasdiskface.png',
                 width=30,vmin=3,vmax=7)
    s.gas['hiden'] = s.gas['rho']*s.gas['HI']
    s.gas['hiden'].units = s.gas['rho'].units
    pynbody.plot.image(s.gas,qty='hiden',units='m_p cm^-2',width=1000,
                       center=False,filename=simname+'.hi500kpc.png',
                       vmin=14,vmax=22)
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
    s.gas['oxden'] = s.gas['rho']*s.gas['OxMassFrac']
    s.gas['oxden'].units = s.gas['rho'].units
    pynbody.plot.image(s.gas,qty='oxden',units='16 m_p cm^-2',
                       width=500,center=False,
                       filename=simname+'.ox500kpc.png',vmin=12,vmax=18)
    pynbody.analysis.angmom.sideon(h[i])
    pp.sph.image(h[i].gas,filename=simname+'.sidegas.png',width=30,
                 units='m_p cm^-3')
    pp.stars.render(h[i].star,file=simname+'.sidestar.png')
    pp.sph.image(s.gas,qty='temp',filename=simname+'.tempgasside.png',
                 width=320,vmin=3,vmax=7)
    pp.sph.image(s.gas,qty='temp',filename=simname+'.tempgasdiskside.png',
                 width=30,vmin=3,vmax=7)
    pynbody.plot.image(s.gas,qty='temp',width=500,center=False,
                       filename=simname+'.temp500kpc.png',vmin=3,vmax=7)
    pynbody.plot.image(s.gas,qty='hiden',units='m_p cm^-2',width=200,
                       center=False,filename=simname+'.sidehi100kpc.png',
                       vmin=14,vmax=22)
except:
    pass
