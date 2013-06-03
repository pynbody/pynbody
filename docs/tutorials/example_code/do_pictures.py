diskgas=s.gas[diskf]
### Make pictures: 
try:
    pp.sph.image(h[i].gas,filename=simname+'.facegas.png',width=30)
    pp.sph.image(h[i].star,filename=simname+'.facestar.png',width=30)
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

    # Turn galaxy for side-on pictures
    pynbody.analysis.angmom.sideon(h[i])
    pp.sph.image(h[i].gas,filename=simname+'.sidegas.png',width=30)
    pp.sph.image(h[i].star,filename=simname+'.sidestar.png',width=30)
    pp.sph.image(s.gas,qty='temp',filename=simname+'.tempgasside.png',
                 width=320,vmin=3,vmax=7)
    pp.sph.image(s.gas,qty='temp',filename=simname+'.tempgasdiskside.png',
                 width=30,vmin=3,vmax=7)
    pynbody.plot.image(s.gas,qty='temp',width=500,center=False,
                       filename=simname+'.temp500kpc.png',vmin=3,vmax=7)
except:
    pass
