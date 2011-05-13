import pynbody
import pynbody.plot as pp
import pynbody.filt as filt
import os, glob

files = glob.glob('*.0????')
for tfile in files:
    sim,step = tfile.split('.')
    if not os.path.exists(step):   os.mkdir(step)
    simname = step+'/'+sim+'.'+step
    if not os.path.exists(simname+'.sfh.png'):
        s = pynbody.load(simname)
        h = s.halos()
        try:
            pp.satlf(h[1],filename=simname+'.satlf.png')
            pp.schmidtlaw(h[1],filename=simname+'.schmidt.png')
            pp.sbprofile(h[1],filename=simname+'.sbprof.png')
            pp.sfh(h[1],filename=simname+'.sfh.png')
            pp.rotation_curve(h[1],filename=simname+'.rc.png',quick=True,
                              max='40 kpc')
            pp.rho_T(h[1],filename=simname+'.phase.png')
            pp.ofefeh(h[1][filt.SolarNeighborhood()],
                      filename=simname+'.ofefeh.png',
                      x_range=[-3,0.3],y_range=[-0.5,1.0])
            pp.mdf(h[1][filt.SolarNeighborhood()],filename=simname+'.mdf.png',
                   range=[-4,0.5])
            pp.density_profile(h[1].dark,filename=simname+'.dmprof.png')
        except:
            pass
        
