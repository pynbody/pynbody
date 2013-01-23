
### Make plots
try:
	pp.sbprofile(h[i],filename=simname+'.sbprof.png',center=False)
	pp.sfh(h[i],filename=simname+'.sfh.png',nbins=500)
	pp.rotation_curve(h[i],filename=simname+'.rc.png',quick=True,
					  max='40 kpc',center=False)
	pp.rotation_curve(h[i],filename=simname+'.rcparts.png',quick=True,
                          parts=True, legend=True, max='40 kpc',center=False)
	pp.rho_T(h[i],filename=simname+'.phase.png')
	pp.ofefeh(h[i].stars, filename=simname+'.ofefeh.png',
                  weights=h[i].stars['mass'].in_units('Msol'), scalemin=1e3,
                  scalemax=1e6, x_range=[-3,0.3],y_range=[-0.5,1.0])
	pp.mdf(h[i].stars,filename=simname+'.mdf.png', range=[-4,0.5])
	pp.density_profile(h[i].dark,filename=simname+'.dmprof.png',center=False)
	pp.guo(h,baryfrac=True,filename=simname+'.guo.png')
	pp.schmidtlaw(h[i],filename=simname+'.schmidt.png',center=False)
	pp.satlf(h[i],filename=simname+'.satlf.png')
except:
	pass

