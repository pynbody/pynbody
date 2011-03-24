import numpy as np
import os

def calc_mags(sim) :
    # find data file in PYTHONPATH
    # data is from http://stev.oapd.inaf.it/cgi-bin/cmd
    # Padova group stellar populations Marigo et al (2008), Girardi et al (2010)
    for directory in os.environ["PYTHONPATH"].split(os.pathsep) :
        lumfile = os.path.join(directory,"pynbody/analysis/cmdlum.npz")
        if os.path.exists(lumfile) :
            # import data
            print "Loading luminosity data"
            lums=np.load(lumfile)
            break
    #calculate star age
    age_star=(sim.properties['time'].in_units('yr', **sim.conversion_context())-sim.star['tform'].in_units('yr'))
    # allocate temporary metals that we can play with
    metals = sim.star['metals']
    # get values off grid to minmax
    age_star[np.where(age_star < np.min(lums['ages']))] = np.min(lums['ages'])
    age_star[np.where(age_star > np.max(lums['ages']))] = np.max(lums['ages'])
    metals[np.where(metals < np.min(lums['mets']))] = np.min(lums['mets'])
    metals[np.where(metals > np.max(lums['mets']))] = np.max(lums['mets'])
    #interpolate
    import scipy.interpolate
    interp_u = scipy.interpolate.interp2d(lums['mets'],np.log10(lums['ages']), lums['u']) 
    print "made interp_u"
    #
    max_block_size=1024
    n_left = len(metals)
    n_done = 0
    while n_left>0 :
        n_block = min(n_left,max_block_size)
        newu = np.zeros(n_block)
        import pdb; pdb.set_trace()
        newu = interp_u(metals[n_done:n_done+n_block],np.log10(age_star[n_done:n_done+n_block])) 
        sim.star['u_mag'][n_done:n_done+n_block] = newu.diagonal() - 2.5*np.log10(sim.star['massform'][n_done:n_done+n_block].in_units('Msol'))
        n_left-=n_block
        n_done+=n_block
