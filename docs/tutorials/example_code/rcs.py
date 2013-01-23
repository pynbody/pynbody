import pynbody, sys
import matplotlib.pyplot as plt
import pickle, glob, numpy as np

runs = ['c.1td.05rp.1','mugs','esn1.2','td.01c.1rp.1','rp.175','noradpres','2crasy']
labels = ['Fiducial','MUGS','120% SN energy','Low Diffusion','High Diffusion','No ESF',
          'High ESF']
cs = ['r','k','c','g','y','m','b']
for j,r in enumerate(runs):
    simname=r+'/01024/g1536.01024'

    d = pickle.load(open(simname+'.data'))

    if j>0:
        plt.plot(d['rotcur']['r'],d['rotcur']['vc'],color=cs[j], 
                 lw=1, label=labels[j],alpha=0.5)
        if j==6:
            rad = d['rotcur']['r']
            plt.plot(rad,(2.0/np.pi)*np.max(d['rotcur']['vc'])*np.arctan(rad),
                     '--b',label='arctangent fit')
    else:
        plt.plot(d['rotcur']['r'],d['rotcur']['vc'],color=cs[j], 
                 lw=2, label=labels[j])
        rad = d['rotcur']['r']
        plt.plot(rad,(2.0/np.pi)*np.max(d['rotcur']['vc'])*np.arctan(4*rad),
                 '--r',label='arctangent fit')

plt.savefig('rc.eps')


