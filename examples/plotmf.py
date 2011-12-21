import pynbody,sys, glob
import matplotlib.pyplot as plt
import numpy as np

tfile = sys.argv[1]

s = pynbody.load(tfile, only_header=True)
hfile = glob.glob(tfile+'*_halos')
masses = np.loadtxt(hfile[0],usecols=[8],unpack=True)
mhist, mbin_edges = np.histogram(np.log10(masses),bins=20)
mbinmps = np.zeros(len(mhist))
mbinsize = np.zeros(len(mhist))
for i in np.arange(len(mhist)):
    mbinmps[i] = np.mean([mbin_edges[i],mbin_edges[i+1]])
    mbinsize[i] = mbin_edges[i+1] - mbin_edges[i]

stms, stsig, stmf = pynbody.analysis.halo_mass_function(s)

plt.semilogy(np.log10(stms),stmf,label="Sheth-Tormen")
plt.semilogy(mbinmps,mhist/(80.0**3)/mbinsize,'o')
plt.xlabel('log$_{10}$(M)')
plt.ylabel('dN / dlog$_{10}$(M)')
