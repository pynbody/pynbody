import matplotlib.pylab as plt
import pynbody;

s = pynbody.load('testdata/tutorial_gadget/snapshot_020');
s.physical_units()
s.properties['sigma8'] = 0.8288
my_cosmology = pynbody.analysis.hmf.PowerSpectrumCAMB(s, filename='../pynbody/analysis/CAMB_WMAP7')
m, sig, dn_dlogm = pynbody.analysis.hmf.halo_mass_function(s, log_M_min=8, log_M_max=15, delta_log_M=0.1, kern="ST", pspec=my_cosmology)
bin_center, bin_counts, err = pynbody.analysis.hmf.simulation_halo_mass_function(s, log_M_min=10, log_M_max=15, delta_log_M=0.1)


plt.figure()
plt.errorbar(bin_center, bin_counts, yerr=err, fmt='o', capthick=2, elinewidth=2, color='darkgoldenrod')
plt.plot(m, dn_dlogm, color='darkmagenta', linewidth=2)
plt.ylabel(r'$\frac{dN}{d\logM}$ ($h^{3}Mpc^{-3}$)')
plt.xlabel('Mass ($h^{-1} M_{\odot}$)')
plt.yscale('log', nonposy='clip');
plt.xscale('log')
plt.show()