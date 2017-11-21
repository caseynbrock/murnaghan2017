#!/usr/bin/env python
import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import murnaghan2017 as m

# read E and v data from file
energy_volume_data = np.loadtxt('energies.dat')
a    = energy_volume_data[:,0]
b    = energy_volume_data[:,1]
c    = energy_volume_data[:,2]
vols = energy_volume_data[:,3]
E    = energy_volume_data[:,4]

# read murnaghan equation paramters from file
with open('murnaghan_parameters.dat') as printed_results:
    for line in printed_results:
        if 'E_0:' in line:
            E0 = float(line.split()[-1])
        if 'B_0 (bulk modulus):' in line:
            B0 = float(line.split()[-1])
        if 'B_0p:' in line:
            B0p = float(line.split()[-1])
        if 'V_0:' in line:
            V0 = float(line.split()[-1])
    

# plot E vs volume
plt.figure()
plt.plot(vols, E, 'ro', label='calculated energies')
vfit = np.linspace(vols[0], vols[-1], 200)
plt.plot(vfit, m.murnaghan_equation([E0, B0, B0p, V0], vfit), 
         label='Murnaghan EOS fit')
plt.xlabel('volume (bohr^3)')
plt.ylabel('Energy (Ha)')
plt.legend(loc='best')

# plot E vs a 
plt.figure()
plt.plot(a, E, 'ro', label='calculated energies')
afit = m.abc_of_vol(vfit, vols[0], [a[0],b[0],c[0]])[0,:] # needed because 
print afit.shape, len(vfit)
plt.plot(afit, m.murnaghan_equation([E0, B0, B0p, V0], vfit), 
         label='Murnaghan EOS fit')
plt.xlabel('scale a (bohr)')
plt.ylabel('Energy (Ha)')
plt.legend(loc='best')

plt.show()
