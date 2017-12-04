__author__ = 'Elliott_Warren'


import numpy as np
import matplotlib.pyplot as plt
from pymiecoated import Mie

ceil_lambda = 905.0e-09

step = 0.005
r_range_um = np.arange(0.000 + step, 10.000 + step, step)
r_range_m = r_range_um * 1.0e-06
x_range = (2.0 * np.pi * r_range_m)/ceil_lambda



#S_r = lidar ratio
S_r = np.empty(len(r_range_m))
S_r[:] = np.nan
for r_idx, r_i in enumerate(r_range_m):

    x_i = x_range[r_idx]  # size parameter_i
    n_i = complex(1.47 + 0.099j)  # fixed complex index of refraction i

    # print loop progress
    if r_idx in np.arange(0, 2100, 100):
        print r_idx

    # calculate Q_back and Q_ext efficiency for current size parameter and complex index of refraction
    particle = Mie(x=x_i, m=n_i)
    Q_ext = particle.qext()
    Q_back = particle.qb()

    # calculate the lidar ratio
    S_r[r_idx] = Q_ext / Q_back


# simple plot of S
fig, ax = plt.subplots(1,1, figsize=(6,5))
plt.loglog(r_range_um * 2.0, S_r) # diameter [microns]
plt.xlim([0.01, 100.0])
plt.ylim([1.0, 10.0e7])
plt.ylabel('Lidar Ratio')
plt.xlabel('Diameter [microns]')
plt.tight_layout()
plt.show()
#plt.savefig(savedir + 'quickplot_S_vs_r.png')
#plt.close(fig)

# ---------------------------------------------------------


