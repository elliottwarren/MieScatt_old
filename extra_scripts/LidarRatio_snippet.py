__author__ = 'Elliott_Warren'


import numpy as np
import matplotlib.pyplot as plt
from pymiecoated import Mie

# directories
maindir = '/home/nerc/Documents/MieScatt/'
datadir = '/home/nerc/Documents/MieScatt/data/'

# save dir
savedir = maindir + 'figures/LidarRatio/'

ceil_lambda = 905.0e-09

step = 0.005
r_range_um = np.arange(0.000 + step, 10.000 + step, step)
r_range_m = r_range_um * 1.0e-06
x_range = (2.0 * np.pi * r_range_m)/ceil_lambda

# lambda = 905 nm!
n_species_905 = {'(NH4)2SO4': (1.5328749999999998+2.3125000000000006e-07j),
                 'CBLK': (1.85+0.6976j),
                 'CORG': (1.5404635+0.006j),
                 'H2O': (1.328+6.00800000000001e-07j),
                 'NH4NO3': (1.611+1.7041000000000008e-06j),
                 'NaCl': (1.5662+1e-07j)}

aer_particles = ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CORG', 'CBLK']


## LINEPLOT - S for each species at 905 nm for monodisperse aerosol, with legend


#S_r = lidar ratio
S_r = {}

for aer_i in n_species_905:

    print ''
    print aer_i

    S_r[aer_i] = np.empty(len(r_range_m))
    S_r[aer_i][:] = np.nan

    # n_i = complex(1.47 + 0.099j)  # fixed complex index of refraction i
    n_i = n_species_905[aer_i]

    for r_idx, r_i in enumerate(r_range_m):

        x_i = x_range[r_idx]  # size parameter_i

        # print loop progress
        if r_idx in np.arange(0, 2100, 100):
            print r_idx

        # calculate Q_back and Q_ext efficiency for current size parameter and complex index of refraction
        particle = Mie(x=x_i, m=n_i)
        Q_ext = particle.qext()
        Q_back = particle.qb()
        Q_back_alt = Q_back / (4.0 * np.pi)

        # calculate the lidar ratio
        S_r[aer_i][r_idx] = Q_ext / Q_back_alt


# simple plot of S
fig, ax = plt.subplots(1,1, figsize=(7,5))
for key, data in S_r.iteritems():
    plt.loglog(r_range_um * 2.0, data, label=key) # diameter [microns]
# plt.xlim([0.01, 100.0])
# plt.ylim([1.0, 10.0e7])
plt.ylabel('Lidar Ratio [sr]')
plt.xlabel('Diameter [microns]')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
plt.savefig(savedir + 'quickplot_S_vs_r_bySpecies.png')
plt.close(fig)

# -----------------------------------------------------

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
    Q_back_alt = Q_back / (4.0 * np.pi)

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


