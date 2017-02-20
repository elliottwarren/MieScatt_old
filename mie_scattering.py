__author__ = 'nerc'




def linear_interpolate_n(particle, aim_lambda):

    """
    linearly interpolate the complex index of refraction for a wavelength of the aerosol and water

    :input: dict: contains lower_lambda, upper_lambda, lower_n, upper_n, lower_k, upper_k for particle type
    :input: aim_lambda: what wavelength the values are being interpolated to

    :return:n: interpolated complex index of refraction
    :return:dict_parts: dictionary with the index refraction parts and how far it was interpolated
    """

    import numpy as np

    def part_file_read(particle):

        """
        Locate and read the particle file. STore wavelength, n and k parts in dictionary
        """

        from numpy import array

        # particle data dir
        part_datadir = '/media/sf_HostGuestShared/complex index of refraction/'

        # find particle filename
        if particle == 'ammonium_nitrate':
            part_file = 'refract_ammoniumnitrate'
        elif particle == 'ammonium_sulphate':
            part_file = 'refract_ammoniumsulphate'
        elif particle == 'organic_carbon':
            part_file = 'refract_ocff'
        elif particle == 'water':
            part_file = 'refract_water.txt'
        else:
            raise ValueError("incorrect species or not yet included in particle list")

        # full path
        file_path = part_datadir + part_file

        # empty dictionary to hold data
        data = {'lambda': [],
                'real': [],
                'imaginary': []}

        # open file and read down to when data starts
        file = open(file_path, "r")
        s = file.readline()
        s = s.rstrip('\n\r')

        while s != '*BEGIN_DATA':
            s = file.readline()
            s = s.rstrip('\n\r')

        line = file.readline() # read line
        line = line.rstrip('\n\r')

        while line != '*END':
            line = ' '.join(line.split()) # remove leading and trailing spaces. Replace multiple spaces in the middle with one.

            # if line isn't last line in file

            line_split = line.split(' ')
            data['lambda'] += [float(line_split[0])]
            data['real'] += [float(line_split[1])]
            data['imaginary'] += [float(line_split[2])]

            # read next line
            line = file.readline()
            line = line.rstrip('\n\r')

        # convert to numpy array
        for key, value in data.iteritems():
            data[key] = array(value)

        return data


    # read in the particle file data
    data = part_file_read(particle)

    # find locaiton of lambda within the spectral file
    idx = np.searchsorted(data['lambda'], aim_lambda)

    # find adjacent wavelengths
    # if lambda is same as one in spectral file, extract
    if data['lambda'][idx] == aim_lambda:

        lambda_n = data['real'][idx]
        lambda_k = data['imaginary'][idx]
        frac = np.nan

    # else interpolate to it
    else:
        upper_lambda = data['lambda'][idx]
        lower_lambda = data['lambda'][idx-1]
        upper_n = data['real'][idx]
        lower_n = data['real'][idx-1]
        upper_k = data['imaginary'][idx]
        lower_k = data['imaginary'][idx-1]

        # differences
        diff_lambda = upper_lambda - lower_lambda
        diff_n = upper_n - lower_n
        diff_k = upper_k - lower_k

        # distance aim_lambda is along linear interpolation [fraction] from the lower limit
        frac = ((aim_lambda - lower_lambda) / diff_lambda)

        # calc interpolated values for n and k
        lambda_n = lower_n + (frac * abs(diff_n))
        lambda_k = lower_k + (frac * abs(diff_k))


    # Complex index of refraction using lambda_n and k
    n = complex(lambda_n, lambda_k)

    dict_parts = {'lambda_n': lambda_n,
                'lambda_k': lambda_k,
                'frac': frac}

    return n, dict_parts

def CIR_Hanel(n_w, n_0, r_0, r):

    """
    Calculate the complex index of refraction (CIR) for the swollen aerosol. Based off of Hanel 1976.

    :param n_w: CIR for pure water
    :param n_0: CIR for pure aerosol (before growth)
    :param r_0: radius before growth
    :param r: radius after growth
    :return: n_swoll: CIR of the swollen aerosol
    """

    n_swoll = n_w + (n_0 - n_w) * (r_0/r)**(3.0)

    return n_swoll

def calc_n_aerosol(rel_vol, ceil_lambda):

    """
    Calculate n for each of the aerosol in rel_vol
    :param rel_vol:
    :return:
    """

    n_aerosol = {}
    for key in rel_vol.iterkeys():
        n_aerosol[key], _ = linear_interpolate_n(key, ceil_lambda)

    return n_aerosol

def calc_n_murk(rel_vol, n):

    """
    Calculate n_murk when inputed relative volumes

    :param rel_vol:
    :return: n_murk
    """

    # Absolute volume of each species used in calculating murk [m3]
    murk_vol_amm_sulph = rel_vol['ammonium_sulphate'] * 0.33
    murk_vol_amm_nit = rel_vol['ammonium_nitrate'] * 0.15
    murk_vol_org_carb = rel_vol['organic_carbon'] * 0.34

    # scale absolute amounts to find relative volume of each used in MURK (so amm_sulph + amm_nit + org_carb = 1) [m3]
    scaler = 1.0/(murk_vol_amm_sulph + murk_vol_amm_nit + murk_vol_org_carb)
    rel_murk_vol_amm_sulph = scaler * murk_vol_amm_sulph
    rel_murk_vol_amm_nit = scaler * murk_vol_amm_nit
    rel_murk_vol_org_carb = scaler * murk_vol_org_carb

    # volume mixing for CIR (eq. 12, Liu and Daum 2008) -> seem pretty good to quote for this and alt. methods
    n_murk = (rel_murk_vol_amm_sulph * n['ammonium_sulphate']) + (rel_murk_vol_amm_nit * n['ammonium_nitrate']) \
              + (rel_murk_vol_org_carb * n['organic_carbon'])

    return n_murk

def plot_radius(savedir, r_md, r_m):

    import matplotlib.pyplot as plt

    # plot radius for wet and dry
    plt.plot(r_m*1.0e6, label='r_m (wet)')
    plt.plot(r_md*1.0e6, label='r_md (dry)')
    plt.legend(loc=0)
    plt.savefig(savedir + 'radii.png')
    plt.close()

    return

def calc_Q(x, m,  type, y=[], m2=[],):

    """

    :param x: dry particle size parameter
    :param m: complex index of refraction for particle
    :param y: wet particle size parameter
    :param m2: complex index of refraction for water
    :return: Q_ext
    :return: calc_type: how was particle treated? (e.g. dry, coated)
    """

    from pymiecoated import Mie
    import numpy as np

    # Coated aerosol (insoluble aerosol that gets coated as it takes on water)
    if type == 'coated':

        if (y != []) & (m2 != []):

            all_particles_coat = [Mie(x=x[i], m=m, y=y[i], m2=m2) for i in np.arange(len(x))]
            Q = np.array([particle.qext() for particle in all_particles_coat])

        else:
            pass

    # Calc extinction efficiency for dry aerosol (using r_md!!!! NOT r_m)
    # if singular, then type == complex, else type == array
    elif type(m) == complex:

        all_particles_dry = [Mie(x=x, m=m) for x_i in x]
        Q = np.array([particle.qext() for particle in all_particles_dry])
        calc_type = 'dry'

    # deliquescent aerosol (solute disolves as it takes on water)
    elif len(x) == len(m):

        all_particles_del = [Mie(x=x[i], m=m[i]) for i in np.arange(len(x))]
        Q = np.array([particle.qext() for particle in all_particles_del])
        calc_type = 'deliquescent'

    # return diagnostic information too?
    if type == True:
        return Q, calc_type
    else:
        return Q


def main():


    """
    Carry out a simple Mie scattering calculation using the pymiecoated code used in Lally (2014).
    Complex index of refraction = CIR

    :return: plots Mie scattering curve of murk, deliquecent and coated spheres at 80 % RH.
    """


    import numpy as np
    from pymiecoated import Mie
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------
    # Setup

    # setup
    ceil_lambda = [0.55e-06] # [m]
    # ceil_lambda = np.arange(0.69e-06, 1.19e-06, 0.05e-06) # [m]
    B = 0.14
    RH_crit = 0.38

    # directories
    savedir = '/home/nerc/Documents/MieScatt/figures/'

    # aerosol with relative volume - average from the 4 Haywood et al 2008 flights
    rel_vol = {'ammonium_sulphate': 0.295,
               'ammonium_nitrate': 0.325,
                'organic_carbon': 0.38}


    # create dry size distribution [m]
    r_md_microm = np.arange(0.03, 5.001, 0.001)
    r_md = r_md_microm*1.0e-06

    # RH array [fraction]
    # This gets fixed for each Q iteration (Q needs to be recalculated for each RH used)
    RH = 0.8

    # densities of MURK constituents # [kg m-3]
    # range of densities of org carb is massive (0.625 - 2 g cm-3)
    # Haywood et al 2003 use 1.35 g cm-3 but Schkolnik et al., 2006 lit review this and claim 1.1 g cm-3
    dens_amm_sulph = 1770
    dens_amm_nit = 1720
    dens_org_carb = 1100 # NOTE ABOVE

    # define array to store Q for each wavelength
    Q_dry = []

    x_store =[]
    n_store=[]

    for lam in ceil_lambda:

        # -------------------------------------------------------------------
        # Process

        # calculate complex index of refraction for MURK species
        # output n is complex index of refraction (n + ik)
        n_aerosol = calc_n_aerosol(rel_vol, lam)

        # NOTE: Able to use volume in MURK equation instead of mass because, if mass is evenly distributed within a volume
        # then taking x of the mass = taking x of the volume.
        # after calculating volumes used in MURK, can find relative % and do volume mixing.

        # bulk complex index of refraction (CIR) for the MURK species using volume mixing method
        # n_murk = calc_n_murk(rel_vol, n_aerosol)
        n_murk = complex(1.53, 0.007)
        n_store += [n_murk]

        # complex indices of refraction (n = n(bar) - ik) at ceilometer wavelength (910 nm) Hesse et al 1998
        # n_water, _ = linear_interpolate_n('water', lam)

        # swell particles using FO method
        # rm = np.ma.ones(RH.shape) - (B / np.ma.log(RH_ge_RHcrit))
        r_m = 1 - (B / np.log(RH))
        r_m2 = np.ma.power(r_m, 1. / 3.)
        r_m = np.ma.array(r_md) * r_m2
        r_m_microm = r_m * 1.0e06

        # calculate size parameter for dry and wet
        x_dry = (2.0 * np.pi * r_md)/lam
        x_store += [x_dry]
        x_wet = (2.0 * np.pi * r_m)/lam


        # calculate swollen index of refraction using MURK
        # n_swoll = CIR_Hanel(n_water, n_murk, r_md, r_m)


        # Calc extinction efficiency for dry aerosol (using r_md!!!! NOT r_m)
        all_particles_dry = [Mie(x=x_i, m=n_murk) for x_i in x_dry]
        Q_dry += [np.array([particle.qext() for particle in all_particles_dry])]

        # original set
        # # Calc extinction efficiency for dry aerosol (using r_md!!!! NOT r_m)
        # all_particles_dry = [Mie(x=x_i, m=n_murk) for x_i in x_dry]
        # Q_dry = np.array([particle.qext() for particle in all_particles_dry])
        #
        # # deliquescent aerosol (solute disolves as it takes on water)
        # all_particles_del = [Mie(x=x_wet[i], m=n_swoll[i]) for i in np.arange(len(x_wet))]
        # Q_del = np.array([particle.qext() for particle in all_particles_del])
        #
        # # coated aerosol (insoluble aerosol that gets coated as it takes on water)
        # all_particles_coat = [Mie(x=x_dry[i], m=n_murk, y=x_wet[i], m2=n_water) for i in np.arange(len(x_wet))]
        # Q_coat = np.array([particle.qext() for particle in all_particles_coat])

    # plot
    fig = plt.figure(figsize=(7, 4.5))

    for Q_i, lam in zip(Q_dry, ceil_lambda):

        # plot it
        ax = plt.semilogx(r_md_microm, Q_i, label=str(lam) + 'm')
        # plt.semilogx(r_md_microm, Q_dry, label='dry murk', color=[0,0,0])
        # plt.semilogx(r_m_microm, Q_del, label='deliquescent murk (RH = ' + str(RH) + ')')
        # plt.semilogx(r_m_microm, Q_coat, label='coated murk (RH = ' + str(RH) + ')')

    plt.title('lambda = ' + str(ceil_lambda[0]) + '-' + str(ceil_lambda[-1]) + 'm, n = murk')
    plt.xlabel('radius [micrometer]', labelpad=-5)
    plt.xlim([0.05, 5.0])
    plt.ylim([0.0, 5.0])
    plt.ylabel('Q_ext')
    # plt.legend(fontsize=8, loc='best')
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.grid(b=True, which='minor', color=[0.85, 0.85, 0.85], linestyle='--')
    plt.savefig(savedir + 'Q_ext_(1.57-0.007i)_' + str(ceil_lambda[0]) + '-' + str(ceil_lambda[-1]) + 'lam.png')
    plt.tight_layout()
    # plt.close()

    # plot the radius
    # plot_radius(savedir, r_md, r_m)

    print 'END PROGRAM'

if __name__ == '__main__':
    main()


