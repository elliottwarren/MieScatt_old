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
            print s

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


def main():


    """
    Carry out a simple Mie scattering calculation using the pymiecoated code used in Lally (2014).
    Complex index of refraction = CIR

    :return:
    """



    import numpy as np
    from pymiecoated import Mie
    import matplotlib.pyplot as plt

    # setup
    ceil_lambda = 0.91e-06 # [m]
    B = 0.14
    RH_crit = 0.38

    # directories
    savedir = '/home/nerc/Documents/MieScatt/figures/'

    # densities of MURK constituents # [kg m-3]
    # range of densities of org carb is massive (0.625 - 2 g cm-3)
    # Haywood et al 2003 use 1.35 g cm-3 but Schkolnik et al., 2006 lit review this and claim 1.1 g cm-3
    dens_amm_sulph = 1770
    dens_amm_nit = 1720
    dens_org_carb = 1100 # NOTE ABOVE


    # calculate complex index of refraction for MURK species
    # input n from dictionary is actually n(bar) from: n = n(bar) - ik
    # output n is complex index of refraction

    # bulk complex index of refraction (CIR) for the MURK species using volume mixing method

    # calculate complex index of refraction for ceil wavelength
    n_amm_sulph, _ = linear_interpolate_n('ammonium_sulphate', ceil_lambda)
    n_amm_nit, _= linear_interpolate_n('ammonium_nitrate', ceil_lambda)
    n_org_carb, _ = linear_interpolate_n('organic_carbon', ceil_lambda)

    # NOTE: Able to use volume in MURK equation instead of mass because, if mass is evenly distributed within a volume
    # then taking x of the mass = taking x of the volume.
    # after calculating volumes used in MURK, can find relative % and do volume mixing.

    # Take average of 4 flights from Haywood for each species relative volume [frac of total volume]:
    vol_amm_sulph = 0.295
    vol_amm_nit = 0.325
    vol_org_carb = 0.38

    # Absolute volume of each species used in calculating murk [m3]
    murk_vol_amm_sulph = vol_amm_sulph * 0.33
    murk_vol_amm_nit = vol_amm_nit * 0.15
    murk_vol_org_carb = vol_org_carb * 0.34

    # scale absolute amounts to find relative volume of each used in MURK (so amm_sulph + amm_nit + org_carb = 1) [m3]
    scaler = 1.0/(murk_vol_amm_sulph + murk_vol_amm_nit + murk_vol_org_carb)
    rel_murk_vol_amm_sulph = scaler * murk_vol_amm_sulph
    rel_murk_vol_amm_nit = scaler * murk_vol_amm_nit
    rel_murk_vol_org_carb = scaler * murk_vol_org_carb

    # volume mixing for CIR (eq. 12, Liu and Daum 2008) -> seem pretty good to quote for this and alt. methods
    n_murk = (rel_murk_vol_amm_sulph * n_amm_sulph) + (rel_murk_vol_amm_nit * n_amm_nit) + (rel_murk_vol_org_carb * n_org_carb)

    # complex indices of refraction (n = n(bar) - ik) at ceilometer wavelength (910 nm) Hesse et al 1998
    n_water, _ = linear_interpolate_n('water', ceil_lambda)
    n_water = complex(1.328, 1.099e-06)

    # create dry size distribution [m]
    r_md_microm = np.arange(0.001, 4.001, 0.001)
    r_md = r_md_microm*1.0e-06

    # RH array [fraction]
    # This gets fixed for each Q iteration (Q needs to be recalculated for each RH used)
    RH = 0.8


    # swell particles using FO method
    # rm = np.ma.ones(RH.shape) - (B / np.ma.log(RH_ge_RHcrit))
    rm = 1 - (B / np.log(RH))
    rm2 = np.ma.power(rm, 1. / 3.)
    rm = np.ma.array(r_md) * rm2
    rm_microm = rm * 1.0e06

    # calculate size parameter for dry
    x_dry = (2.0 * np.pi * r_md)/ceil_lambda
    x_wet = (2.0 * np.pi * rm)/ceil_lambda


    # calculate swollen index of refraction using MURK
    n_swoll = CIR_Hanel(n_water, n_murk, r_md, rm)

    # Calc extinction efficiency for dry aerosol (using r_md!!!! NOT r_m)
    all_particles_dry = [Mie(x=x_i, m=n_murk) for x_i in x_dry]
    Q_dry = np.array([particle.qsca() for particle in all_particles_dry])

    # deliquescent aerosol (solute disolves as it takes on water)
    all_particles_del = [Mie(x=x_wet[i], m=n_swoll[i]) for i in np.arange(len(x_wet))]
    Q_del = np.array([particle.qsca() for particle in all_particles_del])

    # coated aerosol (insoluble aerosol that gets coated as it takes on water)
    all_particles_coat = [Mie(x=x_dry[i], m=n_murk, y=x_wet[i], m2=n_water) for i in np.arange(len(x_wet))]
    Q_coat = np.array([particle.qsca() for particle in all_particles_coat])

    # plot it
    plt.plot(r_md_microm, Q_dry, label='dry murk')
    plt.plot(rm_microm, Q_del, label='deliquenscent murk (RH = ' + str(RH) + ')')
    plt.plot(rm_microm, Q_coat, label='coated murk (RH = ' + str(RH) + ')')
    plt.title('lambda = interpolated to 910 nm, n = murk')
    plt.xlabel('radius [micrometer]')
    plt.xlim([0.0, 4.0])
    plt.ylabel('Q')
    plt.legend()
    plt.savefig(savedir + 'Q_ext_dry_murk_.png')
    plt.close()

    plt.plot(rm*1.0e6, label='r_m (wet)')
    plt.plot(r_md*1.0e6, label='r_md (dry)')
    plt.legend(loc=0)
    plt.savefig(savedir + 'radii.png')
    plt.close()

    print 'END PROGRAM'

if __name__ == '__main__':
    main()


