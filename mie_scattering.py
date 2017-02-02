__author__ = 'nerc'


def linear_interpolate_n(dict, aim_lambda):

    """
    linearly interpolate the complex index of refraction for a wavelength, given the complex index of refraction for two
    surrounding wavelengths

    :input: dict: contains lower_lambda, upper_lambda, lower_n, upper_n, lower_k, upper_k for particle type
    :input: aim_lambda: what wavelength the values are being interpolated to

    :return:n: interpolated complex index of refraction
    :return:dict_parts: dictionary with the index refraction parts and how far it was interpolated
    """

    # # test - ammonium sulphate
    # lower_lambda = 8.0e-07
    # upper_lambda = 1.0e-06
    # aim_lambda = 0.91e-06
    # lower_n = 1.525
    # upper_n = 1.51
    # lower_k = 1.0e-07
    # upper_k = 3.5e-07

    # differences

    # wavelength
    diff_lambda = dict['upper_lambda'] - dict['lower_lambda']

    # n
    diff_n = dict['upper_n'] - dict['lower_n']

    # k
    diff_k = dict['upper_k'] - dict['lower_k']


    # distance aim_lambda is along linear interpolation [fraction] from the lower limit
    frac = ((aim_lambda - dict['lower_lambda']) / diff_lambda)

    # calc interpolated values for n and k
    interp_n = dict['lower_n'] + (frac * abs(diff_n))
    interp_k = dict['lower_k'] + (frac * abs(diff_k))

    # interpolated complex index of refraction
    n = complex(interp_n, interp_k)

    dict_parts = {'interp_n': interp_n,
            'interp_k': interp_k,
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

    # calculate complex index of refraction for MURK species
    # input n from dictionary is actually n(bar) from: n = n(bar) - ik
    # output n is complex index of refraction

    # Toon et al, 1976
    ammonium_sulphate_dict = {
        'lower_lambda': 8.0e-07,
        'upper_lambda': 1.0e-06,
        'lower_n': 1.525,
        'upper_n': 1.51,
        'lower_k': 1.0e-07,
        'upper_k': 3.5e-07}

    # n: CRC handbook of Chem. and Phys., 58th Ed., 1971 (lambda is actually for 587.6 nm, though used for 910 nm in UM)
    # k: Gosse et al., Applied Optics 1997 though in solution at 25 %
    # jump in k is large, ideally want k for wavelengths closer to the ceil
    ammonium_nitrate_dict = {
        'lower_lambda': 8.0e-07,
        'upper_lambda': 1.0e-06,
        'lower_n': 1.611,
        'upper_n': 1.611,
        'lower_k': 1.06e-07,
        'upper_k': 3.15e-06}

    # taken from fossil fuel organic carbon
    organic_carbon_dict = {
        'lower_lambda': 8.6e-07,
        'upper_lambda': 1.06e-06,
        'lower_n': 1.54045,
        'upper_n': 1.54051,
        'lower_k': 0.006,
        'upper_k': 0.006}

    # bulk complex index of refraction (CIR) for the MURK species using volume mixing method

    # calculate complex index of refraction for ceil wavelength
    n_amm_sulph, _ = linear_interpolate_n(ammonium_sulphate_dict, ceil_lambda)
    n_amm_nit, _= linear_interpolate_n(ammonium_nitrate_dict, ceil_lambda)
    n_org_carb, _ = linear_interpolate_n(organic_carbon_dict, ceil_lambda)

    # Take average of 4 flights from Haywood for each species relative volume:
    vol_amm_sulph = 0.295
    vol_amm_nit = 0.325
    vol_org_carb = 0.38

    # calculate absolute value of each species used in calculating murk
    #! NOTE, not sure if MURK calculation was meant to use masses. Therefore, might need to convert volume above to
    # mass, do the MURK calculation, then convert back to volume.
    # Currently, just assumes the MURK equation uses relative volume.
    abs_vol_amm_sulph = vol_amm_sulph * 0.33
    abs_vol_amm_nit = vol_amm_nit * 0.15
    abs_vol_org_carb = vol_org_carb * 0.34

    # scale absolute amounts to find relative amounts of each used in MURK (so amm_sulph + amm_nit + org_carb = 1),
    scaler = 1.0/(abs_vol_amm_sulph + abs_vol_amm_nit + abs_vol_org_carb)
    rel_amm_sulph = scaler * abs_vol_amm_sulph
    rel_amm_nit = scaler *abs_vol_amm_nit
    rel_org_carb = scaler * abs_vol_org_carb


    # volume mixing for CIR (eq. 12, Liu and Daum 2008) -> seem pretty good to quote for this and alt. methods
    n_murk = (rel_amm_sulph * n_amm_sulph) + (rel_amm_nit * n_amm_nit) + (rel_org_carb * n_org_carb)


    # complex indices of refraction (n = n(bar) - ik) at ceilometer wavelength (910 nm) Hesse et al 1998
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


