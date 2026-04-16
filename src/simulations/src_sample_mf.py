import os
import argparse
import numpy as np
import h5py

from src.generators.utils import source_data, get_source_data
from src.simulations.utils import get_deflection_map

from angdist import getAngDist


def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)


cline_parser = argparse.ArgumentParser(
    description='E,Z pair sampling',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

add_arg('--source_id', type=str, help='one of ' + ' , '.join(source_data.keys()), default='M82')
add_arg('--GMF', type=str, help='pt or jf', default='kstt')
add_arg('--Emin', type=str, help='Emin in EeV', default=28)
add_arg('--Nini', type=int, help='Number of cosmic ray nuclei to form a sample', default=100_000)
add_arg('--Nside', type=int, help='Nside for output healpix grid', default=32)
add_arg('--shiftA', type=str, help='A factor to shift and atomic mass by', default='1')

gmf_full_names = {
    'jf': 'JF12ST',
    'kstt': 'KST24T',
    'uf': 'UF23',
    'uf23t': 'UF23'
}


if __name__ == '__main__':
    args = cline_parser.parse_args()


    source_id = args.source_id
    GMF = args.GMF

    # Minimum energy used to create a sample, EeV:
    Emin = args.Emin

    # This is the same number as used in psample.py
    Nini = args.Nini

    # Healpix grid used in simulations:
    Nside = args.Nside

    # CR's mass composition:
    # '1' uses composition model, fitting Auger data.
    # '0.5' - 2 times lighter composition
    # '2.0' - 2 times heavier composition
    shiftA = float(args.shiftA)

    # Radius of the source neighborhood, deg
    source_vicinity_radius = 1

    print('Parameters used:\n \tGMF: ', GMF,
          '\n\tSource id: ', source_id,
          '\n\tEmin: ', Emin,
          '\n\tNini: ', Nini,
          '\n\tSiftA: ', shiftA
          )

    source_lon, source_lat, D_src = get_source_data(source_id)
    gmf_dir = GMF + '/'

    # File with the spectrum that was "propagated" with CRPropa
    infile = '../data/sample_D' + D_src + '_Emin' + str(Emin) + '_' + str(Nini)

    if shiftA != 1.0:
        infile += '_shift' + args.shiftA

    infile += 'nuclei_sorted.txt'

    outfile = ('src_sample_' + source_id + '_D' + D_src
                + '_Emin' + str(Emin)
                + '_N' + str(Nini) + '_R'
                + str(source_vicinity_radius) + '_Nside' + str(Nside))

    if shiftA != 1.0:
        outfile += '_shift' + args.shiftA

    # Read the file and extract data
    data = np.loadtxt(infile, dtype=int)

    # Z, E/EeV, multiplicity of the nuclei
    Z = data[:, 0]
    E = data[:, 1]
    M = data[:, 2]

    # Number of nucleus_energy.txt.xz files to be proceeded
    N_files = len(Z)
    # N_files = 2

    # A list of nuclei to prepare the name of a file to read on the fly
    nuclei = ['proton', 'helium', 'lithium', 'beryllium', 'boron', 'carbon',
              'nitrogen', 'oxygen', 'fluorine', 'neon', 'sodium', 'magnesium',
              'aluminium', 'silicon', 'phosphorus', 'sulfur', 'chlorine', 'argon',
              'potassium', 'calcium', 'scandium', 'titanium', 'vanadium',
              'chromium', 'manganese', 'iron']

    # A counter of EECRs added to the output file (result)

    # A counter for missing (Z,E) pairs
    err_no = 0

    hdf5_path =  f"../data/{GMF}_Nside{Nside}_deflection_maps.h5"
    with h5py.File(hdf5_path, 'r') as file:

        seeds = []
        for group_name in file.keys():
            if group_name.startswith('group_'):
                seeds.append(file[group_name].attrs['random_seed'])

        seeds = np.unique(seeds)
        print('unique seeds', seeds)

        # A template for the output file
        # lat, lon (ini); lat, lon (res); angsep; Z, E, # in the healpix map. total N_turb_seeds maps
        outdata = np.zeros([Nini, 8, len(seeds)], dtype=float)

        for n, seed in enumerate(seeds):
            print(50 * '-', '\n')
            print(f'{n}/{len(seeds)}. Sampling events for turbulent random seed {seed}')
            print(50 * '-', '\n')
            k = 0

            for i in range(N_files):
                nucleus = nuclei[Z[i] - 1]
                print(f'{nucleus} (Z = {Z[i]}) {E[i]}  EeV,  R = {E[i]/Z[i]:.3f}')

                mf_params = {'model': gmf_full_names[GMF], 'random_seed': 8388608}
                if GMF == 'jf':
                    mf_params.update(dict(striated=1, turbulent=1))

                _map = get_deflection_map(
                    nucleus_params={'E': E[i], 'Z': Z[i]},
                    mf_params=mf_params,
                    _h5file=file
                )

                if _map.shape[1] == 6:
                    print('Using maximum propagation length mask. Initial number of cells: ', len(_map))
                    mask = np.where(np.logical_not(_map[:, 5]))[0] # max_lenght_reached
                    _map = _map[mask]
                    print('Number of cells left: ', len(_map))

                # Coordinates at the boundary of the Galaxy
                lat_gal_deg = _map[:, 2]
                lon_gal_deg = _map[:, 3]

                # We need angular separation between the source and
                # arrival directions at the Galaxy boundary
                ang_sep = getAngDist(source_lon, source_lat, lon_gal_deg, lat_gal_deg)
                close_dirs = np.asarray(
                    np.where(ang_sep <= source_vicinity_radius + 0.01))

                N_close = np.size(close_dirs)
                if N_close:
                    print('{:4d} close EECRs'.format(N_close))
                    print('To be selected: ' + str(M[i]) + '\n')

                    # Version 2. Allow replacement in case there are not enough
                    # close arrival directions. This will give an output closer
                    # to the given spectrum
                    close_dirs = np.reshape(close_dirs, N_close)
                    sample_size = M[i]
                    if M[i] <= N_close:
                        sample = np.random.choice(close_dirs, size=sample_size,
                                                  replace=False)
                    else:
                        sample = np.random.choice(close_dirs, size=sample_size,
                                                  replace=True)

                    # OK, this is for the output file
                    for j in sample:
                        try:
                            outdata[k, 0:4, n] = _map[j, 0:4]
                            outdata[k, 4, n] = ang_sep[j]
                            outdata[k, 5, n] = Z[i]
                            outdata[k, 6, n] = E[i]
                            outdata[k, 7, n] = j
                            k += 1
                        except IndexError as er:  # temporary fix or rare error TODO: find out the reason of
                            print('skipping sample', j, ':', er)

                else:
                    print('No close EECRs')
                    print('To be selected: ' + str(M[i]) + '\n')

            print(50 * '-', '\n')

    header = ('#   lat_earth    lon_earth    lat_gal    lon_gal     angsep   '
              'Z   E   cell_no\n')

    np.savez(f'../data/{GMF}/sources/{outfile}.npz',
             data=outdata,
             seeds=seeds,
             meta=np.array([header], dtype=object))

    os.system(f'xz ../data/{GMF}/sources/{outfile}.npz')
