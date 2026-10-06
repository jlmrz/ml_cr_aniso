import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Sequence
from argparse import Namespace

import lzma

train_seed = 0
val_seed = 2 ** 20
test_seed = 2 ** 26


source_data = {
    # Name : [source_lon, source_lat, D_src]
    'M82': [141.4095, 40.5670, '3.5'],
    'NGC1068': [172.10401114, -51.93361399, '16.0'],
    'NGC1052': [182.01946755, -57.92547291, '21.5'],
    'CenA': [309.5159, 19.4173, '3.5'],
    # 'NGC6946': [95.71873,11.6729,'6.0'],
    'M87': [283.7777, 74.4912, '18.5'],
    'FornaxA': [240.1627, -56.6898, '20.0'],
    'NGC4102':	[138.078537, 63.072269, '18.0'],
    'NGC3079': [157.809932, 48.359931, '17.0'],
    'NGC2655': [134.923512, 32.688444, '22.0'],
    # AUGER SEYFERT
    'NGC4941': [308.805657, 57.174319, '17.0'],
    'NGC5643': [321.442860, 15.027954, '17.5'],
    'CircinusGalaxy': [311.325977, -3.807854, '6.5'],

}


def get_source_data(source_id):
    if source_id in source_data:
        return tuple(source_data[source_id])
    else:
        raise ValueError('Unknown source!')


def load_turbulence_src_samples(
        args: Namespace, mfs: Sequence[str],
        sources: Optional[Sequence[str]] = None,
        seeds: Optional[Sequence[int | str]] = None,
        suffix: str = "",
) -> Generator[tuple[str, str, list[np.ndarray]], None, None]:
    """
    Load source samples stored as:

        data/<gmf>/sources/turbulence/<seed>/src_sample_<source>_...txt.xz

    Each yielded item contains all requested sources for one consistent
    (GMF, turbulent seed) realization.

    Yields
    ------
    gmf : str
    seed : str
    data_list : list[np.ndarray]
        One array per source, in the same order as `sources`.
    """
    if sources is None:
        sources = args.source_id.split(",")

    requested_seeds = (
        None
        if seeds is None
        else {str(seed) for seed in seeds}
    )

    for gmf in mfs:
        turbulence_root = (
            Path(args.data_dir)
            / gmf
            / "sources"
            / "turbulence"
        )

        if not turbulence_root.is_dir():
            raise ValueError(
                f"Turbulence directory not found: {turbulence_root}"
            )

        seed_directories = sorted(
            path
            for path in turbulence_root.iterdir()
            if path.is_dir()
            and (
                requested_seeds is None
                or path.name in requested_seeds
            )
        )

        if not seed_directories:
            raise ValueError(
                f"No turbulent realizations found in {turbulence_root}"
            )

        for seed_directory in seed_directories:
            data_list = []

            for source_id in sources:
                _, _, distance = get_source_data(source_id)

                filename = (
                    f"src_sample_{source_id}"
                    f"_D{distance}"
                    f"_Emin{args.Emin}"
                    f"_N{args.Nini}"
                    f"_R{args.source_vicinity_radius}"
                    f"_Nside{args.Nside}"
                    f"{suffix}.txt.xz"
                )

                infile = seed_directory / args.spectrum_scenario / filename

                if not infile.is_file():
                    raise ValueError(
                        "Incomplete turbulent realization: "
                        f"{infile} not found"
                    )

                with lzma.open(infile, "rt") as stream:
                    data_list.append(
                        np.genfromtxt(stream, dtype=float)
                    )

            yield gmf, seed_directory.name, data_list

def load_src_sample(
        args: Namespace,
        suffix: str = '',
        sources: Optional[list] = None,
        mf: Optional[str] = None
) -> Generator[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from src_sample files and yield numpy arrays

    Parameters
    ----------
    args : Namespace
        Command line arguments or configuration object containing parameters:

    suffix : str, optional
        May be used to select a ranage of files (e.g. suffix='*') or specific realizations

    sources : List[str], optional
        Explicit list of sources or file paths to load. If None, uses args.source_id.
        Example: ['source1', 'source2'] or ['/path/to/src_sample_*.txt.xz']

    mf : str, optional
        Magnetic field model. If None, uses args.mf.

    Yields
    ------
    np.ndarray
        Numpy array containing the data loaded from each source sample file.
        The array dtype is float and shape depends on the content of each file.
    """
    import lzma
    import glob

    if sources is None:
        sources = args.source_id.split(',')

    if mf is None:
        mf = args.mf

    if args.turbulence_samples:
        file_extension = 'npz'
    else:
        file_extension = 'txt'  # will be deprecated

    for source_id in sources:
        if 'src_sample_' in source_id:
            infiles = source_id
        else:
            _, _, D_src = get_source_data(source_id)
            infiles = ('src_sample_' + source_id + '_D' + D_src
                       + '_Emin' + str(args.Emin)
                       + '_N' + str(args.Nini)
                       + '_R' + str(args.source_vicinity_radius)
                       + '_Nside' + str(args.Nside) + suffix
                       + '.' + file_extension + '.xz')
            infiles = args.data_dir + '/' + mf + '/sources/' + infiles

        files = list(glob.glob(infiles))
        if len(files) == 0:
            raise ValueError(infiles + ' file(s) not found!')

        for infile in files:
            if args.turbulence_samples: # new format
                for infile in files:
                    with lzma.open(infile, 'rb') as f:
                        with np.load(f) as npz:
                            if 'data' not in npz:
                                raise KeyError(f"{infile}: key 'data' not found in npz file")

                            data = npz['data']
                            yield data
            else:
                with lzma.open(infile, 'rt') as f:
                    yield np.genfromtxt(f, dtype=float)


def f_sampler(
        args: Namespace, n_samples: int = -1,  # if < 0, sample forever
        exclude_iso: bool = False
    ) -> Generator[Tuple[int, int], None, None]:
    """
    Generator function that samples source and isotropic counts based on configuration parameters.

    This function yields tuples of (Nsrc, Niso) counts where Nsrc + Niso = Neecr (total events).
    The sampling behavior is controlled by parameters.

    Parameters
    ----------
    args : Namespace
        - Neecr (int): Total number of events
        - f_src (float): If between 0 and 1, fixed fraction of source events
        - f_src_min (float): Minimum fraction of source events (used when f_src is not specified)
        - f_src_max (float): Maximum fraction of source events (used when f_src is not specified)
        - log_sample (bool): If True, sample logarithmically in source count space

    n_samples : int, optional
        Number of samples to generate. If negative, generates samples indefinitely.
        Default: -1 (sample forever)

    exclude_iso : bool, optional
        If True, ensures at least one source event (Nsrc >= 1) to exclude pure isotropic case.
        Default: False

    Yields
    ------
    tuple
        (Nsrc, Niso) where:
        - Nsrc (int): Number of source events
        - Niso (int): Number of isotropic events
        Always satisfies: Nsrc + Niso = args.Neecr

    Notes
    -----
    The function supports three sampling modes:
    1. Fixed fraction: When 0 <= args.f_src <= 1, uses fixed fraction for all samples
    2. Linear sampling: When args.log_sample is False, samples uniformly between f_src_min and f_src_max
    3. Logarithmic sampling: When args.log_sample is True, samples logarithmically between f_src_min and f_src_max
    """

    Neecr = args.Neecr
    Fsrc = None
    if 0 <= args.f_src <= 1.:
        Fsrc = args.f_src
    else:
        assert 0 <= args.f_src_min < 1
        assert 0 < args.f_src_max <= 1
        assert args.f_src_min < args.f_src_max

        N_src_min = np.round(args.f_src_min * Neecr)
        N_src_max = np.round(args.f_src_max * Neecr)
        if exclude_iso:
            N_src_min = max(1, N_src_min)
            N_src_max = max(1, N_src_max)

        if N_src_min == N_src_max:
            Fsrc = N_src_min / Neecr
        elif args.log_sample:
            logNmax = np.log((N_src_max + 0.49))
            if args.f_src_min == 0 and not exclude_iso:
                # make sure roughly equal amount of isotropic and mixture samples are generated
                logNmin = -logNmax + np.log(0.5)  # boundary between iso and source is at 0.5 since we use np.round below
            else:
                logNmin = np.log((N_src_min-0.49))
    n = 0
    if Fsrc is not None:
        Nsrc = int(np.round(Neecr*Fsrc))

    while n != n_samples:
        if Fsrc is None:
            if args.log_sample:
                logN = logNmin + (logNmax-logNmin)*np.random.rand()
                Nsrc = int(np.round(np.exp(logN)))
                assert N_src_min <= Nsrc <= N_src_max
            else:
                Nsrc = np.random.randint(N_src_min, N_src_max+1)

        Niso = Neecr - Nsrc
        yield Nsrc, Niso
        n += 1


