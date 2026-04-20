"""
Managing h5py files with deflections maps:
creating, writing, checking groups

Scheme

gmf_model_nside_deflection_map.h5
├── /metadata
│   └──  parameter_table  # All unique parameter combinations (sorted key, values in str format)
│   ├── /group_0000
│   │   │   ├── .attrs (parameters)
│   │   │   └── coordinates [N, 6]
│   │   └── ...
│   ├── /group_0001
│   │   │   ├── .attrs (parameters)
│   │   │   └── coordinates [N, 6]
│   │   └── ...
│   └──
└── ...

"""
# GMF - Galactic Magnetic Field

import h5py
import numpy as np
from pathlib import Path
from typing import Optional

from astropy.coordinates import SkyCoord
import astropy.units as u

DATA_PATH = Path('../data/')

R_BINS = np.logspace(np.log10(0.5), np.log10(50), 30)


def get_bin_index(rigidity: float) -> int:
    return np.digitize(rigidity, R_BINS)


def setup_hdf5_file(mf_model: str, Nside: int) -> h5py.File:
    """
    Create or open HDF5 file and return file handle.

    Parameters
    ----------
    mf_model: gmf model jf | jf_sol | jf_pl | tf | pt | kst | uf
    Nside: HEALPix Nside parameter

    Returns
    ----------
    h5py File handle
    """
    hdf5_path = DATA_PATH / f"{mf_model}_Nside{Nside}_deflection_maps_Rbins{len(R_BINS)}.h5"

    if not hdf5_path.parent.exists():
        hdf5_path.parent.mkdir(parents=True)

    if not hdf5_path.exists():
        with h5py.File(hdf5_path, 'w') as f:
            f.create_group('metadata')
            f['metadata'].create_dataset('parameter_table', dtype=h5py.special_dtype(vlen=str))
            f['metadata'].attrs['R_bin_edges'] = R_BINS

    return h5py.File(hdf5_path, 'a')


def find_h5file(mf_model, Nside):
    hdf5_path = DATA_PATH / f"{mf_model}_Nside{Nside}_deflection_maps_Rbins{len(R_BINS)}.h5"
    if not hdf5_path.parent.exists():
        raise ValueError('File ', hdf5_path, ' does not exist.')
    return h5py.File(hdf5_path, 'r')


def find_group(h5file: h5py.File, mf_params: dict, nucleus_params: dict) -> str:
    """
    Returns group name if the group is found else returns empty string

    Parameters
    ----------
    h5file:
    mf_params:
    nucleus_params:

    Returns
    ----------
        name of a group or empty string
    """
    param_sig = create_param_signature(nucleus_params, mf_params)
    # print('Current parameters signature:', param_sig)

    for group_name in h5file.keys():
        if group_name.startswith('group_'):
            if 'param_signature' in h5file[group_name].attrs:
                if h5file[group_name].attrs['param_signature'] == param_sig:
                    if 'deflection_map' in h5file[group_name].keys():
                        print(h5file[group_name]['deflection_map'])
                        return group_name

    return ''


def find_or_create_group(h5file: h5py.File, mf_params: dict, nucleus_params: dict) -> tuple[str, bool]:
    """
    Find existing group or create new one for given parameters

    Parameters
    ----------
    h5file: file containing deflection maps obtained with given GMF model
    mf_params: parameters of the GMF model
    nucleus_params: dict with keys nucleus charge Z, energy E

    Returns
    ----------
        group_name[str]: name of created/existing group of parameters
        status[bool]: whether a new group is created
    """
    param_sig = create_param_signature(nucleus_params, mf_params)

    metadata = h5file['metadata']

    group_name = find_group(h5file, mf_params, nucleus_params)

    if group_name:
        return group_name, False

    else:
        # If not found, create new group
        # Find next available index
        existing_indices = []
        for group_name in h5file.keys():
            if group_name.startswith('group_'):
                try:
                    idx = int(group_name.split('_')[1])
                    existing_indices.append(idx)
                except (IndexError, ValueError):
                    continue

        next_idx = max(existing_indices) + 1 if existing_indices else 0
        group_name = f"group_{next_idx:04d}"

        # Create group and store signature
        group = h5file.create_group(group_name)
        group.attrs['param_signature'] = param_sig
        group.attrs['mf_params'] = str(mf_params)
        group.attrs['nucleus_params'] = str(nucleus_params)

        # Update parameter_table # TODO: get rid of? use group attrs instead

        if metadata['parameter_table'].shape is not None:
            param_table = list(h5file['metadata']['parameter_table'][:])
            param_table.append(param_sig)

            del metadata['parameter_table']

            h5file['metadata'].create_dataset(
                'parameter_table',
                data=np.array(param_table, dtype=object),
                dtype=h5py.special_dtype(vlen=str)
            )

            return group_name, True

        del metadata['parameter_table']
        metadata.create_dataset(
            'parameter_table',
            data=np.array([param_sig], dtype=object),
            dtype=h5py.special_dtype(vlen=str)
        )

        return group_name, True


def store_results(
        h5file: h5py.File, group_path: str,
        results: np.ndarray,
        mf_params: dict,
        nucleus_params: dict
) -> None:
    """
    Store results in HDF5 group

    Parameters
    ----------
    h5file:  file containing deflection maps obtained with given GMF model
    group_path: name of the group
    results: array with corresponding [lat_ini, lon_ini, lat_res, lon_res, deflection]
    mf_params: parameters of the GMF model
    nucleus_params: parameters of the nucleus
    """

    if group_path in h5file:
        del h5file[group_path]  # just in case something went wrong

    param_sig = create_param_signature(nucleus_params, mf_params)
    group = h5file.create_group(group_path)

    # let it be for a while
    group.attrs.update(mf_params)
    group.attrs.update(nucleus_params)

    group.attrs['param_signature'] = param_sig

    # single dataset might be more convenient for further sampling
    # group.create_dataset('coordinates_gal', data=results[:, [2, 3]], compression='gzip') # lat_gal, lon_gal
    # id 4, 5 below -- deflection and flag of maximum length: [lat_e, lon_e, deflection, max_len_mask]
    # group.create_dataset('coordinates_earth', data=results[:, [0, 1, 4, 5]], compression='gzip')

    # [lat_ini (earth), lon_ini, lat_res (gal), lon_res, deflection]
    group.create_dataset('deflection_map', data=results, compression='gzip')

    indices = [hash((lat, lon)) for lat, lon in results[:, [2, 3]]]
    group.create_dataset('index', data=indices)


def create_param_signature(nucleus_params: dict, mf_params: dict) -> str:
    if not 'R' in nucleus_params.keys():
        rb = get_bin_index(rigidity=nucleus_params['E']/nucleus_params['Z'])
    else:
        rb = get_bin_index(rigidity=nucleus_params['R'])
    # Sort mf_params by key for consistent ordering
    mf_str = "_".join([f"{k}={v}" for k, v in sorted(mf_params.items())])
    return f"Rbin{rb:03d}_mf_{mf_str}"


def find_deflection_in_group(lat: float, lon: float, group: h5py.Group) -> Optional[np.ndarray]:
    """
    Finds the index of corresponding entry by given lat_gal, lon_gal, rigidity[E_EeV/Z]
    File must be opened when looking for an entry

    Parameters
    ----------
        lat[float] - latitude at the boundary of the Galaxy
        lon[float] - longitude at the boundary of the Galaxy

    Returns
    ----------
        fields[float] | None - if found, lat_earth[deg], lon_earth[deg], deflection[deg], else None
    """
    target_hash = hash((lat, lon))

    indices = group['index'][:]
    idx = np.where(indices == target_hash)[0]

    if len(idx) > 0:
        fields = group['deflection_map'][idx[0]]
        return fields

    return None


def get_interpolated_deflection_map(r: float, rbins: list, maps: list):
    """

    :param r: rigidity
    :return:
    """
    r1, r2 = rbins
    m1, m2 = maps

    lat_ini, lon_ini = m1[:, 0],  m1[:, 1]
    lat_res1, lon_res1 = m1[:, 2], m1[:, 3]
    lat_res2, lon_res2 = m2[:, 2], m2[:, 3]

    c1 = SkyCoord(l=lon_res1 * u.deg, b=lat_res1 * u.deg, frame='galactic')
    c2 = SkyCoord(l=lon_res2 * u.deg, b=lat_res2 * u.deg, frame='galactic')

    v1 = np.array(c1.cartesian.xyz).T  # (N, 3)
    v2 = np.array(c2.cartesian.xyz).T  # (N, 3)

    t = (r - r1) / (r2 - r1)
    v = (1 - t) * v1 + t * v2
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    cx = SkyCoord(v[:, 0], v[:, 1], v[:, 2],
                  unit=(u.one, u.one, u.one),
                  representation_type='cartesian',
                  frame='galactic')

    # Recompute deflection as angular separation between init and res
    c_ini = SkyCoord(l=lon_ini * u.deg, b=lat_ini * u.deg, frame='galactic')
    sep = c_ini.separation(cx)
    deflection_deg = sep.deg

    cx = cx.represent_as('spherical')
    lon_res_deg = cx.lon.deg
    lat_res_deg = cx.lat.deg

    lon_res_deg = ((lon_res_deg + 180.0) % 360.0) - 180.0

    return np.column_stack([lat_ini, lon_ini, lat_res_deg, lon_res_deg, deflection_deg, m1[:, -1] + m2[:, -1]])


def get_neighboring_rbins(r: float) -> tuple[float, float]:
    """
    Find the neighboring bins for a given rigidity value r.

    :param r: rigidity value to find neighbors for
    :param rbins: array of rigidity bin edges (sorted in increasing order)
    :return: tuple (r_left, r_right, idx_left, idx_right, maps_left, maps_right)
             where idx_left and idx_right are the indices in the rbins array
    """
    global R_BINS

    if r <= R_BINS[0]:
        r1, r2 = R_BINS[0], R_BINS[1]
    elif r >= R_BINS[-1]:
        r1, r2 = R_BINS[-2], R_BINS[-1]
    else:
        idx2 = np.searchsorted(R_BINS, r)
        idx1 = idx2 - 1
        r1, r2 = R_BINS[idx1], R_BINS[idx2]

    return r1, r2


def get_deflection_map(nucleus_params: dict, _h5file: h5py.File, mf_params: dict) -> np.ndarray:
    """
    Searches for closest rigidity bins and interpolates coordinate map for given R value
    :param nucleus_params:
    :param _h5file:
    :param mf_params:
    :return:
    """
    r = nucleus_params['E']/nucleus_params['Z']
    r1, r2 = get_neighboring_rbins(r=r)

    nucleus_params['R'] = r1
    group1 = find_group(_h5file, mf_params=mf_params, nucleus_params=nucleus_params)

    nucleus_params['R'] = r2
    group2 = find_group(_h5file, mf_params=mf_params, nucleus_params=nucleus_params)

    nucleus_params.pop('R')

    map1 = _h5file[group1]['deflection_map'][:]
    map2 = _h5file[group2]['deflection_map'][:]

    _map = get_interpolated_deflection_map(r, [r1, r2], [map1, map2])

    return _map
