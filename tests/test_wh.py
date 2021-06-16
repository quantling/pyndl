#!/usr/bin/env python3

# pylint: disable=C0111, redefined-outer-name


from collections import defaultdict, OrderedDict
import os
import tempfile

import numpy as np
import xarray as xr
import pandas as pd
import pytest

from pyndl import ndl, count, wh

TEST_ROOT = os.path.join(os.path.pardir, os.path.dirname(__file__))
FILE_PATH_WH = os.path.join(TEST_ROOT, "resources/event_file_wh.tab.gz")

TMP_PATH = tempfile.mkdtemp()

ETA = 0.01

CONTINUE_SPLIT_POINT = 3


@pytest.mark.nolinux
def test_consistency_wh():
    events = FILE_PATH_WH

    cue_vectors = xr.DataArray(np.array([[0.2, 0.11, 0.5, 0, 0], [0, 0, 0.5, 0.11, 0.2], [0.2, 0, 0, 0, 0.2]]),
                               dims=('cues', 'cue_vector_dimensions'),
                               coords={'cues': ['a', 'b', 'c'],
                                       'cue_vector_dimensions': ['dim1', 'dim2', 'dim3', 'dim4', 'dim5']})
    outcome_vectors = xr.DataArray(np.array([[0.2, 1.], [0.5, 0], [0, 0.5], [1., 0.2]]),
                                   dims=('outcomes', 'outcome_vector_dimensions'),
                                   coords={'outcomes': ['A', 'B', 'C', 'D'],
                                           'outcome_vector_dimensions': ['o_dim1', 'o_dim2']})

    _ = wh.dict_wh(events, ETA, cue_vectors=cue_vectors, outcome_vectors=outcome_vectors)
    weights = wh.dict_wh(events, ETA, cue_vectors=cue_vectors, outcome_vectors=outcome_vectors,
                         make_data_array=True, verbose=True)

    weights_np = wh.wh(events, ETA, cue_vectors=cue_vectors, outcome_vectors=outcome_vectors,
                       method='numpy', verbose=True)
    weights_openmp = wh.wh(events, ETA, cue_vectors=cue_vectors,
                           outcome_vectors=outcome_vectors, method='openmp', verbose=True)
    assert np.all(weights == weights_np)
    assert np.all(weights == weights_openmp)

    cue_vectors = xr.DataArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                               dims=('cues', 'cue_vector_dimensions'),
                               coords={'cues': ['a', 'b', 'c'], 'cue_vector_dimensions': ['dim1', 'dim2', 'dim3']})
    outcome_vectors = xr.DataArray(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                                   dims=('outcomes', 'outcome_vector_dimensions'),
                                   coords={'outcomes': ['A', 'B', 'C', 'D'],
                                           'outcome_vector_dimensions': ['dim1', 'dim2', 'dim3', 'dim4']})

    weights_wh = wh.dict_wh(events, ETA, cue_vectors, outcome_vectors, make_data_array=True)
    weights_ndl = ndl.dict_ndl(events, alphas=defaultdict(lambda: 1), betas=(ETA, ETA), make_data_array=True)

    weights_wh = weights_wh.loc[{'outcome_vector_dimensions': ['dim1', 'dim2', 'dim3', 'dim4'],
                                 'cue_vector_dimensions': ['dim1', 'dim2', 'dim3']}]
    weights_wh.coords['outcome_vector_dimensions'] = ['A', 'B', 'C', 'D']
    weights_wh.coords['cue_vector_dimensions'] = ['a', 'b', 'c']
    weights_wh = weights_wh.rename({'outcome_vector_dimensions': 'outcomes', 'cue_vector_dimensions': 'cues'})
    unequal, unequal_ratio = compare_arrays(events, weights_wh, weights_ndl)
    print(unequal)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition


@pytest.mark.nolinux
@pytest.mark.runslow
def test_real_to_real_wh_large():
    events = FILE_PATH_WH
    eta = 0.001

    n_cue_vec_dims = 1000
    n_outcome_vec_dims = 4000
    cue_vectors = xr.DataArray(np.random.random((3, n_cue_vec_dims)), dims=('cues', 'cue_vector_dimensions'),
                               coords={'cues': ['a', 'b', 'c'],
                                       'cue_vector_dimensions': [f'c_dim{ii}' for ii in range(n_cue_vec_dims)]})
    outcome_vectors = xr.DataArray(np.random.random((4, n_outcome_vec_dims)),
                                   dims=('outcomes', 'outcome_vector_dimensions'),
                                   coords={'outcomes': ['A', 'B', 'C', 'D'],
                                           'outcome_vector_dimensions': [f'o_dim{ii}' for ii
                                                                         in range(n_outcome_vec_dims)]})

    # 1 min 41 sec
    weights_np = wh.wh(events, eta, cue_vectors=cue_vectors,
                       outcome_vectors=outcome_vectors, method='numpy')
    # 18 sec
    weights_openmp = wh.wh(events, eta, cue_vectors=cue_vectors,
                           outcome_vectors=outcome_vectors, method='openmp')
    assert np.allclose(weights_openmp.data, weights_np.data)

    # 1 min 41 sec
    weights_split_small = wh.wh(events, eta, outcome_vectors=outcome_vectors,
                                n_outcomes_per_job=100)
    # 18 sec
    weights_openmp = wh.wh(events, eta, outcome_vectors=outcome_vectors, method='openmp')
    assert np.allclose(weights_openmp.data, weights_split_small.data)


@pytest.mark.nolinux
@pytest.mark.runslow
def test_binary_to_real_wh_continue_learning():
    cue_vectors = xr.DataArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
                               dims=('cues', 'cue_vector_dimensions'),
                               coords={'cues': ['a', 'b', 'c'],
                                       'cue_vector_dimensions': ['c_dim1', 'c_dim2', 'c_dim3']})
    outcome_vectors = xr.DataArray(np.array([[0.2, 1., 0.7, -0.2], [0.5, 0, 0.1, 0.0],
                                             [0, 0.5, 0.0, 0.1], [1., 0.2, -0.8, 0.11]]),
                                   dims=('outcomes', 'outcome_vector_dimensions'),
                                   coords={'outcomes': ['A', 'B', 'C', 'D'],
                                           'outcome_vector_dimensions': ['o_dim1', 'o_dim2', 'o_dim3', 'o_dim4']})

    weights = wh.wh(FILE_PATH_WH, ETA, outcome_vectors=outcome_vectors, verbose=True)
    for ii in range(10):
        weights = wh.wh(FILE_PATH_WH, ETA, outcome_vectors=outcome_vectors,
                        weights=weights, n_jobs=ii % 4 + 1,
                        n_outcomes_per_job=4 - ii % 4)

    weights_real_to_real = wh.wh(FILE_PATH_WH, ETA, cue_vectors=cue_vectors,
                                 outcome_vectors=outcome_vectors)
    for _ in range(10):
        weights_real_to_real = wh.wh(FILE_PATH_WH, ETA,
                                     cue_vectors=cue_vectors, outcome_vectors=outcome_vectors,
                                     weights=weights_real_to_real)

    weights_real_to_real.coords['cue_vector_dimensions'] = ['a', 'b', 'c']
    weights_real_to_real = weights_real_to_real.rename({'cue_vector_dimensions': 'cues'})

    unequal, unequal_ratio = compare_arrays(FILE_PATH_WH, weights, weights_real_to_real)
    print(unequal)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition


@pytest.mark.nolinux
def test_binary_to_real_wh():
    cue_vectors = xr.DataArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
                               dims=('cues', 'cue_vector_dimensions'),
                               coords={'cues': ['a', 'b', 'c'],
                                       'cue_vector_dimensions': ['c_dim1', 'c_dim2', 'c_dim3']})
    outcome_vectors = xr.DataArray(np.array([[0.2, 1., 0.7, -0.2], [0.5, 0, 0.1, 0.0],
                                             [0, 0.5, 0.0, 0.1], [1., 0.2, -0.8, 0.11]]),
                                   dims=('outcomes', 'outcome_vector_dimensions'),
                                   coords={'outcomes': ['A', 'B', 'C', 'D'],
                                           'outcome_vector_dimensions': ['o_dim1', 'o_dim2', 'o_dim3', 'o_dim4']})

    weights = wh.wh(FILE_PATH_WH, ETA, outcome_vectors=outcome_vectors, verbose=True)
    # weights.to_netcdf(os.path.join(TEST_ROOT, 'reference/binary_to_real_weights.nc'))
    reference_weights = xr.open_dataarray(os.path.join(TEST_ROOT, 'reference/binary_to_real_weights.nc'))

    unequal, unequal_ratio = compare_arrays(FILE_PATH_WH, weights, reference_weights)
    print(unequal)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition

    weights_real_to_real = wh.wh(FILE_PATH_WH, ETA, cue_vectors=cue_vectors, outcome_vectors=outcome_vectors)

    weights_real_to_real.coords['cue_vector_dimensions'] = ['a', 'b', 'c']
    weights_real_to_real = weights_real_to_real.rename({'cue_vector_dimensions': 'cues'})

    unequal, unequal_ratio = compare_arrays(FILE_PATH_WH, weights, weights_real_to_real)
    print(unequal)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition

    weights_split = wh.wh(FILE_PATH_WH, ETA, outcome_vectors=outcome_vectors, verbose=True, n_outcomes_per_job=2)
    unequal, unequal_ratio = compare_arrays(FILE_PATH_WH, weights, weights_split)
    print(unequal)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition

    for _ in range(1):
        weights_split = wh.wh(FILE_PATH_WH, ETA, weights=weights_split,
                              outcome_vectors=outcome_vectors, verbose=True,
                              n_outcomes_per_job=2)
        weights = wh.wh(FILE_PATH_WH, ETA, weights=weights,
                        outcome_vectors=outcome_vectors, verbose=True)
    unequal, unequal_ratio = compare_arrays(FILE_PATH_WH, weights, weights_split)
    print(unequal)
    print('%.2f ratio unequal' % unequal_ratio)
    print(weights_split)
    print(weights)
    assert len(unequal) == 0  # pylint: disable=len-as-condition


@pytest.mark.nolinux
def test_real_to_binary_wh():
    cue_vectors = xr.DataArray(np.array([[0.2, 0.11, 0.5, 0, 0], [0, 0, 0.5, 0.11, 0.2], [0.2, 0, 0, 0, 0.2]]),
                               dims=('cues', 'cue_vector_dimensions'),
                               coords={'cues': ['a', 'b', 'c'],
                                       'cue_vector_dimensions': ['c_dim1', 'c_dim2', 'c_dim3', 'c_dim4', 'c_dim5']})
    outcome_vectors = xr.DataArray(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
                                   dims=('outcomes', 'outcome_vector_dimensions'),
                                   coords={'outcomes': ['A', 'B', 'C', 'D'],
                                           'outcome_vector_dimensions': ['dim1', 'dim2', 'dim3', 'dim4']})

    weights = wh.wh(FILE_PATH_WH, ETA, cue_vectors=cue_vectors, verbose=True)
    # weights.to_netcdf(os.path.join(TEST_ROOT, 'reference/real_to_binary_weights.nc'))
    reference_weights = xr.open_dataarray(os.path.join(TEST_ROOT, 'reference/real_to_binary_weights.nc'))

    unequal, unequal_ratio = compare_arrays(FILE_PATH_WH, weights, reference_weights)
    print(unequal)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition

    weights_real_to_real = wh.wh(FILE_PATH_WH, ETA, cue_vectors=cue_vectors, outcome_vectors=outcome_vectors)

    weights_real_to_real.coords['outcome_vector_dimensions'] = ['A', 'B', 'C', 'D']
    weights_real_to_real = weights_real_to_real.rename({'outcome_vector_dimensions': 'outcomes'})

    unequal, unequal_ratio = compare_arrays(FILE_PATH_WH, weights, weights_real_to_real)
    print(unequal)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition


@pytest.mark.nolinux
def test_continue_learning_wh():
    events = FILE_PATH_WH

    cue_vectors = xr.DataArray(np.array([[0.2, 0.11, 0.5, 0, 0], [0, 0, 0.5, 0.11, 0.2], [0.2, 0, 0, 0, 0.2]]),
                               dims=('cues', 'cue_vector_dimensions'),
                               coords={'cues': ['a', 'b', 'c'],
                                       'cue_vector_dimensions': ['dim1', 'dim2', 'dim3', 'dim4', 'dim5']})
    outcome_vectors = xr.DataArray(np.array([[0.2, 1.], [0.5, 0], [0, 0.5], [1., 0.2]]),
                                   dims=('outcomes', 'outcome_vector_dimensions'),
                                   coords={'outcomes': ['A', 'B', 'C', 'D'],
                                           'outcome_vector_dimensions': ['o_dim1', 'o_dim2']})

    weights = wh.wh(events, ETA, cue_vectors=cue_vectors, outcome_vectors=outcome_vectors)
    weights = wh.wh(events, ETA, cue_vectors=cue_vectors, outcome_vectors=outcome_vectors, weights=weights)
    # TODO: insert assert

    weights = wh.wh(events, ETA, outcome_vectors=outcome_vectors)
    weights = wh.wh(events, ETA, outcome_vectors=outcome_vectors, weights=weights)
    # TODO: insert assert

    weights = wh.wh(events, ETA, cue_vectors=cue_vectors)
    weights = wh.wh(events, ETA, cue_vectors=cue_vectors, weights=weights)
    # TODO: insert assert


def compare_arrays(file_path, arr1, arr2):
    _, cues, outcomes = count.cues_outcomes(file_path)
    cue_map, outcome_map, _ = generate_mapping(file_path)

    unequal = list()

    if isinstance(arr1, xr.DataArray):
        outcome_dim_name, cue_dim_name = arr1.dims

    if outcome_dim_name == 'outcome_vector_dimensions':
        outcomes = list(arr1[outcome_dim_name].data)
    if cue_dim_name == 'cue_vector_dimensions':
        cues = list(arr1[cue_dim_name].data)

    for outcome in outcomes:
        for cue in cues:
            values = list()
            for array in (arr1, arr2):
                if isinstance(array, np.ndarray):
                    outcome_index = outcome_map[outcome]
                    cue_index = cue_map[cue]
                    values.append(array[outcome_index][cue_index])
                elif isinstance(array, xr.DataArray):
                    values.append(array.loc[{outcome_dim_name: outcome, cue_dim_name: cue}].values)
                elif isinstance(array, pd.DataFrame):
                    values.append(array.loc[outcome][cue])
                else:
                    values.append(array[outcome][cue])

            value1, value2 = values  # pylint: disable=unbalanced-tuple-unpacking
            if not np.isclose(value1, value2, rtol=1e-02, atol=1e-05):
                unequal.append((outcome, cue, value1, value2))

    unequal_ratio = len(unequal) / (len(outcomes) * len(cues))
    return (unequal, unequal_ratio)


def generate_mapping(event_path):
    _, cues, outcomes = count.cues_outcomes(event_path)
    all_cues = list(cues.keys())
    all_outcomes = list(outcomes.keys())
    cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(all_cues)))
    outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(all_outcomes)))

    return (cue_map, outcome_map, all_outcomes)
