import pytest
import h5py
import numpy as np
import os
import random
import tempfile as tmp
import uuid

import biggie
import biggie.core as core
import biggie.util as util

MIN_ROUNDS = 2000
MAX_ITEMS = 5000


@pytest.fixture(scope='module')
def data():
    some_int = 3
    some_str = 'im_a_string'
    some_list = [1, 2, 3]
    some_ndarray = np.arange(5)

    class Sample(object):
        a = some_int
        b = some_str
        c = some_list
        d = some_ndarray
        e = core.Entity(
            a=some_int, b=some_str,
            c=some_list, d=some_ndarray)

    return Sample


@pytest.mark.unit
def test_Field___init__():
    field = core.Field('test')
    assert field is not None


@pytest.mark.unit
def test_Field_value():
    field = core.Field('test')
    assert field.value == 'test'


@pytest.mark.unit
def test_Field_value_setter():
    field = core.Field([1, 2, 3])
    assert isinstance(field.value, np.ndarray)
    np.testing.assert_array_equal(field.value, np.arange(1, 4))


@pytest.mark.unit
def test_Field_slice():
    field = core.Field(np.arange(20).reshape(4, 5))
    slidx = [slice(1, 3), slice(1, 3)]
    exp_res = np.array([[6, 7],
                        [11, 12]])
    np.testing.assert_array_equal(field.slice(slidx), exp_res)


@pytest.mark.skipif(True)
def testbench_Field_slice(benchmark):
    ndim = 128
    value = np.arange(ndim*ndim).reshape(ndim, ndim)
    field = core.Field(value)
    slidx = (slice(32, 36), slice(32, 36))
    exp_res = value[slidx]
    act_res = benchmark(field.slice, slidx)
    np.testing.assert_array_equal(act_res, exp_res)


@pytest.mark.unit
def test_Field_from_hdf5_dataset():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    fh = h5py.File(fp.name)
    key = 'test'
    value = np.arange(20).reshape(4, 5)
    dset = fh.create_dataset(key, data=value)
    field = core.Field.from_hdf5_dataset(dset)
    np.testing.assert_array_equal(field.value, value)


@pytest.mark.unit
def test_LazyField___init__():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    fh = h5py.File(fp.name)
    key = 'test'
    value = np.arange(20).reshape(4, 5)
    dset = fh.create_dataset(key, data=value)
    field = core.LazyField(dset)
    assert field is not None


@pytest.mark.unit
def test_LazyField_value():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    fh = h5py.File(fp.name)
    key = 'test'
    value = np.arange(20).reshape(4, 5)
    dset = fh.create_dataset(key, data=value)
    field = core.LazyField(dset)
    np.testing.assert_array_equal(field.value, value)


@pytest.mark.unit
def test_LazyField_slice():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    fh = h5py.File(fp.name)
    key = 'test'
    value = np.arange(20).reshape(4, 5)
    dset = fh.create_dataset(key, data=value)
    field = core.LazyField(dset)
    slidx = (slice(1, 3), slice(1, 3))
    np.testing.assert_array_equal(field.slice(slidx), value[slidx])


@pytest.fixture
def h5py_data():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    fh = h5py.File(fp.name)

    value = np.arange(2000*200).reshape(2000, 200)
    keys = []
    for key in util.uniform_hexgen(3, 256):
        fh.create_dataset(name=key, data=value)
        keys += [key]
        if len(keys) >= MAX_ITEMS:
            break

    fh.close()
    slidx = (slice(32, 36), slice(32, 36))
    return (fp, keys, value, slidx)


@pytest.mark.benchmark(min_rounds=MIN_ROUNDS)
def testbench_h5py_slice(benchmark, h5py_data):
    def fx(fh, keys, slidx):
        key = keys[random.randint(0, len(keys) - 1)]
        return fh[key][slidx]

    fp, keys, value, slidx = h5py_data
    fh = h5py.File(fp.name)
    exp_res = value[slidx]
    act_res = benchmark(fx, fh, keys, slidx)
    np.testing.assert_array_equal(act_res, exp_res)


@pytest.mark.benchmark(min_rounds=MIN_ROUNDS)
def testbench_h5py_value_slice(benchmark, h5py_data):
    def fx(fh, keys, slidx):
        key = keys[random.randint(0, len(keys) - 1)]
        return fh[key].value[slidx]

    fp, keys, value, slidx = h5py_data
    fh = h5py.File(fp.name)
    exp_res = value[slidx]
    act_res = benchmark(fx, fh, keys, slidx)
    np.testing.assert_array_equal(act_res, exp_res)


@pytest.mark.benchmark(min_rounds=MIN_ROUNDS)
def testbench_h5py_asarray_slice(benchmark, h5py_data):
    def fx(fh, keys, slidx):
        key = keys[random.randint(0, len(keys) - 1)]
        return np.asarray(fh[key].value)[slidx]

    fp, keys, value, slidx = h5py_data
    fh = h5py.File(fp.name)
    exp_res = value[slidx]
    act_res = benchmark(fx, fh, keys, slidx)
    np.testing.assert_array_equal(act_res, exp_res)


@pytest.mark.benchmark(min_rounds=MIN_ROUNDS)
def testbench_LazyField_slice(benchmark, h5py_data):
    def fx(fields, slidx):
        idx = random.randint(0, len(fields)-1)
        return fields[idx].slice(slidx)

    fp, keys, value, slidx = h5py_data
    fh = h5py.File(fp.name)
    exp_res = value[slidx]
    fields = [core.LazyField(fh[key]) for key in keys]
    act_res = benchmark(fx, fields, slidx)
    np.testing.assert_array_equal(act_res, exp_res)


@pytest.mark.benchmark(min_rounds=MIN_ROUNDS)
def testbench_LazyField_slice__dataset(benchmark, h5py_data):
    def fx(fields, slidx):
        idx = random.randint(0, len(fields)-1)
        return fields[idx]._dataset[slidx]

    fp, keys, value, slidx = h5py_data
    fh = h5py.File(fp.name)
    exp_res = value[slidx]
    fields = [core.LazyField(fh[key]) for key in keys]
    act_res = benchmark(fx, fields, slidx)
    np.testing.assert_array_equal(act_res, exp_res)


@pytest.fixture
def npz_data():
    """Populate a directory of NPZ files."""
    tdir = tmp.TemporaryDirectory()
    value = np.arange(2000*200).reshape(2000, 200)
    fpaths = []
    for n in range(MAX_ITEMS):
        basename = uuid.uuid4()
        fpath = os.path.join(tdir.name, "{}.npz".format(basename))
        fpaths += [fpath]
        np.savez(fpath, data=value)
        if len(fpaths) >= MAX_ITEMS:
            break

    slidx = (slice(32, 36), slice(32, 36))
    return tdir, fpaths, value, slidx


@pytest.mark.benchmark(min_rounds=MIN_ROUNDS)
def testbench_npz_slice(benchmark, npz_data):
    def npz_slice(fpaths, slidx):
        fpath = fpaths[random.randint(0, len(fpaths)-1)]
        return np.load(fpath)['data'][slidx]

    tdir, fpaths, value, slidx = npz_data
    exp_res = value[slidx]
    act_res = benchmark(npz_slice, fpaths, slidx)
    np.testing.assert_array_equal(act_res, exp_res)


@pytest.mark.unit
def test_Entity_getitem(data):
    assert data.e.a == data.e['a'].value, \
        "Failed to initialize keys / attributes."


@pytest.mark.unit
def test_Entity_setitem(data):
    e = biggie.Entity(a=42)
    assert e.a == 42, "Failed to initialize int."
    e.a = 13
    assert e.a == 13


@pytest.mark.unit
def test_Entity_int_field(data):
    assert data.e.a == data.a, "Failed to initialize an int."


@pytest.mark.unit
def test_Entity_str_field(data):
    # import pdb;pdb.set_trace()
    assert data.e.b == data.b, "Failed to initialize a string."


@pytest.mark.unit
def test_Entity_list_field(data):
    assert data.e.c.tolist() == data.c, "Failed to initialize a list."


@pytest.mark.unit
def test_Entity_ndarray_field(data):
    np.testing.assert_array_equal(
        data.e.d, data.d, "Failed to initialize a numpy array.")


@pytest.mark.unit
def test_Entity_keys(data):
    keys = sorted(data.e.keys())
    exp_keys = [k for k in 'abcd']
    assert keys == exp_keys


@pytest.mark.unit
def test_Entity_values():
    data = biggie.Entity(a=0, b=1, c=2, d=3)
    values = list(range(4))
    exp_values = [data[k].value for k in 'abcd']
    assert values == exp_values


@pytest.mark.unit
def test_Entity_items(data):
    items = dict(data.e.items())
    exp_items = dict([(k, getattr(data, k)) for k in 'abcd'])
    assert len(items) == len(exp_items)
    assert items['a'] == exp_items['a']
    assert items['b'] == exp_items['b']
    assert items['c'].tolist() == exp_items['c']
    np.testing.assert_array_equal(
        items['d'], exp_items['d'], "Failed to initialize a numpy array.")
