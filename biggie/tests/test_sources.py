import pytest

import glob
import h5py
from joblib import Parallel, delayed
import json
import numpy as np
import os
import tempfile as tmp

import biggie
import biggie.util as util


@pytest.mark.unit
def test_Stash__init__():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    stash = biggie.Stash(fp.name)
    assert stash is not None


@pytest.mark.unit
def test_Stash___load_keymap__():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    fh = h5py.File(fp.name)
    keymap = dict(test='00')
    fh[biggie.Stash.__KEYMAP__] = json.dumps(keymap)

    fh = biggie.Stash(fp.name)
    assert fh._keymap == keymap


@pytest.mark.unit
def test_Stash___dump_keymap__():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    stash = biggie.Stash(fp.name)
    keymap = dict(test='00')
    stash._keymap = keymap
    stash.close()

    fh = h5py.File(fp.name)
    dset = fh.get(biggie.Stash.__KEYMAP__)
    assert dset.value == json.dumps(keymap)


@pytest.mark.unit
def test_Stash_add():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    stash = biggie.Stash(fp.name)

    key = 'foo'
    entity = biggie.Entity(a=3, b='im_a_string', c=[1, 2, 3], d=np.arange(5))
    stash.add(key, entity)
    assert key in stash.keys()


@pytest.mark.unit
@pytest.fixture(scope='module')
def data():
    class Data(object):
        entity = biggie.Entity(
            a=3, b='im_a_string', c=[1, 2, 3], d=np.arange(5))
        key = 'my_key'
        fp = tmp.NamedTemporaryFile(suffix=".hdf5")

    stash = biggie.Stash(Data.fp.name)
    stash.add(Data.key, Data.entity)
    stash.close()
    return Data


@pytest.mark.unit
def test_Stash_get(data):
    stash = biggie.Stash(data.fp.name)
    loaded_entity = stash.get(data.key)

    assert data.entity.a == loaded_entity.a, \
        "Could not reconstitute entity.a"

    assert data.entity.b == loaded_entity.b, \
        "Could not reconstitute entity.b"

    assert data.entity.c.tolist() == loaded_entity.c.tolist(), \
        "Could not reconstitute entity.c"

    np.testing.assert_array_equal(
        data.entity.d,
        loaded_entity.d,
        "Could not reconstitute entity.d")


@pytest.mark.unit
def test_Stash_overwrite(data):
    stash = biggie.Stash(data.fp.name)
    loaded_entity = stash.get(data.key)
    loaded_entity.e = 4
    with pytest.raises(ValueError):
        stash.add(data.key, loaded_entity)

    stash.add(data.key, loaded_entity, True)
    stash.close()

    stash = biggie.Stash(data.fp.name)
    another_entity = stash.get(data.key)
    assert len(stash) == 1
    assert another_entity.e == 4


@pytest.mark.unit
def test_Stash_cache(data):
    stash = biggie.Stash(data.fp.name, cache_size=100)
    loaded_entity = stash.get(data.key)

    np.testing.assert_array_equal(
        data.entity.d,
        loaded_entity.d,
        "Could not reconstitute entity.d")

    np.testing.assert_array_equal(
        data.entity.d,
        stash.__local__[data.key].d,
        "Failed to cache entity")


# Helper function
def touch_one(stash, keys=None, key=None):
    key = np.random.choice(list(keys)) if keys else key
    entity = stash.get(key)
    np.asarray(entity.data)
    # entity.data
    return True


@pytest.mark.unit
def test_Stash_thread_safe():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    stash = biggie.Stash(fp.name, cache_size=0)
    shape = (64, 64)
    data_gen = util.random_ndarray_generator(shape, max_items=25)
    for key, value in data_gen:
        stash.add(key, biggie.Entity(data=value))

    pool = Parallel(n_jobs=2)
    fx = delayed(touch_one)
    stash = biggie.Stash(fp.name, cache_size=0)
    res = pool(fx(stash=stash, key=key) for key in stash.keys())
    assert all(res)


data_params = [(10, (64, 64)),
               (100, (64, 64)),
               (1000, (64, 64)),
               (10000, (64, 64)),
               (10, (1024, 128)),
               (100, (1024, 128)),
               (1000, (1024, 128)),
               (10000, (1024, 128))]


@pytest.fixture(params=data_params,
                ids=["(n={}, ndim={})".format(*p) for p in data_params],
                scope="module",)
def stash_fp(request):
    """Populate a temporary stash file with data."""
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    stash = biggie.Stash(fp.name, cache_size=0)

    data_gen = util.random_ndarray_generator(
        request.param[1], max_items=request.param[0])
    for key, value in data_gen:
        stash.add(key, biggie.Entity(data=value))
    stash.close()
    return fp


@pytest.mark.benchmark
def test_Stash_stress_random(benchmark, stash_fp):
    """Stress test random-access reads on a Stash file."""
    stash = biggie.Stash(stash_fp.name, cache_size=0)
    assert benchmark(touch_one, stash, keys=stash.keys())
    # stash.close()
    # stash_fp.close()


@pytest.mark.benchmark
def test_Stash_stress_ordered(benchmark, stash_fp):
    """Stress test random-access reads on a Stash file."""
    stash = biggie.Stash(stash_fp.name, cache_size=0)
    # print(list(stash._keymap.items()))
    addrmap = sorted([(a, k) for k, a in stash._keymap.items()])
    print(addrmap[:5])
    keys = [pair[1] for pair in addrmap]
    assert benchmark(touch_one, stash, keys=keys)
    # stash.close()
    # stash_fp.close()


@pytest.fixture(params=data_params,
                ids=["(n={}, ndim={})".format(*p) for p in data_params],
                scope="module",)
def npz_dir(request):
    """Populate a directory of NPZ files."""
    tdir = tmp.TemporaryDirectory()
    data_gen = util.random_ndarray_generator(
        request.param[1], max_items=request.param[0])
    for basename, value in data_gen:
        fout = os.path.join(tdir.name, "{}.npz".format(basename))
        np.savez(fout, data=value)

    return tdir


# Helper function
def touch_one_npz(fpaths=None, fpath=None):
    fpath = np.random.choice(fpaths) if fpaths else fpath
    arc = np.load(fpath)
    np.asarray(arc['data'])
    return True


@pytest.mark.benchmark
def test_npz_stress_random(benchmark, npz_dir):
    """Stress test random-access reads on NPZ archives."""
    fpaths = glob.glob(os.path.join(npz_dir.name, "*.npz"))
    assert benchmark(touch_one_npz, fpaths=fpaths)
    # npz_dir.cleanup()


# Helper function
def touch_next_npz(fpaths):
    fpath = fpaths.pop(0)
    arc = np.load(fpath)
    np.asarray(arc['data'])
    fpaths.append(fpath)
    return True


@pytest.mark.benchmark
def test_npz_stress_ordered(benchmark, npz_dir):
    """Stress test random-access reads on NPZ archives."""
    fpaths = sorted(glob.glob(os.path.join(npz_dir.name, "*.npz")))
    assert benchmark(touch_next_npz, fpaths=fpaths)
    # npz_dir.cleanup()

# @pytest.mark.benchmark
# def test_Stash__fhandle(benchmark, stash_fp):
#     def verify_handle(stash):
#         return stash._fhandle is not None

#     stash = biggie.Stash(stash_fp.name, cache_size=0)
#     assert benchmark(verify_handle, stash)


# def test_Collection___init__():
#     collec = biggie.Collection('test')
#     assert collec is not None


# def test_Collection___init__with_stash():
#     collec = biggie.Collection(
#         name='test',
#         stash_kwargs=dict(filename='/tmp/deleteme.hdf5'))
#     assert collec is not None
#     assert collec.stash is not None
