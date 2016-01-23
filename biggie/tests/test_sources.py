import pytest

import h5py
from joblib import Parallel, delayed
import json
import numpy as np
import tempfile as tmp

import biggie
import biggie.util as util


def test_Stash__init__():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    stash = biggie.Stash(fp.name)
    assert stash is not None


def test_Stash___load_keymap__():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    fh = h5py.File(fp.name)
    keymap = dict(test='00')
    fh[biggie.Stash.__KEYMAP__] = json.dumps(keymap)

    fh = biggie.Stash(fp.name)
    assert fh._keymap == keymap


def test_Stash___dump_keymap__():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    stash = biggie.Stash(fp.name)
    keymap = dict(test='00')
    stash._keymap = keymap
    stash.close()

    fh = h5py.File(fp.name)
    dset = fh.get(biggie.Stash.__KEYMAP__)
    assert dset.value == json.dumps(keymap)


def test_Stash_add():
    fp = tmp.NamedTemporaryFile(suffix=".hdf5")
    stash = biggie.Stash(fp.name)

    key = 'foo'
    entity = biggie.Entity(a=3, b='im_a_string', c=[1, 2, 3], d=np.arange(5))
    stash.add(key, entity)
    assert key in stash.keys()


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


def test_Stash_thread_safe():
    pass


# def test_Collection___init__():
#     collec = biggie.Collection('test')
#     assert collec is not None


# def test_Collection___init__with_stash():
#     collec = biggie.Collection(
#         name='test',
#         stash_kwargs=dict(filename='/tmp/deleteme.hdf5'))
#     assert collec is not None
#     assert collec.stash is not None
