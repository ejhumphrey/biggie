import pytest

import tempfile as tmp
import numpy as np

import biggie


@pytest.fixture(scope='module')
def data():
    class Data(object):
        entity = biggie.Entity(
            a=3, b='im_a_string', c=[1, 2, 3], d=np.arange(5))
        key = 'my_key'
        fpath = tmp.mktemp(suffix=".hdf5", dir=tmp.gettempdir())

    stash = biggie.Stash(Data.fpath)
    stash.add(Data.key, Data.entity)
    stash.close()

    return Data


def test_stash_persistence(data):
    stash = biggie.Stash(data.fpath)
    loaded_entity = stash.get(data.key)

    assert data.entity.a == loaded_entity.a, \
        "Could not reconstitute entity.a"

    assert data.entity.b.tostring() == loaded_entity.b.tostring(), \
        "Could not reconstitute entity.b"

    assert data.entity.c.tolist() == loaded_entity.c.tolist(), \
        "Could not reconstitute entity.c"

    np.testing.assert_array_equal(
        data.entity.d,
        loaded_entity.d,
        "Could not reconstitute entity.d")


def test_stash_overwrite(data):
    stash = biggie.Stash(data.fpath)
    loaded_entity = stash.get(data.key)
    loaded_entity.e = 4
    with pytest.raises(ValueError):
        stash.add(data.key, loaded_entity)

    stash.add(data.key, loaded_entity, True)
    stash.close()

    stash = biggie.Stash(data.fpath)
    another_entity = stash.get(data.key)
    assert len(stash) == 1
    assert another_entity.e == 4


def test_stash_cache(data):
    stash = biggie.Stash(data.fpath, cache=True)
    loaded_entity = stash.get(data.key)

    np.testing.assert_array_equal(
        data.entity.d,
        loaded_entity.d,
        "Could not reconstitute entity.d")

    np.testing.assert_array_equal(
        data.entity.d,
        stash.__local__[data.key].d,
        "Failed to cache entity")
