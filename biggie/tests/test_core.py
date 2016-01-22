import pytest
import numpy as np

import biggie


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
        e = biggie.Entity(
            a=some_int, b=some_str,
            c=some_list, d=some_ndarray)

    return Sample


def test_Entity_getitem(data):
    assert data.e.a == data.e['a'].value, \
        "Failed to initialize keys / attributes."


def test_Entity_int_field(data):
    assert data.e.a == data.a, "Failed to initialize an int."


def test_Entity_str_field(data):
    assert data.e.b.tostring() == data.b, "Failed to initialize a string."


def test_Entity_list_field(data):
    assert data.e.c.tolist() == data.c, "Failed to initialize a list."


def test_Entity_ndarray_field(data):
    np.testing.assert_array_equal(
        data.e.d, data.d, "Failed to initialize a numpy array.")


def test_Entity_values(data):
    values = data.e.values()
    assert values['a'] == data.a, "Failed to initialize a numpy array."
