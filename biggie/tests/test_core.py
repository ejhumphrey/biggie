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


def test_Field___init__():
    pass


def test_Field_value():
    pass


def test_Field_value_setter():
    pass


def test_Field_from_hdf5_dataset():
    pass


def test_LazyField___init__():
    pass


def test_LazyField_value():
    pass


def test_LazyField_attrs():
    pass


def test_Entity_getitem(data):
    assert data.e.a == data.e['a'].value, \
        "Failed to initialize keys / attributes."


def test_Entity_setitem(data):
    e = biggie.Entity(a=42)
    assert e.a == 42, "Failed to initialize int."
    e.a = 13
    assert e.a == 13


def test_Entity_int_field(data):
    assert data.e.a == data.a, "Failed to initialize an int."


def test_Entity_str_field(data):
    # import pdb;pdb.set_trace()
    assert data.e.b == data.b, "Failed to initialize a string."


def test_Entity_list_field(data):
    assert data.e.c.tolist() == data.c, "Failed to initialize a list."


def test_Entity_ndarray_field(data):
    np.testing.assert_array_equal(
        data.e.d, data.d, "Failed to initialize a numpy array.")


def test_Entity_keys(data):
    keys = sorted(data.e.keys())
    exp_keys = [k for k in 'abcd']
    assert keys == exp_keys


def test_Entity_values():
    data = biggie.Entity(a=0, b=1, c=2, d=3)
    values = list(range(4))
    exp_values = [data[k].value for k in 'abcd']
    assert values == exp_values


def test_Entity_items(data):
    # TODO: This test could be better, matching c & d are tough.
    items = sorted(data.e.items())
    exp_items = [(k, getattr(data, k)) for k in 'abcd']
    assert len(items) == len(exp_items)
    assert exp_items[0] == ('a', data.a)
    assert exp_items[1] == ('b', data.b)
