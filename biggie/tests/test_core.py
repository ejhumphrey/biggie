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


@pytest.mark.unit
def test_Field___init__():
    pass


@pytest.mark.unit
def test_Field_value():
    pass


@pytest.mark.unit
def test_Field_value_setter():
    pass


@pytest.mark.unit
def test_Field_from_hdf5_dataset():
    pass


@pytest.mark.unit
def test_LazyField___init__():
    pass


@pytest.mark.unit
def test_LazyField_value():
    pass


@pytest.mark.unit
def test_LazyField_attrs():
    pass


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
