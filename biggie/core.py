"""Core Data Objects


The primary object is an Entity, which consists of named Fields. Each Field
returns the data (scalar, string, list, or array) saved to the attribute.

The goal of Entities / Fields is to provide a seamless de/serialization
data structure for scaling well with potentially massive datasets.
"""

import numpy as np
import six


class Field(object):
    """Data wrapper.

    You should seldom, if ever, need to create Fields explicitly.

    TODO: In the future, be a bit more clever about caching data types in
    the `attrs` attribute.
    """
    def __init__(self, value):
        """Entity field.

        Note : Embedded entities (nesting) is not currently supported.

        Parameters
        ----------
        value : object
            Supported types include all builtins and np.ndarrays.
        """
        self._attrs = dict()
        self.value = value

    @property
    def attrs(self):
        return self._attrs

    @property
    def value(self):
        value = self._value
        if isinstance(value, np.ndarray) and value.dtype.type is np.str_:
            self._value = str(value)
        return self._value

    @value.setter
    def value(self, value):
        """All values get mapped through numpy datatypes for h5py safety."""
        self._value = np.asarray(value)

    @classmethod
    def from_hdf5_dataset(cls, hdf5_dataset):
        """This might be poor practice."""
        return _LazyField(hdf5_dataset)


class _LazyField(Field):
    """Lazy-loading Field for reading data from HDF5 files.

    Note: Do not use directly. This class provides a common interface with
    Fields, but only returns information as needed, wrapping h5py types."""
    def __init__(self, hdf5_dataset):
        self._dataset = hdf5_dataset
        self._value = None
        self._attrs = dict()

    @property
    def value(self):
        """LazyFields only pull data into the namespace when accessed."""
        if self._value is None:
            self._value = np.asarray(self._dataset.value)
        return super(_LazyField, self).value()

    @property
    def attrs(self):
        """Manual override for pulling attributes into the namespace."""
        if self._attrs is None:
            self._attrs = dict(self._dataset.attrs)
        return self._attrs


class Entity(object):
    """Struct-like object for getting named fields into and out of a Stash.

    Much like a native Python dictionary, the keyword arguments of an Entity
    will become keys of the object. Additionally, and more importantly, these
    keys are also named attributes:

    >>> x = Entity(a=3, b=5)
    >>> x.a == 3
    True
    >>> x['a'].value == 3
    True

    See tests/test_core.py for more examples.
    """
    def __init__(self, **kwargs):
        object.__init__(self)
        for key, value in six.iteritems(kwargs):
            self[key] = value

    def __repr__(self):
        """Render the Field names of the Entity as a string."""
        return '{}<{}>'.format(self.__class__.__name__, ", ".join(self.keys()))

    def __setitem__(self, key, value):
        """Set value for the given key.

        Parameters
        ----------
        key: str
            Attribute name, must be valid python attribute syntax.

        value: scalar, string, list, or np.ndarray
            Data corresponding to the given key, stored as a Field.
        """
        self.__dict__[key] = Field(value)

    def __setattr__(self, key, value):
        """Set value for key.

        Parameters
        ----------
        key: str
            Attribute name, must be valid python attribute syntax.

        value: scalar, string, list, or np.ndarray
            Data corresponding to the given key, stored as a Field.
        """
        self[key] = value

    def __getitem__(self, key):
        """Get value for key."""
        return self.__dict__[key]

    def __getattribute__(self, key):
        """Unnecessary, but could make Fields more transparent?"""
        obj = object.__getattribute__(self, key)
        if isinstance(obj, Field) or isinstance(obj, _LazyField):
            # Back out Fields transparently.
            obj = obj.value
        return obj

    def __delitem__(self, key):
        """Remove an attribute from the Entity."""
        del self.__dict__[key]

    def __len__(self):
        return len(self.keys())

    def keys(self):
        """Returns a list of field names (keys) of the entity."""
        return self.__dict__.keys()

    def values(self):
        """Returns a list of the values in the entity."""
        return [self[k].value for k in self.keys()]

    def items(self):
        """Return the (key, value) items of the entity."""
        return dict([(k, getattr(self, k)) for k in self.keys()])

    @classmethod
    def from_hdf5_group(cls, group):
        """writeme."""
        new_grp = cls()
        for key in group:
            new_grp.__dict__[key] = _LazyField(group[key])
        return new_grp
