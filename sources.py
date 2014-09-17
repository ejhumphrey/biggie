import json
import h5py
import numpy as np

from .util import uniform_hexgen
from .core import Entity


class Stash(h5py.File):
    """On-disk dictionary-like object."""
    __KEYMAP__ = "__KEYMAP__"
    __WIDTH__ = 256
    __DEPTH__ = 3

    def __init__(self, name, mode=None, entity_class=None, cache=False,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Path to file on disk
        mode: str
            Filemode for the object.
        entity: Entity subclass
            Entity class for interpreting objects in the stash.
        """
        h5py.File.__init__(self, name=name, mode=mode, **kwargs)

        if entity_class is None:
            entity_class = Entity
        self._entity_cls = entity_class
        self._cache = cache
        self.__local__ = dict()
        self._keymap = self.__decode_keymap__()
        self._agu = uniform_hexgen(self.__DEPTH__, self.__WIDTH__)

    def __decode_keymap__(self):
        keymap_array = np.array(h5py.File.get(self, self.__KEYMAP__, '{}'))
        return json.loads(keymap_array.tostring())

    def __encode_keymap__(self, keymap):
        return np.array(json.dumps(keymap))

    def __del__(self):
        """Safe default destructor"""
        if self.fid.valid:
            self.close()

    def close(self):
        """write keys and paths to disk"""
        if self.__KEYMAP__ in self:
            del self[self.__KEYMAP__]
        keymap_str = self.__encode_keymap__(self._keymap)
        h5py.File.create_dataset(
            self, name=self.__KEYMAP__, data=keymap_str)

        h5py.File.close(self)

    def __load__(self, key):
        """Deeply load an entity from the underlying HDF5 file."""
        addr = self._keymap.get(key, None)
        if addr is None:
            return addr
        raw_group = h5py.File.get(self, addr)
        raw_key = raw_group.attrs.get("key")
        if raw_key != key:
            raise ValueError(
                "Key inconsistency: received '%s'"
                ", expected '%s'" % (raw_key, key))
        print "Loading %s" % key
        return self._entity_cls.from_hdf5_group(raw_group)

    def get(self, key):
        """Fetch the entity for a given key."""
        value = self.__local__.get(key, self.__load__(key))
        if self._cache:
            self.__local__[key] = value

        return value

    def add(self, key, entity, overwrite=False):
        """Add a key-entity pair to the File.

        Parameters
        ----------
        key: str
            Key to write the value under.
        entity: Entity
            Object to write to file.
        overwrite: bool, default=False
            Overwrite the key-entity pair if the key currently exists.
        """
        # TODO(ejhumphrey): update locals!!
        key = str(key)
        if key in self._keymap:
            if not overwrite:
                raise ValueError(
                    "Data exists for '%s'; did you mean overwrite=True?" % key)
            else:
                addr = self.remove(key)
        else:
            addr = self._agu.next()

        while addr in self:
            addr = self._agu.next()

        self._keymap[key] = addr
        new_grp = self.create_group(addr)
        new_grp.attrs.create(name='key', data=key)
        for dset_key, dset in dict(**entity).iteritems():
            new_dset = new_grp.create_dataset(name=dset_key, data=dset.value)
            for k, v in new_dset.attrs.iteritems():
                dset.attrs.create(name=k, data=v)

    def remove(self, key):
        """Delete a key-entity pair from the stash.

        Parameters
        ----------
        key: str
            Key to remove from the Stash.

        Returns
        -------
        address: str
            Absolute internal address freed in the process.
        """
        addr = self._keymap.pop(key, None)
        if addr is None:
            raise ValueError("The key '%s' does not exist." % key)

        del self[addr]
        return addr

    def keys(self):
        """Return a list of all keys in the Stash."""
        return self._keymap.keys()

    def __paths__(self):
        """Return a list of all absolute archive paths in the Stash."""
        return self._keymap.values()

    def __len__(self):
        return len(self.keys())
