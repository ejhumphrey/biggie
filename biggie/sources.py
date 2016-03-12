"""Data sources in the biggie ecosystem."""

from __future__ import print_function
import h5py
import json
import logging
import numpy as np

import biggie.core as core
import biggie.util as util


class Stash(object):
    """On-disk dictionary-like object."""
    __KEYMAP__ = "__KEYMAP__"
    __WIDTH__ = 256
    __DEPTH__ = 3

    def __init__(self, filename, mode=None, cache_size=False,
                 log_level=logging.INFO, keep_open=True):
        """Create a Stash object, pointing to an hdf5 file on-disk.

        Parameters
        ----------
        filename : str
            Path to file on disk

        mode : str
            Filemode for the object.

        cache_size : int or False-equivalent, default=False
            Number of items to keep cached internally (for speed).

        log_level : int, default=logging.INFO
            Level for setting the internal logger; see logging.X for more info.

        keep_open : bool, default=True
            If True, maintain a reference to the HDF5 file, otherwise re-open
            it when necessary; trades a slight drop in efficiency for parallel
            reads.
        """
        self._filename = filename
        self._mode = mode
        self._keep_open = keep_open
        self.__handle__ = None
        self._cache_size = cache_size
        self.__local__ = dict()
        self._agu = None

        self._logger = logging.getLogger('Stash')
        self._logger.setLevel(log_level)
        self.__load_keymap__()

    @property
    def _fhandle(self):
        fh = None
        if self.__handle__ is None:
            fh = h5py.File(name=self._filename, mode=self._mode)
        if self._keep_open and fh:
            self.__handle__ = fh
        return self.__handle__ if self.__handle__ is not None else fh

    def __load_keymap__(self):
        if not self._fhandle or self.__KEYMAP__ not in self._fhandle:
            self._keymap = dict()
        else:
            keymap_dset = self._fhandle.get(self.__KEYMAP__)
            self._keymap = json.loads(str(keymap_dset.value))

    def __dump_keymap__(self):
        if self.__KEYMAP__ in self._fhandle:
            del self._fhandle[self.__KEYMAP__]

        self._fhandle.create_dataset(
            name=self.__KEYMAP__,
            data=np.str(json.dumps(self._keymap)))

    @property
    def agu(self):
        """Currently, this lil guy causes problems with parallelization.

        TODO: The AGU should be superseded by a "figure-out-the-next-address"
        function, based on the state of the keymap. In the meantime, converting
        the generator to a class might suffice...
        """
        if self._agu is None:
            self._agu = util.uniform_hexgen(self.__DEPTH__, self.__WIDTH__)
        return self._agu

    def __del__(self):
        """Safe default destructor"""
        self.close()

    def close(self):
        """write keys and paths to disk"""
        self.__dump_keymap__()
        if self.__handle__:
            self.__handle__ = self.__handle__.close()

    def __load__(self, key):
        """Deeply load an entity from the base HDF5 file."""
        addr = self._keymap[key]
        raw_group = self._fhandle.get(addr)
        raw_key = raw_group.attrs.get("key")
        if raw_key != key:
            raise ValueError("Key inconsistency: received '{}'"
                             ", expected '{}'".format(raw_key, key))
        self._logger.debug("Loading {}".format(key))
        return core.Entity.from_hdf5_group(raw_group)

    def get(self, key, default=None):
        """Fetch the entity for a given key.

        Parameters
        ----------
        key : str
            Key of the entity to get.

        default : object
            If given, default return entity on unfound keys.
        """
        # Check local cache for the data first.
        entity = self.__local__.get(key, None)

        # If key is not in local (entity == None), go get that sucker.
        entity = self.__load__(key) if entity is None else entity

        # Caching logic.
        if self._cache_size > 0 and (len(self.__local__) >= self._cache_size):
            # TODO: Pick a entity and ditch it.
            pass

        if self._cache_size > 0:
            self.__local__[key] = entity

        return entity

    def add(self, key, entity, overwrite=False):
        """Add a key-entity pair to the Stash.

        Parameters
        ----------
        key : str
            Key to write the entity under.

        entity : Entity or dict
            Data to write to file.

        overwrite : bool, default=False
            Overwrite the key-entity pair if the key currently exists.
        """
        # TODO(ejhumphrey): update locals!!
        key = str(key)
        if key in self._keymap:
            if not overwrite:
                raise ValueError(
                    "Data exists for '{}'; did you mean `overwrite=True?`"
                    "".format(key))
            else:
                addr = self.remove(key)
        else:
            addr = next(self.agu)

        while addr in self._fhandle:
            addr = next(self.agu)

        self._keymap[key] = addr

        grp = self._fhandle.create_group(addr)
        grp.attrs['key'] = key
        for field, value in entity.items():
            grp.create_dataset(name=field, data=value)
            # for k, v in six.iteritems(dict(**dset.attrs)):
            #     dset.attrs.create(name=k, data=v)

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
            raise KeyError("The key '{}' does not exist.".format(key))

        del self._fhandle[addr]
        return addr

    def keys(self):
        """Return a list of all keys in the Stash."""
        return self._keymap.keys()

    def __addrs__(self):
        """Return a list of all absolute archive paths in the Stash."""
        return self._keymap.values()

    def __len__(self):
        return len(self.keys())
