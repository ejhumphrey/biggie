"""
"""

from __future__ import print_function
import h5py
import json
import logging
import numpy as np
import pymongo
import warnings

import biggie.core as core
import biggie.util as util


class Stash(object):
    """On-disk dictionary-like object."""
    __KEYMAP__ = "__KEYMAP__"
    __WIDTH__ = 256
    __DEPTH__ = 3

    def __init__(self, filename, mode=None, cache_size=False,
                 log_level=logging.INFO, refresh_handle=False):
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
        """
        self._filename = filename
        self._mode = mode
        self._refresh_handle = refresh_handle
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
        if not self._refresh_handle and fh:
            self.__handle__ = fh
        return self.__handle__ if self.__handle__ is not None else fh

    def __load_keymap__(self):
        if self.__KEYMAP__ not in self._fhandle:
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
        self._fhandle.close()

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
            Key of the value to get.

        default : object
            If given, default return value on unfound keys.
        """
        # Check local cache for the data first.
        value = self.__local__.get(key, None)

        # If key is not in local (value == None), go get that sucker.
        value = self.__load__(key) if value is None else value

        # Caching logic.
        if self._cache_size > 0 and (len(self.__local__) >= self._cache_size):
            # TODO: Pick a value and ditch it.
            pass

        if self._cache_size > 0:
            self.__local__[key] = value

        return value

    def add(self, key, entity, overwrite=False):
        """Add a key-object pair to the Stash.

        Parameters
        ----------
        key : str
            Key to write the value under.

        entity : Entity
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


class Collection(object):
    """Stash controller."""
    __STASH_KEY__ = '__stash_data__'

    def __init__(self, name,
                 database='biggie-default', host='localhost', port=None,
                 stash_kwargs=None):

        # Some combination of the following may be necessary to test for
        # thread-safety.
        self.__host__ = host
        self.__port__ = port
        self.__database__ = database
        self.__name__ = name

        self._stash_kwargs = self._collection.find_one(
            dict(_id=self.__STASH_KEY__), dict(_id=False))

        if not self._stash_kwargs and stash_kwargs:
            # First time!
            self._stash_kwargs = dict(_id=self.__STASH_KEY__, **stash_kwargs)
            # Now check that the data is valid.
            if not self.stash:
                raise ValueError(
                    "Could not instantiate the stash given provided arguments:"
                    "\n{}".format(self._stash_kwargs))
            self._collection.insert_one(self._stash_kwargs)
        elif not any([self._stash_kwargs, stash_kwargs]):
            warnings.warn(
                "This Collection lacks a default stash! You will not be able "
                "to store numpy data!")
        elif stash_kwargs and self._stash_kwargs != stash_kwargs:
            raise NotImplementedError(
                "Recovered stash kwargs conflict with those provided!\n"
                "Set: {}\n Given:{}".format(self._stash_kwargs, stash_kwargs))

    @property
    def _collection(self):
        client = pymongo.mongo_client.MongoClient(self.__host__, self.__port__)
        db_conn = client.get_database(self.__database__)
        return db_conn.get_collection(self.__name__)

    @property
    def name(self):
        return self._collection.name

    @property
    def stash(self):
        """On-demand instantiation of the stash for thread-safeness!"""
        return Stash(**self._stash_kwargs)

    @classmethod
    def from_config(cls, filename):
        return cls(**json.load(open(filename)))

    def insert(self, obj):
        """

        Note: In traversing the object, all np.ndarray datatypes are migrated
        to this collection's stash.

        Parameters
        ----------
        obj : dict
            Object to add to the biggie collection.

        """
        pass

    def get(self, oid):
        pass

    def find(self, query):
        pass

    def update(self, oid, update):
        pass
