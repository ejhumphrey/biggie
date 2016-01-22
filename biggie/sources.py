from __future__ import print_function
import h5py
import json
import logging
import numpy as np
import pymongo
import warnings

import biggie.core as core
import biggie.util as util


class Stash(h5py.File):
    """On-disk dictionary-like object."""
    __KEYMAP__ = "__KEYMAP__"
    __WIDTH__ = 256
    __DEPTH__ = 3

    def __init__(self, filename, mode=None, cache_size=False,
                 log_level=logging.INFO):
        """Create a Stash object, pointing to an hdf5 file on-disk.

        Parameters
        ----------
        filename : str
            Path to file on disk

        mode : str
            Filemode for the object.

        cache : int or False-equivalent, default=False


        """
        super(Stash, self).__init__(name=filename, mode=mode)
        self._cache_size = cache_size
        self._logger = logging.getLogger('Stash')
        self._logger.setLevel(log_level)

        self.__local__ = dict()
        self._keymap = self.__decode_keymap__()
        self._agu = None

    @property
    def agu(self):
        if self._agu is None:
            self._agu = util.uniform_hexgen(self.__DEPTH__, self.__WIDTH__)
        return self._agu

    def __decode_keymap__(self):
        keymap_array = np.array(super(Stash, self).get(self.__KEYMAP__, '{}'))
        return json.loads(keymap_array.tostring())

    def __encode_keymap__(self, keymap):
        return np.array(json.dumps(keymap))

    def __del__(self):
        """Safe default destructor"""
        # Old h5py consideration, version 2.0.1
        if not hasattr(self.fid, 'valid'):
            return
        elif self.fid.valid:
            self.close()

    def close(self):
        """write keys and paths to disk"""
        if self.__KEYMAP__ in self:
            del self[self.__KEYMAP__]
        keymap_str = self.__encode_keymap__(self._keymap)
        super(Stash, self).create_dataset(
            name=self.__KEYMAP__, data=keymap_str)

        super(Stash, self).close()

    def __load__(self, key):
        """Deeply load an entity from the base HDF5 file."""
        addr = self._keymap.get(key, None)
        if addr is None:
            return addr
        raw_group = super(Stash, self).get(addr)
        raw_key = raw_group.attrs.get("key")
        if raw_key != key:
            raise ValueError(
                "Key inconsistency: received '{}'"
                ", expected '{}'".format(raw_key, key))
        self._logger.debug("Loading {}".format(key))
        return core.Entity.from_hdf5_group(raw_group)

    def get(self, key):
        """Fetch the entity for a given key."""
        value = self.__local__.get(key, None)
        value = self.__load__(key) if value is None else value
        if self._cache_size > 0 and (len(self.__local__) >= self._cache_size):
            # TODO: Pick a value and ditch it.
            pass
        if self._cache_size > 0:
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
            addr = self.agu.next()

        while addr in self:
            addr = self.agu.next()

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
            raise ValueError("The key '{}' does not exist.".format(key))

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
