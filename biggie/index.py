import json
import pymongo

import biggie.sources


class Index(object):
    """Index view over a stash."""

    def __init__(self, name, stash,
                 database='biggie-default', host='localhost', port=None):

        # Some combination of the following may be necessary to test for
        # thread-safety.
        self._stash = stash
        self.__host__ = host
        self.__port__ = port
        self.__database__ = database
        self.__name__ = name

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
        return self._stash

    @classmethod
    def from_config(cls, filename):
        """Create an index from a given configuration file.

        Parameters
        ----------
        filename : str
            JSON file containing kwargs for both the `stash` and `index`.
        """
        kwargs = json.load(open(filename))
        stash = biggie.sources.Stash(**kwargs['stash'])
        return cls(stash=stash, **kwargs['index'])
