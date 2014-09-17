"""
"""

import unittest
import tempfile as tmp
import numpy as np

import biggie


class TestSources(unittest.TestCase):

    def setUp(self):
        self.entity = biggie.Entity(
            a=3, b='im_a_string', c=[1, 2, 3], d=np.arange(5))
        self.key = 'my_key'
        self.fpath = tmp.mktemp(suffix=".hdf5", dir=tmp.gettempdir())
        fh = biggie.Stash(self.fpath)
        fh.add(self.key, self.entity)
        fh.close()

    def tearDown(self):
        pass

    def test_stash_persistence(self):
        fh = biggie.Stash(self.fpath)
        loaded_entity = fh.get(self.key)

        self.assertEqual(
            self.entity.a,
            loaded_entity.a,
            "Could not reconstitute entity.a")

        self.assertEqual(
            self.entity.b.tostring(),
            loaded_entity.b.tostring(),
            "Could not reconstitute entity.b")

        self.assertEqual(
            self.entity.c.tolist(),
            loaded_entity.c.tolist(),
            "Could not reconstitute entity.c")

        np.testing.assert_array_equal(
            self.entity.d,
            loaded_entity.d,
            "Could not reconstitute entity.d")

    def test_stash_overwrite(self):
        fh = biggie.Stash(self.fpath)
        loaded_entity = fh.get(self.key)
        loaded_entity.e = 4
        self.assertRaises(ValueError, fh.add, self.key, loaded_entity)

        fh.add(self.key, loaded_entity, True)
        fh.close()

        fh = biggie.Stash(self.fpath)
        another_entity = fh.get(self.key)
        self.assertEqual(len(fh), 1)
        self.assertEqual(another_entity.e, 4)

    def test_stash_cache(self):
        fh = biggie.Stash(self.fpath, cache=True)
        loaded_entity = fh.get(self.key)

        np.testing.assert_array_equal(
            self.entity.d,
            loaded_entity.d,
            "Could not reconstitute entity.d")

        np.testing.assert_array_equal(
            self.entity.d,
            fh.__local__[self.key].d,
            "Failed to cache entity")

if __name__ == "__main__":
    unittest.main()
