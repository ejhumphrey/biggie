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

    def tearDown(self):
        pass

    def test_stash(self):
        key = 'my_key'
        fpath = tmp.mktemp(suffix=".hdf5", dir=tmp.gettempdir())
        fh = biggie.Stash(fpath)
        fh.add(key, self.entity)
        fh.close()

        fh = biggie.Stash(fpath)
        loaded_entity = fh.get(key)

        self.assertEqual(
            self.entity.a.value,
            loaded_entity.a.value,
            "Could not reconstitute entity.a")

        self.assertEqual(
            self.entity.b.value.tostring(),
            loaded_entity.b.value.tostring(),
            "Could not reconstitute entity.b")

        self.assertEqual(
            self.entity.c.value.tolist(),
            loaded_entity.c.value.tolist(),
            "Could not reconstitute entity.c")

        self.assertEqual(
            self.entity.d.value.tolist(),
            loaded_entity.d.value.tolist(),
            "Could not reconstitute entity.d")

        loaded_entity.a.value = 4
        self.assertRaises(ValueError, fh.add, key, loaded_entity)

        fh.add(key, loaded_entity, True)
        fh.close()

        fh = biggie.Stash(fpath)
        another_entity = fh.get(key)
        self.assertEqual(len(fh), 1)
        self.assertEqual(another_entity.a.value, 4)

if __name__ == "__main__":
    unittest.main()
