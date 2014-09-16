import unittest
import numpy as np

import biggie


class TestCore(unittest.TestCase):

    def setUp(self):
        self.some_int = 3
        self.some_str = 'im_a_string'
        self.some_list = [1, 2, 3]
        self.some_ndarray = np.arange(5)
        self.entity = biggie.Entity(
            a=self.some_int, b=self.some_str,
            c=self.some_list, d=self.some_ndarray)

    def tearDown(self):
        pass

    def test_getitem(self):
        self.assertEqual(
            self.entity.a,
            self.entity['a'].value,
            "Failed to initialize keys / attributes.")

    def test_int_field(self):
        self.assertEqual(
            self.entity.a,
            self.some_int,
            "Failed to initialize an int.")

    def test_str_field(self):
        self.assertEqual(
            self.entity.b.tostring(),
            self.some_str,
            "Failed to initialize a string.")

    def test_list_field(self):
        self.assertEqual(
            self.entity.c.tolist(),
            self.some_list,
            "Failed to initialize a list.")

    def test_ndarray_field(self):
        np.testing.assert_array_equal(
            self.entity.d,
            self.some_ndarray,
            "Failed to initialize a numpy array.")

if __name__ == "__main__":
    unittest.main()
