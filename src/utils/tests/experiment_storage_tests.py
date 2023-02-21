import unittest
from src.utils.experiment_storage import ExperimentStorage
from src.utils.keygen import get_key
import numpy as np
import os


class ExperimentStorageTests(unittest.TestCase):

    def setUp(self) -> None:
        self.file_path = "/tmp/experiment_storage_test.csv"
        self.experiment_storage = ExperimentStorage(file_path=self.file_path, mode="replace")

    def test_experiment_storage_saving(self):
        key = get_key()
        self.experiment_storage.store(key, {'lambdas': np.ones(shape=100)})

        self.assertTrue(self.experiment_storage._has_key(key))
        obj = self.experiment_storage.get_item(key)

        self.assertTrue("lambdas" in obj)
        self.assertTrue(np.array_equal(obj['lambdas'], np.ones(shape=100)))

        self.experiment_storage.save()
        self.assertTrue(os.path.exists(self.file_path))

    def test_storing_multiple_items(self):
        key = get_key()
        self.experiment_storage.store(key, {'lambdas': np.eye(10)})
        self.experiment_storage.store(key, {'divergence': np.zeros(shape=100)})

        obj = self.experiment_storage.get_item(key)

        self.assertTrue("lambdas" in obj)
        self.assertTrue(np.array_equal(obj['lambdas'], np.eye(10)))

        self.assertTrue("divergence" in obj)
        self.assertTrue(np.array_equal(obj['divergence'], np.zeros(shape=100)))

        self.experiment_storage.save()
        self.assertTrue(os.path.exists(self.file_path))



