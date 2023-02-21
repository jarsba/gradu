import unittest
from src.utils.timer import Timer
from time import sleep
import os

class TimerTests(unittest.TestCase):

    def setUp(self) -> None:
        self.file_path = "/tmp/timer_tests.csv"
        self.timer = Timer(file_path=self.file_path, mode="replace")

    def test_timer_storage(self):
        pid = self.timer.start("MCMC")
        sleep(1)
        self.timer.stop(pid)

        df = self.timer.to_df()

        self.assertTrue(df is not None)

        self.timer.save()

        self.assertTrue(os.path.exists(self.file_path))