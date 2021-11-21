import os
import json
import unittest
import sys

import numpy as np
import luigi
import z5py

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest

try:
    from .failing_task import FailingTaskLocal
except ImportError:
    from failing_task import FailingTaskLocal


class TestRetry(BaseTest):
    output_key = "data"
    shape = (100, 1024, 1024)

    def setUp(self):
        super().setUp()
        conf_path = os.path.join(self.config_folder, "global.config")
        with open(conf_path) as f:
            global_config = json.load(f)
        global_config["max_num_retries"] = 2
        with open(conf_path, "w") as f:
            json.dump(global_config, f)

    def test_retry(self):
        task = FailingTaskLocal
        ret = luigi.build([task(output_path=self.output_path,
                                output_key=self.output_key,
                                shape=self.shape,
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                max_jobs=self.max_jobs)], local_scheduler=True)
        self.assertTrue(ret)
        with z5py.File(self.output_path) as f:
            data = f[self.output_key][:]
        self.assertTrue(np.allclose(data, 1))


if __name__ == "__main__":
    unittest.main()
