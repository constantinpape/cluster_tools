import json
import multiprocessing
import os
import unittest
from shutil import rmtree
from zipfile import ZipFile

import luigi
from cluster_tools.cluster_tasks import BaseClusterTask
from cluster_tools.graph import GraphWorkflow

INPUT_PATH = os.environ.get("CLUSTER_TOOLS_TEST_PATH",
                            os.path.join(os.path.split(__file__)[0], "../test_data/sampleA.n5"))
SHEBANG = os.environ.get("CLUSTER_TOOLS_TEST_SHEBANG", None)
MAX_JOBS = os.environ.get("CLUSTER_TOOS_TEST_MAX_JOBS", min(multiprocessing.cpu_count(), 16))
TARGET = os.environ.get("CLUSTER_TOOLS_TEST_TARGET", "local")


def prepare_test():
    import z5py

    if os.path.exists(INPUT_PATH):
        print("Test data at", INPUT_PATH, "exists already")
    else:
        import gdown

        # download the test data
        print("Downloading test data to", INPUT_PATH)
        out_folder = os.path.split(INPUT_PATH)[0]
        out_path = os.path.join(out_folder, "tmp.zip")
        os.makedirs(out_folder, exist_ok=True)
        url = "https://drive.google.com/u/0/uc?export=download&confirm=crb1&id=1E_Wpw9u8E4foYKk7wvx5RPSWvg_NCN7U"
        gdown.download(url, out_path, quiet=False)

        # unzip it
        with ZipFile(out_path, "r") as f:
            f.extractall(out_folder)
        assert os.path.exists(INPUT_PATH)

        # clean up
        os.remove(out_path)

    with z5py.File(INPUT_PATH, "r") as f:
        assert "volumes/raw" in f, "Test data is invalid"


class BaseTest(unittest.TestCase):
    input_path = os.path.abspath(INPUT_PATH)
    shebang = SHEBANG
    max_jobs = MAX_JOBS
    target = TARGET

    tmp_folder = "./tmp"
    config_folder = "./tmp/config"
    output_path = "./tmp/data.n5"
    block_shape = [32, 256, 256]

    graph_key = "graph"
    ws_key = "volumes/segmentation/watershed"
    boundary_key = "volumes/boundaries"
    aff_key = "volumes/affinities"

    def setUp(self):
        os.makedirs(self.config_folder, exist_ok=True)
        config = BaseClusterTask.default_global_config()
        config.update({"block_shape": self.block_shape})
        if self.shebang is not None:
            config["shebang"] = self.shebang
        with open(os.path.join(self.config_folder, "global.config"), "w") as f:
            json.dump(config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def compute_graph(self, ignore_label=True):
        task = GraphWorkflow

        config = task.get_config()["initial_sub_graphs"]
        config.update({"ignore_label": ignore_label})
        with open(os.path.join(self.config_folder, "initial_sub_graphs.config"), "w") as f:
            json.dump(config, f)

        ret = luigi.build([task(input_path=self.input_path,
                                input_key=self.ws_key,
                                graph_path=self.output_path,
                                output_key=self.graph_key,
                                n_scales=1,
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)], local_scheduler=True)
        self.assertTrue(ret)

    def get_target_name(self):
        name_dict = {"local": "Local", "slurm": "Slurm", "lsf": "LSF"}
        return name_dict[self.target]


if __name__ == "__main__":
    prepare_test()
