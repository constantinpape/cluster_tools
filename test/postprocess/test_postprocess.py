import unittest
import sys
import luigi

try:
    from ..base import BaseTest
except ImportError:
    sys.path.append('..')
    from base import BaseTest


class TestPostprocess(BaseTest):
    input_key = 'volumes/watershed'
    output_key = 'filtered'

    def test_size_filter_bg(self):
        from cluster_tools.postprocess import SizeFilterWorkflow
        thresh = 250
        task = SizeFilterWorkflow(tmp_folder=self.tmp_folder,
                                  config_dir=self.config_folder,
                                  target=self.target, max_jobs=self.max_jobs,
                                  input_path=self.path, input_key=self.input_key,
                                  output_path=self.output_path, output_key=self.output_key,
                                  size_threshold=thresh, relabel=False)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)


if __name__ == '__main__':
    unittest.main()
