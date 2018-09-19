import os
import json
import luigi

from ..utils.volume_utils import file_reader
from ..cluster_tasks import WorkflowBase
from . import downscaling as downscale_tasks


# TODO write proper metadata for the format:
# n5, bdv, or imaris
class WriteDownscalingMetadata(luigi.Task):
    tmp_folder = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key_prefix = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def _write_metadata(self, out_key, scale_factor, effective_scale):
        with file_reader(self.output_path) as f:
            ds = f[out_key]
            ds.attrs['scale_factor'] = scale_factor
            ds.attrs['effective_scale'] = effective_scale

    def run(self):
        effective_scale = [1, 1, 1]
        for scale, scale_factor in enumerate(self.scale_factors):
            # we offset the scale by 1 because
            # 's0' indicates the original resoulution
            prefix = 's%i' % (scale + 1,)
            out_key = os.path.join(self.output_key_prefix, prefix)
            # write metadata for this scale level,
            # most importantly the effective downsampling factor
            # compared to the original resolution
            if isinstance(scale_factor, int):
                effective_scale = [eff * scale_factor for eff in effective_scale]
            else:
                effective_scale = [eff * sf for sf, eff in zip(scale_factor, effective_scale)]
            self._write_metadata(out_key, scale_factor, effective_scale)
        with open(self.output().path, 'w') as f:
            f.write('write metadata successfull')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              'write_downscaling_metadata.log'))



class DownscalingWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key_prefix = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    halos = luigi.ListParameter()
    # TODO support different downsampling formats:
    # n5, bigdataviewer-spim, imaris
    # and write the corresponding meta-data

    @staticmethod
    def validate_scale_factors(scale_factors):
        assert all(isinstance(sf, (int, list, tuple)) for sf in scale_factors)
        assert all(len(sf) == 3 for sf in scale_factors if isinstance(sf, (tuple, list)))

    @staticmethod
    def validate_halos(halos, n_scales):
        assert len(halos) == n_scales
        # normalize halos to be correc input
        halos = [[] if halo is None else list(halo) for halo in halos]
        # check halos for correctness
        assert all(isinstance(halo, list) for halo in halos)
        assert all(len(halo) == 3 for halo in halos if halo)
        return halos

    def requires(self):
        self.validate_scale_factors(self.scale_factors)
        halos = self.validate_halos(self.halos, len(self.scale_factors))

        ds_task = getattr(downscale_tasks,
                          self._get_task_name('Downscaling'))

        t_prev = self.dependency
        in_path = self.input_path
        in_key = self.input_key

        effective_scale = [1, 1, 1]
        for scale, scale_factor in enumerate(self.scale_factors):
            # we offset the scale by 1 because
            # 's0' indicates the original resoulution
            prefix = 's%i' % (scale + 1,)
            out_key = os.path.join(self.output_key_prefix, prefix)

            if isinstance(scale_factor, int):
                effective_scale = [eff * scale_factor for eff in effective_scale]
            else:
                effective_scale = [eff * sf for sf, eff in zip(scale_factor, effective_scale)]

            t = ds_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                        config_dir=self.config_dir,
                        input_path=in_path, input_key=in_key,
                        output_path=self.output_path, output_key=out_key,
                        scale_factor=scale_factor, scale_prefix=prefix,
                        effective_scale_factor=effective_scale,
                        halo=halos[scale],
                        dependency=t_prev)

            t_prev = t
            in_path = self.output_path
            in_key = out_key

        # task to write the metadata
        t_meta = WriteDownscalingMetadata(tmp_folder=self.tmp_folder,
                                          output_path=self.output_path,
                                          output_key_prefix=self.output_key_prefix,
                                          scale_factors=self.scale_factors,
                                          dependency=t)
        return t_meta

    @staticmethod
    def get_config():
        configs = super(DownscalingWorkflow, DownscalingWorkflow).get_config()
        configs.update({'downscaling': downscale_tasks.DownscalingLocal.default_task_config()})
        return configs
