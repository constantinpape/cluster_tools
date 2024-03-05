import os
from copy import deepcopy
from datetime import datetime

import luigi

from ..cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from .. import copy_volume as copy_tasks
from . import downscaling as downscale_tasks


class WriteDownscalingMetadata(luigi.Task):
    tmp_folder = luigi.Parameter()
    output_path = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    dependency = luigi.TaskParameter()
    metadata_format = luigi.Parameter()
    metadata_dict = luigi.DictParameter(default={})
    output_key_prefix = luigi.Parameter(default="")
    scale_offset = luigi.IntParameter(default=0)
    prefix = luigi.Parameter(default="")

    def requires(self):
        return self.dependency

    def _write_log(self, msg):
        log_file = self.output().path
        with open(log_file, "a") as f:
            f.write("%s: %s\n" % (str(datetime.now()), msg))

    def run(self):
        vu.write_format_metadata(self.metadata_format, self.output_path, self.metadata_dict,
                                 self.scale_factors, self.scale_offset, self.output_key_prefix)
        self._write_log("write metadata successfull")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              "write_downscaling_metadata_%s.log" % self.prefix))


class DownscalingWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    halos = luigi.ListParameter()
    dtype = luigi.Parameter(default=None)
    int_to_uint = luigi.BoolParameter(default=False)
    metadata_format = luigi.Parameter(default="paintera")
    metadata_dict = luigi.DictParameter(default={})
    # the output path is optional, if not given, we take the same as input path
    output_path = luigi.Parameter(default="")
    output_key_prefix = luigi.Parameter(default="")
    force_copy = luigi.BoolParameter(default=False)
    skip_existing_levels = luigi.BoolParameter(default=False)
    scale_offset = luigi.IntParameter(default=0)

    formats = vu.get_formats()

    @staticmethod
    def validate_scale_factors(scale_factors, metadata_format):
        assert all(isinstance(sf, (int, list, tuple)) for sf in scale_factors)
        ndims = (2, 3) if metadata_format == "ome.zarr" else (3,)
        assert all(len(sf) in ndims for sf in scale_factors if isinstance(sf, (tuple, list)))

    @staticmethod
    def validate_halos(halos, n_scales, ndim=3):
        assert len(halos) == n_scales, "%i, %i" % (len(halos), n_scales)
        # normalize halos
        halos = [[] if halo is None else (ndim * [halo] if isinstance(halo, int) else list(halo))
                 for halo in halos]
        # check halos for correctness
        assert all(isinstance(halo, list) for halo in halos)
        assert all(len(halo) == ndim for halo in halos if halo)
        return halos

    @staticmethod
    def validate_resolution(metadata_dict, ndim=3):
        resolution = metadata_dict["resolution"]
        assert isinstance(resolution, (list, tuple))
        resolution = list(resolution)
        assert len(resolution) == ndim

        for idx, pxs in enumerate(resolution):
            assert isinstance(pxs, (str, int, float))
            resolution[idx] = float(pxs)
            assert resolution[idx] > 0

        tmp_dict = dict(metadata_dict)
        tmp_dict["resolution"] = resolution
        out_dict = luigi.freezing.recursively_freeze(tmp_dict)
        return out_dict

    def _is_h5(self):
        h5_exts = (".h5", ".hdf5", ".hdf")
        out_path = self.input_path if self.output_path == "" else self.output_path
        return os.path.splitext(out_path)[1].lower() in h5_exts

    def _is_n5(self):
        n5_exts = (".n5",)
        out_path = self.input_path if self.output_path == "" else self.output_path
        return os.path.splitext(out_path)[1].lower() in n5_exts

    def _is_zarr(self):
        zarr_exts = (".zarr", ".zr")
        out_path = self.input_path if self.output_path == "" else self.output_path
        return os.path.splitext(out_path)[1].lower() in zarr_exts

    def validate_format(self):
        assert self.metadata_format in self.formats,\
                "Invalid format: %s not in %s" % (self.metadata_format, str(self.formats))
        if self.metadata_format == "paintera":
            assert self.output_key_prefix != "",\
                "Need output_key_prefix for paintera data format"
            assert self._is_n5(), "paintera format only supports n5 output"
        # for now, we only support a single "setup" and a single
        # time-point for the bdv format
        elif self.metadata_format == "ome.zarr":
            assert self._is_zarr(), "ome.zarr format only supports zarr output"
        else:
            msg = f"Must not give output_key_prefix for bdv data format, got {self.output_key_prefix}"
            assert self.output_key_prefix == "", msg
            if self.metadata_format in ("bdv", "bdv.hdf5"):
                assert self._is_h5(), "%s format only supports hdf5 output" % self.metadata_format
            elif self.metadata_format == "bdv.n5":
                assert self._is_n5(), "bdv.n5 format only supports n5 output"
            else:
                raise RuntimeError  # this should never happen

    def _link_scale_zero_h5(self, trgt):
        with vu.file_reader(self.input_path) as f:
            if trgt not in f:
                f[trgt] = f[self.input_key]

    def _link_scale_zero_n5(self, trgt):
        with vu.file_reader(self.input_path) as f:
            if trgt not in f:
                os.makedirs(os.path.split(os.path.join(self.input_path, trgt))[0], exist_ok=True)
                src_path = os.path.abspath(os.path.realpath(os.path.join(self.input_path,
                                                                         self.input_key)))
                trgt_path = os.path.abspath(os.path.realpath(os.path.join(self.input_path,
                                                                          trgt)))
                os.symlink(src_path, trgt_path)

    def _have_scale(self, scale):
        key = vu.get_format_key(self.metadata_format, scale, self.output_key_prefix)
        with vu.file_reader(self.input_path) as f:
            return key in f

    def _copy_scale_zero(self, out_path, out_key, dep, dtype, int_to_uint):
        task = getattr(copy_tasks, self._get_task_name("CopyVolume"))
        prefix = "initial_scale"
        dimension_separator = "/" if self.metadata_format == "ome.zarr" else None
        dep = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                   config_dir=self.config_dir,
                   input_path=self.input_path, input_key=self.input_key,
                   output_path=out_path, output_key=out_key,
                   int_to_uint=int_to_uint, dtype=dtype,
                   prefix=prefix, dependency=dep, dimension_separator=dimension_separator)
        return dep

    def require_initial_scale(self, out_path, out_key, dep, dtype, int_to_uint):
        """ Link or copy the initial dataset to self.output_key_prefix.
        We copy if input_path != output_path or force_copy is set.
        """
        copy_initial_ds = True if self.force_copy else out_path != self.input_path

        if copy_initial_ds:
            dep = self._copy_scale_zero(out_path, out_key, dep, dtype, int_to_uint)
        else:
            # make a link in the h5 file
            if self.metadata_format in ("bdv", "bdv.hdf5"):
                self._link_scale_zero_h5(out_key)
            # make a link on the file system
            elif self.metadata_format in ("bdv.n5", "ome.zarr", "paintera"):
                self._link_scale_zero_n5(out_key)
            else:
                raise RuntimeError  # this should never happen
        return dep

    def requires(self):
        self.validate_scale_factors(self.scale_factors, self.metadata_format)
        ndim = len(self.scale_factors[0])
        halos = self.validate_halos(self.halos, len(self.scale_factors), ndim)
        self.metadata_dict = self.validate_resolution(self.metadata_dict, ndim)
        self.validate_format()

        out_path = self.input_path if self.output_path == "" else self.output_path
        in_key = vu.get_format_key(self.metadata_format, self.scale_offset, self.output_key_prefix)
        # require the initial scale dataset
        dep = self.require_initial_scale(out_path, in_key, self.dependency, self.dtype, self.int_to_uint)

        dimension_separator = "/" if self.metadata_format == "ome.zarr" else None
        task = getattr(downscale_tasks, self._get_task_name("Downscaling"))
        effective_scale = [1] * ndim
        for scale, (scale_factor, halo) in enumerate(zip(self.scale_factors, halos),
                                                     self.scale_offset + 1):
            out_key = vu.get_format_key(self.metadata_format, scale, self.output_key_prefix)

            if isinstance(scale_factor, int):
                effective_scale = [eff * scale_factor for eff in effective_scale]
            else:
                effective_scale = [eff * sf for sf, eff in zip(scale_factor, effective_scale)]

            # check if this scale already exists.
            # if so, skip it if we have `skip_existing_levels` set to True
            if self.skip_existing_levels and self._have_scale(scale):
                in_key = out_key
                continue

            dep = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       input_path=out_path, input_key=in_key,
                       output_path=out_path, output_key=out_key,
                       scale_factor=scale_factor, scale_prefix="s%i" % scale,
                       effective_scale_factor=effective_scale,
                       halo=halo, dimension_separator=dimension_separator, dependency=dep)
            in_key = out_key

        # task to write the metadata
        dep = WriteDownscalingMetadata(tmp_folder=self.tmp_folder,
                                       output_path=out_path,
                                       output_key_prefix=self.output_key_prefix,
                                       metadata_format=self.metadata_format,
                                       metadata_dict=self.metadata_dict,
                                       scale_factors=self.scale_factors,
                                       scale_offset=self.scale_offset,
                                       dependency=dep, prefix="downscaling")
        return dep

    @staticmethod
    def get_config():
        configs = super(DownscalingWorkflow, DownscalingWorkflow).get_config()
        configs.update({"downscaling": downscale_tasks.DownscalingLocal.default_task_config(),
                        "copy_volume": copy_tasks.CopyVolumeLocal.default_task_config()})
        return configs


# HDF5 is frickin slow, so it seems to be better to do the
# computations in n5 and then copy data to h5
class PainteraToBdvWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key_prefix = luigi.Parameter()
    output_path = luigi.Parameter()
    dtype = luigi.Parameter(default=None)
    metadata_dict = luigi.DictParameter(default={})
    skip_existing_levels = luigi.BoolParameter(default=True)
    metadata_format = luigi.Parameter("bdv")

    def get_scales(self):
        with vu.file_reader(self.input_path, "r") as f:
            g = f[self.input_key_prefix]
            scale_names = list(g.keys())
        scale_levels = [int(name[1:]) for name in scale_names]
        return list(sorted(scale_levels))

    def requires(self):
        task = getattr(copy_tasks, self._get_task_name("CopyVolume"))

        # get scales that need to be copied
        scales = self.get_scales()

        dep = self.dependency
        prev_scale = None
        scale_factors = []
        for scale in scales:
            in_key = vu.get_format_key("paintera", scale, self.input_key_prefix)
            out_key = self.get_scale_key(self.metadata_format, scale)

            # read the downsampling factors
            with vu.file_reader(self.input_path) as f:
                effective_scale = f[in_key].attrs.get("downsamplingFactors", [1, 1, 1])
                if isinstance(effective_scale, int):
                    effective_scale = 3 * [effective_scale]

            if scale > 0:
                assert prev_scale is not None
                scale_factors.append([eff / prev for eff, prev in zip(effective_scale,
                                                                      prev_scale)])
            prev_scale = deepcopy(effective_scale)

            if self.skip_existing_levels and os.path.exists(self.output_path):
                with vu.file_reader(self.output_path, "r") as f:
                    if out_key in f:
                        print("have out_key", out_key)
                        continue

            prefix = "s%i" % scale
            dep = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       input_path=self.input_path, input_key=in_key,
                       output_path=self.output_path, output_key=out_key,
                       prefix=prefix, effective_scale_factor=effective_scale,
                       dtype=self.dtype, dependency=dep)

        # get the metadata for this dataset
        # if we have the `resolution` or `offset` attribute
        # in the dataset, we load them and add them to
        # the metadata dict. However if these keys
        # are already in the metadata dict, the existing values
        # have priority
        metadata_dict = {**self.metadata_dict}
        with vu.file_reader(self.input_path) as f:
            attrs = f[self.input_key_prefix].attrs
            offsets = attrs.get("offset", None)
            resolution = attrs.get("resolution", None)

        if "offsets" not in metadata_dict and offsets is not None:
            metadata_dict.update({"offsets": offsets})
        if "resolution" not in metadata_dict and resolution is not None:
            metadata_dict.update({"resolution": resolution})

        # task to write the metadata
        dep = WriteDownscalingMetadata(tmp_folder=self.tmp_folder,
                                       output_path=self.output_path,
                                       metadata_format=self.metadata_format,
                                       metadata_dict=metadata_dict,
                                       scale_factors=scale_factors,
                                       dependency=dep, prefix="paintera-to-bdv")
        return dep

    @staticmethod
    def get_config():
        configs = super(PainteraToBdvWorkflow, PainteraToBdvWorkflow).get_config()
        configs.update({"copy_volume": copy_tasks.CopyVolumeLocal.default_task_config()})
        return configs
