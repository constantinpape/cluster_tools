import os
import json
import xml.etree.ElementTree as ET

import numpy as np
import luigi

from ..utils.volume_utils import file_reader
from ..cluster_tasks import WorkflowBase
from . import downscaling as downscale_tasks


# pretty print xml, from:
# http://effbot.org/zone/element-lib.htm#prettyprint
def indent_xml(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# TODO write proper metadata for the format:
# n5, bdv, or imaris
class WriteDownscalingMetadata(luigi.Task):
    tmp_folder = luigi.Parameter()
    output_path = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    dependency = luigi.TaskParameter()
    metadata_format = luigi.Parameter(default='custom')
    metadata_dict = luigi.DictParameter(default={})
    output_key_prefix = luigi.Parameter(default='')

    def requires(self):
        return self.dependency

    def _write_metadata_custom(self, out_key, scale_factor, effective_scale):
        with file_reader(self.output_path) as f:
            ds = f[out_key]
            ds.attrs['scale_factor'] = scale_factor
            ds.attrs['effective_scale'] = effective_scale

    def _custom_metadata(self):
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

            self._write_metadata_custom(out_key, scale_factor, effective_scale)

    def _write_metadata_bdv(self, scales, chunks):
        # we need to reorder scales and resoultions,
        # because bdv uses 'xyz' axis convention and we use 'zyx'
        assert scales.shape == chunks.shape
        scales_ = np.zeros_like(scales, dtype='float32')
        for sc in range(len(scales)):
            scales_[sc] = scales[sc][::-1]
        chunks_ = np.zeros_like(chunks, dtype='int')
        for ch in range(len(chunks)):
            chunks_[ch] = chunks[ch][::-1]
        with file_reader(self.output_path) as f:
            dsr = f.require_dataset('s00/resolutions', shape=scales_.shape, dtype=scales_.dtype)
            dsr[:] = scales_
            dsc = f.require_dataset('s00/subdivisions', shape=chunks_.shape, dtype=chunks_.dtype)
            dsc[:] = chunks_

    # write bdv xml, from:
    # https://github.com/tlambert03/imarispy/blob/master/imarispy/bdv.py#L136
    def _write_bdv_xml(self):
        # TODO we have hardcoded the number of
        # channels and  time points to 1, but should support more channels
        nt, nc = 1, 1
        key = 't00000/s00/0/cells'
        with file_reader(self.output_path, 'r') as f:
            shape = f[key].shape
        nz, ny, nx = tuple(shape)

        # write top-level data
        root = ET.Element('SpimData')
        root.set('version', '0.2')
        bp = ET.SubElement(root, 'BasePath')
        bp.set('type', 'relative')
        bp.text = '.'

        # read metadata from dict
        unit = self.metadata_dict.get('unit', 'micrometer')
        resolution = self.metadata_dict.get('resolution', (1., 1., 1.))
        dz, dy, dx = resolution
        offsets = self.metadata_dict.get('offsets', (0., 0., 0.))
        oz, oy, ox = offsets

        seqdesc = ET.SubElement(root, 'SequenceDescription')
        imgload = ET.SubElement(seqdesc, 'ImageLoader')
        imgload.set('format', 'bdv.hdf5')
        el = ET.SubElement(imgload, 'hdf5')
        el.set('type', 'relative')
        el.text = os.path.basename(self.output_path)
        viewsets = ET.SubElement(seqdesc, 'ViewSetups')
        attrs = ET.SubElement(viewsets, 'Attributes')
        attrs.set('name', 'channel')
        for c in range(nc):
            vs = ET.SubElement(viewsets, 'ViewSetup')
            ET.SubElement(vs, 'id').text = str(c)
            ET.SubElement(vs, 'name').text = 'channel {}'.format(c + 1)
            ET.SubElement(vs, 'size').text = '{} {} {}'.format(nx, ny, nz)
            vox = ET.SubElement(vs, 'voxelSize')
            ET.SubElement(vox, 'unit').text = unit
            ET.SubElement(vox, 'size').text = '{} {} {}'.format(dx, dy, dz)
            a = ET.SubElement(vs, 'attributes')
            ET.SubElement(a, 'channel').text = str(c + 1)
            chan = ET.SubElement(attrs, 'Channel')
            ET.SubElement(chan, 'id').text = str(c + 1)
            ET.SubElement(chan, 'name').text = str(c + 1)
        tpoints = ET.SubElement(seqdesc, 'Timepoints')
        tpoints.set('type', 'range')
        ET.SubElement(tpoints, 'first').text = str(0)
        ET.SubElement(tpoints, 'last').text = str(nt - 1)

        vregs = ET.SubElement(root, 'ViewRegistrations')
        for t in range(nt):
            for c in range(nc):
                vreg = ET.SubElement(vregs, 'ViewRegistration')
                vreg.set('timepoint', str(t))
                vreg.set('setup', str(c))
                vt = ET.SubElement(vreg, 'ViewTransform')
                vt.set('type', 'affine')
                ET.SubElement(vt, 'affine').text = '{} 0.0 0.0 {} 0.0 {} 0.0 {} 0.0 0.0 {} {}'.format(dx, ox, dy, oy, dz, oz)

        indent_xml(root)
        tree = ET.ElementTree(root)
        tree.write(os.path.splitext(self.output_path)[0] + ".xml")

    def _bdv_metadata(self):
        effective_scale = [1, 1, 1]
        scales = []
        chunks = []
        for scale, scale_factor in enumerate(self.scale_factors):
            # compute the effective scale at this level
            if isinstance(scale_factor, int):
                effective_scale = [eff * scale_factor for eff in effective_scale]
            else:
                effective_scale = [eff * sf for sf, eff in zip(scale_factor, effective_scale)]
            scales.append(effective_scale)
            # get the chunk size for this level
            out_key = 't00000/s00/%i/cells' % (scale + 1,)
            with file_reader(self.output_path, 'r') as f:
                chunk = f[out_key].chunks
            chunks.append(chunk)
        self._write_metadata_bdv(np.array(scales), np.array(chunks))
        self._write_bdv_xml()

    def run(self):
        if self.metadata_format == 'custom':
            self._custom_metadata()
        elif self.metadata_format == 'bdv':
            self._bdv_metadata()
        with open(self.output().path, 'w') as f:
            f.write('write metadata successfull')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              'write_downscaling_metadata.log'))



class DownscalingWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    halos = luigi.ListParameter()
    metadata_format = luigi.Parameter(default='custom')
    metadata_dict = luigi.DictParameter(default={})
    output_key_prefix = luigi.Parameter(default='')

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

    def validate_format(self):
        # TODO support more downsampling formats:
        # n5-viewer, imaris
        assert self.metadata_format in ('custom', 'bdv')
        if self.metadata_format == 'custom':
            assert self.output_key_prefix != '', "Need output_key_prefix for custom data format"
        # for now, we only support a single 'setup' and a single
        # time-point for the bdv format
        elif self.metadata_format == 'bdv':
            # TODO we could use hdf5 external links to drop this requirement
            assert self.input_path == self.output_path
            assert self.output_key_prefix == '', "Must not give output_key_prefix for custom data format"
            # TODO do we need to check for some relevant meta-data ???

    # we offset the scale by 1 because
    # 0 indicates the original resoulution
    def get_scale_key(self, scale):
        if self.metadata_format == 'custom':
            prefix = 's%i' % (scale + 1,)
            out_key = os.path.join(self.output_key_prefix, prefix)
        elif self.metadata_format == 'bdv':
            # we only support a single time-point and single set-up for now
            # TODO support multiple set-ups for multi-channel data
            out_key = 't00000/s00/%i/cells' % (scale + 1,)
        return out_key

    def _link_scale_zero(self, trgt):
        with file_reader(self.input_path) as f:
            if trgt not in f:
                f[trgt] = f[self.input_key]

    def requires(self):
        self.validate_scale_factors(self.scale_factors)
        halos = self.validate_halos(self.halos, len(self.scale_factors))
        self.validate_format()

        ds_task = getattr(downscale_tasks,
                          self._get_task_name('Downscaling'))

        if self.metadata_format == 'bdv':
            in_path = self.output_path
            in_key = self.get_scale_key(-1)
            self._link_scale_zero(in_key)
        else:
            t_prev = self.dependency
            in_path = self.input_path
            in_key = self.input_key
        t_prev = self.dependency

        effective_scale = [1, 1, 1]
        for scale, scale_factor in enumerate(self.scale_factors):
            out_key = self.get_scale_key(scale)

            if isinstance(scale_factor, int):
                effective_scale = [eff * scale_factor for eff in effective_scale]
            else:
                effective_scale = [eff * sf for sf, eff in zip(scale_factor, effective_scale)]

            prefix = 's%i' % (scale + 1,)
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
                                          metadata_format=self.metadata_format,
                                          metadata_dict=self.metadata_dict,
                                          scale_factors=self.scale_factors,
                                          dependency=t)
        return t_meta

    @staticmethod
    def get_config():
        configs = super(DownscalingWorkflow, DownscalingWorkflow).get_config()
        configs.update({'downscaling': downscale_tasks.DownscalingLocal.default_task_config()})
        return configs
