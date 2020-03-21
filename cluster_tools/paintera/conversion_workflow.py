import os
from datetime import datetime

import numpy as np
import luigi
import nifty.tools as nt

# we don't need to bother with the file reader
# wrapper here, because paintera needs n5 files anyway.
import z5py

from ..cluster_tasks import WorkflowBase
from ..label_multisets import LabelMultisetWorkflow
from ..downscaling import DownscalingWorkflow

from . import unique_block_labels as unique_tasks
from . import label_block_mapping as labels_to_block_tasks


class WritePainteraMetadata(luigi.Task):
    tmp_folder = luigi.Parameter()

    # path and keys
    path = luigi.Parameter()
    raw_key = luigi.Parameter()
    label_group = luigi.Parameter()

    # resolutions and n-scales
    raw_resolution = luigi.ListParameter()
    label_resolution = luigi.ListParameter()
    n_scales = luigi.IntParameter()

    offset = luigi.ListParameter()
    max_id = luigi.IntParameter()
    dependency = luigi.TaskParameter()

    def _write_log(self, msg):
        log_file = self.output().path
        with open(log_file, 'a') as f:
            f.write('%s: %s\n' % (str(datetime.now()), msg))

    def requires(self):
        return self.dependency

    def _write_downsampling_factors(self, data_group, group):
        for scale in range(1, self.n_scales):
            scale_key = 's%i' % scale
            factor = data_group[scale_key].attrs['downsamplingFactors']
            group[scale_key].attrs['downsamplingFactors'] = factor

    def run(self):

        # compute the correct resolutions for raw data and labels
        label_resolution = self.label_resolution
        raw_resolution = self.raw_resolution

        with z5py.File(self.path) as f:

            # write metadata for the top-level label group
            g = f[self.label_group]
            g.attrs['painteraData'] = {'type': 'label'}
            g.attrs['maxId'] = self.max_id
            # add the metadata referencing the label to block lookup
            scale_ds_pattern = os.path.join(self.label_group, 'label-to-block-mapping', 's%d')
            g.attrs["labelBlockLookup"] = {"type": "n5-filesystem",
                                           "root": os.path.abspath(os.path.realpath(self.path)),
                                           "scaleDatasetPattern": scale_ds_pattern}

            # write metadata for the label-data group
            data_group = g['data']
            data_group.attrs['maxId'] = self.max_id
            # we revese resolution and offset because java n5 uses axis
            # convention XYZ and we use ZYX
            data_group.attrs['offset'] = self.offset[::-1]
            data_group.attrs['resolution'] = label_resolution[::-1]
            # note: we have the downsampling factors for the data group already

            # add metadata for unique labels group
            unique_group = g['unique-labels']
            unique_group.attrs['multiScale'] = True
            self._write_downsampling_factors(data_group, unique_group)

            # add metadata for label to block mapping
            mapping_group = g['label-to-block-mapping']
            mapping_group.attrs['multiScale'] = True
            self._write_downsampling_factors(data_group, mapping_group)

            # add metadata for the raw data
            raw_group = f[self.raw_key]
            raw_group.attrs['resolution'] = raw_resolution[::-1]

        self._write_log('write metadata successfull')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              'write_paintera_metadata.log'))


class ConversionWorkflow(WorkflowBase):
    path = luigi.Parameter()
    raw_key = luigi.Parameter()
    label_in_key = luigi.Parameter()
    label_out_key = luigi.Parameter()
    label_scale = luigi.IntParameter()
    assignment_path = luigi.Parameter(default='')
    assignment_key = luigi.Parameter(default='')
    use_label_multiset = luigi.BoolParameter(default=False)
    copy_labels = luigi.BoolParameter(default=False)
    offset = luigi.ListParameter(default=[0, 0, 0])
    resolution = luigi.ListParameter(default=[1, 1, 1])
    restrict_sets = luigi.ListParameter(default=[])
    restrict_scales = luigi.IntParameter(default=None)

    ##############################################################
    # Step 1 Implementations: align scales and make label datasets
    ##############################################################

    def _downsample_labels(self, scale_factors, dep):
        task = DownscalingWorkflow
        out_key_prefix = os.path.join(self.label_out_key, 'data')
        halos = len(scale_factors) * [[0, 0, 0]]
        dep = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                   config_dir=self.config_dir, target=self.target,
                   input_path=self.path, input_key=self.label_in_key,
                   output_path=self.path, output_key_prefix=out_key_prefix,
                   scale_factors=scale_factors, halos=halos,
                   metadata_format='paintera', force_copy=self.copy_labels,
                   dependency=dep)
        return dep

    def _make_label_multisets(self, scale_factors, dep):
        task = LabelMultisetWorkflow

        restrict_sets = self.restrict_sets
        assert len(restrict_sets) == len(scale_factors),\
            "Need restrict_sets for label-multisets: %i, %i" % (len(restrict_sets), len(scale_factors))

        out_key_prefix = os.path.join(self.label_out_key, 'data')
        dep = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                   config_dir=self.config_dir, target=self.target,
                   input_path=self.path, input_key=self.label_in_key,
                   output_path=self.path, output_prefix=out_key_prefix,
                   scale_factors=scale_factors, restrict_sets=restrict_sets,
                   dependency=dep)
        return dep

    def _align_scales(self, dep):
        g_raw = z5py.File(self.path)[self.raw_key]
        ds_labels = z5py.File(self.path)[self.label_in_key]

        # make sure that the shape of the raw data and the
        # labels agree at the specified scale level
        label_scale_prefix = 's%i' % self.label_scale
        assert label_scale_prefix in g_raw, "Cannot find label scale in raw data"
        shape_raw = g_raw[label_scale_prefix].shape
        shape_labels = ds_labels.shape
        assert shape_raw == shape_labels, "%s, %s" % (str(shape_raw), str(shape_labels))

        # get and sort the raw scales
        raw_scales = list(g_raw.keys())
        raw_scales = np.array([int(rscale[1:]) for rscale in raw_scales])
        raw_scales = np.sort(raw_scales)

        # compute the scale factors from the raw datasets
        scale_factors = [[1, 1, 1]]
        effective_scale_factors = [[1, 1, 1]]
        for scale in raw_scales[1:]:
            # we need to reverse the scale factors because paintera has axis order
            # XYZ and we have axis order ZYX
            effective_scale_factor = g_raw['s%i' % scale].attrs['downsamplingFactors'][::-1]

            # find the relative scale factor
            scale_factor = [int(sf_out // sf_in) for sf_out, sf_in
                            in zip(effective_scale_factor, effective_scale_factors[-1])]

            effective_scale_factors.append(effective_scale_factor)
            scale_factors.append(scale_factor)

        # compute the label resolution
        label_scale_factor = effective_scale_factors[self.label_scale]
        label_resolution = [res * eff for res, eff in zip(self.resolution, label_scale_factor)]

        # restrict to the scale factors we need for down-sampling the labels
        scale_factors = scale_factors[(self.label_scale + 1):]
        if self.restrict_scales is not None:
            scale_factors = scale_factors[:self.restrict_scales]

        # create downsampled labels in label-multiset format
        # or by default downsampling
        if self.use_label_multiset:
            dep = self._make_label_multisets(scale_factors, dep)
        else:
            dep = self._downsample_labels(scale_factors, dep)

        # prepend scale factor for scale 0
        scale_factors = [[1, 1, 1]] + scale_factors
        return dep, scale_factors, label_resolution

    ############################################
    # Step 2 Implementations: make block uniques
    ############################################

    def _uniques_in_blocks(self, dep, scale_factors):
        task = getattr(unique_tasks, self._get_task_name('UniqueBlockLabels'))
        effective_scale = [1, 1, 1]
        for scale, factor in enumerate(scale_factors):
            in_key = os.path.join(self.label_out_key, 'data', 's%i' % scale)
            out_key = os.path.join(self.label_out_key, 'unique-labels', 's%i' % scale)
            effective_scale = [eff * sf for eff, sf in zip(effective_scale, factor)]
            dep = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       input_path=self.path, output_path=self.path,
                       input_key=in_key, output_key=out_key,
                       effective_scale_factor=effective_scale,
                       dependency=dep, prefix='s%i' % scale)
        return dep

    ##############################################
    # Step 3 Implementations: invert block uniques
    ##############################################

    def _label_block_mapping(self, dep, scale_factors):
        task = getattr(labels_to_block_tasks,
                       self._get_task_name('LabelBlockMapping'))

        # get the framgent max id
        with z5py.File(self.path) as f:
            max_id = f[self.label_in_key].attrs['maxId']

        # compte the label to block mapping for all scales
        effective_scale = [1, 1, 1]
        for scale, factor in enumerate(scale_factors):
            in_key = os.path.join(self.label_out_key, 'unique-labels', 's%i' % scale)
            out_key = os.path.join(self.label_out_key, 'label-to-block-mapping', 's%i' % scale)
            effective_scale = [eff * sf for eff, sf in zip(effective_scale, factor)]
            dep = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       input_path=self.path, output_path=self.path,
                       input_key=in_key, output_key=out_key,
                       number_of_labels=max_id + 1, dependency=dep,
                       effective_scale_factor=effective_scale,
                       prefix='s%i' % scale)
        return dep

    #####################################################
    # Step 4 Implementations: fragment segment assignment
    #####################################################

    def _fragment_segment_assignment(self, dep):
        if self.assignment_path == '':
            # get the framgent max id
            with z5py.File(self.path) as f:
                max_id = f[self.label_in_key].attrs['maxId']
            return dep, max_id
        else:
            assert self.assignment_key != ''
            assert os.path.exists(self.assignment_path), self.assignment_path
            # TODO should make this a task
            with z5py.File(self.assignment_path) as f, z5py.File(self.path) as f_out:
                assignments = f[self.assignment_key][:]
                n_fragments = len(assignments)

                # find the fragments which have non-trivial assignment
                segment_ids, counts = np.unique(assignments,
                                                return_counts=True)
                seg_ids_to_counts = {seg_id: count
                                     for seg_id, count in zip(segment_ids, counts)}
                fragment_ids_to_counts = nt.takeDict(seg_ids_to_counts, assignments)
                fragment_ids = np.arange(n_fragments, dtype='uint64')

                non_triv_fragments = fragment_ids[fragment_ids_to_counts > 1]
                non_triv_segments = assignments[non_triv_fragments]
                non_triv_segments += n_fragments

                # determine the overall max id
                max_id = int(non_triv_segments.max())

                # TODO do we need to assign a special value to ignore label (0) ?
                frag_to_seg = np.vstack((non_triv_fragments, non_triv_segments))

                # fragment_ids = np.arange(n_fragments, dtype='uint64')
                # assignments += n_fragments
                # frag_to_seg = np.vstack((fragment_ids, assignments))

                # max_id = int(frag_to_seg.max())

                out_key = os.path.join(self.label_out_key, 'fragment-segment-assignment')
                chunks = (1, frag_to_seg.shape[1])
                f_out.require_dataset(out_key, data=frag_to_seg, shape=frag_to_seg.shape,
                                      compression='gzip', chunks=chunks)
            return dep, max_id

    def requires(self):

        # align the scales of labels and raw data and make label datasets
        dep, scale_factors, label_resolution = self._align_scales(self.dependency)

        # # next, compute the mapping of unique labels to blocks
        dep = self._uniques_in_blocks(dep, scale_factors)
        # # next, compute the inverse mapping
        dep = self._label_block_mapping(dep, scale_factors)
        # # next, compute the fragment-segment-assignment
        dep, max_id = self._fragment_segment_assignment(dep)

        # # finally, write metadata
        dep = WritePainteraMetadata(tmp_folder=self.tmp_folder, path=self.path,
                                    raw_key=self.raw_key, label_group=self.label_out_key,
                                    raw_resolution=self.resolution, label_resolution=label_resolution,
                                    n_scales=len(scale_factors), offset=self.offset, max_id=max_id,
                                    dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(ConversionWorkflow, ConversionWorkflow).get_config()
        configs.update({'unique_block_labels': unique_tasks.UniqueBlockLabelsLocal.default_task_config(),
                        'label_block_mapping': labels_to_block_tasks.LabelBlockMappingLocal.default_task_config(),
                        **DownscalingWorkflow.get_config(), **LabelMultisetWorkflow.get_config()})
        return configs
