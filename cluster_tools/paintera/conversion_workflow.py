import os
import json
from datetime import datetime

import numpy as np
import luigi
# NOTE we don't need to bother with the file reader
# wrapper here, because paintera needs n5 files anyway.
import z5py

from ..import downscaling as sampling_tasks
from ..cluster_tasks import WorkflowBase
# TODO
# from ..label_multisets import

from . import unique_block_labels as unique_tasks
from . import label_block_mapping as labels_to_block_tasks


class WritePainteraMetadata(luigi.Task):
    tmp_folder = luigi.Parameter()
    path = luigi.Parameter()
    raw_key = luigi.Parameter()
    label_group = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    label_scale = luigi.IntParameter()
    is_label_multiset = luigi.BoolParameter()
    resolution = luigi.ListParameter()
    offset = luigi.ListParameter()
    max_id = luigi.IntParameter()
    dependency = luigi.TaskParameter()

    def _write_log(self, msg):
        log_file = self.output().path
        with open(log_file, 'a') as f:
            f.write('%s: %s\n' % (str(datetime.now()), msg))

    def requires(self):
        return self.dependency

    def _write_downsampling_factors(self, group):
        # get the actual scales we have in the segmentation
        scale_factors = [[1, 1, 1]] + list(self.scale_factors[self.label_scale+1:])
        effective_scale = [1, 1, 1]
        # write the scale factors
        for scale, scale_factor in enumerate(scale_factors):
            ds = group['s%i' % scale]
            effective_scale = [sf * eff for sf, eff in zip(scale_factor, effective_scale)]
            # we need to reverse the scale factors because paintera has axis order
            # XYZ and we have axis order ZYX
            ds.attrs['downsamplingFactors'] = effective_scale[::-1]

    def run(self):

        # compute the correct resolutions for raw data and labels
        label_resolution = [res * sf for res, sf in zip(self.resolution,
                                                        self.scale_factors[self.label_scale])]
        raw_resolution = self.resolution

        with z5py.File(self.path) as f:
            # write metadata for the top-level label group
            label_group = f[self.label_group]
            label_group.attrs['painteraData'] = {'type': 'label'}
            label_group.attrs['maxId'] = self.max_id
            # add the metadata referencing the label to block lookup
            scale_ds_pattern = os.path.join(self.label_group, 'label-to-block-mapping', 's%d')
            label_group.attrs["labelBlockLookup"] = {"type": "n5-filesystem",
                                                     "root": os.path.abspath(os.path.realpath(self.path)),
                                                     "scaleDatasetPattern": scale_ds_pattern}
            # write metadata for the label-data group
            data_group = f[os.path.join(self.label_group, 'data')]
            data_group.attrs['maxId'] = self.max_id
            data_group.attrs['multiScale'] = True
            # we revese resolution and offset because java n5 uses axis
            # convention XYZ and we use ZYX
            data_group.attrs['offset'] = self.offset[::-1]
            data_group.attrs['resolution'] = label_resolution[::-1]
            data_group.attrs['isLabelMultiset'] = self.is_label_multiset
            self._write_downsampling_factors(data_group)
            # add metadata for unique labels group
            unique_group = f[os.path.join(self.label_group, 'unique-labels')]
            unique_group.attrs['multiScale'] = True
            self._write_downsampling_factors(unique_group)
            # add metadata for label to block mapping
            mapping_group = f[os.path.join(self.label_group, 'label-to-block-mapping')]
            mapping_group.attrs['multiScale'] = True
            self._write_downsampling_factors(mapping_group)
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
    assignment_key = luigi.Parameter(default='')
    use_label_multiset = luigi.BoolParameter(default=False)
    offset = luigi.ListParameter(default=[0, 0, 0])
    resolution = luigi.ListParameter(default=[1, 1, 1])

    #####################################
    # Step 1 Implementations: make_labels
    #####################################

    def _link_labels(self, data_path, dependency):
        norm_path = os.path.abspath(os.path.realpath(self.path))
        src = os.path.join(norm_path, self.label_in_key)
        dst = os.path.join(data_path, 's0')
        # self._write_log("linking label dataset from %s to %s" % (src, dst))
        os.symlink(src, dst)
        return dependency

    # TODO implement
    def _make_label_multiset(self):
        raise NotImplementedError("Label multi-set not implemented yet")

    def _make_labels(self, dependency):

        # check if we have output labels already
        dst_key = os.path.join(self.label_out_key, 'data', 's0')
        with z5py.File(self.path) as f:
            assert self.label_in_key in f, "key %s not in input file" % self.label_in_key
            if dst_key in f:
                return dependency

        # we make the label output group
        with z5py.File(self.path) as f:
            g = f.require_group(self.label_out_key)
            dgroup = g.require_group('data')
            # resolve relative paths and links
            data_path = os.path.abspath(os.path.realpath(dgroup.path))

        # if we use label-multisets, we need to create the label multiset for this scale
        # otherwise, we just make a symlink
        # make symlink from input dataset to output dataset
        return self._make_label_multiset(dependency) if self.use_label_multiset\
            else self._link_labels(data_path, dependency)

    ######################################
    # Step 2 Implementations: align scales
    ######################################

    # TODO implement for label-multi-set
    def _downsample_labels(self, downsample_scales, scale_factors, dependency):
        task = getattr(sampling_tasks, self._get_task_name('Downscaling'))

        # run downsampling
        in_scale = self.label_scale
        in_key = os.path.join(self.label_out_key, 'data', 's0')
        dep = dependency

        effective_scale = [1, 1, 1]
        label_scales = range(1, len(downsample_scales) + 1)
        for scale, out_scale in zip(label_scales, downsample_scales):
            out_key = os.path.join(self.label_out_key, 'data', 's%i' % scale)
            scale_factor = scale_factors[out_scale]
            effective_scale = [eff * scf for eff, scf in zip(effective_scale, scale_factor)]
            dep = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       input_path=self.path, input_key=in_key,
                       output_path=self.path, output_key=out_key,
                       scale_factor=scale_factor, scale_prefix='s%i' % scale,
                       effective_scale_factor=effective_scale,
                       dependency=dep)

            in_scale = out_scale
            in_key = out_key
        return dep

    def _align_scales(self, dependency):
        # check which sales we have in the raw data
        raw_dir = os.path.join(self.path, self.raw_key)
        raw_scales = os.listdir(raw_dir)
        raw_scales = [rscale for rscale in raw_scales
                      if os.path.isdir(os.path.join(raw_dir, rscale))]

        def isint(inp):
            try:
                int(inp)
                return True
            except ValueError:
                return False

        raw_scales = np.array([int(rscale[1:]) for rscale in raw_scales if isint(rscale[1:])])
        raw_scales = np.sort(raw_scales)

        # match the label scale and determine which scales we have to compute
        # via downsampling
        downsample_scales = raw_scales[self.label_scale+1:]

        # load the scale factors from the raw dataset
        scale_factors = []
        relative_scale_factors = []
        with z5py.File(self.path) as f:
            for scale in raw_scales:
                scale_key = os.path.join(self.raw_key, 's%i' % scale)
                # we need to reverse the scale factors because paintera has axis order
                # XYZ and we have axis order ZYX
                if scale == 0:
                    scale_factors.append([1, 1, 1])
                    relative_scale_factors.append([1, 1, 1])
                else:
                    scale_factor = f[scale_key].attrs['downsamplingFactors'][::-1]
                    # find the relative scale factor
                    rel_scale = [int(sf_out // sf_in) for sf_out, sf_in
                                 in zip(scale_factor, scale_factors[-1])]

                    scale_factors.append(scale_factor)
                    relative_scale_factors.append(rel_scale)

        # downsample segmentations
        t_down = self._downsample_labels(downsample_scales,
                                         relative_scale_factors, dependency)
        return t_down, relative_scale_factors

    ############################################
    # Step 3 Implementations: make block uniques
    ############################################

    def _uniques_in_blocks(self, dependency, scale_factors):
        task = getattr(unique_tasks, self._get_task_name('UniqueBlockLabels'))
        # require the unique-labels group
        with z5py.File(self.path) as f:
            f.require_group(os.path.join(self.label_out_key, 'unique-labels'))
        dep = dependency

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
    # Step 4 Implementations: invert block uniques
    ##############################################

    def _label_block_mapping(self, dependency, scale_factors):
        task = getattr(labels_to_block_tasks, self._get_task_name('LabelBlockMapping'))
        # require the labels-to-blocks group
        with z5py.File(self.path) as f:
            f.require_group(os.path.join(self.label_out_key, 'label-to-block-mapping'))
        # get the framgent max id
        with z5py.File(self.path) as f:
            max_id = f[self.label_in_key].attrs['maxId']

        # compte the label to block mapping for all scales
        n_scales = len(scale_factors)
        dep = dependency
        for scale in range(n_scales):
            in_key = os.path.join(self.label_out_key, 'unique-labels', 's%i' % scale)
            out_key = os.path.join(self.label_out_key, 'label-to-block-mapping', 's%i' % scale)
            dep = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       input_path=self.path, output_path=self.path,
                       input_key=in_key, output_key=out_key,
                       number_of_labels=max_id + 1, dependency=dep,
                       prefix='s%i' % scale)
        return dep

    #####################################################
    # Step 5 Implementations: fragment segment assignment
    #####################################################

    def _fragment_segment_assignment(self, dependency):
        if self.assignment_key == '':
            # get the framgent max id
            with z5py.File(self.path) as f:
                max_id = f[self.label_in_key].attrs['maxId']
            return dependency, max_id
        else:
            # TODO should make this a task
            with z5py.File(self.path) as f:
                assignments = f[self.assignment_key][:]
                n_fragments = len(assignments)

                # find the fragments which have non-trivial assignment
                segment_ids, counts = np.unique(assignments, return_counts=True)
                fragment_ids = np.arange(n_fragments, dtype='uint64')
                fragment_ids_to_counts = counts[segment_ids[fragment_ids]]
                non_triv_fragments = fragment_ids[fragment_ids_to_counts > 1]
                non_triv_segments = assignments[non_triv_fragment_ids]
                non_triv_segments += n_fragments

                # determine the overall max id
                max_id = int(non_triv_segments).max()

                # TODO do we need to assign a special value to ignore label (0) ?
                frag_to_seg = np.vstack((non_triv_fragments, non_triv_segments))

                out_key = os.path.join(self.label_out_key, 'fragment-segment-assignment')
                chunks = (1, n_fragments)
                f.require_dataset(out_key, data=frag_to_seg, compression='gzip', chunks=chunks)
            return dependency, max_id

    def requires(self):
        # first, we make the labels at label_out_key
        # (as label-multi-set if specified)
        t1 = self._make_labels(self.dependency)
        # next, align the scales of labels and raw data
        t2, scale_factors = self._align_scales(t1)
        downsampling_factors = [[1, 1, 1]] + scale_factors[self.label_scale+1:]

        # # next, compute the mapping of unique labels to blocks
        t3 = self._uniques_in_blocks(t2, downsampling_factors)
        # # next, compute the inverse mapping
        t4 = self._label_block_mapping(t3, downsampling_factors)
        # # next, compute the fragment-segment-assignment
        t5, max_id = self._fragment_segment_assignment(t4)

        # finally, write metadata
        t6 = WritePainteraMetadata(tmp_folder=self.tmp_folder, path=self.path,
                                   raw_key=self.raw_key,
                                   label_group=self.label_out_key, scale_factors=scale_factors,
                                   label_scale=self.label_scale,
                                   is_label_multiset=self.use_label_multiset,
                                   resolution=self.resolution, offset=self.offset,
                                   max_id=max_id, dependency=t5)
        return t6

    @staticmethod
    def get_config():
        configs = super(ConversionWorkflow, ConversionWorkflow).get_config()
        configs.update({'unique_block_labels': unique_tasks.UniqueBlockLabelsLocal.default_task_config(),
                        'label_block_mapping': labels_to_block_tasks.LabelBlockMappingLocal.default_task_config(),
                        'downscaling': sampling_tasks.DownscalingLocal.default_task_config()})
        return configs
