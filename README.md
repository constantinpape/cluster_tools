# Cluster Tools

Workflows for distributed Bio Image Analysis and Segmentation.
Supports Slurm, LSF and local execution, easy to extend to more scheduling systems.


## Workflows

- [Hierarchical Multicut](http:/openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w1/Pape_Solving_Large_Multicut_ICCV_2017_paper.pdf) / [Hierarchical lifted Multicut](https://arxiv.org/abs/1905.10535)
  - Distance Transform Watersheds
  - Region Adjacency Graph
  - Edge Feature Extraction from Boundary-or-Affinity Maps
  - Agglomeration via (lifted) Multicut
- [Sparse lifted Multicut from biological priors](https://arxiv.org/abs/1905.10535)
- [Mutex Watershed](https://link.springer.com/chapter/10.1007/978-3-030-01225-0_34)
- Connected Components
- Downscaling and Pyramids
  - [Paintera Format](https://github.com/saalfeldlab/paintera)
  - [BigDataViewer Format](https://imagej.net/BigDataViewer)
  - [Bigcat Format](https://github.com/saalfeldlab/bigcat)
- [Ilastik Prediction](https://www.ilastik.org/)
- Skeletonization
- Distributed Neural Network Prediction (originally implemented [here](https://github.com/constantinpape/simpleference))
- Validation with Rand Index and Variation of Information


## Installation

You can install the package via conda:
```
conda install -c conda-forge -c cpape cluster_tools
```

To set-up a develoment environment with all necessary dependencies, you can use the `environment.yml` file:
```
conda env create -f environment.yml
```
In this case, the package itself must be added to the python environment manually, e.g. by
creating a softlink in the `lib/python3.7/site-packages` folder of the conda env.


## Citation

If you use this software in a publication, please cite
```
Pape, Constantin, et al. "Solving large multicut problems for connectomics via domain decomposition." Proceedings of the IEEE International Conference on Computer Vision. 2017.
```

For the lifted multicut workflows, please cite
```
Pape, Constantin, et al. "Leveraging Domain Knowledge to improve EM image segmentation with Lifted Multicuts." arXiv preprint. 2019.
```
You can find code for the experiments in `publications/lifted_domain_knowledge`.

If you are using another algorithom not part of these two publications, please also cite the appropriate publication ([see the links here](https://github.com/constantinpape/cluster_tools#workflows)).


## Getting Started

This repository uses [luigi](https://github.com/spotify/luigi) for workflow management.
We support different cluster schedulers, so far 
- [`slurm`](https://slurm.schedmd.com/documentation.html)
- [`lsf`](https://www.ibm.com/support/knowledgecenter/en/SSWRJV_10.1.0/lsf_welcome/lsf_kc_ss.html)
- `local` (local execution based on `ProcessPool`)

The scheduler can be selected by the keyword `target`.
Inter-process communication is achieved through files which are stored in a temporary folder and
most workflows use [n5](https://github.com/saalfeldlab/n5) storage. You can use [z5](https://github.com/constantinpape/z5) to convert files to it with python.

Simplified, running a workflow from this repository looks like this:
```py
import json
import luigi
from cluster_tools import SimpleWorkflow  # this is just a mock class, not actually part of this repository

# folder for temporary scripts and files
tmp_folder = 'tmp_wf'

# directory for configurations for workflow sub-tasks stored as json
config_dir = 'configs'

# get the default configurations for all sub-tasks
default_configs = SimpleWorkflow.get_config()

# global configuration for shebang to proper python interpreter with all dependencies,
# group name and block-shape
global_config = default_configs['global']
shebang = '#! /path/to/bin/python'
global_config.update({'shebang': shebang, 'groupname': 'mygroup'})
with open('configs/global.config', 'w') as f:
  json.dump(global_config, f)
  
# run the example workflow with `max_jobs` number of jobs
max_jobs = 100
task = SimpleWorkflow(tmp_folder=tmp_folder, config_dir=config_dir,
                      target='slurm', max_jobs=max_jobs,
                      input_path='/path/to/input.n5', input_key='data',
                      output_path='/path/to/output.n5', output_key='data')
luigi.build([task])
 ```
For a list of the available segmentation worklfows, have a look at [this](https://github.com/constantinpape/cluster_tools/blob/master/cluster_tools/workflows.py).
Unfortunately, there is no proper documentation yet. For more details, have a look at the
[examples](https://github.com/constantinpape/cluster_tools/blob/master/example), in particular
[this example](https://github.com/constantinpape/cluster_tools/blob/master/example/multicut.py).
You can donwload the example data (also used for the tests) [here](https://drive.google.com/file/d/1E_Wpw9u8E4foYKk7wvx5RPSWvg_NCN7U/view?usp=sharing).
