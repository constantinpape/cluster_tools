from distutils.core import setup
from pkgutil import walk_packages

import cluster_tools
from cluster_tools import __version__


def find_packages(path=__path__, prefix=""):
    yield prefix
    prefix = prefix + "."
    for _, name, ispkg in walk_packages(path, prefix):
        if ispkg:
            yield name


setup(name='cluster_tools',
      version=__version__,
      description='Workflows for distributed bio-image analysis and segmentation',
      author='Constantin Pape',
      packages=list(find_packages(mypackage.__path__, mypackage.__name__)))
