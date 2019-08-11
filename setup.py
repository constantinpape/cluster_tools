import runpy
from setuptools import setup, find_packages

version = runpy.run_path('cluster_tools/version.py')['__version__']
setup(name='cluster_tools',
      packages=find_packages(include='cluster_tools'),
      version=version,
      author='Constantin Pape',
      url='https://github.com/constantinpape/cluster_tools',
      license='MIT')
