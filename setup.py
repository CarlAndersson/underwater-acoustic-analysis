from setuptools import setup
import sys
import os.path

package_name = 'uwacan'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_name))
from _version import version_manager  # We cannot import the _version module, but we can import from it.

with version_manager() as version:
    setup(
        name=package_name,
        version=version,
        description='Underwater Acoustic Analysis tools',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/CarlAndersson/underwater-acoustics-analysis',
        author='Carl Andersson',
        author_email='carl.andersson@ivl.se',
        packages=[package_name],
        python_requires=">=3.8",
        install_requires=[
            'numpy',
            'scipy',
            'geographiclib',
            'pendulum>=3',
            'xarray',
            'python-dotenv',
            'pysoundfile',
        ],
    )
