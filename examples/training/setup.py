from setuptools import find_packages
from setuptools import setup

#REQUIRED_PACKAGES = ["pandas", "numpy", "scikit-learn"]
REQUIRED_PACKAGES = []

setup(
    name='sklearn_cluster',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='A trainer application package for generic clustering/segmentation'
)