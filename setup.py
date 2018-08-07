from setuptools import find_packages, setup

exclude_dirs = ["exmples",
                "doc",
                "scripts"]

PACKAGES = find_packages(exclude=exclude_dirs)

setup(name='pytorch_integrated_cell', 
                 version='0.1',
                 packages=PACKAGES)
