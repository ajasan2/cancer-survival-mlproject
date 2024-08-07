from setuptools import setup, find_packages
from typing import List

def get_requirements(filename:str)->List[str]:
    requirements = []
    with open(filename) as file:
        requirements = file.readlines()
        requirements = [r.strip() for r in requirements]
        requirements.remove('-e .') if '-e .' in requirements else None
    return requirements


setup(
    name='mlproject',
    author='ajasan2',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)