from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path) -> List[str]:

    """This function will return the list of requirements mentioned in requirements.txt file"""
    with open(file_path, mode='r') as file:
        requirements = file.readlines()
        
    l=[i.strip() for i in requirements]

    return l


setup(
    name='Insurance Cost Prediction',
    version='0.1.0',
    packages=find_packages(),
    install_requires=get_requirements(file_path='requirements.txt'),
    author='SunnyCodex',
    description='A project to predict insurance costs using machine learning techniques.',
)