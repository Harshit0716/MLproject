from setuptools import setup, find_packages
from typing import List
hyphen='-e .'
def get_requirements(path:str) -> List[str]:
    requirements=[]
    with open(path) as f:
        requirements=f.readlines()
        requirements=[r.replace('\n','') for r in requirements]
        if hyphen in requirements:
            requirements.remove(hyphen)
    return requirements
setup(
name="MLproject",
version="0.0.1",
author="Harshit",
author_email="harshitsta03@gmail.com",
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
)