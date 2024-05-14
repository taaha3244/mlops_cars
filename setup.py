from setuptools import find_packages,setup
from typing import List

Hypen="-e ."
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if Hypen in requirements:
            requirements.remove(Hypen)
    return requirements



setup(
name='mlops',
version='0.0.1',
author='Taaha',
author_email='taahamushtaq1998@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')


)