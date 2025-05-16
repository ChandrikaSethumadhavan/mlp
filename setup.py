from setuptools import setup, find_packages,setup
from typing import List

def get_requirements(file_path:str)->List [str]:
    '''
    This function will return a list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() # readlines() will return a list of strings 
        requirements = [req.replace("\n", "") for req in requirements] # replace the \n with empty string   
        # if '-e .' in requirements:  # if -e . is present in the requirements, remove it (check requirements.txt for explaination)
        #     requirements.remove('-e .')
    return requirements


setup(name ="mlp",
      version = "0.0.1",
      author= "Chandrika",
      author_email = "chandrikasethu06@gmail.com",
      packages = find_packages(),
      install_requires = get_requirements('requirements.txt'))