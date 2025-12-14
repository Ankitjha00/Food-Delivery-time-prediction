from setuptools import setup,find_packages
from typing import List

def get_requirements(file_name:str)->List[str]:
    
    requirements=[]
    
    try:
        with open(file_name,'r') as file_obj:
            file_content=file_obj.readline()
            
            for line in file_content:
                requirement=line.strip()
                if requirement and requirement!='-e .':
                    requirements.append(requirement)
        
    except FileNotFoundError:
        print("Requirements.txt file not found")
                
    return requirements
    
            
            
            
setup(
    name='Food Delivery Time Prediction',
    version='0.0.1',
    author='Sarvesh Chhabra',
    author_email='sarveshpoker@gmail.com',
    packages=find_packages(),
    install_packages=get_requirements('reqirements.txt')    
)
        
        