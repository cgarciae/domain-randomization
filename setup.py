import os
from setuptools import setup, find_packages


setup(
    name = "domain_randomization",
    version = "0.0.1",
    author = "Cristian Garcia",
    author_email = "cgarcia.e88@gmail.com",
    description = ("An object-like dictionary"),
    license = "MIT",
    keywords = [],
    url = "https://github.com/cgarciae/domain_randomization",
   	packages = find_packages(),
    package_data={
        '': ['LICENCE', 'requirements.txt', 'README.md', 'CHANGELOG.md'],
    },
    include_package_data = True,
    install_requires = [
        
    ],
    entry_points = {
        'console_scripts': [
            'domain-randomization = domain_randomization.cli:main'
        ],
    }
)
