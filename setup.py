from setuptools import setup

setup(name='semnet',
	version='0.1.1',
	description='A package for working with Semantic Medline data',
	url='https://github.gatech.edu/asedler3/semnet',
	author='Andrew Sedler',
	author_email='asedler3@gatech.edu',
	packages=['semnet'],
	install_requires=['hetio==0.2.8', 'xarray==0.16.2', 'numpy==1.19.5','py2neo==3.1.2', 'pandas==1.1.5', 
	'scikit-learn>=0.19.1', 'scipy==1.5.4', 'matplotlib==3.3.3', 'tqdm>=4.23.4', 'seaborn==0.8.1'],
	include_package_data=True,
	package_data={
		'semnet': ['data/*']
	}
)
