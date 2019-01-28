from setuptools import setup

setup(name='semnet',
	version='0.1.0',
	description='A package for working with Semantic Medline data',
	url='https://github.gatech.edu/asedler3/semnet',
	author='Andrew Sedler',
	author_email='asedler3@gatech.edu',
	packages=['semnet'],
	install_requires=['hetio', 'xarray', 'numpy','py2neo', 'pandas', 
	'sklearn', 'scipy', 'matplotlib', 'tqdm'],
	include_package_data=True,
	package_data={
		'semnet': ['data/*']
	}
)
