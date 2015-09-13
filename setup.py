from setuptools import setup

long_description = \
    """A python package for storing structs of numpy arrays in a persistent
    key-value object, offering the advantages of HDF5 (single files on disk,
    reasonably fast random access) with the simplicity of a 'memory-mapped'
    dictionary."""

setup(
    name='biggie',
    version='0.0.1',
    description='A library for managing notoriously big data.',
    author='Eric J. Humphrey',
    author_email='ejhumphrey@nyu.edu',
    url='http://github.com/ejhumphrey/biggie',
    download_url='http://github.com/ejhumphrey/biggie/releases',
    packages=['biggie'],
    package_data={},
    long_description=long_description,
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7"
    ],
    keywords='',
    license='ISC',
    install_requires=[
        'numpy >= 1.9.0',
        'h5py >= 2.2.1'
    ]
)
