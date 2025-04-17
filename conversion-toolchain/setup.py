from setuptools import setup, find_packages

setup(
    name='conversion_toolchain',
    version='0.1.0',
    author='Ayoub Assis',
    author_email='aassis@networkoptix.com',

    packages=find_packages(),
    python_requires='>=3.8',
    entry_points={
        'console_scripts': ['conversion_toolchain=conversion_toolchain.main:cli']
    },

)
