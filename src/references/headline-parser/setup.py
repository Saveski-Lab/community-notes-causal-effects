

from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    'spacy>=2.0',
    'boltons',
]

CLASSIFIERS = [
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]


setup(
    name='headline-parser',
    version='0.1.0',
    description='Standardize news article headlines.',
    url='https://github.com/davidmcclure/headline-parser',
    license='MIT',
    author='David McClure',
    author_email='dclure@mit.edu',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=CLASSIFIERS,
)
