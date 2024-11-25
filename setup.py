from setuptools import setup, find_packages

setup(
    name='OmniDoublet',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/mmmads/OmniDoublet',
    license='MIT',
    author='L Liu',
    author_email='11918162@zju.edu.cn',
    description='A doublet detection method for multimodal single cell data',
    long_description=open("README.md").read(),
)
