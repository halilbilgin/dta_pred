from setuptools import setup, find_packages

setup(name='dta_pred',
      version='0.5',
      description='A specialized deep learning library for protein drug affinity prediction.',
      author='Halil Bilgin',
      author_email='bilginhalil@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)