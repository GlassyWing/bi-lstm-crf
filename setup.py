from setuptools import setup, find_packages

setup(name='dltokenizer',

      version='0.1-SNAPSHOT',

      url='https://github.com/GlassyWing/bi-lstm-crf',

      license='Apache License 2.0',

      author='Manlier',

      author_email='dengjiaxim@gmail.com',

      description='inset pest predict model',

      packages=find_packages(exclude=['tests', 'examples']),

      package_data={'dltokenizer': ['*.*', 'checkpoints/*', 'config/*']},

      long_description=open('README.md', encoding="utf-8").read(),

      zip_safe=False,

      install_requires=['keras', 'keras-contrib'],

      )
