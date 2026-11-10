import re
import sys
from pathlib import Path
from setuptools import Extension, find_packages, setup

# Get the version number:
with open('hotcent/__init__.py') as f:
    version = re.search("__version__ = '(.*)'", f.read()).group(1)

extensions = [
    Extension('_hotcent',
              sources=['hotcent/extensions.c'],
              language='c',
              extra_compile_args=['-O3', '-march=native'],
              ),
]

files = [
    'hotcent-basis',
    'hotcent-concat',
    'hotcent-tables',
]
scripts = [str(Path('tools') / f) for f in files]

install_requires = [
    'ase>=3.22.0',
    'lxml',
    'matplotlib',
    'numpy>=2.0.0',
    'pyyaml',
    'scipy>=1.14.0',
]

extras_require = {
    'dev': [
        'pytest',
    ]
}

setup(
  ext_modules=extensions,
  install_requires=install_requires,
  extras_require=extras_require,
  license='LICENSE',
  name='hotcent',
  packages=find_packages(),
  python_requires='>=3.10',
  scripts=scripts,
  url='https://gitlab.com/mvdb/hotcent',
  version=version,
)
