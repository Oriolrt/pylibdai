from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

libdaidir = '../../libDAI-0.3.1'

files = ["dai.pyx"]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension('dai', files, language = 'c++',
        include_dirs = [os.path.join(libdaidir, 'include')],
        library_dirs = [os.path.join(libdaidir, 'lib')],
        libraries = ['dai', 'gmp', 'gmpxx'])]
)

