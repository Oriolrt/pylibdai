from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy
import subprocess
import zipfile

NAME = "dai"
VERSION = "1.0"
DESCR = "Wrapper for Python to use libDAI"
URL = "https://github.com/Oriolrt/pylibdai"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Oriol Ramos Terrades"
EMAIL = "oriolrt@cvc.uab.cat"

LICENSE = "BSD 2-clause"

SRC_DIR = "libdai"
PACKAGES = [SRC_DIR]


system = subprocess.check_output('uname').splitlines()[0].decode()
libdaidir = 'libdai'

if system == 'Linux':
  libdai = 'dai-'+system

  with zipfile.ZipFile(libdaidir + '/lib/lib' + libdai + '.zip', 'r') as zip_ref:
    zip_ref.extractall(libdaidir + '/lib/')
  lib_boost_searched = "libboost_program_options.so"
  boost_inc = [ x[:x.decode().find("blank.hpp")].decode() for x in subprocess.check_output("find /usr -name  blank.hpp", shell=True).splitlines()]
  boost_inc = boost_inc if len(boost_inc) > 1 else boost_inc[0]
  boost_lib = [ x[:x.decode().find(lib_boost_searched)].decode() for x in subprocess.check_output("find /usr -name  "+lib_boost_searched, shell=True).splitlines()]
  boost_lib = boost_lib if len(boost_lib) > 1 else boost_lib[0]
  gmp_inc = []
  gmp_lib = []

if system == 'Darwin':
  libdai = 'dai-'+system
  boost_path = '/opt/local/libexec/boost/1.81/include'
  boost_inc = boost_path[:boost_path.find('include')] +'include/' 
  boost_lib = boost_path + '/lib'
  gmp_inc = [ x[2:].decode() for x in subprocess.check_output('pkg-config --cflags  gmp',shell=True).splitlines() ]
  gmp_lib = [ x[2:].decode().split()[0] for x in subprocess.check_output('pkg-config --libs  gmp',shell=True).splitlines() ]




files = ["dai.pyx"]

setup(name="dai",
    version='1.0',
    install_requires=['numpy','cython'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension('dai', files, language = 'c++',
        include_dirs = [os.path.join(libdaidir, 'include')]+[numpy.get_include()] + gmp_inc + [boost_inc],
        library_dirs = [os.path.join(libdaidir, 'lib')] + gmp_lib + [boost_lib],
        libraries = [libdai, 'gmp', 'gmpxx'])]
)

