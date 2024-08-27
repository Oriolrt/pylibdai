

# pylibDAI
This Cython wrapper for libDAI have been adapted from [this repository](https:github.com/samehkhamis/pylibdai). Similar to libDAI, it is licensed under the BSD 2-clause license.


# Requirements

We need a Python interpreter with version at least 3.10 or greater. The following dependencies have to be installed to link properly with the libdai library.

```
apt-get install libboost-dev libboost-graph-dev libboost-program-options-dev libboost-test-dev libgmp-dev cimg-dev libboost-program-options-dev
```

Python dependencies are found in the [requirements.txt](requirements.txt) file.

# Install

The libdai library has been compiled separately and saved in the [libdai/lib](libdai/lib) folder for MacOSX and Linux ubuntu. The [setup.py](setup.py) script allow to install de **dai** library in any Python virtual environment or conda environment as follows:

```angular2html
python -m pip install git+https://github.com/Oriolrt/pylibdai.git
```

# Example scripts

The [test](test.py) and [test_tree_order](test_tree_order.py) are simple examples to run inference algorithms in 2 order and 3 order PGMs. The [ExamplePyLibDAI](ExamplePyLiBDAI.ipynb) notebook is a simple example like the [test](test.py) script. The [ExampleMiddlebury](ExampleMiddlebury.py) script is an adaptation borrowed from  [this repository](https://github.com/amueller/daimrf) of the classic Middelbury task. 