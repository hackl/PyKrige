__author__ = 'Benjamin S. Murphy'
__version__ = '1.4.dev1'
__doc__ = """
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Kriging toolkit for Python.

ok: Contains class OrdinaryKriging, which is a convenience class for easy
    access to 2D ordinary kriging.
uk: Contains class UniversalKriging, which provides more control over
    2D kriging by utilizing drift terms. Supported drift terms currently
    include point-logarithmic, regional linear, and external z-scalar.
    Generic functions of the spatial coordinates may also be supplied to
    provide drift terms, or the point-by-point values of a drift term
    may be supplied.
ok3d: Contains class OrdinaryKriging3D, which provides support for
    3D ordinary kriging.
uk3d: Contains class UniversalKriging3D, which provide support for
    3D universal kriging. A regional linear drift is the only drift term
    currently supported, but generic drift functions or point-by-point
    values of a drift term may also be supplied.
kriging_tools: Contains a set of functions to work with *.asc files.
variogram_models: Contains the definitions for the implemented variogram
    models. Note that the utilized formulas are as presented in Kitanidis,
    so the exact definition of the range (specifically, the associated
    scaling of that value) may differ slightly from other sources.
core: Contains the backbone functions of the package that are called by both
    the various kriging classes. The functions were consolidated here
    in order to reduce redundancy in the code.
test: Contains the test script.

References
----------
.. [1] P.K. Kitanidis, Introduction to Geostatistics: Applications in
    Hydrogeology, (Cambridge University Press, 1997) 272 p.
    
Copyright (c) 2015-2018, PyKrige Developers
"""

import os
import sys
import collections
import configparser
import logging

from . import kriging_tools as kt
from .ok import OrdinaryKriging
from .uk import UniversalKriging
from .ok3d import OrdinaryKriging3D
from .uk3d import UniversalKriging3D

__all__ = ['ok', 'uk', 'ok3d', 'uk3d', 'kriging_tools']

class DotDict(collections.OrderedDict):
    """
    A string-valued dictionary that can be accessed with the "." notation
    """
    def __getattr__(self, key):
        try:
            if self[key] == 'True':
                return True
            elif self[key] == 'False':
                return False
            elif self.is_number(self[key]):
                if self.is_int(self[key]):
                    return int(self[key])
                else:
                    return float(self[key])
            else:
                return self[key]
        except KeyError:
            raise AttributeError(key)

    def is_number(self, s):
        try:
            float(s)
            return True
        except TypeError:
            return False
        except ValueError:
            return False

    def is_int(self,x):
        try:
            a = float(x)
            b = int(a)
        except ValueError:
            return False
        else:
            return a == b

    # def __getattr__(self, key):
    #     try:
    #         return self[key]
    #     except KeyError:
    #         raise AttributeError(key)

config = DotDict() # global configuration

d = os.path.dirname
base = os.path.join(d(d(__file__)),'pykrige', 'pykrige.cfg')

config.paths = [base,'pykrige.cfg']

def read(*paths, **validators):
    """
    Load the configuration, make each section available in a separate dict.

    The configuration location is where the script is executed:
       - pykrige.cfg

    If this file is missing, the fallback is the source code:
       - pykrige/pykrige.cfg

    Please note: settings in the site configuration file are overridden
    by settings with the same key names in the pykrige.cfg.
    """
    paths = config.paths + list(paths)
    parser = configparser.SafeConfigParser()
    found = parser.read(os.path.normpath(os.path.expanduser(p)) for p in paths)
    if not found:
        raise IOError('No configuration file found in %s' % str(paths))
    config.found = found
    config.clear()
    for section in parser.sections():
        config[section] = sec = DotDict(parser.items(section))
        for k, v in sec.items():
            sec[k] = validators.get(k, lambda x: x)(v)

config.read = read

config.read()

# set up logging to file - see previous section for more details
if config.logging.enabled:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)12s:%(lineno)4d - %(levelname)8s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=config.logging.logfile,
                        filemode='w')
else:
    logging.basicConfig()
if config.logging.verbose:
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging._nameToLevel[config.logging.level])

    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s: %(levelname)-5s] %(message)s',
                              datefmt='%m-%d %H:%M:%S')

    # tell the handler to use this format
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

