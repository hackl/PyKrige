#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : sandbox.py 
# Creation  : 22 Mar 2018
# Time-stamp: <Don 2018-03-22 14:17 juergen>
#
# Copyright (c) 2018 Jürgen Hackl <hackl@ibi.baug.ethz.ch>
#               http://www.ibi.ethz.ch
# $Id$ 
#
# Description : Sandbox to test new functions
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 
# =============================================================================

# import modules
# ==============

import logging
import pykrige
#from pykrige.ok import OrdinaryKriging
from pykrige.onk import OrdinaryKriging
import numpy as np
import pykrige.kriging_tools as kt
import matplotlib.pyplot as plt

# initialize logger
# =================
log = logging.getLogger(__name__)

def main():
    log.info('============= START =============')
    test_onk()
    log.info('============== END ==============')
    pass

def test_onk():
    data = np.array([[0.3, 1.2, 0.47],
                     [1.9, 0.6, 0.56],
                     [1.1, 3.2, 0.74],
                     [3.3, 4.4, 1.47],
                     [4.7, 3.8, 1.74]])

    gridx = np.arange(0.0, 5.5, 0.5)
    gridy = np.arange(0.0, 5.5, 0.5)

    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                         variogram_model = 'gaussian',
                         #verbose = True,
                         #enable_plotting = True,
                         anisotropy_scaling=1.1,
                         coordinates_type = 'euclidean',
    )
    

    z, ss = OK.execute('grid', gridx, gridy)


def test_ok():

    data = np.array([[0.3, 1.2, 0.47],
                     [1.9, 0.6, 0.56],
                     [1.1, 3.2, 0.74],
                     [3.3, 4.4, 1.47],
                     [4.7, 3.8, 1.74]])

    gridx = np.arange(0.0, 5.5, 0.5)
    gridy = np.arange(0.0, 5.5, 0.5)

    # Create the ordinary kriging object. Required inputs are the X-coordinates
    # of the data points, the Y-coordinates of the data points, and the Z-values
    # of the data points. If no variogram model is specified, defaults to a
    # linear variogram model. If no variogram model parameters are specified,
    # then the code automatically calculates the parameters by fitting the
    # variogram model to the binned experimental semivariogram. The verbose
    # kwarg controls code talk-back, and the enable_plotting kwarg controls the
    # display of the semivariogram. 
    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                         enable_plotting=True)

    # Creates the kriged grid and the variance grid. Allows for kriging on a
    # rectangular grid of points, on a masked rectangular grid of points, or
    # with arbitrary points. (See OrdinaryKriging.__doc__ for more information.)
    z, ss = OK.execute('grid', gridx, gridy)

    # plt.clf()
    # plt.imshow(z);
    # plt.colorbar()
    # plt.savefig('result.png',fmt='png',dpi=200)

    # plt.clf()
    # fig, ax = plt.subplots()
    # ax.scatter( data[:, 0], data[:, 1], c=data[:, 2]) cmap='gray' )
    # plt.colorbar()
    # ax.set_aspect(1)
    # plt.xlabel('Easting [m]')
    # plt.ylabel('Northing [m]')
    # plt.title('Porosity %') ;
    # plt.savefig('data.png',fmt='png',dpi=200)
    # # Writes the kriged grid to an ASCII grid file.
    # kt.write_asc_grid(gridx, gridy, z, filename="output.asc")
    pass

if __name__ == '__main__':
    main()


# =============================================================================
# eof
#
# Local Variables: 
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 80
# End:  
