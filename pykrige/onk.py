#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : onk.py 
# Creation  : 22 Mar 2018
# Time-stamp: <Die 2018-03-27 10:07 juergen>
#
# Copyright (c) 2018 Jürgen Hackl <hackl@ibi.baug.ethz.ch>
#               http://www.ibi.ethz.ch
# $Id$ 
#
# Description : Ordinary Network Kriging
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

import logging
import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from . import variogram_models
from . import core, config
from .core import _adjust_for_anisotropy, _initialize_variogram_model, \
    _make_variogram_parameter_list, _find_statistics

log = logging.getLogger(__name__)

class OrdinaryKriging:
    """Convenience class for easy access to 2D Ordinary Kriging.

    Parameters
    ----------
    x : array_like
        X-coordinates of data points or network nodes
    y : array_like
        Y-coordinates of data points or network nodes
    z : array_like
        Values at data points or network nodes
    variogram_model : str, optional
        Specifies which variogram model to use; may be one of the following:
        linear, power, gaussian, spherical, exponential, hole-effect.
        Default is linear variogram model. To utilize a custom variogram model,
        specify 'custom'; you must also provide variogram_parameters and
        variogram_function. Note that the hole-effect model is only technically
        correct for one-dimensional problems.
    variogram_parameters : list or dict, optional
        Parameters that define the specified variogram model. If not provided,
        parameters will be automatically calculated using a "soft" L1 norm
        minimization scheme. For variogram model parameters provided in a dict,
        the required dict keys vary according to the specified variogram
        model: ::
            linear - {'slope': slope, 'nugget': nugget}
            power - {'scale': scale, 'exponent': exponent, 'nugget': nugget}
            gaussian - {'sill': s, 'range': r, 'nugget': n}
                        OR
                       {'psill': p, 'range': r, 'nugget':n}
            spherical - {'sill': s, 'range': r, 'nugget': n}
                         OR
                        {'psill': p, 'range': r, 'nugget':n}
            exponential - {'sill': s, 'range': r, 'nugget': n}
                           OR
                          {'psill': p, 'range': r, 'nugget':n}
            hole-effect - {'sill': s, 'range': r, 'nugget': n}
                           OR
                          {'psill': p, 'range': r, 'nugget':n}
        Note that either the full sill or the partial sill
        (psill = sill - nugget) can be specified in the dict.
        For variogram model parameters provided in a list, the entries
        must be as follows: ::
            linear - [slope, nugget]
            power - [scale, exponent, nugget]
            gaussian - [sill, range, nugget]
            spherical - [sill, range, nugget]
            exponential - [sill, range, nugget]
            hole-effect - [sill, range, nugget]
        Note that the full sill (NOT the partial sill) must be specified
        in the list format.
        For a custom variogram model, the parameters are required, as custom
        variogram models will not automatically be fit to the data.
        Furthermore, the parameters must be specified in list format, in the
        order in which they are used in the callable function (see
        variogram_function for more information). The code does not check
        that the provided list contains the appropriate number of parameters
        for the custom variogram model, so an incorrect parameter list in
        such a case will probably trigger an esoteric exception someplace
        deep in the code.
        NOTE that, while the list format expects the full sill, the code
        itself works internally with the partial sill.
    variogram_function : callable, optional
        A callable function that must be provided if variogram_model is
        specified as 'custom'. The function must take only two arguments:
        first, a list of parameters for the variogram model; second, the
        distances at which to calculate the variogram model. The list
        provided in variogram_parameters will be passed to the function
        as the first argument.
    nlags : int, optional
        Number of averaging bins for the semivariogram. Default is 6.
    weight : bool, optional
        Flag that specifies if semivariance at smaller lags should be weighted
        more heavily when automatically calculating variogram model.
        The routine is currently hard-coded such that the weights are
        calculated from a logistic function, so weights at small lags are ~1
        and weights at the longest lags are ~0; the center of the logistic
        weighting is hard-coded to be at 70% of the distance from the shortest
        lag to the largest lag. Setting this parameter to True indicates that
        weights will be applied. Default is False. (Kitanidis suggests that the
        values at smaller lags are more important in fitting a variogram model,
        so the option is provided to enable such weighting.)
    anisotropy_scaling : float, optional
        Scalar stretching value to take into account anisotropy.
        Default is 1 (effectively no stretching).
        Scaling is applied in the y-direction in the rotated data frame
        (i.e., after adjusting for the anisotropy_angle, if anisotropy_angle
        is not 0). This parameter has no effect if coordinate_types is
        set to 'geographic' or 'network'.
    anisotropy_angle : float, optional
        CCW angle (in degrees) by which to rotate coordinate system in
        order to take into account anisotropy. Default is 0 (no rotation).
        Note that the coordinate system is rotated. This parameter has
        no effect if coordinate_types is set to 'geographic' or 'network'
    coordinates_type : str, optional
        One of 'euclidean', 'geographic' or 'network'. Determines if the x and y
        coordinates are interpreted as on a plane ('euclidean'), as coordinates
        on a sphere ('geographic'), or as coordinates along edges ('network').
        In case of geographic coordinates, x is interpreted as longitude and y
        as latitude coordinates, both given in degree. Longitudes are expected
        in [0, 360] and latitudes in [-90, 90]. Default is 'euclidean'.
    network : network, optional
        If network kriging is chosen by enabling the coordinate_type 'network',
        the network topology has to be provided as a Network class object.
    
    References
    ----------
    .. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
        Hydrogeology, (Cambridge University Press, 1997) 272 p.
    """

    eps = config.kriging.cutoff   # Cutoff for comparison to zero
    variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

    def __init__(self, x, y, z, variogram_model='linear',
                 variogram_parameters=None, variogram_function=None, nlags=6,
                 weight=False, anisotropy_scaling=1.0, anisotropy_angle=0.0,
                 coordinates_type='euclidean',network=None):

        # Code assumes 1D input arrays of floats. Ensures that any extraneous
        # dimensions don't get in the way. Copies are created to avoid any
        # problems with referencing the original passed arguments.
        # Also, values are forced to be float... in the future, might be worth
        # developing complex-number kriging (useful for vector field kriging)
        self.X_ORIG = \
            np.atleast_1d(np.squeeze(np.array(x, copy=True, dtype=np.float64)))
        self.Y_ORIG = \
            np.atleast_1d(np.squeeze(np.array(y, copy=True, dtype=np.float64)))
        self.Z = \
            np.atleast_1d(np.squeeze(np.array(z, copy=True, dtype=np.float64)))

        self.network = network

        self.verbose = config.logging.verbose
        self.enable_plotting = config.plotting.enabled
        self.enable_statistics = config.statistics.enabled

        self.coordinates_type = coordinates_type

        log.info("Initializing Ordinary Kriging")
        if self.enable_plotting:
            log.info("Plotting Enabled")

        # check coordinates and adjust for anisotropy
        self._adjust_coordinates(anisotropy_scaling,anisotropy_angle)

        # set up variogram model and parameters...
        self.update_variogram_model(variogram_model,
                                    variogram_parameters=variogram_parameters,
                                    variogram_function=variogram_function,
                                    nlags=nlags, weight=weight,
                                    anisotropy_scaling=anisotropy_scaling,
                                    anisotropy_angle=anisotropy_angle)

        pass

    def _adjust_coordinates(self,anisotropy_scaling,anisotropy_angle):
        """Adjust coordinates for anisotropy.

        Notes
        -----
        Only implemented for euclidean (rectangular) coordinates, as anisotropy
        is ambiguous for geographic coordinates...

        Parameters
        ----------
        anisotropy_scaling : float, optional
            Scalar stretching value to take into account anisotropy.
            Default is 1 (effectively no stretching).
            Scaling is applied in the y-direction in the rotated data frame
            (i.e., after adjusting for the anisotropy_angle, if anisotropy_angle
            is not 0). This parameter has no effect if coordinate_types is
            set to 'geographic' or 'network'.
        anisotropy_angle : float, optional
            CCW angle (in degrees) by which to rotate coordinate system in
            order to take into account anisotropy. Default is 0 (no rotation).
            Note that the coordinate system is rotated. This parameter has
            no effect if coordinate_types is set to 'geographic' or 'network'
        """
        if self.coordinates_type == 'euclidean':
            self.XCENTER = (np.amax(self.X_ORIG) + np.amin(self.X_ORIG))/2.0
            self.YCENTER = (np.amax(self.Y_ORIG) + np.amin(self.Y_ORIG))/2.0
            self.anisotropy_scaling = anisotropy_scaling
            self.anisotropy_angle = anisotropy_angle
            log.info("Adjusting data for anisotropy...")
            self.X_ADJUSTED, self.Y_ADJUSTED = \
                _adjust_for_anisotropy(np.vstack((self.X_ORIG, self.Y_ORIG)).T,
                                       [self.XCENTER, self.YCENTER],
                                       [self.anisotropy_scaling],
                                       [self.anisotropy_angle]).T
        elif self.coordinates_type == 'geographic':
            # Leave everything as is in geographic case.
            # May be open to discussion?
            if anisotropy_scaling != 1.0:
                log.warn("Anisotropy is not compatible with geographic "
                         "coordinates. Ignoring user set anisotropy.")
            self.XCENTER= 0.0
            self.YCENTER= 0.0
            self.anisotropy_scaling = 1.0
            self.anisotropy_angle = 0.0
            self.X_ADJUSTED = self.X_ORIG
            self.Y_ADJUSTED = self.Y_ORIG
        elif self.coordinates_type == 'network':
            if self.network is None:
                log.error("No 'network' is defined.")
                raise ValueError
            # TODO: add test if network is a network
            if anisotropy_scaling != 1.0:
                log.warn("Anisotropy is not compatible with network "
                         "coordinates. Ignoring user set anisotropy.")
            self.XCENTER= 0.0
            self.YCENTER= 0.0
            self.anisotropy_scaling = 1.0
            self.anisotropy_angle = 0.0
            self.X_ADJUSTED = self.X_ORIG
            self.Y_ADJUSTED = self.Y_ORIG
        else:
            log.error("Only 'euclidean', 'geographic' and 'network' are valid "
                      +"values for coordinates-keyword.")
            raise ValueError

        pass

    def update_variogram_model(self, variogram_model, variogram_parameters=None,
                               variogram_function=None, nlags=6, weight=False,
                               anisotropy_scaling=1., anisotropy_angle=0.):
        """Allows user to update variogram type and/or
        variogram model parameters.

        Parameters
        ----------
        variogram_model : str, optional
            Specifies which variogram model to use; may be one of the following:
            linear, power, gaussian, spherical, exponential, hole-effect.
            Default is linear variogram model. To utilize a custom variogram model,
            specify 'custom'; you must also provide variogram_parameters and
            variogram_function. Note that the hole-effect model is only technically
            correct for one-dimensional problems.
        variogram_parameters : list or dict, optional
            Parameters that define the specified variogram model. If not provided,
            parameters will be automatically calculated using a "soft" L1 norm
            minimization scheme. For variogram model parameters provided in a dict,
            the required dict keys vary according to the specified variogram
            model: ::
                linear - {'slope': slope, 'nugget': nugget}
                power - {'scale': scale, 'exponent': exponent, 'nugget': nugget}
                gaussian - {'sill': s, 'range': r, 'nugget': n}
                            OR
                           {'psill': p, 'range': r, 'nugget':n}
                spherical - {'sill': s, 'range': r, 'nugget': n}
                             OR
                            {'psill': p, 'range': r, 'nugget':n}
                exponential - {'sill': s, 'range': r, 'nugget': n}
                               OR
                              {'psill': p, 'range': r, 'nugget':n}
                hole-effect - {'sill': s, 'range': r, 'nugget': n}
                               OR
                              {'psill': p, 'range': r, 'nugget':n}
            Note that either the full sill or the partial sill
            (psill = sill - nugget) can be specified in the dict.
            For variogram model parameters provided in a list, the entries
            must be as follows: ::
                linear - [slope, nugget]
                power - [scale, exponent, nugget]
                gaussian - [sill, range, nugget]
                spherical - [sill, range, nugget]
                exponential - [sill, range, nugget]
                hole-effect - [sill, range, nugget]
            Note that the full sill (NOT the partial sill) must be specified
            in the list format.
            For a custom variogram model, the parameters are required, as custom
            variogram models will not automatically be fit to the data.
            Furthermore, the parameters must be specified in list format, in the
            order in which they are used in the callable function (see
            variogram_function for more information). The code does not check
            that the provided list contains the appropriate number of parameters
            for the custom variogram model, so an incorrect parameter list in
            such a case will probably trigger an esoteric exception someplace
            deep in the code.
            NOTE that, while the list format expects the full sill, the code
            itself works internally with the partial sill.
        variogram_function : callable, optional
            A callable function that must be provided if variogram_model is
            specified as 'custom'. The function must take only two arguments:
            first, a list of parameters for the variogram model; second, the
            distances at which to calculate the variogram model. The list
            provided in variogram_parameters will be passed to the function
            as the first argument.
        nlags : int, optional
            Number of averaging bins for the semivariogram. Default is 6.
        weight : bool, optional
            Flag that specifies if semivariance at smaller lags should be weighted
            more heavily when automatically calculating variogram model.
            The routine is currently hard-coded such that the weights are
            calculated from a logistic function, so weights at small lags are ~1
            and weights at the longest lags are ~0; the center of the logistic
            weighting is hard-coded to be at 70% of the distance from the shortest
            lag to the largest lag. Setting this parameter to True indicates that
            weights will be applied. Default is False. (Kitanidis suggests that the
            values at smaller lags are more important in fitting a variogram model,
            so the option is provided to enable such weighting.)
        anisotropy_scaling : float, optional
            Scalar stretching value to take into account anisotropy.
            Default is 1 (effectively no stretching).
            Scaling is applied in the y-direction in the rotated data frame
            (i.e., after adjusting for the anisotropy_angle, if anisotropy_angle
            is not 0). This parameter has no effect if coordinate_types is
            set to 'geographic' or 'network'.
        anisotropy_angle : float, optional
            CCW angle (in degrees) by which to rotate coordinate system in
            order to take into account anisotropy. Default is 0 (no rotation).
            Note that the coordinate system is rotated. This parameter has
            no effect if coordinate_types is set to 'geographic' or 'network'
        """

        if anisotropy_scaling != self.anisotropy_scaling or \
           anisotropy_angle != self.anisotropy_angle:
            self._adjust_coordinates(anisotropy_scaling,anisotropy_angle)

        self.variogram_model = variogram_model
        if self.variogram_model not in self.variogram_dict.keys() and \
           self.variogram_model != 'custom':
            log.error("Specified variogram model '{}' is not supported.".format(
                variogram_model))
            raise ValueError
        elif self.variogram_model == 'custom':
            if variogram_function is None or not callable(variogram_function):
                log.error("Must specify callable function for custom variogram model.")
                raise ValueError
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]

        log.info("Updating variogram mode...")

        vp_temp = _make_variogram_parameter_list(self.variogram_model,
                                                 variogram_parameters)
        self.lags, self.semivariance, self.variogram_model_parameters = \
            _initialize_variogram_model(np.vstack((self.X_ADJUSTED,
                                                   self.Y_ADJUSTED)).T,
                                        self.Z, self.variogram_model, vp_temp,
                                        self.variogram_function, nlags,
                                        weight, self.coordinates_type,
                                        self.network)

        log.info("Coordinates type: '{}'".format(self.coordinates_type))
        if self.variogram_model == 'linear':
            log.info("Using '{}' Variogram Model".format(self.variogram_model))
            log.info("Slope: {}".format(self.variogram_model_parameters[0]))
            log.info("Nugget: {}".format(self.variogram_model_parameters[1]))
        elif self.variogram_model == 'power':
            log.info("Using '{}' Variogram Model".format(self.variogram_model))
            log.info("Scale: {}".format(self.variogram_model_parameters[0]))
            log.info("Exponent: {}".format(self.variogram_model_parameters[1]))
            log.info("Nugget: {}".format(self.variogram_model_parameters[2]))
        elif self.variogram_model == 'custom':
            log.info("Using Custom Variogram Model")
        else:
            log.info("Using '{}' Variogram Model".format(self.variogram_model))
            log.info("Partial Sill:{}".format(self.variogram_model_parameters[0]))
            log.info("Full Sill: {}".format(self.variogram_model_parameters[0]
                                            + self.variogram_model_parameters[2]))
            log.info("Range: {}".format(self.variogram_model_parameters[1]))
            log.info("Nugget: {}".format(self.variogram_model_parameters[2]))
        if self.enable_plotting:
            self.display_variogram_model()

        log.info("Calculating statistics on variogram model fit...")

        if self.enable_statistics:
            self.delta, self.sigma, self.epsilon = \
                _find_statistics(np.vstack((self.X_ADJUSTED,
                                            self.Y_ADJUSTED)).T,
                                 self.Z, self.variogram_function,
                                 self.variogram_model_parameters,
                                 self.coordinates_type,self.network)
            self.Q1 = core.calcQ1(self.epsilon)
            self.Q2 = core.calcQ2(self.epsilon)
            self.cR = core.calc_cR(self.Q2, self.sigma)
            log.info("Q1 = {}".format(self.Q1))
            log.info("Q2 = {}".format(self.Q2))
            log.info("cR = {}".format(self.cR))
        else:
            self.delta, self.sigma, self.epsilon, self.Q1, self.Q2, self.cR = [None]*6
        pass
    
    def display_variogram_model(self):
        """Displays variogram model with the actual binned data."""
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.lags, self.semivariance, 'r*')
        ax.plot(self.lags,
                self.variogram_function(self.variogram_model_parameters,
                                        self.lags), 'k-')
        if config.plotting.show:
            plt.show()
        plt.savefig(config.plotting.path + 'variogram.png')

    def get_epsilon_residuals(self):
        """Returns the epsilon residuals for the variogram fit."""
        return self.epsilon

    def plot_epsilon_residuals(self):
        """Plots the epsilon residuals for the variogram fit."""
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(self.epsilon.size), self.epsilon, c='k', marker='*')
        ax.axhline(y=0.0)
        if config.plotting.show:
            plt.show()
        plt.savefig(config.plotting.path + 'epsilon_residuals.png')


    def get_statistics(self):
        """Returns the Q1, Q2, and cR statistics for the variogram fit
        (in that order). No arguments.
        """
        return self.Q1, self.Q2, self.cR

    def print_statistics(self):
        """Prints out the Q1, Q2, and cR statistics for the variogram fit.
        NOTE that ideally Q1 is close to zero, Q2 is close to 1,
        and cR is as small as possible.
        """
        print("Q1 =", self.Q1)
        print("Q2 =", self.Q2)
        print("cR =", self.cR)

    def _get_kriging_matrix(self, n):
        """Assembles the kriging matrix."""

        xy = np.concatenate((self.X_ADJUSTED[:, np.newaxis],
                             self.Y_ADJUSTED[:, np.newaxis]), axis=1)
        d = cdist(xy, xy, 'euclidean')
        a = np.zeros((n+1, n+1))
        a[:n, :n] = - self.variogram_function(self.variogram_model_parameters,
                                              d)
        np.fill_diagonal(a, 0.)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0

        return a

    def _exec_vector(self, a, bd, mask):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""

        npt = bd.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zero_index = None
        zero_value = False

        a_inv = scipy.linalg.inv(a)

        if np.any(np.absolute(bd) <= self.eps):
            zero_value = True
            zero_index = np.where(np.absolute(bd) <= self.eps)

        b = np.zeros((npt, n+1, 1))
        b[:, :n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
        if zero_value:
            b[zero_index[0], zero_index[1], 0] = 0.0
        b[:, n, 0] = 1.0

        if (~mask).any():
            mask_b = np.repeat(mask[:, np.newaxis, np.newaxis], n+1, axis=1)
            b = np.ma.array(b, mask=mask_b)

        x = np.dot(a_inv, b.reshape((npt, n+1)).T).reshape((1, n+1, npt)).T
        zvalues = np.sum(x[:, :n, 0] * self.Z, axis=1)
        sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        return zvalues, sigmasq

    def _exec_loop(self, a, bd_all, mask):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""

        npt = bd_all.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        a_inv = scipy.linalg.inv(a)

        for j in np.nonzero(~mask)[0]:   # Note that this is the same thing as range(npt) if mask is not defined,
            bd = bd_all[j]               # otherwise it takes the non-masked elements.
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_index = None
                zero_value = False

            b = np.zeros((n+1, 1))
            b[:n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0
            x = np.dot(a_inv, b)
            zvalues[j] = np.sum(x[:n, 0] * self.Z)
            sigmasq[j] = np.sum(x[:, 0] * -b[:, 0])

        return zvalues, sigmasq

    def _exec_loop_moving_window(self, a_all, bd_all, mask, bd_idx):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""
        import scipy.linalg.lapack

        npt = bd_all.shape[0]
        n = bd_idx.shape[1]
        zvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        for i in np.nonzero(~mask)[0]:   # Note that this is the same thing as range(npt) if mask is not defined,
            b_selector = bd_idx[i]       # otherwise it takes the non-masked elements.
            bd = bd_all[i]

            a_selector = np.concatenate((b_selector, np.array([a_all.shape[0] - 1])))
            a = a_all[a_selector[:, None], a_selector]

            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_index = None
                zero_value = False
            b = np.zeros((n+1, 1))
            b[:n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0

            x = scipy.linalg.solve(a, b)

            zvalues[i] = x[:n, 0].dot(self.Z[b_selector])
            sigmasq[i] = - x[:, 0].dot(b[:, 0])

        return zvalues, sigmasq

    def execute(self, style, xpoints, ypoints, mask=None, backend='vectorized',
                n_closest_points=None):
        """Calculates a kriged grid and the associated variance.

        This is now the method that performs the main kriging calculation.
        Note that currently measurements (i.e., z values) are considered
        'exact'. This means that, when a specified coordinate for interpolation
        is exactly the same as one of the data points, the variogram evaluated
        at the point is forced to be zero. Also, the diagonal of the kriging
        matrix is also always forced to be zero. In forcing the variogram
        evaluated at data points to be zero, we are effectively saying that
        there is no variance at that point (no uncertainty,
        so the value is 'exact').

        In the future, the code may include an extra 'exact_values' boolean
        flag that can be adjusted to specify whether to treat the measurements
        as 'exact'. Setting the flag to false would indicate that the
        variogram should not be forced to be zero at zero distance
        (i.e., when evaluated at data points). Instead, the uncertainty in
        the point will be equal to the nugget. This would mean that the
        diagonal of the kriging matrix would be set to
        the nugget instead of to zero.

        Parameters
        ----------
        style : str
            Specifies how to treat input kriging points. Specifying 'grid'
            treats xpoints and ypoints as two arrays of x and y coordinates
            that define a rectangular grid. Specifying 'points' treats
            xpoints and ypoints as two arrays that provide coordinate pairs
            at which to solve the kriging system. Specifying 'masked'
            treats xpoints and ypoints as two arrays of x and y coordinates
            that define a rectangular grid and uses mask to only evaluate
            specific points in the grid.
        xpoints : array_like, shape (N,) or (N, 1)
            If style is specific as 'grid' or 'masked',
            x-coordinates of MxN grid. If style is specified as 'points',
            x-coordinates of specific points at which to solve
            kriging system.
        ypoints : array_like, shape (M,) or (M, 1)
            If style is specified as 'grid' or 'masked',
            y-coordinates of MxN grid. If style is specified as 'points',
            y-coordinates of specific points at which to solve kriging
            system. Note that in this case, xpoints and ypoints must have
            the same dimensions (i.e., M = N).
        mask : bool, array_like, shape (M, N), optional
            Specifies the points in the rectangular grid defined
            by xpoints and ypoints that are to be excluded in the
            kriging calculations. Must be provided if style is specified
            as 'masked'. False indicates that the point should not be
            masked, so the kriging system will be solved at the point.
            True indicates that the point should be masked, so the kriging
            system should will not be solved at the point.
        backend : str, optional
            Specifies which approach to use in kriging.
            Specifying 'vectorized' will solve the entire kriging problem
            at once in a vectorized operation. This approach is faster but
            also can consume a significant amount of memory for large grids
            and/or large datasets. Specifying 'loop' will loop through each
            point at which the kriging system is to be solved.
            This approach is slower but also less memory-intensive.
            Specifying 'C' will utilize a loop in Cython.
            Default is 'vectorized'.
        n_closest_points : int, optional
            For kriging with a moving window, specifies the number of
            nearby points to use in the calculation. This can speed up the
            calculation for large datasets, but should be used
            with caution. As Kitanidis notes, kriging with a moving window
            can produce unexpected oddities if the variogram model
            is not carefully chosen.

        Returns
        -------
        zvalues : ndarray, shape (M, N) or (N, 1)
            Z-values of specified grid or at the specified set of points.
            If style was specified as 'masked', zvalues will
            be a numpy masked array.
        sigmasq : ndarray, shape (M, N) or (N, 1)
            Variance at specified grid points or at the specified
            set of points. If style was specified as 'masked', sigmasq
            will be a numpy masked array.
        """

        log.info("Executing Ordinary Kriging...")

        if style != 'grid' and style != 'masked' and style != 'points':
            log.error("style argument must be 'grid', 'points', or 'masked'")
            raise ValueError

        xpts = np.atleast_1d(np.squeeze(np.array(xpoints, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(ypoints, copy=True)))
        n = self.X_ADJUSTED.shape[0]
        nx = xpts.size
        ny = ypts.size
        a = self._get_kriging_matrix(n)

        if style in ['grid', 'masked']:
            if style == 'masked':
                if mask is None:
                    log.error("Must specify boolean masking array when style "+
                              "is 'masked'.")
                    raise IOError
                if mask.shape[0] != ny or mask.shape[1] != nx:
                    if mask.shape[0] == nx and mask.shape[1] == ny:
                        mask = mask.T
                    else:
                        log.error("Mask dimensions do not match specified "+
                                  "grid dimensions.")
                        raise ValueError
                mask = mask.flatten()
            npt = ny*nx
            grid_x, grid_y = np.meshgrid(xpts, ypts)
            xpts = grid_x.flatten()
            ypts = grid_y.flatten()

        elif style == 'points':
            if xpts.size != ypts.size:
                log.error("xpoints and ypoints must have same dimensions "+
                          "when treated as listing discrete points.")
                raise ValueError
            npt = nx
        else:
            log.error("style argument must be 'grid', 'points', or 'masked'")
            raise ValueError

        if self.coordinates_type == 'euclidean':
            xpts, ypts = _adjust_for_anisotropy(np.vstack((xpts, ypts)).T,
                                                [self.XCENTER, self.YCENTER],
                                                [self.anisotropy_scaling],
                                                [self.anisotropy_angle]).T
            xy_data = np.concatenate((self.X_ADJUSTED[:, np.newaxis],
                                      self.Y_ADJUSTED[:, np.newaxis]), axis=1)
            xy_points = np.concatenate((xpts[:, np.newaxis],
                                        ypts[:, np.newaxis]), axis=1)
        elif self.coordinates_type == 'geographic':
            # Quick version: Only difference between euclidean and spherical
            # space regarding kriging is the distance metric.
            # Since the relationship between three dimensional euclidean
            # distance and great circle distance on the sphere is monotonous,
            # use the existing (euclidean) infrastructure for nearest neighbour
            # search and distance calculation in euclidean three space and
            # convert distances to great circle distances afterwards.
            lon_d = self.X_ADJUSTED[:, np.newaxis] * np.pi / 180.0
            lat_d = self.Y_ADJUSTED[:, np.newaxis] * np.pi / 180.0
            xy_data = np.concatenate((np.cos(lon_d) * np.cos(lat_d),
                                      np.sin(lon_d) * np.cos(lat_d),
                                      np.sin(lat_d)), axis=1)
            lon_p = xpts[:, np.newaxis] * np.pi / 180.0
            lat_p = ypts[:, np.newaxis] * np.pi / 180.0
            xy_points = np.concatenate((np.cos(lon_p) * np.cos(lat_p),
                                        np.sin(lon_p) * np.cos(lat_p),
                                        np.sin(lat_p)), axis=1)
        elif self.coordinates_type == 'network':
            xy_data = np.concatenate((self.X_ADJUSTED[:, np.newaxis],
                                      self.Y_ADJUSTED[:, np.newaxis]), axis=1)
            xy_points = np.concatenate((xpts[:, np.newaxis],
                                        ypts[:, np.newaxis]), axis=1)

        if style != 'masked':
            mask = np.zeros(npt, dtype='bool')

        c_pars = None
        if backend == 'C':
            try:
                from .lib.cok import _c_exec_loop, _c_exec_loop_moving_window
            except ImportError:
                log.warn('Warning: failed to load Cython extensions.\n'
                      '   See https://github.com/bsmurphy/PyKrige/issues/8 \n'
                      '   Falling back to a pure python backend...')
                backend = 'loop'
            except:
                log.error("Unknown error in trying to load Cython extension.")
                raise RuntimeError

            c_pars = {key: getattr(self, key) for key in ['Z', 'eps', 'variogram_model_parameters', 'variogram_function']}

        if n_closest_points is not None:
            # TODO: add this option also for networks
            if self.coordinates_type == 'network':
                log.error("'n closest points' are not yet supported for"+
                          "networks!")
                raise ValueError

            from scipy.spatial import cKDTree
            tree = cKDTree(xy_data)
            bd, bd_idx = tree.query(xy_points, k=n_closest_points, eps=0.0)
            if self.coordinates_type == 'geographic':
                # Convert euclidean distances to great circle distances:
                bd = core.euclid3_to_great_circle(bd)

            if backend == 'loop':
                zvalues, sigmasq = \
                    self._exec_loop_moving_window(a, bd, mask, bd_idx)
            elif backend == 'C':
                zvalues, sigmasq = \
                    _c_exec_loop_moving_window(a, bd, mask.astype('int8'),
                                               bd_idx, self.X_ADJUSTED.shape[0],
                                               c_pars)
            else:
                log.error('Specified backend {} for a moving window '+
                          'is not supported.'.format(backend))
                raise ValueError
        else:
            if self.coordinates_type == 'network':
                bd = self.network.ndist(xy_points,  a=xy_data, mode='cdist')
            else:
                bd = cdist(xy_points,  xy_data, 'euclidean')
            if self.coordinates_type == 'geographic':
                # Convert euclidean distances to great circle distances:
                bd = core.euclid3_to_great_circle(bd)

            if backend == 'vectorized':
                zvalues, sigmasq = self._exec_vector(a, bd, mask)
            elif backend == 'loop':
                zvalues, sigmasq = self._exec_loop(a, bd, mask)
            elif backend == 'C':
                zvalues, sigmasq = _c_exec_loop(a, bd, mask.astype('int8'),
                                                self.X_ADJUSTED.shape[0],
                                                c_pars)
            else:
                log.error('Specified backend {} is not supported for '+
                          '2D ordinary kriging.'.format(backend))
                raise ValueError

        if style == 'masked':
            zvalues = np.ma.array(zvalues, mask=mask)
            sigmasq = np.ma.array(sigmasq, mask=mask)

        if style in ['masked', 'grid']:
            zvalues = zvalues.reshape((ny, nx))
            sigmasq = sigmasq.reshape((ny, nx))

        return zvalues, sigmasq






# =============================================================================
# eof
#
# Local Variables: 
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 80
# End:  
