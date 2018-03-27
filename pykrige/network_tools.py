#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : network_tools.py 
# Creation  : 26 Mar 2018
# Time-stamp: <Die 2018-03-27 08:50 juergen>
#
# Copyright (c) 2018 Jürgen Hackl <hackl@ibi.baug.ethz.ch>
#               http://www.ibi.ethz.ch
# $Id$ 
#
# Description : Network tools for network kriging
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
import networkx as nx


# initialize logger
# =================
log = logging.getLogger(__name__)

class Network:
    """ Generic network model for network kriging """

    def __init__(self,network=None,x='x',y='y',z='z',weight=None):
        log.info('Initialize network ...')
        self.network = network
        self.x_attribute_name = x
        self.y_attribute_name = y
        self.z_attribute_name = z
        self.weight = weight
        
        if self.network is not None:
            self._check_network()
        pass

    def _check_network(self):
        log.info('Check network ...')

        x_attributes = \
        nx.get_node_attributes(self.network,self.x_attribute_name)
        y_attributes = \
        nx.get_node_attributes(self.network,self.y_attribute_name)
        z_attributes = \
        nx.get_node_attributes(self.network,self.z_attribute_name)

        # check if x,y and z data is available in the network
        if len(x_attributes) == 0:
            log.error("No 'x' values are found in the network under the name"+
                      " '{}'".format(self.x_attribute_name))
            raise KeyError
        elif len(y_attributes) == 0:
            log.error("No 'y' values are found in the network under the name"+
                      " '{}'".format(self.y_attribute_name))
            raise KeyError
        elif len(z_attributes) == 0:
            log.error("No 'z' values are found in the network under the name"+
                      " '{}'".format(self.z_attribute_name))
            raise KeyError

        # assign x,y and z
        x = []
        y = []
        z = []
        self.node_dict_coord = {}
        self.node_dict_id = {}
        if self.x_attribute_name == self.y_attribute_name == \
           self.z_attribute_name:
            for n,a in self.network.nodes(data=True):
                x.append(a[self.x_attribute_name][0])
                y.append(a[self.y_attribute_name][1])
                z.append(a[self.z_attribute_name][2])
                self.node_dict_coord[(x[-1],y[-1])] = n
                self.node_dict_id[n] = len(x)-1
        elif self.x_attribute_name == self.y_attribute_name:
            for n,a in self.network.nodes(data=True):
                x.append(a[self.x_attribute_name][0])
                y.append(a[self.y_attribute_name][1])
                z.append(a[self.z_attribute_name])
                self.node_dict_coord[(x[-1],y[-1])] = n
                self.node_dict_id[n] = len(x)-1
        else:
            for n,a in self.network.nodes(data=True):
                x.append(a[self.x_attribute_name])
                y.append(a[self.y_attribute_name])
                z.append(a[self.z_attribute_name])
                self.node_dict_coord[(x[-1],y[-1])] = n
                self.node_dict_id[len(x)-1] = n
        # create numpy vectors
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.xyz = np.vstack((self.x,self.y,self.z)).T
        pass

    def load(self,filepath):
        """ Load external network """
        self.network = nx.read_gpickle(filepath)
        self._check_network()
        pass

    def get_xyz(self,nodes=None):
        """ Get the x y and z values of the network """
        log.debug('Return x, y and z values ...')
        if nodes is None:
            return self.xyz
        else:
            V = [self.node_dict_id[u] for u in nodes]
            return self.xyz[V]

    def get_nodes(self,X=None):
        """ Get the node ids the network """
        log.debug('Return x, y and z values ...')
        if X is None:
            return list(self.network.nodes())
        elif len(X.shape) > 1:
            return [self.node_dict_coord[(X[i][0],X[i][1])] for i in range(len(X))]
        else:
            return [self.node_dict_coord[(X[0],X[1])]]


    def ndist(self,X,a=None,mode='pdist'):
        dist = []
        V = self.get_nodes(X)
        if mode == 'pdist':
            # TODO: Slow NetworkX algorithm
            # TODO: Implement faster one
            for i in range(0,len(V)-1):
                for j in range(i+1,len(V)):
                    dist.append(nx.shortest_path_length(self.network, source=V[i],
                                                        target=V[j],
                                                        weight=self.weight))
        elif mode == 'cdist':
            U = self.get_nodes(a)
            if len(U) == 1:
                for v in V:
                    dist.append(nx.shortest_path_length(self.network, source=v,
                                                        target=U[0],
                                                        weight=self.weight))
            else:
                for v in V:
                    temp_dist = []
                    for u in U:
                        temp_dist.append(nx.shortest_path_length(self.network,
                                                            source=v,
                                                            target=u,
                                                            weight=self.weight))
                    dist.append(temp_dist)
        return np.array(dist).astype(np.float)



# =============================================================================
# eof
#
# Local Variables: 
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 80
# End:  
