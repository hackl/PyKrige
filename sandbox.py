#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : sandbox.py 
# Creation  : 22 Mar 2018
# Time-stamp: <Fre 2018-03-23 16:37 juergen>
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

from timeit import Timer
import logging
import pykrige
import networkx as nx
#from pykrige.ok import OrdinaryKriging
from pykrige.onk import OrdinaryKriging
import numpy as np
import pykrige.kriging_tools as kt
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist

# initialize logger
# =================
log = logging.getLogger(__name__)

def main():
    log.info('============= START =============')
    test_onk()
    log.info('============== END ==============')
    pass

def dfun(u, v):
    return np.sqrt(((u-v)**2).sum())

def ndist_x(X,network):
    for i in range(0,len(X)-1):
        for j in range(i+1,len(X)):
            # print(i,j)
            print(dfun(X[i],X[j]))

def ndist_old(V,network):
    dist = []
    for i in range(0,len(V)-1):
        for j in range(i+1,len(V)):
            # print(i,j)
            # print(V[i],V[j])
            dist.append(nx.shortest_path_length(network, source=V[i], target=V[j],weight='length'))
            #print(dfun(X[i],X[j]))
    return(np.array(dist))


def ndist(V,network,weight=None,algorithm='single'):
    dist = []
    if algorithm is 'all':
        paths = dict(nx.all_pairs_dijkstra_path_length(network,weight=weight))
    for i in range(0,len(V)-1):
        for j in range(i+1,len(V)):
            if algorithm is 'single':
                dist.append(nx.shortest_path_length(network, source=V[i],
    target=V[j],weight=weight))
            elif algorithm is 'all':
                dist.append(paths[V[i]][V[j]])
    return(np.array(dist))

def ndist2(V,u,network,weight=None):
    dist = []
    for v in V:
        dist.append(nx.shortest_path_length(network, source=v, target=u,weight=weight))
    return(np.array(dist))




def get_nodes(X,network):
    node_dict = {}
    for n,a in network.nodes(data=True):
        node_dict[(a['x'],a['y'])] = n

    V = [node_dict[(X[i][0],X[i][1])] for i in range(len(X))]
    return(V)

def test_onk():

    graph = nx.read_gpickle('./data/network.gpickle')
    print(nx.info(graph))

 
    V = []
    for n,a in graph.nodes(data=True):
        graph.node[n]['x'] = a['coordinate'][0]
        graph.node[n]['y'] = a['coordinate'][1]
        graph.node[n]['z'] = float(n)
        V.append(n)

    #V = ['1','2','3','4','5']


    data = np.empty((0,3))
    node_dict = {}
    for n,a in graph.nodes(data=True):
        if n in V:
            node = np.array([[a['x'],a['y'],a['z']]])
            data = np.append(data,node,axis=0)
            node_dict[(a['x'],a['y'])] = n

    # print(data)
    # X = np.vstack((data[:,0],data[:,1])).T
    # print(X)
    # V = get_nodes(X,graph)
    # print(V)
    # d = ndist(V,graph,weight='length')
    # print(d)
    # print(node_dict)
    # print(data)
    # print(data[:,0])

    # X = np.array([[679909.39040124,4831236.17646105]])
    # coords = np.array([679686.00965448,4826823.21079346])

    # print(cdist(X, coords[None, :], metric='euclidean'))

    # print(data)
    # X = np.array([[679909.39040124 , 4831236.17646105],
    #               [679686.00965448 , 4826823.21079346],
    #               [680827.02756151 , 4825897.1672747 ],
    #               [684732.05571324 , 4830618.74355952]])
    
    # coords = np.array([683228.05203023,4825927.86958611])

    # print(get_nodes(X,graph))
    # print(cdist(X, coords[None, :], metric='euclidean'))


    # print(cdist([X[1]], coords[None, :], metric='euclidean'))

    
    # d = ndist(V,graph,weight='length')
    # print(d)
    
    # t = Timer(lambda: ndist(V,graph))
    # print(t.timeit(number=1))

    # t = Timer(lambda: ndist(V,graph,algorithm='all'))
    # print(t.timeit(number=1))

    # t = Timer(lambda: nx.all_pairs_dijkstra_path_length(graph,weight='length'))
    # print(t.timeit(number=1))


    # t = Timer(lambda: nx.shortest_path_length(graph, source=V[0], target=V[1],weight='length'))
    # print(t.timeit(number=1))

    # #print(dict(nx.all_pairs_dijkstra_path_length(graph,weight='length')))

    # #print(d)
    # data = np.array([[0.3, 1.2, 0.47],
    #                  [1.9, 0.6, 0.56],
    #                  [1.1, 3.2, 0.74],
    #                  [3.3, 4.4, 1.47],
    #                  [4.7, 3.8, 1.74]])

    # X = np.vstack((data[:,0],data[:,1])).T
    # ndist_x(X,network=None)
    # print(dfun(X[3],X[4]))
    # d = pdist(X, metric='euclidean')
    # print(X)
    # print(d)

    # y = data[:,2]
    # print(y)
    # g = 0.5 * pdist(y[:, None], metric='sqeuclidean')
    # print(g)
    
    # gridx = np.arange(0.0, 5.5, 0.5)
    # gridy = np.arange(0.0, 5.5, 0.5)

    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                         variogram_model = 'gaussian',
                         # verbose = True,
                         # enable_plotting = True,
                         # anisotropy_scaling=1.0,
                         coordinates_type = 'network',#'euclidean',#'network',
                         network = graph
    )

    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                         variogram_model = 'linear',
                         # verbose = True,
                         # enable_plotting = True,
                         # anisotropy_scaling=1.0,
                         coordinates_type = 'euclidean',#'network',
                         network = graph
    )


    # z, ss = OK.execute('grid', gridx, gridy)


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
main()
# if __name__ == '__main__':
#     main()


# =============================================================================
# eof
#
# Local Variables: 
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 80
# End:  
