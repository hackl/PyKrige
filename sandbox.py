#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : sandbox.py 
# Creation  : 22 Mar 2018
# Time-stamp: <Die 2018-03-27 09:34 juergen>
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
import pykrige.network_tools as nt
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
plt.style.use('ggplot')
# initialize logger
# =================
log = logging.getLogger(__name__)


def main():
    #pykrige.config.log.verbose = False
    log.info('============= START =============')
    # test_onk()
    # test_nt()
    test_1d()
    #test_c()
    log.info('============== END ==============')
    pass

def test_c():
    from pykrige.lib.cok import _c_exec_loop, _c_exec_loop_moving_window
    # import pykrige.lib as clib
    # clib.cok._c_exec_loop

def test_1d():
    # X, y = np.array([[-5.01, 1.06], [-4.90, 0.92], [-4.82, 0.35], [-4.69, 0.49], [-4.56, 0.52], 
    #                  [-4.52, 0.12], [-4.39, 0.47], [-4.32,-0.19], [-4.19, 0.08], [-4.11,-0.19],
    #                  [-4.00,-0.03], [-3.89,-0.03], [-3.78,-0.05], [-3.67, 0.10], [-3.59, 0.44],
    #                  [-3.50, 0.66], [-3.39,-0.12], [-3.28, 0.45], [-3.20, 0.14], [-3.07,-0.28],
    #                  [-3.01,-0.46], [-2.90,-0.32], [-2.77,-1.58], [-2.69,-1.44], [-2.60,-1.51],
    #                  [-2.49,-1.50], [-2.41,-2.04], [-2.28,-1.57], [-2.19,-1.25], [-2.10,-1.50],
    #                  [-2.00,-1.42], [-1.91,-1.10], [-1.80,-0.58], [-1.67,-1.08], [-1.61,-0.79],
    #                  [-1.50,-1.00], [-1.37,-0.04], [-1.30,-0.54], [-1.19,-0.15], [-1.06,-0.18],
    #                  [-0.98,-0.25], [-0.87,-1.20], [-0.78,-0.49], [-0.68,-0.83], [-0.57,-0.15],
    #                  [-0.50, 0.00], [-0.38,-1.10], [-0.29,-0.32], [-0.18,-0.60], [-0.09,-0.49],
    #                  [0.03 ,-0.50], [0.09 ,-0.02], [0.20 ,-0.47], [0.31 ,-0.11], [0.41 ,-0.28],
    #                  [0.53 , 0.40], [0.61 , 0.11], [0.70 , 0.32], [0.94 , 0.42], [1.02 , 0.57],
    #                  [1.13 , 0.82], [1.24 , 1.18], [1.30 , 0.86], [1.43 , 1.11], [1.50 , 0.74],
    #                  [1.63 , 0.75], [1.74 , 1.15], [1.80 , 0.76], [1.93 , 0.68], [2.03 , 0.03],
    #                  [2.12 , 0.31], [2.23 ,-0.14], [2.31 ,-0.88], [2.40 ,-1.25], [2.50 ,-1.62],
    #                  [2.63 ,-1.37], [2.72 ,-0.99], [2.80 ,-1.92], [2.83 ,-1.94], [2.91 ,-1.32],
    #                  [3.00 ,-1.69], [3.13 ,-1.84], [3.21 ,-2.05], [3.30 ,-1.69], [3.41 ,-0.53],
    #                  [3.52 ,-0.55], [3.63 ,-0.92], [3.72 ,-0.76], [3.80 ,-0.41], [3.91 , 0.12],
    #                  [4.04 , 0.25], [4.13 , 0.16], [4.24 , 0.26], [4.32 , 0.62], [4.44 , 1.69],
    #                  [4.52 , 1.11], [4.65 , 0.36], [4.74 , 0.79], [4.84 , 0.87], [4.93 , 1.01],
    #                  [5.02 , 0.55]]).T
    #X_pred = np.linspace(0, 100, 200)

    G = nx.Graph()
    X_pred = []
    for i in range(101):
        X_pred.append(i)
        G.add_node(str(i))
        G.node[str(i)]['coordinate'] = (i,0)
        G.node[str(i)]['z'] = 0.0
        if i == 0:
            continue
        G.add_edge(str(i-1),str(i))

    X_pred = np.array(X_pred)
    y = [0.7842928345871103, 0.69504241423575497, 0.78666633463117397, 0.13294867055326676, -0.36546487121376786, -0.25100252113997262, -0.77595583702806747, -0.63070694727239696, -1.0223707272332583]

    V = []
    for i,j in enumerate(range(10,100,10)):
        # a = 30
        # y.append(np.cos(j/a)+np.random.normal(0,.2))
        G.node[str(j)]['z'] = y[i]
        V.append(str(j))
    

    network = nt.Network(G,x='coordinate',y='coordinate')
    data = network.get_xyz(nodes=V)
    #print(data)

    X = data[:, 0]
    Y = data[:, 1]
    y = data[:, 2]
    
    uk = OrdinaryKriging(X, Y, y, variogram_model='linear',)
    uk.plot_epsilon_residuals()
#    uk.update_variogram_model('gaussian')
    y_pred, y_std = uk.execute('points', X_pred, np.zeros(101),backend='C')


    y_pred = np.squeeze(y_pred)
    y_std = np.squeeze(y_std)
    #print(y_std)
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.scatter(X, y, s=40, label='Input data')
    ax.plot(X_pred, y_pred, label='Predicted values')
    ax.fill_between(X_pred, y_pred - 3*y_std, y_pred + 3*y_std, alpha=.5, label='Confidence interval')
    ax.legend(loc=9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.2, 1.2)
    plt.savefig('test_1d_euclidean.png')





    uk = OrdinaryKriging(X, Y, y, variogram_model='linear',coordinates_type = 'network',network = network)

    y_pred, y_std = uk.execute('points', X_pred, np.zeros(101),backend='C')

    y_pred = np.squeeze(y_pred)
    y_std = np.squeeze(y_std)
    #print(y_std)
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.scatter(X, y, s=40, label='Input data')
    ax.plot(X_pred, y_pred, label='Predicted values')
    ax.fill_between(X_pred, y_pred - 3*y_std, y_pred + 3*y_std, alpha=.5, label='Confidence interval')
    ax.legend(loc=9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.2, 1.2)
    plt.savefig('test_1d_network.png')

def test_nt():
    graph = nx.read_gpickle('./data/network.gpickle')
    for n,a in graph.nodes(data=True):
        # graph.node[n]['x'] = a['coordinate'][0]
        # graph.node[n]['y'] = a['coordinate'][1]
        graph.node[n]['z'] = float(n)
        #V.append(n)

    # print(nx.info(graph))
    V = ['1','2','3','4','5']
    network = nt.Network(graph,x='coordinate',y='coordinate',weight='length')
    data = network.get_xyz(nodes=V)

    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                         variogram_model = 'linear',
                         coordinates_type = 'euclidean',#'network',
                         network = network
    )

    
    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                         variogram_model = 'linear',
                         coordinates_type = 'network',#'euclidean',#'network',
                         network = network
    )


    
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
