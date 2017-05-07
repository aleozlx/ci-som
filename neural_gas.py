import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def v_network_ctor(N_UNITS, DIM_DATA, copy_ref = None):
    """ Network constructor """
    if copy_ref:
        # print('v_network_ctor copying', N_UNITS)
        _W, _C, _T, _E = copy_ref

        # Weights
        W = np.zeros((N_UNITS, DIM_DATA))
        W[:_W.shape[0]] = _W

        # Connections
        C = np.zeros((N_UNITS, N_UNITS), dtype = bool)
        C[:_C.shape[0], :_C.shape[1]] = _C

        # Connection ages
        T = np.zeros(C.shape, dtype = int)
        T[:_T.shape[0], :_T.shape[1]] = _T

        # Errors
        E = np.zeros((N_UNITS, ))
        E[:_E.shape[0]] = _E
    else:
        W = np.random.rand(N_UNITS, DIM_DATA) * 4.0 - 2.0
        C = np.ones((N_UNITS, N_UNITS), dtype = bool)
        np.fill_diagonal(C, False)
        T = np.zeros(C.shape, dtype = int)
        E = np.zeros((N_UNITS, ))
    return (W, C, T, E)

def v_network_nonzero_degree(copy_ref):
    return copy_ref
    _W, _C, _T, _E = copy_ref
    nz = np.array(np.sum(_C, axis = 1), dtype = bool)
    # print('v_network_nonzero_degree', nz)
    if np.all(nz):
        return copy_ref
    else:
        print('shrinking down')
        new_network = (_W[nz, :].copy(), _C[nz, :][:, nz].copy(), _T[nz, :][:, nz].copy(), _E[nz].copy())
        print('W', _W.shape, new_network[0].shape)
        print('C', _C.shape, new_network[1].shape)
        print('T', _T.shape, new_network[2].shape)
        print('E', _E.shape, new_network[3].shape)
        return new_network

class vSOMBase(object):
    """ Base class for SOM-like algorithms """

    def __init__(self, input_data, N_units):
        self.data = input_data

        # start with two units a and b at random positions
        DIM_DATA = self.data.shape[1]
        self._network = v_network_ctor(N_units, DIM_DATA)
        (W, C, T, E) = self._network
        self.N_units = W.shape[0]
        # clusters of neurons and data
        self._clusters = None
        self.clusters = None

    def plot_network(self, file_path, weights, connections):
        plt.clf()
        plt.scatter(self.data[:, 0], self.data[:, 1], c='r')
        plt.scatter(weights[:, 0], weights[:, 1], c='b', marker='x')
        edge_ctor = lambda edge: edge if edge[0]<edge[1] else (edge[1], edge[0])
        edges = set(edge_ctor((i,j))
            for i in range(connections.shape[0]) for j in range(connections.shape[1]) if connections[i,j] and i!=j)
        points = weights[np.array(list(edges))]
        for edge in points:
            plt.plot(edge[:, 0], edge[:, 1], 'k-', alpha = 0.9)
        plt.savefig(file_path)

    def update_clusters(self, outlier_tolerance = 1):
        (W, C, T, E) = self._network
        v = np.zeros((self.N_units, ), dtype = bool)
        self._clusters = list()
        from queue import Queue
        bfs_queue = Queue()
        while np.any(~v):
            root = np.where(~v)[0][0]
            bfs_queue.put(root); v[root] = True
            current_cluster = set()
            self._clusters.append(current_cluster)
            while not bfs_queue.empty():
                p = bfs_queue.get(); current_cluster.add(p)
                neighbors = np.where(C[p])[0]
                new_nodes = np.where(~v[neighbors])[0]
                current_cluster.update(neighbors[new_nodes])
                for i in neighbors[new_nodes]:
                    bfs_queue.put(i)
                v[neighbors[new_nodes]] = True

        self._clusters = list(filter(lambda cluster: len(cluster)>outlier_tolerance, self._clusters))
        self.clusters = list(set() for i in self._clusters)
        cluster_index = { unit: self.clusters[ci]
            for ci, cluster in enumerate(self._clusters) for unit in cluster}
        for i, v in enumerate(self.data):
            ranked_indices = np.argsort(np.linalg.norm(W-v[np.newaxis, ...], ord = 2, axis = 1))
            for j in ranked_indices:
                if j in cluster_index:
                    cluster_index[j].add(i)
                    break

    def plot_clusters(self):
        import matplotlib.colors
        import matplotlib.cm
        plt.clf()
        plt.title('Cluster affectation')
        scalarMap = matplotlib.cm.ScalarMappable(
            norm = matplotlib.colors.Normalize(vmin=0, vmax=len(self.clusters)-1),
            cmap= 'viridis')
        for i, cluster in enumerate(self.clusters):
            points = self.data[list(cluster), :]
            plt.scatter(points[:, 0], points[:, 1],
                color = scalarMap.to_rgba(i), label = 'cluster #'+str(i))
        plt.legend()
        plt.savefig('visualization/clusters.png')

class vGNG(vSOMBase):
    """ Growing Neural Gas (vectorized online update) 
     ref:
        https://github.com/AdrienGuille/GrowingNeuralGas
        http://stackoverflow.com/questions/34072772/what-is-the-difference-between-self-organizing-maps-and-neural-gas
        https://www.quora.com/When-I-should-I-use-competitive-learning-or-neural-gas-algorithms-instead-of-k-means-to-cluster-data
    """

    def __init__(self, input_data):
        super(vGNG, self).__init__(input_data, 2)

    def fit(self, E_NEAREST, E_NEIBOR, MAX_AGE, STEP_NEW_UNIT, ERR_DECAY_LOCAL, ERR_DECAY_GLOBAL, N_PASS, plot_evolution=False):
        global_error = []; total_units = []
        DIM_DATA = self.data.shape[1]
        (W, C, T, E) = self._network

        # 1. iterate through the data
        sequence = 0
        for p in range(N_PASS):
            print('   Pass #%d' % (p + 1))
            np.random.shuffle(self.data)
            steps = 0
            for observation in self.data:
                # 2. find the nearest unit and the second
                ranked_indices = np.argsort(np.linalg.norm(W-observation[np.newaxis, ...], ord = 2, axis = 1))
                i0 = ranked_indices[0]
                i1 = ranked_indices[1]
                # 3. increment the age of all edges emanating from i0
                T[i0, C[i0]] += 1
                T[:,i0] = T[i0,:]
                # 4. add the squared distance between the observation and i0 in feature space
                difference = observation - W[i1]
                E[i0] += np.linalg.norm(difference, ord = 2) ** 2
                # 5. move i0 and its direct topological neighbors towards the observation
                W[i1] += E_NEAREST * difference
                W[C[i1]] += (E_NEIBOR * difference)[np.newaxis, ...]
                # 6. if i0 and i1 are connected by an edge, set the age of this edge to zero
                #    if such an edge doesn't exist, create it
                C[i0, i1] = C[i1, i0] = True
                T[i0, i1] = T[i1, i0] = 0
                # 7. remove edges with an age larger than MAX_AGE
                #    if this results in units having no emanating edges, remove them as well
                expired = T > MAX_AGE
                C[expired] = False; T[expired] = 0
                (W, C, T, E) = v_network_nonzero_degree((W, C, T, E))
                self.N_units = W.shape[0]
                # 8. if the number of steps so far is an integer multiple of parameter STEP_NEW_UNIT, insert a new unit
                steps += 1
                if steps % STEP_NEW_UNIT == 0:
                    if plot_evolution:
                        self.plot_network('visualization/sequence/%s.png' % str(sequence), W, C)
                    sequence += 1
                    # 8.a determine the unit q with the maximum accumulated error
                    q = np.argmax(E)
                    # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
                    neighbors = np.where(C[q])[0]
                    if neighbors.shape[0]: # ignore this process if there're no neighbors
                        f = neighbors[np.argmax(E[neighbors])]
                        (W, C, T, E) = v_network_ctor(self.N_units + 1, DIM_DATA, (W, C, T, E))
                        W[-1] = (W[q] + W[f]) * 0.5
                        self.N_units += 1
                        # 8.c insert edges connecting the new unit r with q and f
                        #     remove the original edge between q and f
                        C[q, -1] = C[-1, q] = True
                        C[f, -1] = C[-1, f] = True
                        C[q, f] = C[f, q] = False
                        # 8.d decrease the error variables of q and f by multiplying them with a
                        #     initialize the error variable of r with the new value of the error variable of q
                        E[q] *= ERR_DECAY_LOCAL
                        E[f] *= ERR_DECAY_LOCAL
                        E[-1] = E[q] # why?
                    else: # no neighbors would be a bug. gracefully handle anyway.
                        print('warning: no neighbors in step 8')
                        (W, C, T, E) = v_network_ctor(self.N_units + 1, DIM_DATA, (W, C, T, E))
                        W[-1] = W[q] + (np.random.rand(DIM_DATA) * 4.0 - 2.0)
                        self.N_units += 1
                        C[q, -1] = C[-1, q] = True
                        E[q] *= ERR_DECAY_LOCAL
                        E[-1] = E[q]

                # 9. decrease all error variables by multiplying them with a constant d
                total_units.append(self.N_units)
                E *= ERR_DECAY_GLOBAL
            global_error.append(sum(np.min(np.linalg.norm(W-v[np.newaxis, ...], ord = 2, axis = 1)) for v in self.data))
        plt.clf()
        plt.title('Global error')
        plt.xlabel('N_PASS')
        plt.plot(range(len(global_error)), global_error)
        plt.savefig('visualization/global_error.png')
        self._network = (W, C, T, E)

class vNG(vSOMBase):
    """ Neural Gas (vectorized online update) """

    def __init__(self, input_data, N_units = 32):
        super(vNG, self).__init__(input_data, N_units)

    def fit(self, e_epsilon, e_lambda, MAX_AGE, STEP_NEW_UNIT, a, d, N_PASS, plot_evolution=False):
        global_error = []; total_units = []
        DIM_DATA = self.data.shape[1]
        (W, C, T, E) = self._network

        sequence = 0
        for p in range(N_PASS):
            print('   Pass #%d' % (p + 1))
            np.random.shuffle(self.data)
            steps = 0
            for observation in self.data:
                # 2. find the nearest unit and the second
                ranked_indices = np.argsort(np.linalg.norm(W-observation[np.newaxis, ...], ord = 2, axis = 1))
                i0 = ranked_indices[0]
                i1 = ranked_indices[1]
                # 3. increment the age of all edges emanating from i0
                T[i0, C[i0]] += 1
                T[:,i0] = T[i0,:]
                # 4. add the squared distance between the observation and i0 in feature space
                difference = observation - W[i1]
                E[i0] += np.linalg.norm(difference, ord = 2) ** 2
                # 5. move i0 and its direct topological neighbors towards the observation
                W[i1] += e_epsilon * difference
                W[C[i1]] += (e_lambda * difference)[np.newaxis, ...]
                # 6. if i0 and i1 are connected by an edge, set the age of this edge to zero
                #    if such an edge doesn't exist, create it
                C[i0, i1] = C[i1, i0] = True
                T[i0, i1] = T[i1, i0] = 0
                # 7. remove edges with an age larger than MAX_AGE
                #    if this results in units having no emanating edges, remove them as well
                expired = T > MAX_AGE
                C[expired] = False; T[expired] = 0
                # (W, C, T, E) = v_network_nonzero_degree((W, C, T, E))
                # self.N_units = W.shape[0]
                # 8. Update plots
                steps += 1
                if steps % STEP_NEW_UNIT == 0:
                    if plot_evolution:
                        self.plot_network('visualization/sequence/%s.png' % str(sequence), W, C)
                    sequence += 1
                # 9. decrease all error variables by multiplying them with a constant d
                total_units.append(self.N_units)
                E *= d
            global_error.append(sum(np.min(np.linalg.norm(W-v[np.newaxis, ...], ord = 2, axis = 1)) for v in self.data))
        plt.clf()
        plt.title('Global error')
        plt.xlabel('N_PASS')
        plt.plot(range(len(global_error)), global_error)
        plt.savefig('visualization/global_error.png')
        self._network = (W, C, T, E)