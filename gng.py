import numpy as np
# import networkx as nx
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def find_nearest_units(network, observation):
    distance = []
    for u, attributes in network.nodes(data=True):
        vector = attributes['vector']
        dist = spatial.distance.euclidean(vector, observation)
        distance.append((u, dist))
    distance.sort(key=lambda x: x[1])
    ranking = [u for u, dist in distance]
    return ranking

class GrowingNeuralGas:
    def plot_clusters(self, clustered_data):
        number_of_clusters = nx.number_connected_components(self.network)
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(number_of_clusters):
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.array(observations)
                plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
        plt.legend()
        plt.savefig('visualization/clusters.png')

def v_network_ctor(N_UNITS, DIM_DATA, copy_ref = None):
    """ Network constructor """
    if copy_ref:
        _W, _C, _T, _E = copy_ref
    # Weights
    if copy_ref:
        W = np.zeros((N_UNITS, DIM_DATA))
        W[:_W.shape[0]] = _W
        W[-1] = np.random.rand(DIM_DATA) * 4.0 - 2.0
    else:
        W = np.random.rand(N_UNITS, DIM_DATA) * 4.0 - 2.0
    # Connections
    C = np.zeros((N_UNITS, N_UNITS), dtype = bool)
    # Connection ages
    T = np.zeros(C.shape, dtype = int)
    # Errors
    E = np.zeros((N_UNITS, ))
    if copy_ref:
        C[:_C.shape[0], :_C.shape[1]] = _C
        T[:_T.shape[0], :_T.shape[1]] = _T
        E[:_E.shape[0]] = _E
    return (W, C, T, E)

class vGNG(object):
    """ Growing Neural Gas (vectorized online update) 
     ref:
        https://github.com/AdrienGuille/GrowingNeuralGas
        http://stackoverflow.com/questions/34072772/what-is-the-difference-between-self-organizing-maps-and-neural-gas
        https://www.quora.com/When-I-should-I-use-competitive-learning-or-neural-gas-algorithms-instead-of-k-means-to-cluster-data
    """
    def __init__(self, input_data):
        self.data = input_data

        # start with two units a and b at random positions
        DIM_DATA = self.data.shape[1]
        self._network = v_network_ctor(2, DIM_DATA)
        (W, C, T, E) = self._network
        self.N_units = W.shape[0]

        self.clusters = None

    def fit(self, e_b, e_n, MAX_AGE, STEP_NEW_UNIT, a, d, N_PASS=1, plot_evolution=False):
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
                # 2. find the nearest unit s_1 and the second nearest unit s_2
                ranked_indices = np.argsort(np.linalg.norm(W-observation[np.newaxis, ...], ord = 2, axis = 1))
                i0 = ranked_indices[0]
                i1 = ranked_indices[1]
                # 3. increment the age of all edges emanating from s_1
                T[i0, C[i0]] += 1
                T[:,i0] = T[i0,:]
                # 4. add the squared distance between the observation and the nearest unit in input space
                difference = observation - W[i1]
                distance = np.linalg.norm(difference, ord = 2)
                E[i0] += distance ** 2
                # 5 .move s_1 and its direct topological neighbors towards the observation by the fractions
                #    e_b and e_n, respectively, of the total distance
                W[i1] += e_b * difference
                W[C[i1]] += (e_n * difference)[np.newaxis, ...]
                # 6. if s_1 and s_2 are connected by an edge, set the age of this edge to zero
                #    if such an edge doesn't exist, create it
                C[i0, i1] = C[i1, i0] = True
                T[i0, i1] = T[i1, i0] = 0
                # 7. remove edges with an age larger than MAX_AGE
                #    if this results in units having no emanating edges, remove them as well
                expired = T > MAX_AGE
                C[expired] = False
                T[expired] = 0
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
                    if neighbors.shape[0]: # no neighbors?
                        f = neighbors[np.argmax(E[neighbors])]
                    else:
                        f = None

                    (W, C, T, E) = v_network_ctor(self.N_units + 1, DIM_DATA, (W, C, T, E))
                    if f == None:
                        W[-1] = W[q] + (np.random.rand(DIM_DATA) * 4.0 - 2.0)
                    else:
                        W[-1] = (W[q] + W[f]) * 0.5
                    self.N_units += 1

                    # 8.c insert edges connecting the new unit r with q and f
                    #     remove the original edge between q and f
                    if f == None:
                        C[q, -1] = C[-1, q] = True
                    else:
                        C[q, -1] = C[-1, q] = True
                        C[f, -1] = C[-1, f] = True
                        C[q, f] = C[f, q] = False
                    # 8.d decrease the error variables of q and f by multiplying them with a
                    #     initialize the error variable of r with the new value of the error variable of q
                    E[q] *= a
                    if f != None:
                        E[f] *= a
                    E[-1] *= E[q] # why?
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
        self._update_clusters(C)

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

    def _update_clusters(self, connections):
        v = np.zeros((self.N_units, ), dtype = bool)
        self.clusters = list()
        from queue import Queue
        bfs_queue = Queue()
        while np.any(~v):
            root = np.where(~v)[0][0]
            bfs_queue.put(root); v[root] = True
            current_cluster = set()
            self.clusters.append(current_cluster)
            while not bfs_queue.empty():
                p = bfs_queue.get(); current_cluster.add(p)
                neighbors = np.where(connections[p])[0]
                new_nodes = np.where(~v[neighbors])[0]
                current_cluster.update(neighbors[new_nodes])
                for i in neighbors[new_nodes]:
                    bfs_queue.put(i)
                v[neighbors[new_nodes]] = True

    def remove_outliers(self):
        self.clusters = list(filter(lambda cluster: len(cluster)>2, self.clusters))

    def plot_clusters(self, clustered_data):
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(nx.number_connected_components(self.network)):
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.array(observations)
                plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
        plt.legend()
        plt.savefig('visualization/clusters.png')
