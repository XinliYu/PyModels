import numpy as np
import networkx as nx
from util.stat_ext import *
import random


# the `Node2VecGraph` class is based on https://github.com/lucashu1/link-prediction/blob/master/node2vec.py


class Node2VecGraph:
    """
    Wraps a graph for node2vec random samples.
    """

    def __init__(self, graph: nx.classes.graph.Graph, p, q, directed=False, weighted=True, preprocess=True):
        """
        Wraps a graph for node2vec random samples.
        node2vec random walk is a mixture of BFS and DFS, adding two control parameters `p` and `q` to adjust how likely the random walk will linger in the neighborhood.
        :param graph: the graph.
        :param directed: `True` if the `graph` is directed; otherwise, `False`.
        :param weighted: `True` if the `graph` is weighted; otherwise, `False`.
        :param p: one of the two node2vec control parameters. A smaller `p` make the random walk more likely to repeat previous nodes.
        :param q: one of the two node2vec control parameters. A smaller `q` make the random walk more likely to explore far-away nodes.
        :param preprocess: immediately pre-computes the random walk parameters. If this is set `False`, must call the `preprocess` method before generating random walks.
        """
        self.g = graph
        self.directed = directed
        self.p = p
        self.q = q
        self.weighted = weighted
        if preprocess:
            self.preprocess()
        else:
            self.node_alias_paras = self.edge_alias_paras = None

    def node2vec_walk(self, start_node, walk_length):
        """
        Simulate a node2vec random walk starting of length `walk_length` from start node. NOTE this graph must be preprocessed in order to generate a random walk.
        :param start_node: the starting node of the random walk.
        :param walk_length: the length of the random walk.
        :return: a single sequence of node2vec random walk of length `walk_length` starting from `start_node`.
        """
        walk = [start_node]
        while len(walk) < walk_length:
            curr_node = walk[-1]
            curr_node_neighbors = sorted(self.g.neighbors(curr_node))
            if len(curr_node_neighbors) > 0:
                if len(walk) == 1:
                    walk.append(curr_node_neighbors[alias_sample(self.node_alias_paras[curr_node][0], self.node_alias_paras[curr_node][1])])
                else:
                    prev_noe = walk[-2]
                    next_node = curr_node_neighbors[alias_sample(*self.edge_alias_paras[(prev_noe, curr_node)])]
                    walk.append(next_node)
            else:
                break

        return walk

    def sample_node2vec_walks(self, num_walks, walk_length):
        """
        Sample sequences of node2vec random walks. All node2vec random walks have the same length `walk_length`.
        NOTE this graph must be preprocessed in order to generate the random walks.
        :param num_walks: the number of random walks to sample for each node.
        :param walk_length: the length of each random walk.
        :return: the sampled random walks.
        """
        walks = []
        nodes = list(self.g.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(start_node=node, walk_length=walk_length))

        return walks

    def preprocess(self):
        """
        Pre-computes the random walk parameters by the technique of alias-sampling.
        For alias sampling, see https://en.wikipedia.org/wiki/Alias_method.
        """

        g = self.g
        p = self.p
        q = self.q

        def _get_alias_edge_dist(src_node, dst_node):
            node2vec_transition_dist = []
            if self.weighted:
                for dst_neighbor in sorted(g.neighbors(dst_node)):
                    if dst_neighbor == src_node:
                        node2vec_transition_dist.append(g[dst_node][dst_neighbor]['weight'] / p)
                    elif g.has_edge(dst_neighbor, src_node):
                        node2vec_transition_dist.append(g[dst_node][dst_neighbor]['weight'])
                    else:
                        node2vec_transition_dist.append(g[dst_node][dst_neighbor]['weight'] / q)
            else:
                for dst_neighbor in sorted(g.neighbors(dst_node)):
                    if dst_neighbor == src_node:
                        node2vec_transition_dist.append(1 / p)
                    elif g.has_edge(dst_neighbor, src_node):
                        node2vec_transition_dist.append(1)
                    else:
                        node2vec_transition_dist.append(1 / q)
            normalizing_const = sum(node2vec_transition_dist)
            node2vec_transition_dist = [u_prob / normalizing_const for u_prob in node2vec_transition_dist]
            return alias_sample_preprocess(node2vec_transition_dist)

        alias_nodes = {}
        for node in g.nodes():
            random_walk_transition_dist = [g[node][nbr]['weight'] if self.weighted else 1 for nbr in sorted(g.neighbors(node))]
            norm_const = sum(random_walk_transition_dist)
            random_walk_transition_dist = [u_prob / norm_const for u_prob in random_walk_transition_dist]
            alias_nodes[node] = alias_sample_preprocess(random_walk_transition_dist)

        alias_edges = {}

        if self.directed:
            for edge in g.edges():
                alias_edges[edge] = _get_alias_edge_dist(edge[0], edge[1])
        else:
            for edge in g.edges():
                alias_edges[edge] = _get_alias_edge_dist(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = _get_alias_edge_dist(edge[1], edge[0])

        self.node_alias_paras = alias_nodes
        self.edge_alias_paras = alias_edges
