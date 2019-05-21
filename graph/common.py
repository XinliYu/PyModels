import collections

import numpy as np
import scipy.sparse as sp
import networkx as nx

from data.exp_common import train_model, test_model
from util.general_ext import *
from enum import Enum
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from data.data_util import sparse_to_tuple


# Perform train-test split
# Takes in adjacency matrix in sparse format
# Returns: adj_train, train_edges, val_edges, val_edges_false,
# test_edges, test_edges_false

class EdgeSampleAlgorithms(Enum):
    RandomSample = 0
    BreathFirstSearch = 1
    NaiveRemoval = 2


def edge_train_test_sample(adj_mat, train_frac=0.8, val_frac=.1, test_frac=.1, false_rate=1.0, algorithm: EdgeSampleAlgorithms = EdgeSampleAlgorithms.RandomSample, undirected=True, return_weights=True):
    """
    Performs edge sampling into a training set, a validation set and a test set.
    :param adj_mat: the adjacency matrix for a graph; can be a weighted adjacency matrix.
    :param train_frac: the percentage of edges to be extracted as the training set.
    :param val_frac: the percentage of edges to be extracted as the validation set.
    :param test_frac: the percentage of edges to be extracted as the test set.
    :param false_rate: indicate how many negative samples will be in each of the training set, the validation set and the test set.
    For example, 1.0 means each set will have the same number of positive sample and the negative samples, while 0.5 means each set have negative samples as many as half of the positive samples.
    :param algorithm: chooses one of the algorithms to perform the sample.
    :param undirected: `True` if the graph is an undirected graph; `False` if the graph is a directed graph
    :param return_weights: `True` if the method returns edges weights; otherwise, `False`.
    :return: the training set, negative training set, validation set, negative validation set, test set, negative test set;
    in additional returns the training set weights, validation set weights and the test set weights if `return_weights` is `True`
    """

    # converts the adjacency matrix to a coordinate sparse matrix if necessary
    if type(adj_mat) is not sp.coo.coo_matrix:
        try:
            adj_mat = sp.coo_matrix(adj_mat)
        except Exception as e:
            error_print(edge_train_test_sample.__name__, "unable to convert the adjacency matrix to a coordinate sparse (COO) matrix")
            raise e

    # remove diagonal elements to avoid self-loops in the adjacency matrix
    adj_mat = adj_mat - sp.dia_matrix((adj_mat.diagonal()[np.newaxis, :], [0]), shape=adj_mat.shape)

    # removes zeros if any
    adj_mat.eliminate_zeros()

    # for undirected graph, only keeps the upper triangular matrix
    if undirected:
        adj_mat = sp.triu(adj_mat)

    # gets the number of edges and the number of nodes
    edge_count = adj_mat.getnnz()
    node_count = adj_mat.shape[0]

    # gets the edge tuples from the coordinate sparse matrix
    edge_tuples = np.vstack((adj_mat.row, adj_mat.col)).transpose()

    # determines sizes
    total_sample_frac = train_frac + val_frac + test_frac
    total_sample_size = int(edge_count * total_sample_frac)
    train_set_size = int(edge_count * train_frac)
    validation_set_size = int(edge_count * val_frac)
    test_set_size = total_sample_size - train_set_size - validation_set_size

    if algorithm == EdgeSampleAlgorithms.RandomSample:
        # For the straightforward edge random sampling,
        # in which case the training set might turn out to have more connected components than the input graph (because it is a sample of subset of edges)

        # randomly shuffle edges
        edge_idxes = np.random.permutation(edge_count)
        edge_tuples = edge_tuples[edge_idxes]
        if return_weights:
            weights = adj_mat.data[edge_idxes]

        # region sample training edges
        start = 0
        train_edges = edge_tuples[start:train_set_size]
        if return_weights:
            train_edge_weights = weights[start:train_set_size]
        # endregion

        # region sample validation edges and test edges

        # NOTE the training set might be cover the whole graph.
        # In this case the test set and validation set has to be constrained in this sub-graph.

        # STEP1: sample all edges constrained by the training set
        start = train_set_size
        train_nodes = set(train_edges.flatten())
        edge_tuples_constrained = []
        if return_weights:
            edge_weights_constrained = []
        num_expected_to_sample = validation_set_size + test_set_size
        while start < edge_count and num_expected_to_sample > 0:
            edge = edge_tuples[start]
            if edge[0] in train_nodes and edge[1] in train_nodes:
                num_expected_to_sample -= 1
                edge_tuples_constrained.append(edge)
                if return_weights:
                    edge_weights_constrained.append(weights[start])
            start += 1

        num_actual = len(edge_tuples_constrained)
        ratio_actual = num_actual / (validation_set_size + test_set_size)
        edge_tuples_constrained = np.array(edge_tuples_constrained)
        if return_weights:
            edge_weights_constrained = np.array(edge_weights_constrained)

        # STEP2: sample the validation set, number of edges adjusted by `ratio_actual`
        if val_frac != 0:
            validation_set_size = int(validation_set_size * ratio_actual)
            validation_edges = edge_tuples_constrained[:validation_set_size]
            if return_weights:
                validation_edge_weights = edge_weights_constrained[:validation_set_size]
        else:
            validation_edges = None

        # STEP3: sample the test set, number of edges adjusted by `ratio_actual`
        test_set_size = num_actual - validation_set_size
        test_edges = edge_tuples_constrained[validation_set_size:]
        if return_weights:
            test_edge_weights = edge_weights_constrained[validation_set_size:]

        # endregion

    elif algorithm == EdgeSampleAlgorithms.BreathFirstSearch:
        # Do breath first search with a random start node;
        # ensures maximum connectivity, and the train set will have no more connected components than the original graph

        edge_idx = 0
        validation_set_end = train_set_size + validation_set_size
        test_set_end = validation_set_end + test_set_size
        q = collections.deque()  # the queue for BFS
        q.append(np.random.randint(node_count))
        flags = set(range(node_count))
        train_edges = []
        if val_frac != 0:
            validation_edges = []
        test_edges = []

        if return_weights:
            train_edge_weights = []
            if val_frac != 0:
                validation_edge_weights = []
            test_edge_weights = []

        while True:
            n1 = q.popleft()
            flags.remove(n1)
            row = adj_mat.getrow(n1)
            row_indices = row.indices
            if return_weights:
                row_data = row.data
            for i in range(row_indices):
                n2 = row_indices[i]
                if n2 in flags:
                    edge = (n1, n2)
                    if edge_idx < train_set_size:
                        train_edges.append(edge)
                        if return_weights:
                            train_edge_weights.append(row_data[i])
                    elif edge_idx < validation_set_end:
                        validation_edges.append(edge)
                        if return_weights:
                            validation_edge_weights.append(row_data[i])
                    elif edge_idx < test_set_end:
                        test_edges.append(edge)
                        if return_weights:
                            test_edge_weights.append(row_data[i])
                    else:
                        break
                    edge_idx += 1
                    q.append(n2)
            if len(q) == 0:
                if len(flags) > 0:
                    q.append(next(iter(flags)))
                else:
                    break
        train_edges = np.array(train_edges)
        if val_frac != 0:
            validation_edges = np.array(validation_edges)
        else:
            validation_edges = None
        test_edges = np.array(test_edges)
        if return_weights:
            train_edge_weights = np.array(train_edge_weights)
            if val_frac != 0:
                validation_edge_weights = np.array(validation_edge_weights)
            else:
                validation_edge_weights = None
            test_edge_weights = np.array(test_edge_weights)
        else:
            train_edge_weights = validation_edge_weights = test_edge_weights = None
    elif algorithm == EdgeSampleAlgorithms.NaiveRemoval:
        # Sample edges by removal, a brutal force algorithm as implemented in https://github.com/tkipf/gae/blob/master/gae/preprocessing.py;
        # this algorithm ensures the train set will have no more connected components than the original graph
        # could be slow for large graphs as it need to repeatedly check if the number of connected components increase.
        g = nx.from_scipy_sparse_matrix(adj_mat)
        count_cc = nx.number_connected_components if undirected else nx.number_strongly_connected_components
        num_cc = count_cc(g)

        edge_idxes = np.random.permutation(edge_count)
        i = 0

        def _remove_edge_with_cc_check(val_set_or_test_set):
            # tries to remove an edge from the graph and add it to the validation set or test set
            n1, n2 = edge_tuples[edge_idxes[i]]
            g.remove_edge(n1, n2)
            if count_cc(g) > num_cc:  # if the removal causes a new component, then undo the removal
                g.add_edge(n1, n2)
                return False
            else:
                if val_set_or_test_set is not None:
                    val_set_or_test_set.append((n1, n2))
                return True

        # first removes extra edges
        start = 0
        end = edge_count - total_sample_size
        while start < end and i < edge_count:
            if _remove_edge_with_cc_check(None):
                start += 1
            i += 1

        # then sample the test set
        end += test_set_size
        test_edges = []
        while start < end and i < edge_count:
            if _remove_edge_with_cc_check(test_edges):
                start += 1
            i += 1
        if return_weights:
            test_edge_weights = []
            for edge in test_edges:
                test_edge_weights.append(g[edge[0]][edge[1]]['weight'])
        test_edges = np.array(test_edges)

        # then sample the validation set
        if validation_set_size != 0:
            end += validation_set_size
            validation_edges = []
            while start < end and i < edge_count:
                if _remove_edge_with_cc_check(validation_edges):
                    start += 1
                i += 1
            if return_weights:
                validation_edge_weights = []
                for edge in validation_edges:
                    validation_edge_weights.append(g[edge[0]][edge[1]]['weight'])
            validation_edges = np.array(validation_edges)
        else:
            validation_edges = None

        # train set is the remaining edges
        if return_weights:
            train_edges = []
            train_edge_weights = []
            for edge in g.edges:
                train_edges.append(edge)
                train_edge_weights.append(g[edge[0]][edge[1]]['weight'])
        else:
            train_edges = np.array(g.edges)

    # negative sampling (sample "edges" that do not exist in the graph)
    # TODO if graph is dense, then the following random sample is not efficient
    edge_tuples = set(map(tuple, edge_tuples))

    def _negative_sample(size):
        i = 0
        sample = set()
        while i < size:
            edges = np.random.randint(0, node_count, size=(size - i, 2))
            for edge in edges:
                n1, n2 = edge
                if n1 == n2:
                    continue
                false_edge = (n1, n2) if n1 < n2 or not undirected else (n2, n1)
                if false_edge in edge_tuples:
                    continue
                i += 1
                sample.add(false_edge)
        return np.array(list(sample))

    negative_train_set_count = int(train_set_size * false_rate)
    negative_train_edges = _negative_sample(negative_train_set_count)

    negative_test_count = int(test_set_size * false_rate)
    negative_test_edges = _negative_sample(negative_test_count)

    if validation_set_size != 0:
        negative_val_count = int(validation_set_size * false_rate)
        negative_validation_edges = _negative_sample(negative_val_count)
    else:
        negative_validation_edges = None

    if return_weights:
        return train_edges, negative_train_edges, \
               validation_edges, negative_validation_edges, \
               test_edges, negative_test_edges, \
               train_edge_weights, validation_edge_weights, test_edge_weights
    else:
        return train_edges, negative_train_edges, validation_edges, negative_validation_edges, test_edges, negative_test_edges


def link_prediction_by_node_embeddings(positive_train_edges,
                                       negative_train_edges,
                                       node_embeddings,
                                       node_embedding_merger=np.multiply,
                                       classifier=None,
                                       extra_train_arguments=None,
                                       positive_test_edges=None,
                                       negative_test_edges=None,
                                       test_metric=None,
                                       extra_test_arguments=None):
    """
    Runs embedding based network link prediction.
    :param positive_train_edges: the positive training edges,
            represented by an Nx2 matrix where each row is a pair of node indices of an existent edge in the network.
    :param negative_train_edges: the negative training edges,
            represented by an Nx2 matrix where each row is a pair of node indices of a non-existent "edge" (i.e there is no edge in the network between this pair of nodes).
    :param node_embeddings: provides the node embedding matrix; edge embeddings are computed based on a pair of node embeddings.
    :param node_embedding_merger: provides a function to merge a pair of node embeddings to an edge embedding.
    :param classifier: provides a classifier.
            The classifier must have a function named "fit" or "train" that takes at least two positional arguments,
            the first being the edge embeddings, and the second being the labels.
    :param extra_train_arguments: extra training argument to input the classifier if necessary.
    :param positive_test_edges: positive test edges.
    :param negative_test_edges: negative test edges.
    :param test_metric: a function to measure the test performance.
    :param extra_test_arguments: extra argument to input the classifier during test.
    :return: the training results returned by the classifier, as well as the test results if either `positive_test_edges` and/or `negative_test_edges` are provided.
    """
    n1_embeddings = np.vstack((node_embeddings[positive_train_edges[:, 0]], node_embeddings[negative_train_edges[:, 0]]))
    n2_embeddings = np.vstack((node_embeddings[positive_train_edges[:, 1]], node_embeddings[negative_train_edges[:, 1]]))
    labels = np.concatenate([np.ones(len(positive_train_edges)), np.zeros(len(negative_train_edges))])
    edge_embeddings = node_embedding_merger(n1_embeddings, n2_embeddings)
    if classifier is None:
        classifier = LogisticRegression(random_state=0, solver='lbfgs')
    if test_metric is None:
        def _default_test_metric(truth, prediction):
            return roc_auc_score(truth, prediction[:, 1])

        test_metric = _default_test_metric

    train_results = train_model(model=classifier,
                                training_set=edge_embeddings,
                                targets=labels,
                                extra_arguments=extra_train_arguments,
                                error_tag=link_prediction_by_node_embeddings.__name__)

    if positive_test_edges is None:
        if negative_test_edges is None:  # no test
            return train_results
        else:  # just negative test edges
            n1_embeddings = node_embeddings[negative_test_edges[:, 0]]
            n2_embeddings = node_embeddings[negative_test_edges[:, 1]]
            labels = np.zeros(len(negative_test_edges))
    elif negative_test_edges is None:  # just positive test edges
        n1_embeddings = node_embeddings[positive_test_edges[:, 0]]
        n2_embeddings = node_embeddings[positive_test_edges[:, 1]]
        labels = np.ones(len(positive_test_edges))
    else:  # both positive and negative test edges
        n1_embeddings = np.vstack((node_embeddings[positive_test_edges[:, 0]], node_embeddings[negative_test_edges[:, 0]]))
        n2_embeddings = np.vstack((node_embeddings[positive_test_edges[:, 1]], node_embeddings[negative_test_edges[:, 1]]))
        labels = np.concatenate([np.ones(len(positive_test_edges)), np.zeros(len(negative_test_edges))])

    edge_embeddings = node_embedding_merger(n1_embeddings, n2_embeddings)
    return train_results, test_model(model=classifier,
                                     test_set=edge_embeddings,
                                     targets=labels,
                                     metric=test_metric,
                                     extra_arguments=extra_test_arguments,
                                     error_tag=link_prediction_by_node_embeddings.__name__)
