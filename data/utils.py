import codecs
import numpy as np
import pickle as pkl
import networkx as nx
from os import path
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    # index = []
    with open(filename, 'rb') as f:
        index = pkl.load(f)
    # for line in open(filename):
    #    index.append(int(line.strip()))
    return index


# sample_mask(idx_train, labels.shape[0])
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_f(f, add_data, delimiter=None, type_or_format_fun=float):
    if delimiter is None or delimiter == '':
        if type_or_format_fun in (int, float, str):
            for line in f:
                splits = line.split()
                add_data([type_or_format_fun(cell) for cell in splits])
        else:
            for line in f:
                add_data(type_or_format_fun(line.split()))
    else:
        if type_or_format_fun in (int, float, str):
            for line in f:
                splits = line.split(delimiter)
                add_data([type_or_format_fun(cell) for cell in splits])
        else:
            for line in f:
                add_data(type_or_format_fun(line.split(delimiter)))


def load_csv(f, delimiter=None, type_or_format=float):
    data = []
    load_f(f=f, add_data=data.append,
           delimiter=delimiter,
           type_or_format_fun=type_or_format)
    return data


def load_graph_from_edge_tuples(f, delimiter=None, type_or_format_fun=int, weight_transform=None):
    G = nx.Graph()

    def _add_edge(tups):
        if len(tups) == 2:
            G.add_edge(tups[0], tups[1])
        else:
            G.add_edge(tups[0], tups[1],
                       weight=tups[2] if weight_transform is None else weight_transform(tups[2]))

    load_f(f=f, add_data=_add_edge,
           delimiter=delimiter,
           type_or_format_fun=type_or_format_fun)

    return G


def load(data_file: str, encoding: str = None, delimiter=None, data_type=None, field_type_or_format_fun=None, **kwargs):
    """
    Loads data from a file.
        1) If the file is a text file ending with extension name `.txt`, then it will be treated as a `delimiter`-separated file,
        and the data will be either read as a list of lists if `data_type` is not specified, where each list corresponds to one line in the text file;
        or each line will be converted to some data object if `data_type` specifies a supported pre-defined data type, and a collection of all data objects will be returned.
        2) Otherwise, the file is treated as a pickle file and the data object read from the file will be returned.
    :param data_file: the path to the data file.
    :param encoding: the encoding of the data file.
    :param delimiter: the delimiter if the data file is a text file consisting of lines of `delimiter`-separated data entries.
    :param data_type: specifies a build-in data type support; currently supported data type.
    :param field_type_or_format_fun: provides a data type for each field of each line, e.g. `int`, `float` or `str`, or a data conversion function for each line.
    :return: a collection of data read from the file.
    """

    if data_file.endswith('.txt'):
        with open(data_file, 'r') if encoding is None \
                else codecs.open(data_file, encoding=encoding) as f:
            if data_type == 'graph_edge_tuples':
                return load_graph_from_edge_tuples(f, delimiter, field_type_or_format_fun, kwargs.get('weight_transform'))
            else:
                return load_csv(f, delimiter, field_type_or_format_fun)

    else:
        with open(data_file, 'rb') as f:
            if encoding is None or sys.version_info < (3, 0):
                return pkl.load(f)
            else:
                return pkl.load(f, encoding=encoding)


def load_data_gcn_dblp(dir: str, edge_tuple_file: str, feature_file: str, node_label_file: str, train_index_file, validation_index_file, test_index_file, weight_transform):
    graph = load(edge_tuple_file, delimiter=None, data_type='graph_edge_tuples', field_type_or_format_fun=int, weight_transform=weight_transform)
    adj = nx.adjacency_matrix(graph)

    if path.exists(feature_file):
        features = np.array(load(feature_file, field_type_or_format_fun=float))






def load_data_aminer(dataset: str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['ty', 'ally', 'graph']
    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("test_data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    ty, ally, graph = tuple(objects)
    # x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("test_data/ind.{}.test.index".format(dataset))
    print("-----inx_reorder------")
    print(test_idx_reorder)
    # 从小到大排序
    test_idx_range = np.sort(test_idx_reorder)

    print("-----inx_range------")
    print(test_idx_range)
    #
    # print("-----x----")
    # print(x)
    # print("-----tx----")
    # print(tx.shape)
    # print("-----allx----")
    # print(allx)
    print("-----ty----")
    print(ty.shape)
    print("-----ally----")
    print(ally.shape)

    # if dataset_str == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended
    #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #     ty_extended[test_idx_range-min(test_idx_range), :] = ty
    #     ty = ty_extended

    # aminer数据集随机构造features向量
    # edges = np.loadtxt("aminer/aminer_edge.txt", dtype=np.int32)
    # features = sp.eye(np.max(edges), dtype=np.float32).tolil()
    features_list = []
    with open("test_data/features.txt") as flistfile:
        for line in flistfile:
            num = list(map(float, line.split()))
            features_list.append(num)
    features = np.array(features_list)
    features = sp.csr_matrix(features).tolil()
    print("features:")
    print(features.shape)

    # vstack函数是按垂直的把数组堆叠起来， tolil把faetures转换成增量矩阵
    # features = sp.vstack((allx, tx)).tolil()

    # 把特征矩阵还原，和对应的邻接矩阵对应起来
    features[test_idx_reorder, :] = features[test_idx_range, :]

    # 邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print("aaaaaaaaaaa")
    print(graph)

    print("-----adj------")
    print(adj.shape)

    # labels.shape:(2708,7)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # len(list(idx_val)) + len(list(idx_train)) + len(idx_test) =  1640
    idx_test = test_idx_range.tolist()
    print("###idx_test")
    print(idx_test)
    idx_train = range(14272)
    # idx_val = range(len(y), len(y)+500)
    idx_val = range(14272, 14272 + 27855)
    print("---idx_train-- labels.shape[0]-------")
    print(idx_train)
    print(labels.shape[0])

    # train_mask.shape : (2708,) , 指示坐标的函数，表明坐标为true的位置将被取出来使用
    train_mask = sample_mask(idx_train, labels.shape[0])
    print("------train_mask-----")
    print(train_mask)
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train y_val y_test .shape : (2708, 7)
    # 替换了true位置，即这些位置的labels是已知，其他的是未知（zeros）
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # print(graph.items())

    # train_mask=140 y_test_mask=1000  val_mask=640-140
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # 图的归一化拉普拉斯矩阵
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
