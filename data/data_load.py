from collections import defaultdict
from enum import Enum
from data.data_util import *
from util.dict_ext import *
from util.io_ext import *
from util.path_ext import *
from util.time_exp import *
from util.general_ext import *
import networkx as nx
import os


# methods for loading data from different data sets


class SupportedDataFormats(Enum):
    DelimiterSeparatedValues = 0,
    GraphEdgeTuplesAsNetwork = 1,
    DictTuples = 2,
    TrainTestMasks = 3,
    CoordinateSparseMatrix = 4,
    GraphEdgeTuplesAsSparseMatrix = 5


def load_f(f, add_data, delimiter=None, value_type_or_format_fun=float, skip_first_line=False):
    def _try_convert(entry):
        try:
            return value_type_or_format_fun(entry)
        except:
            return entry

    if skip_first_line:
        next(f)
    if delimiter is None or delimiter == '':
        if value_type_or_format_fun in (int, float, str, bool):
            for line in f:
                splits = line.split()
                if len(splits) == 1:
                    add_data(_try_convert(splits[0]))
                else:
                    add_data([_try_convert(cell) for cell in splits])
        elif value_type_or_format_fun is bool:
            for line in f:
                splits = line.split()
                if len(splits) == 1:
                    add_data(str2bool(splits[0]))
                else:
                    add_data([str2bool(cell) for cell in splits])
        elif value_type_or_format_fun is None:
            for line in f:
                add_data(line.split())
        else:
            for line in f:
                add_data(value_type_or_format_fun(line.split()))
    else:
        if value_type_or_format_fun in (int, float, str, bool):
            for line in f:
                splits = line.split(delimiter)
                if len(splits) == 1:
                    add_data(_try_convert(splits[0]))
                else:
                    add_data([_try_convert(cell) for cell in splits])
        elif value_type_or_format_fun is bool:
            for line in f:
                splits = line.split()
                if len(splits) == 1:
                    add_data(str2bool(splits[0]))
                else:
                    add_data([str2bool(cell) for cell in splits])
        elif value_type_or_format_fun is None:
            for line in f:
                add_data(line.split(delimiter))
        else:
            for line in f:
                add_data(value_type_or_format_fun(line.split(delimiter)))


def load_csv(f, delimiter=None, value_type_or_format_fun=float, skip_first_line=False):
    data = []
    load_f(f=f, add_data=data.append,
           delimiter=delimiter,
           value_type_or_format_fun=value_type_or_format_fun,
           skip_first_line=skip_first_line)
    return data


def load_graph_from_edge_tuples(f, delimiter=None, node_type_or_format_fun=int, weight_transform=None, raw_data=False, skip_first_line=False):
    G = nx.Graph()

    def _add_graph_edge(tups):
        if len(tups) == 2:
            G.add_edge(tups[0], tups[1])
        else:
            G.add_edge(tups[0], tups[1],
                       weight=tups[2] if weight_transform is None else weight_transform(tups[2]))

    load_f(f=f, add_data=_add_graph_edge,
           delimiter=delimiter,
           value_type_or_format_fun=node_type_or_format_fun,
           skip_first_line=skip_first_line)

    if raw_data:
        graph_nodes = sorted(G.nodes)
        adjacency_matrix = nx.adjacency_matrix(G, nodelist=graph_nodes)
        return adjacency_matrix, graph_nodes
    else:
        return G


def load_spcoo_from_tuples(f, delimiter=None, data_type_or_format_fun=float, square_matrix=True, zero_based_index=True, dtype=np.float64, skip_first_line=False):
    """
    Loads a sparse matrix from a file object. The file should be a text file where each line is a three-tuple: the row index, the column index and the data.
    :param f: the file object.
    :param delimiter: the character or string to separate tuple elements in each line of the file.
    :param data_type_or_format_fun: the data type or a format function for the data.
    :param square_matrix: `True` if the returned matrix is a square matrix; `False` if the returned matrix is a rectangle matrix.
    :param zero_based_index: `True` if the row indices and column indices are zero-based; `False` if they are one-based.
    :return: a `scipy.sparse.coo_matrix` object.
    """
    row = []
    col = []
    data = []
    _max_idx = [0, 0]

    def _add_tuple(tups):
        row_idx = int(tups[0])
        col_idx = int(tups[1])
        if not zero_based_index:
            row_idx -= 1
            col_idx -= 1
        _max_idx[0] = max(row_idx, _max_idx[0])
        _max_idx[1] = max(col_idx, _max_idx[1])
        row.append(row_idx)
        col.append(col_idx)
        if len(tups) == 2:
            data.append(1)
        else:
            data.append(tups[2] if data_type_or_format_fun is None else data_type_or_format_fun(tups[2]))

    load_f(f=f, add_data=_add_tuple, delimiter=delimiter, value_type_or_format_fun=None, skip_first_line=skip_first_line)

    _max_idx[0] += 1
    _max_idx[1] += 1
    if square_matrix:
        if _max_idx[0] < _max_idx[1]:
            _shape = (_max_idx[1] + 1, _max_idx[1] + 1)
        else:
            _shape = (_max_idx[0] + 1, _max_idx[0] + 1)
    else:
        _shape = (_max_idx[0] + 1, _max_idx[1] + 1)
    return sp.coo_matrix((data, (row, col)), shape=_max_idx, dtype=dtype)


def load_edge_tuples_as_spcoo(f, delimiter=None, weight_type_or_format_fun=float, zero_based_index=True, squeeze=True, undirected=True, return_raw_lists=False, dtype=np.float16, skip_first_line=False):
    """
    Loads a graph represented by a coordinate sparse matrix and a list of nodes from a file object.
    The file should be a text file where each line is a three-tuple: the row index, the column index and the data.
    The returned node indices are sorted in a descending order.
    :param f: the file object.
    :param delimiter: the character or string to separate tuple elements in each line of the file.
    :param weight_type_or_format_fun: the weight type or a format function for the weight.
    :param zero_based_index: `True` if the node indices are zero-based; `False` if they are one-based.
    :return: the graph adjacency matrix represented by a `scipy.sparse.coo_matrix` object, and the nodes represented by a list of sorted indices.
    """
    if squeeze:
        row_dict = defaultdict(list)
        nodes = set()

        def _add_tuple(tups):
            row_idx = int(tups[0])
            col_idx = int(tups[1])
            nodes.add(row_idx)
            nodes.add(col_idx)
            if len(tups) == 2:
                weight = 1
            else:
                weight = tups[2] if weight_type_or_format_fun is None else weight_type_or_format_fun(tups[2])
            row_dict[row_idx].append((col_idx, weight))

        load_f(f=f, add_data=_add_tuple, delimiter=delimiter, value_type_or_format_fun=None, skip_first_line=skip_first_line)

        if __debug__:
            highlight_print(load_edge_tuples_as_spcoo.__name__, 'Squeezing edge node indices ...')

        nodes = sorted(nodes)
        node_idx_dict = index_dict(nodes)
        row = []
        col = []
        data = []
        for row_idx, node in enumerate(nodes):
            if node in row_dict:
                for col_node, weight in row_dict[node]:
                    col_idx = node_idx_dict[col_node]
                    row.append(row_idx)
                    col.append(col_idx)
                    data.append(weight)
                    if undirected:
                        row.append(col_idx)
                        col.append(row_idx)
                        data.append(weight)
        if return_raw_lists:
            if __debug__:
                highlight_print(load_edge_tuples_as_spcoo.__name__, 'Edge tuple lists returned.')
            return row, col, data, len(nodes), nodes
        else:
            if __debug__:
                highlight_print(load_edge_tuples_as_spcoo.__name__, 'Constructing sparse matrix ...')
            return sp.coo_matrix((data, (row, col)), shape=(len(nodes), len(nodes)), dtype=dtype), nodes
    else:
        row = []
        col = []
        data = []
        max_idx = [0]
        nodes = set()

        def _add_tuple(tups):
            row_idx = int(tups[0])
            col_idx = int(tups[1])
            nodes.add(row_idx)
            nodes.add(col_idx)
            if not zero_based_index:
                row_idx -= 1
                col_idx -= 1
            max_idx[0] = max(row_idx, col_idx, max_idx[0])
            if len(tups) == 2:
                weight = 1
            else:
                weight = tups[2] if weight_type_or_format_fun is None else weight_type_or_format_fun(tups[2])
            row.append(row_idx)
            col.append(col_idx)
            data.append(weight)
            if undirected:
                row.append(col_idx)
                col.append(row_idx)
                data.append(weight)

        load_f(f=f, add_data=_add_tuple, delimiter=delimiter, value_type_or_format_fun=None, skip_first_line=skip_first_line)
        if return_raw_lists:
            if __debug__:
                highlight_print(load_edge_tuples_as_spcoo.__name__, 'Edge tuple lists returned.')
            return row, col, data, max_idx[0] + 1, nodes
        else:
            if __debug__:
                highlight_print(load_edge_tuples_as_spcoo.__name__, 'Constructing sparse matrix ...')
            return sp.coo_matrix((data, (row, col)), shape=(max_idx[0] + 1, max_idx[0] + 1)), sorted(nodes)


def load_dict_from_tuples(f, delimiter=None, value_type_or_format_fun=int, skip_first_line=False):
    d = {}

    def _add_dict_entry(tups):
        if len(tups) == 2:
            d.update({tups[0]: tups[1]})
        else:
            d.update({tups[0]: tups[1:]})

    load_f(f=f, add_data=_add_dict_entry,
           delimiter=delimiter,
           value_type_or_format_fun=value_type_or_format_fun,
           skip_first_line=skip_first_line)

    return d


def load_train_test_masks(size, mask_file=None, encoding: str = None, delimiter=None, train_ratio=0.9, validation_ratio=0.0, test_ratio=0.1):
    if mask_file is not None and path.exists(mask_file):
        tic("Loading mask file " + mask_file)
        masks = np.array(load(mask_file, encoding=encoding, delimiter=delimiter, data_format=SupportedDataFormats.DelimiterSeparatedValues, value_type_or_format_fun=bool))
        if masks.shape[1] == 2:
            train_mask, test_mask = masks[:, 0], masks[:, 1]
            validation_mask = None
        else:
            train_mask, validation_mask, test_mask = masks[:, 0], masks[:, 1], masks[:, 2]
        toc("Mask file {} loaded".format(mask_file))
    else:
        tic("Creating mask file " + mask_file)
        if validation_ratio == test_ratio == 0:
            test_ratio = 1 - train_ratio
        masks = mask_by_percentage(size, percents=(train_ratio, validation_ratio, test_ratio))
        train_mask, validation_mask, test_mask = masks[0], masks[1], masks[2]

        if mask_file is not None:
            save_lists(file_path=mask_file, lists=masks, encoding=encoding)
        toc("Mask file {} created".format(mask_file))
    return train_mask, validation_mask, test_mask


def load(data_file: str, encoding: str = None, delimiter=None, data_format: SupportedDataFormats = SupportedDataFormats.DelimiterSeparatedValues, value_type_or_format_fun=None,
         skip_first_line=False, text_file_extension=('.txt', '.csv'), binary_file_extension='.dat', compressed: bool = False, create_binary_cache: bool = False, binary_cache_dir: str = 'cache',
         cache_exists=None, cache_save=None, cache_load=None, **kwargs):
    """
    Loads data from a file.
        1) If the file is a text file ending with extension name `.txt`, then it will be treated as a `delimiter`-separated file,
        and the data will be processed according to `data_format` and `value_type_or_format_fun`;
        2) Otherwise, the file is treated as a pickle file and the data object read from the file will be returned.
    :param data_file: the path to the data file.
    :param encoding: the encoding of the data file.
    :param delimiter: the delimiter if the data file is a text file consisting of lines of `delimiter`-separated data entries.
    :param data_format: specifies a build-in supported data format.
    :param value_type_or_format_fun: provides a Python data type for each field of each line, e.g. `int`, `float`, `bool` or `str`, or a data conversion function for each line.
    :return: a collection of data read from the file.
    """

    if data_format == SupportedDataFormats.TrainTestMasks:
        return load_train_test_masks(mask_file=data_file, encoding=encoding, delimiter=delimiter,
                                     train_ratio=kwargs.get('train_ratio'), validation_ratio=kwargs.get('validation_ratio'), test_ratio=kwargs.get('test_ratio'))

    if data_file.endswith(text_file_extension):
        binary_cache_file = replace_ext(data_file, binary_file_extension) if binary_cache_dir is None \
            else add_subfolder_and_replace_ext(data_file, binary_cache_dir, binary_file_extension)

        if (cache_exists is None and path.exists(binary_cache_file)) or (cache_exists is not None and cache_exists(binary_cache_file)):
            return pickle_load(binary_cache_file, compressed=compressed, encoding=encoding) if cache_load is None else cache_load(binary_cache_file, compressed, encoding)

        with open(data_file, 'r') if encoding is None \
                else codecs.open(data_file, encoding=encoding) as f:
            if data_format == SupportedDataFormats.GraphEdgeTuplesAsNetwork:
                obj = load_graph_from_edge_tuples(f=f,
                                                  delimiter=delimiter,
                                                  node_type_or_format_fun=kwargs.get('node_transform'),
                                                  weight_transform=value_type_or_format_fun,
                                                  skip_first_line=skip_first_line)
            elif data_format == SupportedDataFormats.DictTuples:
                obj = load_dict_from_tuples(f=f,
                                            delimiter=delimiter,
                                            value_type_or_format_fun=value_type_or_format_fun,
                                            skip_first_line=skip_first_line)
            elif data_format == SupportedDataFormats.CoordinateSparseMatrix:
                obj = load_spcoo_from_tuples(f=f, delimiter=delimiter,
                                             data_type_or_format_fun=value_type_or_format_fun,
                                             square_matrix=kwargs.get('square_matrix'),
                                             zero_based_index=kwargs.get('zero_based_index'),
                                             dtype=kwargs.get('dtype'),
                                             skip_first_line=skip_first_line)
            elif data_format == SupportedDataFormats.GraphEdgeTuplesAsSparseMatrix:
                obj = load_edge_tuples_as_spcoo(f=f, delimiter=delimiter,
                                                weight_type_or_format_fun=value_type_or_format_fun,
                                                zero_based_index=kwargs.get('zero_based_index'),
                                                squeeze=kwargs.get('squeeze'),
                                                return_raw_lists=kwargs.get('raw_data'),
                                                dtype=kwargs.get('dtype'),
                                                skip_first_line=skip_first_line)
            else:
                obj = load_csv(f, delimiter, value_type_or_format_fun, skip_first_line=skip_first_line)

            if create_binary_cache:
                if __debug__:
                    print("Saving binary cache for {} at {}.".format(data_file, binary_cache_file))
                if cache_save is None:
                    pickle_save(binary_cache_file, data=obj, compressed=compressed)
                else:
                    cache_save(binary_cache_file, obj, compressed)
                if __debug__:
                    print("Binary cache saved.")
            return obj

    else:
        return pickle_load(data_file, compressed=compressed, encoding=encoding) if cache_load is None else cache_load(data_file, compressed, encoding)


def save(data_file: str, data, compressed: bool = False):
    if data_file.endswith('.dat'):
        with open(data_file, 'wb') if not compressed else gzip.open(data_file, 'wb') as f:
            pkl.dump(data, f)


def load_graph_adjacency(data_folder: str,
                         edge_tuple_file: str,
                         node_transform=int,
                         weight_transform=float,
                         zero_based_index: bool = False,
                         squeeze: bool = True,
                         create_binary_cache: bool = True,
                         binary_cache_compressed: bool = True,
                         cache_raw_data=False,
                         use_networkx=False,
                         adjacency_dtype=np.float16):
    tic('Loading graph data from file')

    data_format = SupportedDataFormats.GraphEdgeTuplesAsNetwork if use_networkx \
        else SupportedDataFormats.GraphEdgeTuplesAsSparseMatrix

    def _adj_and_nodes_cache_exists(file):
        return path.exists(file + '.pt0') and path.exists(file + '.pt1')

    def _save_adj_and_nodes(file, obj, compressed):
        file1 = file + '.pt0'
        sp.save_npz(file1, obj[0], compressed)
        os.rename(file1 + '.npz', file1)
        pickle_save(file + '.pt1', obj[1], compressed)

    def _load_adj_and_nodes(file, compressed, encoding):
        return sp.load_npz(file + '.pt0'), pickle_load(file + '.pt1', compressed)

    # protocol: graph_data is a tuple, with its last being the list of nodes.
    # Currently, if `use_networkx = True` and `cache_raw_data = True`, it will return a 5-tuple to be converted to a coordinate matrix;
    # otherwise, it will return a 2-tuple of adjacency matrix and the node list.
    graph_data = load(path.join(data_folder, edge_tuple_file),
                      data_format=data_format,
                      value_type_or_format_fun=weight_transform,
                      zero_based_index=zero_based_index,
                      squeeze=squeeze,
                      create_binary_cache=create_binary_cache,
                      compressed=binary_cache_compressed,
                      raw_data=cache_raw_data,
                      node_transform=node_transform,
                      cache_exists=_adj_and_nodes_cache_exists,
                      cache_save=_save_adj_and_nodes,
                      cache_load=_load_adj_and_nodes,
                      dtype=adjacency_dtype)

    if type(graph_data) is nx.Graph:
        graph_nodes = sorted(graph_data.nodes)
        adjacency_matrix = nx.adjacency_matrix(graph_data, nodelist=graph_nodes)
        graph_data = (adjacency_matrix, graph_nodes)

    toc('Graph nodes and adjacency matrix loaded from file with {} nodes'.format(len(graph_data[-1])))

    if len(graph_data) == 5:
        tic('Graph is loaded as tuples, converting them to a coordinate sparse matrix')
        col, row, weights, size, graph_nodes = graph_data
        adjacency_matrix = sp.coo_matrix((weights, (row, col)), shape=(size, size))
        toc('Graph tuples converted to a coordinate sparse matrix.')
        return adjacency_matrix, graph_nodes
    else:
        return graph_data


def load_graph_data(data_folder: str,
                    edge_tuple_file: str,
                    feature_file: str,
                    node_label_file: str,
                    mask_file: str,
                    train_ratio: float = 0.9,
                    validation_ratio: float = 0.0,
                    test_ratio=0.1,
                    node_transform=int,
                    weight_transform=float,
                    sparse_feature=True,
                    zero_based_index: bool = False,
                    squeeze=True,
                    create_binary_cache=True,
                    binary_cache_compressed=True,
                    cache_raw_data=False,
                    use_networkx=False,
                    normalize_features=True,
                    onehot_labels=True,
                    adjacency_dtype=np.float16):
    """
    Loads DBLP dataset.
    :param data_folder: the folder of the data files.
    :param edge_tuple_file: the name of the file that stores the edge tuples: the two values are the two nodes of an edge, and the third value is the weight of the edge.
    :param feature_file: the name of the file that stores features; if this parameter is `None`, then one-hot features will be generated.
    :param node_label_file: the name of the file providing the labels of the nodes.
    :param mask_file: the name of the file that defines the train/validation/test mask;
                1) if this parameter is `None`, then new mask will be randomly generated for nodes with labels;
                2) if the file is specified but does not exit, then new mask will be randomly generated for nodes with labels, and saved to a file with the provided name;
                NOTE all nodes without labels must be excluded from any mask (i.e. their corresponding value in the mask should be `False`).
    :param train_ratio: the percentage of training set, for generation of masks.
    :param validation_ratio: the percentage of validation set, for generation of masks.
    :param test_ratio: the percentage of test set, for generation of masks.
    :param weight_transform: specifies any weight transformation if necessary.
    :return: the loaded DBLP dataset as a tuple, including the adjacency matrix, the features, the labels, the training mask, the validation mask, and the test mask.
    """
    adjacency_matrix, graph_nodes = load_graph_adjacency(data_folder=data_folder,
                                                         edge_tuple_file=edge_tuple_file,
                                                         node_transform=node_transform,
                                                         weight_transform=weight_transform,
                                                         zero_based_index=zero_based_index,
                                                         squeeze=squeeze,
                                                         create_binary_cache=create_binary_cache,
                                                         binary_cache_compressed=binary_cache_compressed,
                                                         cache_raw_data=cache_raw_data,
                                                         use_networkx=use_networkx,
                                                         adjacency_dtype=adjacency_dtype)

    tic('Loading node labels')
    label_dict = load(path.join(data_folder, node_label_file),
                      data_format=SupportedDataFormats.DictTuples,
                      value_type_or_format_fun=int,
                      create_binary_cache=create_binary_cache,
                      compressed=binary_cache_compressed)

    if squeeze:
        labels = np.array([label_dict[key] if key in label_dict else 0 for key in graph_nodes])
    else:
        labels = np.array([label_dict[key] if key in label_dict else 0 for key in range(adjacency_matrix.shape[0])])
    toc("Node labels loaded")

    tic('Loading node features')
    feature_file = path.join(data_folder, feature_file) if feature_file is not None else None
    if feature_file is not None and path.exists(feature_file):
        features = np.array(load(feature_file,
                                 value_type_or_format_fun=float,
                                 create_binary_cache=create_binary_cache,
                                 compressed=binary_cache_compressed))
        if zero_based_index:
            features = features[np.array(graph_nodes), :]
        else:
            features = features[np.array(graph_nodes) - 1, :]
        if onehot_labels:
            onehot_labels = labels_to_onehot(labels)
        if normalize_features:
            features = row_normalize(features)
    else:
        features = labels_to_onehot(labels)
        if onehot_labels:
            onehot_labels = features.copy()
    toc('Node features loaded')

    mask_file = path.join(data_folder, mask_file if mask_file is not None else path.join('masks', "mask_{}.txt".format(int(time()))))
    labelled_nodes = np.array(list(set(graph_nodes) & set(label_dict.keys())))
    train_mask, validation_mask, test_mask = load_train_test_masks(mask_file=mask_file,
                                                                   size=len(labelled_nodes),
                                                                   train_ratio=train_ratio,
                                                                   validation_ratio=validation_ratio,
                                                                   test_ratio=test_ratio)
    train_mask = mask1d_by_index(mask_idxes=labelled_nodes[train_mask],
                                 size=len(features),
                                 ref_idxes=graph_nodes)
    validation_mask = mask1d_by_index(mask_idxes=labelled_nodes[validation_mask],
                                      size=len(features),
                                      ref_idxes=graph_nodes) if validation_mask is not None else None
    test_mask = mask1d_by_index(mask_idxes=labelled_nodes[test_mask],
                                size=len(features),
                                ref_idxes=graph_nodes)
    return adjacency_matrix, features, labels, onehot_labels, train_mask, validation_mask, test_mask


def load_data_gcn(data_folder: str, dataset_str: str):
    x = load(path.join(data_folder, dataset_str + '.x.dat'), encoding='latin1')
    y = load(path.join(data_folder, dataset_str + '.y.dat'), encoding='latin1')
    tx = load(path.join(data_folder, dataset_str + '.tx.dat'), encoding='latin1')
    ty = load(path.join(data_folder, dataset_str + '.ty.dat'), encoding='latin1')
    allx = load(path.join(data_folder, dataset_str + '.allx.dat'), encoding='latin1')
    ally = load(path.join(data_folder, dataset_str + '.ally.dat'), encoding='latin1')
    graph = load(path.join(data_folder, dataset_str + '.graph.dat'), encoding='latin1')
    test_idx_reorder = load(path.join(data_folder, dataset_str + '.test.index.txt'), value_type_or_format_fun=int)

    test_idx_begin, test_idx_end = min(test_idx_reorder), max(test_idx_reorder) + 1
    test_count = test_idx_end - test_idx_begin
    features = sp.vstack((allx, sp.lil_matrix((test_count, x.shape[1])))).tolil()
    features[test_idx_reorder, :] = tx
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, np.zeros((test_count, y.shape[1]))))
    labels[test_idx_reorder, :] = ty

    train_mask = mask1d_by_index(range(len(y)), labels.shape[0])
    val_mask = mask1d_by_index(range(len(y), len(y) + 500), labels.shape[0])
    test_mask = mask1d_by_index(range(test_idx_begin, test_idx_end), labels.shape[0])

    return adj, features, labels, train_mask, val_mask, test_mask


if __name__ == '__main__':
    test_code = int(input("Input the test code:"))
    if test_code == 0:
        load_data_dblp(data_folder='dblp', edge_tuple_file='AA.txt', feature_file='A_features', mask_file="AA_mask.txt", node_label_file='author_label.txt')
    elif test_code == 1:
        load_data_gcn(data_folder='gcn', dataset_str='citeseer')
