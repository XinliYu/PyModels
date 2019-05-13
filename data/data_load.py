from os import path
import gzip
import sys
import time
import networkx as nx
import scipy.sparse as sp
from enum import Enum
from data.data_util import *


# methods for loading data from different data sets
# methods

class SupportedDataFormats(Enum):
    DelimiterSeparatedValues = 0,
    GraphEdgeTuples = 1,
    DictTuples = 2,
    TrainTestMasks = 3


def load_f(f, add_data, delimiter=None, value_type_or_format_fun=float):
    def _try_convert(entry):
        try:
            return value_type_or_format_fun(entry)
        except:
            return entry

    if delimiter is None or delimiter == '':
        if value_type_or_format_fun in (int, float, str, bool):
            for line in f:
                splits = line.split()
                if len(splits) == 1:
                    add_data(_try_convert(splits[0]))
                else:
                    add_data([_try_convert(cell) for cell in splits])
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
        else:
            for line in f:
                add_data(value_type_or_format_fun(line.split(delimiter)))


def load_csv(f, delimiter=None, value_type_or_format_fun=float):
    data = []
    load_f(f=f, add_data=data.append,
           delimiter=delimiter,
           value_type_or_format_fun=value_type_or_format_fun)
    return data


def load_graph_from_edge_tuples(f, delimiter=None, value_type_or_format_fun=int, weight_transform=None):
    G = nx.Graph()

    def _add_graph_edge(tups):
        if len(tups) == 2:
            G.add_edge(tups[0], tups[1])
        else:
            G.add_edge(tups[0], tups[1],
                       weight=tups[2] if weight_transform is None else weight_transform(tups[2]))

    load_f(f=f, add_data=_add_graph_edge,
           delimiter=delimiter,
           value_type_or_format_fun=value_type_or_format_fun)

    return G


def load_dict_from_tuples(f, delimiter=None, value_type_or_format_fun=int):
    d = {}

    def _add_dict_entry(tups):
        if len(tups) == 2:
            d.update({tups[0]: tups[1]})
        else:
            d.update({tups[0]: tups[1:]})

    load_f(f=f, add_data=_add_dict_entry,
           delimiter=delimiter,
           value_type_or_format_fun=value_type_or_format_fun)

    return d


def load_train_test_masks(size, mask_file=None, encoding: str = None, delimiter=None, train_ratio=0.9, validation_ratio=0.0, test_ratio=0.1):
    if mask_file is not None and path.exists(mask_file):
        masks = np.array(load(mask_file, encoding=encoding, delimiter=delimiter, data_format=SupportedDataFormats.DelimiterSeparatedValues, value_type_or_format_fun=bool))
        if masks.shape[1] == 2:
            train_mask, test_mask = masks[:, 0], masks[:, 1]
            validation_mask = None
        else:
            train_mask, validation_mask, test_mask = masks[:, 0], masks[:, 1], masks[:, 2]
    else:
        if validation_ratio == test_ratio == 0:
            test_ratio = 1 - train_ratio
        masks = mask_by_percentage(size, percents=(train_ratio, validation_ratio, test_ratio))
        train_mask, validation_mask, test_mask = masks[0], masks[1], masks[2]

        if mask_file is not None:
            save_lists(file_path=mask_file, lists=masks, encoding=encoding)
    return train_mask, validation_mask, test_mask


def load(data_file: str, encoding: str = None, delimiter=None, data_format: SupportedDataFormats = SupportedDataFormats.DelimiterSeparatedValues, value_type_or_format_fun=None, compressed: bool = False, **kwargs):
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

    if data_file.endswith('.txt'):
        with open(data_file, 'r') if encoding is None \
                else codecs.open(data_file, encoding=encoding) as f:
            if data_format == SupportedDataFormats.GraphEdgeTuples:
                return load_graph_from_edge_tuples(f=f,
                                                   delimiter=delimiter,
                                                   value_type_or_format_fun=value_type_or_format_fun,
                                                   weight_transform=kwargs.get('weight_transform'))
            elif data_format == SupportedDataFormats.DictTuples:
                return load_dict_from_tuples(f=f,
                                             delimiter=delimiter,
                                             value_type_or_format_fun=value_type_or_format_fun)
            else:
                return load_csv(f, delimiter, value_type_or_format_fun)

    else:
        with open(data_file, 'rb') if not compressed else gzip.open(data_file, 'rb') as f:
            if encoding is None or sys.version_info < (3, 0):
                return pkl.load(f)
            else:
                return pkl.load(f, encoding=encoding)


def save(data_file: str, data, compressed: bool = False):
    if data_file.endswith('.dat'):
        with open(data_file, 'wb') if not compressed else gzip.open(data_file, 'wb') as f:
            pkl.dump(data, f)


def load_data_dblp(data_folder: str,
                   edge_tuple_file: str,
                   feature_file: str,
                   node_label_file: str,
                   mask_file: str,
                   train_ratio: float = 0.9,
                   validation_ratio: float = 0.0,
                   test_ratio=0.1,
                   weight_transform=None,
                   sparse_feature=True):
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
    graph = load(path.join(data_folder, edge_tuple_file), data_format=SupportedDataFormats.GraphEdgeTuples, value_type_or_format_fun=int, weight_transform=weight_transform)
    graph_nodes = sorted(graph.nodes)
    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=graph_nodes)
    label_dict = load(path.join(data_folder, node_label_file), data_format=SupportedDataFormats.DictTuples, value_type_or_format_fun=int)
    labels = np.array([label_dict[key] if key in label_dict else 0 for key in graph_nodes])

    feature_file = path.join(data_folder, feature_file) if feature_file is not None else None
    if feature_file is not None and path.exists(feature_file):
        features = np.array(load(feature_file, value_type_or_format_fun=float))
        features = features[np.array(graph_nodes) - 1, :]
    else:
        features = labels_to_onehot(labels)

    mask_file = path.join(data_folder, mask_file if mask_file is not None else "mask_{}.txt".format(str(int(time.time()))))
    labelled_nodes = np.array(list(set(graph.nodes) & set(label_dict.keys())))
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
    return adjacency_matrix, features, labels, train_mask, validation_mask, test_mask, mask_file


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
