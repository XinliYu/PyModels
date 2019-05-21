from os import path

import networkx as nx

from data.data_load import load_graph_adjacency, load
from graph.common import edge_train_test_sample, link_prediction_by_node_embeddings, EdgeSampleAlgorithms
from info import data_folder_path
import numpy as np

from util.general_ext import highlight_print

dblp_folder = path.join(data_folder_path, 'dblp')

adjacency_matrix, graph_nodes = load_graph_adjacency(data_folder=dblp_folder,
                                                     edge_tuple_file='AA.txt',
                                                     node_transform=int,
                                                     weight_transform=float,
                                                     zero_based_index=False,
                                                     squeeze=True,
                                                     create_binary_cache=True,
                                                     binary_cache_compressed=False,
                                                     cache_raw_data=False,
                                                     use_networkx=False,
                                                     adjacency_dtype=np.float16)
train_edges, negative_train_edges, \
validation_edges, negative_validation_edges, \
test_edges, negative_test_edges \
    = edge_train_test_sample(adjacency_matrix, train_frac=0.8, val_frac=0, test_frac=0.2, false_rate=1.0, return_weights=False, algorithm=EdgeSampleAlgorithms.RandomSample)

embedding_files = ("dblp_A_line1_vec.txt", "dblp_A_line2_vec.txt", "dblp_A_node2vec_vectors.txt",
                   path.join("..", "A_features.txt"), "dblp_A_gcn_vector.csv",
                   "dblp_A_gcn_AVX0.csv", "dblp_A_gcn_AVX1.csv", "dblp_A_gcn_AVXW0.csv", "dblp_A_gcn_AVXW1.csv", "dblp_A_gcn_activatedAVXW0.csv")
skip_first_line_flags = (True, True, True, False, False, False, False, False, False, False, False, False)
first_element_index_flags = (True, True, True, False, False, False, False, False, False, False, False, False)
embedding_mergers = (('Mean', lambda x, y: x + y / 2), ('Hadamard', np.multiply), ('L1', lambda x, y: np.abs(x - y)), ('L2', lambda x, y: np.square(x - y)))

for embedding_file, skip_first_line, first_element_index in zip(embedding_files, skip_first_line_flags, first_element_index_flags):
    embedding_file_path = path.join(dblp_folder, "embeddings", embedding_file)
    node_embeddings = load(embedding_file_path, value_type_or_format_fun=float, skip_first_line=skip_first_line)
    if first_element_index:
        node_embeddings.sort()
    node_embeddings = np.array(node_embeddings)
    if first_element_index:
        node_embeddings = node_embeddings[:, 1:]

    for embedding_merger in embedding_mergers:
        _, test_results = link_prediction_by_node_embeddings(positive_train_edges=train_edges,
                                                             negative_train_edges=negative_train_edges,
                                                             node_embeddings=node_embeddings,
                                                             node_embedding_merger=embedding_merger[1],
                                                             positive_test_edges=test_edges,
                                                             negative_test_edges=negative_test_edges)
        highlight_print('{} ({})'.format(embedding_file, embedding_merger[0]), test_results[1])
