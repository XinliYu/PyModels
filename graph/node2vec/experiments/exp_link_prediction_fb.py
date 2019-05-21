from graph.common import edge_train_test_sample, link_prediction_by_node_embeddings, EdgeSampleAlgorithms
from graph.node2vec.model import Node2VecGraph
from gensim.models import Word2Vec
import numpy as np
import networkx as nx
import pickle


def load_fb_network(ego_user=0):
    network_dir = '../../../data/fb-processed/{0}-adj-feat.pkl'.format(ego_user)
    with open(network_dir, 'rb') as f:
        adj, features = pickle.load(f, encoding='latin1')

    return nx.Graph(adj)


g = load_fb_network()
np.random.seed(0)  # make sure train-test split is roughly consistent for each run
adjacency_matrix = nx.to_scipy_sparse_matrix(g)

# Perform train-test split
train_edges, negative_train_edges, \
validation_edges, negative_validation_edges, \
test_edges, negative_test_edges \
    = edge_train_test_sample(adjacency_matrix, train_frac=0.6, val_frac=0, test_frac=0.4, false_rate=1.0, return_weights=False, algorithm=EdgeSampleAlgorithms.RandomSample)

g_train = nx.Graph()
g_train.add_edges_from(train_edges)

p, q = 1, 1
window_size = 10  # skip-gram window size
num_walks = 10  # number of random walks per node
walk_length = 80  # length of random walks
embedding_dims = 128  # Embedding dimension
workers = 8  # number of parallel workers
epochs = 1  # SGD epochs

g_n2v = Node2VecGraph(g_train, p, q, directed=False, weighted=False)  # create node2vec graph instance
walks = g_n2v.sample_node2vec_walks(num_walks, walk_length)
walks = [list(map(str, walk)) for walk in walks]

# trains the skip-gram model
model = Word2Vec(walks, size=embedding_dims, window=window_size, min_count=0, sg=1, workers=workers, iter=epochs)

# store embeddings
embedding_dict = model.wv

node_embeddings = []
for node_index in range(0, adjacency_matrix.shape[0]):
    node_str = str(node_index)
    if node_str in embedding_dict:
        node_emb = embedding_dict[node_str]
        node_embeddings.append(node_emb)
    else:
        node_embeddings.append(np.zeros(embedding_dims))
node_embeddings = np.vstack(node_embeddings)

_, test_results = link_prediction_by_node_embeddings(positive_train_edges=train_edges,
                                                     negative_train_edges=negative_train_edges,
                                                     node_embeddings=node_embeddings,
                                                     positive_test_edges=test_edges,
                                                     negative_test_edges=negative_test_edges)

print(test_results[1])
