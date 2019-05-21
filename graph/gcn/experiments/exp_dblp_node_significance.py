from data.data_load import *
from data.exp_common import *
from sklearn.metrics import *
from graph.gcn.model import GcnTf
from util.general_ext import *
from util.path_ext import *
from matplotlib import pyplot
import tensorflow as tf
from itertools import product

dblp_folder = path.join(path.dirname(path.abspath(__file__)), 'dblp')
sys.stdout = flogger(log_file_path(dblp_folder))

weighted_convolution = True
hidden_dims = (512,)
reg_lambda = 5e-7
learning_rate = 5e-4
train_drop_out = 0.5
max_iter = 5000

repeat = 3
early_stop_lookback = 0
normalize_features = True
break_after_first_iter = True
use_one_hot_features = False
use_specified_mask = False
save_embeddings = True
plot_plan = 2

adjacency_matrix, features, labels, onehot_labels, train_mask, validation_mask, test_mask = \
    load_graph_data(data_folder=dblp_folder,
                    edge_tuple_file='AA.txt',
                    feature_file='A_features.txt' if not use_one_hot_features else None,
                    mask_file='A_masks.txt' if use_specified_mask else None,
                    node_label_file='author_label.txt',
                    train_ratio=0.2,
                    validation_ratio=0.4,
                    test_ratio=0.4,
                    node_transform=int,
                    weight_transform=float,
                    zero_based_index=False,
                    squeeze=True,
                    create_binary_cache=True,
                    binary_cache_compressed=False,
                    cache_raw_data=False,
                    use_networkx=False,
                    normalize_features=normalize_features,
                    adjacency_dtype=np.float16)

test_true_labels = labels[test_mask]
test_label_set = list(set(test_true_labels))

gcn = GcnTf(adjacency_matrices=adjacency_matrix,
            input_dim=features.shape[1],
            output_dim=onehot_labels.shape[1],
            hidden_dims=hidden_dims,
            sparse_feature=False,
            layer_dropouts=tf.placeholder_with_default(0., shape=()),
            weighted_convolution=weighted_convolution,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda)

batch_data, test_outputs, _ = gcn.train(features=features,
                                        labels=onehot_labels,
                                        train_mask=train_mask,
                                        validation_mask=validation_mask,
                                        test_mask=test_mask,
                                        num_features_nonzero=0,
                                        train_dropouts=train_drop_out,
                                        max_iter=max_iter,
                                        early_stop_lookback=early_stop_lookback)

predicted_labels = gcn.predict(features, num_features_nonzero=0, test_mask=test_mask, argmax=True)

result_time_stamp = time_stamp_for_filename()

if weighted_convolution:
    if plot_plan == 1:
        for layer_idx, layer in enumerate(gcn.model):
            V = gcn.sess.run(layer.V, feed_dict=batch_data)
            pyplot.hist(V, bins=20)
            pyplot.title('Layer {} Author Self-Significance'.format(layer_idx), fontsize=16)
            pyplot.xlabel('Node Self-Significance', fontsize=14)
            pyplot.ylabel('Number of Nodes', fontsize=14)
            pyplot.tight_layout()
            pyplot.savefig(path.join(dblp_folder, 'results', 'V_hist_{}_{}.png'.format(layer_idx, result_time_stamp)), dpi=600)
            pyplot.clf()
            pyplot.hist(V[:500], bins=20)
            pyplot.title('Layer {} First 500 Author Self-Significance'.format(layer_idx), fontsize=16)
            pyplot.xlabel('Node Self-Significance', fontsize=14)
            pyplot.ylabel('Number of Nodes', fontsize=14)
            pyplot.tight_layout()
            pyplot.savefig(path.join(dblp_folder, 'results', 'V_hist_first_{}_{}.png'.format(layer_idx, result_time_stamp)), dpi=600)
            pyplot.clf()
            pyplot.hist(V[-500:-1, ], bins=20)
            pyplot.title('Layer {} Last 500 Author Self-Significance'.format(layer_idx), fontsize=16)
            pyplot.xlabel('Node Self-Significance', fontsize=14)
            pyplot.ylabel('Number of Nodes', fontsize=14)
            pyplot.tight_layout()
            pyplot.savefig(path.join(dblp_folder, 'results', 'V_hist_last_{}_{}.png'.format(layer_idx, result_time_stamp)), dpi=600)
            pyplot.clf()
            node_weight_ranks = np.hstack((np.sort(V, axis=0)[::-1], np.argsort(V, axis=0)[::-1]))
            np.savetxt(fname=path.join(dblp_folder, 'results', 'V_{}_{}.txt'.format(layer_idx, result_time_stamp)),
                       X=node_weight_ranks,
                       fmt='%f %i')
    elif plot_plan == 2:
        V = gcn.sess.run(gcn.model[0].V, feed_dict=batch_data)

sys.stdout.reset()
