from data.data_load import *
from data.exp_common import *
from sklearn.metrics import *
from graph.gcn.model import GcnTf
from info import data_folder_path
from util.general_ext import *
from util.path_ext import *
import tensorflow as tf
from itertools import product

dblp_folder = path.join(data_folder_path, 'dblp')
sys.stdout = flogger(log_file_path(dblp_folder))

weighted_convolution_trials = [True, True]
hidden_dims_trials = [(512,), (512, 256), (256,), (256, 64)]
reg_lambda_trials = [1e-6, 1e-5, 5e-6, 1e-5]
learning_rates_trials = [1e-3, 1e-3, 5e-4, 1e-5]
train_drop_outs_trials = [0.5, 0.7, 0.3, 0.1]
max_iters_trials = [3000, 1000]

repeat = 1
early_stop_lookback = 30
normalize_features = False
break_after_first_iter = True
use_one_hot_features = False
use_specified_mask = True
save_embeddings = True

adjacency_matrix, features, labels, onehot_labels, train_mask, validation_mask, test_mask = \
    load_graph_data(data_folder=dblp_folder,
                    edge_tuple_file='AA.txt',
                    feature_file='A_features.txt' if not use_one_hot_features else None,
                    mask_file='A_masks.txt' if use_specified_mask else None,
                    node_label_file='author_label.txt',
                    train_ratio=0.9,
                    validation_ratio=0,
                    test_ratio=0.1,
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


def _gcn_model_run(para_set):
    weighted_convolution, hidden_dims, reg_lambda, learning_rate, train_drop_out, max_iter = para_set
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
    return gcn, batch_data, test_outputs[0], test_outputs[1], predicted_labels


experiment(exp_name='dblp_gcn',
           para_tune_iter=product(
               weighted_convolution_trials,
               hidden_dims_trials,
               reg_lambda_trials,
               learning_rates_trials,
               train_drop_outs_trials,
               max_iters_trials),
           para_names=('weighted convolution',
                       'hidden dimensions',
                       'regularization penalty',
                       'learning rate',
                       'dropout',
                       'max iter'),
           model_run=_gcn_model_run,
           metrics=(lambda **kwargs: kwargs.get('test_metric'),
                    lambda **kwargs: f1_score(y_true=test_true_labels, y_pred=kwargs.get('test_predictions'), labels=test_true_labels, average='micro'),
                    lambda **kwargs: f1_score(y_true=test_true_labels, y_pred=kwargs.get('test_predictions'), labels=test_true_labels, average='macro')),
           metric_names=('accuracy', 'f1_micro', 'f1_macro'),
           post_model_run=lambda: tf.reset_default_graph(),
           model_run_repeat=repeat,
           break_after_first_para_set=break_after_first_iter,
           exp_saves_folder=path.join(dblp_folder, 'embeddings'),
           get_exp_saves=(lambda model, batch_data: model.eval_embeddings(batch_data)) if save_embeddings else None,
           exp_save_names=('layer', 'activatedAVXW', 'AVXW', 'AVX', 'W'),
           save_fun=np.savetxt,
           exp_save_file_ext='csv')

sys.stdout.reset()
