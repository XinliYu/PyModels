from data.data_load import *
from data.data_common import *
from graph.gcn.model import GcnTf
from util.print_ext import *
from util.path_ext import *
import tensorflow as tf
from itertools import product

dblp_folder = path.join(path.dirname(path.abspath(__file__)), 'dblp')
sys.stdout = flogger(output_file_name(dblp_folder))
weighted_convolution_trials = [False, False]
hidden_dims_trials = [(512,), (512, 256), (256,), (256, 64)]
reg_lambda_trials = [1e-6, 1e-5, 5e-6, 1e-5]
learning_rates_trials = [1e-3, 1e-3, 5e-4, 1e-5]
train_drop_outs_trials = [0.5, 0.7, 0.3, 0.1]
max_iters_trials = [2500, 1000]
repeat = 1
normalize_feature = False
break_after_first_iter = True
use_one_hot_features = False
use_specified_mask = True
save_embeddings = True

adjacency_matrix, features, labels, train_mask, validation_mask, test_mask, mask_file = \
    load_data_dblp(data_folder=dblp_folder,
                   edge_tuple_file='AA.txt',
                   feature_file='A_features.txt' if not use_one_hot_features else None,
                   mask_file='A_masks.txt' if use_specified_mask else None,
                   node_label_file='author_label.txt',
                   train_ratio=0.8,
                   validation_ratio=0.1,
                   test_ratio=0.1)
print("Using mask file " + mask_file)
labels = labels_to_onehot(labels)
if normalize_feature:
    features = row_normalize(features)

best_accuracy = 0
best_accuracy_paras = None
AX = None
for weighted_convolution, hidden_dims, reg_lambda, learning_rate, train_drop_out, max_iter in product(
        weighted_convolution_trials,
        hidden_dims_trials,
        reg_lambda_trials,
        learning_rates_trials,
        train_drop_outs_trials,
        max_iters_trials):
    results = []
    for _ in range(repeat):
        gcn = GcnTf(adjacency_matrices=adjacency_matrix,
                    input_dim=features.shape[1],
                    output_dim=labels.shape[1],
                    hidden_dims=hidden_dims,
                    sparse_feature=False,
                    layer_dropouts=tf.placeholder_with_default(0., shape=()),
                    weighted_convolution=weighted_convolution,
                    learning_rate=learning_rate,
                    reg_lambda=reg_lambda)

        test_outputs, batch_data = gcn.train(features=features,
                                             labels=labels,
                                             train_mask=train_mask,
                                             validation_mask=validation_mask,
                                             test_mask=test_mask,
                                             num_features_nonzero=0,
                                             train_dropouts=train_drop_out,
                                             max_iter=max_iter,
                                             early_stop_lookback=0)
        results.append(test_outputs[1])

        if save_embeddings:
            embedding_collection = gcn.eval_embeddings(batch_data)
            for layer_idx, embeddings in enumerate(embedding_collection):
                accu_str = str(int(results[-1] * 10000))
                if embeddings[0].shape != ():
                    np.savetxt(output_file_name(dblp_folder, extension='_layer{}_accu{}_activatedAVXW.csv'.format(layer_idx, accu_str)), embeddings[0])
                np.savetxt(output_file_name(dblp_folder, extension='_layer{}_accu{}_AVXW.csv'.format(layer_idx, accu_str)), embeddings[1])
                if embeddings[2].shape != ():
                    np.savetxt(output_file_name(dblp_folder, extension='_layer{}_accu{}_AVX.csv'.format(layer_idx, accu_str)), embeddings[2])
                np.savetxt(output_file_name(dblp_folder, extension='_layer{}_accu{}_W.csv'.format(layer_idx, accu_str)), embeddings[3])
        tf.reset_default_graph()
    cur_accuracy = np.mean(results)
    best_accuracy = max(best_accuracy, cur_accuracy)
    cur_msg = 'max_iter:{}, weighted convolution:{}, hidden dimensions:{}, reg lambda:{}, learning rate:{}, dropout:{}, accuracy:{:.5f}, best accuracy:{:.5f}'.format(max_iter, weighted_convolution, hidden_dims, reg_lambda, learning_rate, train_drop_out, cur_accuracy, best_accuracy)
    if best_accuracy == cur_accuracy:
        best_accuracy_paras = cur_msg
    print(cur_msg)
    sys.stdout.flush()

    if break_after_first_iter:
        break

print('\n')
print('-------------best result-------------')
print(best_accuracy_paras)

sys.stdout.reset()
