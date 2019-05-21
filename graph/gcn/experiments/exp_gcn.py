from data.data_load import *
from data.exp_common import *
from graph.gcn.model import GcnTf
import tensorflow as tf

adjacency_matrix, features, labels, train_mask, validation_mask, test_mask = load_data_gcn(data_folder='gcn', dataset_str='citeseer')
features = sparse_row_normalize(features)

gcn = GcnTf(adjacency_matrices=adjacency_matrix,
            input_dim=features[2][1],
            output_dim=labels.shape[1],
            hidden_dims=(32,),
            layer_dropouts=tf.placeholder_with_default(0., shape=()))

gcn.train(features=features,
          labels=labels,
          train_mask=train_mask,
          validation_mask=validation_mask,
          test_mask=test_mask,
          num_features_nonzero=features[1].shape,
          train_dropouts=0.3,
          max_iter=300,
          early_stop_lookback=0)
