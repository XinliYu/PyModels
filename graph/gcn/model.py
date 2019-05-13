from basic.tf.layer import *
from data.data_util import *
from scipy.sparse.linalg.eigen.arpack import eigsh
from typing import Tuple


def _normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def approx_graph_conv(adjacency_matrix):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    return sparse_to_tuple(_normalize_adj(adjacency_matrix + sp.eye(adjacency_matrix.shape[0])))


def approx_graph_conv_k(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    adj_normalized = _normalize_adj(adj)
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

    for i in range(len(t_k)):
        t_k[i] = sparse_to_tuple(t_k[i])

    return t_k


class GcnLayerTf(LayerTf):
    """Graph convolution layer."""

    def __init__(self, node_count: int, input_dim: int, output_dim: int,
                 convolutions_ph,
                 num_nonzero_features_ph=None, featureless: bool = False, weighted_convolution=False,
                 dropout=0., activation=tf.nn.relu, bias=False,
                 name: str = None, logging=False):
        """
        Defines one layer in a graph convolutional (GCN) network. A graph convolutional layer is defined as `activation(∑_i(A^hat_i)XW)`,
            where each `A^hat_i=(D_tilde^(-1/2))A^tilde_i(D_tilde^(-1/2))` represents one convolution,
            `X` are the input features for the current layer with each row as one feature,
            and `W` are the weights over features.
        :param node_count:
        :param input_dim: the dimension of each input feature; also the number of columns `X`.
        :param output_dim: the output dimension of the feature linear transformation `XW`.
        :param convolutions_ph: the place holders for the graph convolutions.
        :param num_nonzero_features_ph:
        :param featureless:
        :param weighted_convolution:
        :param dropout:
        :param activation: the activation function for this GCN layer. By default it is ReLU.
        :param bias:
        :param name:
        :param logging:
        """
        super(GcnLayerTf, self).__init__(name, logging)

        self.dropout = dropout
        self.activation = activation
        self.convolutions_ph = convolutions_ph
        self.num_nonzero_features = num_nonzero_features_ph
        self.featureless = featureless
        self.weighted_convolutions = weighted_convolution
        self.has_bias = bias

        with tf.variable_scope(self.variable_space_name()):
            if type(self.convolutions_ph) is list:
                self._convolution_count = len(self.convolutions_ph)
                if self._convolution_count == 1:
                    self.W = glorot_init([input_dim, output_dim], name='W')
                    self.V = tf.Variable(tf.ones((node_count, 1)), name='V', dtype=tf.float32)
                else:
                    self.W = [None] * self._convolution_count
                    for i in range(self._convolution_count):
                        self.W[i] = glorot_init([input_dim, output_dim], name='W_' + str(i))
                    if self.weighted_convolutions:
                        self.V = [None] * self._convolution_count
                        for i in range(self._convolution_count):
                            self.V[i] = tf.Variable(tf.ones((node_count, 1)), name='V_' + str(i), dtype=tf.float32)  # TODO share V?
            else:
                self._convolution_count = 1
                self.W = glorot_init([input_dim, output_dim], name='W')
                self.V = tf.Variable(tf.ones((node_count, 1)), name='V', dtype=tf.float32)

            if self.has_bias:  # shared bias
                self.bias = tf.Variable(tf.zeros(output_dim, dtype=tf.float32), name="bias")

            self.log_vars()

    def forward(self, x, train_mode=True):

        self.sparse_x = type(x) is tf.SparseTensor

        # region dropout

        # need different functions to treat a sparse tensor and a non-sparse tensor
        if train_mode:
            x = sparse_dropout(x, self.dropout, self.num_nonzero_features) if self.sparse_x \
                else tf.nn.dropout(x, rate=self.dropout)

        # endregion

        AVXW = self._embeddingsAVXW(x)
        AVX = self._embeddingsAVX(x)

        if self.activation:
            activated_AVXW = self.activation(AVXW)
            self.embeddings = [activated_AVXW, AVXW, AVX, self.W]
            return activated_AVXW
        else:
            self.embeddings = [tf.constant(False), AVXW, AVX, self.W]
            return AVXW

    @staticmethod
    def _dot(_x, _y, _sparse: bool):
        if _sparse:
            res = tf.sparse_tensor_dense_matmul(_x, _y)
        else:
            res = tf.matmul(_x, _y)
        return res

    def _embeddingsAVXW(self, x):
        self.sparse_x = type(x) is tf.SparseTensor
        if self._convolution_count == 1:
            if not self.featureless:
                pre_sup = self._dot(x, self.W, self.sparse_x)
            else:
                pre_sup = self.W
            if self.weighted_convolutions:
                pre_sup = tf.multiply(self.V, pre_sup)
            output = self._dot(self.convolutions_ph, pre_sup, True)
        else:
            tmp = []
            for i in range(len(self.convolutions_ph)):
                if not self.featureless:
                    pre_sup = self._dot(x, self.W[i], self.sparse_x)
                else:
                    pre_sup = self.W[i]
                if self.weighted_convolutions:
                    pre_sup = tf.multiply(self.V[i], pre_sup)
                support = self._dot(self.convolutions_ph[i], pre_sup, True)
                tmp.append(support)
            output = tf.add_n(tmp)

        if self.has_bias:
            output += self.bias
        return output

    def _embeddingsAVX(self, x):
        if not self.weighted_convolutions:
            return tf.constant(False)

        self.sparse_x = type(x) is tf.SparseTensor

        if self._convolution_count == 1:
            pre_sup = tf.multiply(self.V, x)
            output = self._dot(self.convolutions_ph, pre_sup, True)
        else:
            tmp = []
            for i in range(len(self.convolutions_ph)):
                pre_sup = tf.multiply(self.V[i], x)
                support = self._dot(self.convolutions_ph[i], pre_sup, True)
                tmp.append(support)
            output = tf.add_n(tmp)

        return output


class GcnTf(ModelTf):
    """
    Defines the graph convolutional network as in **Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016)**.
    """

    def __init__(self, adjacency_matrices, input_dim, output_dim, hidden_dims: Tuple = (16,),
                 sparse_feature=True, convolution_order: int = 1,
                 loss_fun=None, optimizer=None, metric_fun=None,
                 layer_dropouts: Tuple = None,
                 layer_bias: Tuple = None,
                 layer_activations: Tuple = None,
                 layer_names: Tuple = None,
                 layer_loggings: Tuple = None,
                 weighted_convolution=None,
                 reg_lambda=None,
                 learning_rate: float = 1e-2, name: str = None):

        assert convolution_order >= 1, "Convolution order must be equal to or larger than 1."
        self.node_count = adjacency_matrices.shape[0]
        if type(adjacency_matrices) is list:
            if len(adjacency_matrices) > 1:
                self.convolution_values = []
                self.convolutions_ph = []
                for adj in adjacency_matrices:
                    self.convolution_values.append(approx_graph_conv(adj) if convolution_order == 1 else approx_graph_conv_k(adj, k=convolution_order))
                    self.convolutions_ph.append(tf.sparse_placeholder(tf.float32))
            else:
                self.convolution_values = approx_graph_conv(adjacency_matrices[0]) if convolution_order == 1 else approx_graph_conv_k(adjacency_matrices[0], k=convolution_order)
                self.convolutions_ph = tf.sparse_placeholder(tf.float32)
        else:
            self.convolution_values = approx_graph_conv(adjacency_matrices) if convolution_order == 1 else approx_graph_conv_k(adjacency_matrices, k=convolution_order)
            self.convolutions_ph = tf.sparse_placeholder(tf.float32)

        if sparse_feature:
            self.features_ph = tf.sparse_placeholder(tf.float32, shape=tf.constant((self.node_count, input_dim), dtype=tf.int64))
        else:
            self.features_ph = tf.placeholder(tf.float32, shape=(self.node_count, input_dim))

        self.target_ph = tf.placeholder(tf.float32, shape=(None, output_dim))

        self.train_mask_ph = tf.placeholder(tf.int32)
        self.num_nonzero_features_ph = tf.placeholder(tf.int32)
        self.layer_dropouts = layer_dropouts
        self.layer_bias = layer_bias
        self.layer_activations = layer_activations
        self.weighted_convolution = weighted_convolution
        self.layer_names = layer_names
        self.layer_loggings = layer_loggings
        self.reg_lambda = 5e-4 if reg_lambda is None else reg_lambda

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif callable(optimizer):
            optimizer = optimizer(learning_rate=learning_rate)

        super(GcnTf, self).__init__(input_ph=self.features_ph,
                                    target_ph=self.target_ph,
                                    loss_fun=loss_fun,
                                    optimizer=optimizer,
                                    name=name,
                                    metric_fun=metric_fun,
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    hidden_dims=hidden_dims)

    def forward(self, x):
        layer_count = len(self.dimensions) - 1
        layers = [None] * layer_count
        for i in range(layer_count):
            layers[i] = GcnLayerTf(node_count=self.node_count,
                                   input_dim=self.dimensions[i],
                                   output_dim=self.dimensions[i + 1],
                                   convolutions_ph=self.convolutions_ph,
                                   num_nonzero_features_ph=self.num_nonzero_features_ph,
                                   featureless=self.dimensions[0] == 0,
                                   activation=(tf.nn.relu if i != layer_count - 1 else None) if self.layer_activations is None else take_element_if_list(self.layer_activations, i),
                                   dropout=0.0 if self.layer_dropouts is None else take_element_if_list(self.layer_dropouts, i),
                                   weighted_convolution=False if self.weighted_convolution is None else (self.weighted_convolution[i] if isinstance(self.weighted_convolution, list) else self.weighted_convolution),
                                   bias=False if self.layer_bias is None else take_element_if_list(self.layer_bias, i),
                                   name=None if self.layer_names is None else self.layer_names[i],
                                   logging=False if self.layer_loggings is None else take_element_if_list(self.layer_loggings, i))
        return layers

    def batch_data(self, features, labels, num_features_nonzero, mask, dropouts=None):
        return {self.convolutions_ph: self.convolution_values,
                self.features_ph: features,
                self.target_ph: labels,
                self.train_mask_ph: mask,
                self.num_nonzero_features_ph: num_features_nonzero,
                self.layer_dropouts: 0.0 if dropouts is None else dropouts} if mask is not None else None

    def train(self, features, labels, train_mask, test_mask, validation_mask=None, num_features_nonzero=None, train_dropouts=None,
              stop_loss: float = 1e-6, max_iter: int = 1000, verbose=True, print_interval=10, validation_data=None, early_stop_lookback=5):
        return super(GcnTf, self).train(batch_data=self.batch_data(features=features, labels=labels, num_features_nonzero=num_features_nonzero, mask=train_mask, dropouts=train_dropouts),
                                        stop_loss=stop_loss,
                                        max_iter=max_iter,
                                        verbose=verbose,
                                        print_interval=print_interval,
                                        validation_data=self.batch_data(features=features, labels=labels, num_features_nonzero=num_features_nonzero, mask=validation_mask),
                                        test_data=self.batch_data(features=features, labels=labels, num_features_nonzero=num_features_nonzero, mask=test_mask),
                                        early_stop_lookback=early_stop_lookback)


    def default_loss(self):
        loss_fun = 0
        train_vars = tf.trainable_variables()
        if type(self.reg_lambda) in (int, float):
            if self.reg_lambda != 0:
                for var in train_vars:
                    loss_fun += self.reg_lambda * tf.nn.l2_loss(var)
        else:
            var_count = len(train_vars)
            lambdas_count = len(self.reg_lambda)
            for i in range(min(var_count, lambdas_count)):
                if self.reg_lambda[i] != 0:
                    loss_fun += self.reg_lambda[i] * tf.nn.l2_loss(train_vars[i])
        loss_fun += masked_cross_entropy(estimated_scores=self.outputs, label_probabilities=self.target_ph, mask=self.train_mask_ph)
        return loss_fun

    def default_metric(self):
        return masked_accuracy(estimated_scores=self.outputs, ground_truth_label_scores=self.target_ph, mask=self.train_mask_ph)

    def predict(self):
        return tf.nn.softmax(self.outputs)
