import tensorflow as tf
from collections import defaultdict
import numpy as np
from util.time_exp import TicToc
from util.general_ext import *
from typing import Tuple, List
import scipy.sparse as sp

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = defaultdict(int)


def get_layer_uid(layer_name: str):
    """
    Returns an integer id for the given layer name, starting at 1.
    :param layer_name: the layer name.
    :return: an integer id for the given layer name.
    """
    _LAYER_UIDS[layer_name] += 1
    return _LAYER_UIDS[layer_name]


def sparse_dropout(sparse_tensor, rate, nonzero_count, rescale=True):
    """
    Dropout for tensorflow sparse tensors.
    :param sparse_tensor: the sparse tensor.
    :param rate: the dropout rate.
    :param nonzero_count: the number of nonzero elements int the sparse tensor.
    :param rescale the values in the sparse tensor after dropout.
    :return: the sparse tensor after dropout.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random_uniform(nonzero_count)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(sparse_tensor, dropout_mask)
    return pre_out * (1. / (1 - rate)) if rescale else pre_out


def glorot_init(shape, name=None):
    """
    2D Tensor initialization according to the "Glorot & Bengio (AISTATS 2010)" paper.
    :param shape: the shape of the tensor.
    :param name: the name of the initialized tensor.
    :return: the initialized tensor.
    """
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def masked_cross_entropy(estimated_scores, label_probabilities, mask):
    """
    Negative log-likelihood loss with mask.
    :param estimated_scores: the label scores on which the softmax function is applied;
                the highest score corresponds to the label with highest probability.
    :param label_probabilities: the true label probabilities; must be valid probability distribution (including one-hot vectors).
                NOTE discrete labels need to be converted to one-hot vectors.
    :param mask: the mask.
    :return: the masked negative log-likelihood loss.
    """
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=estimated_scores, labels=label_probabilities)
    if mask is not None:
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(estimated_scores, ground_truth_label_scores, mask):
    """
    Accuracy (the percentage of correct predictions) with masking.
    :param estimated_scores: the label scores where the highest score indicates the predicted label.
    :param ground_truth_label_scores: the ground-truth label score vectors reflecting the true labels;
                every score vector must have the index of its highest value be the actual true label;
                for discrete labels, a valid score vector can be one-hot vectors.
    :param mask: the mask.
    :return: the masked accuracy.
    """
    correct_prediction = tf.equal(tf.argmax(estimated_scores, 1), tf.argmax(ground_truth_label_scores, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    if mask is not None:
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def check_early_stop(look_back: int, curr_eval, prev_evals: List):
    if look_back > 0:
        prev_evals.append(curr_eval)
        if len(prev_evals) >= look_back and prev_evals[-1] > np.mean(prev_evals[-(look_back + 1):-1]):
            return True
    return False


class LayerTf(object):
    """
    Base layer class for tensorflow-based models.
    """

    def __init__(self, name: str = None, logging=False):
        if name is None:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.logging = logging
        self.logging_callable = callable(logging)
        self.sparse_x = False

        # the embeddings (placeholders), initialized as `False`, meaning there is no embedding
        self.embeddings = tf.constant(False)

    def forward(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self.forward(inputs)

            if self.logging_callable:
                if not self.sparse_x:
                    self.logging(self.name + '/inputs', inputs)
                if type(outputs) is not tf.SparseTensor:
                    self.logging(self.name + '/outputs', outputs)
            elif self.logging:
                if not self.sparse_x:
                    tf.summary.histogram(self.name + '/inputs', inputs)
                if type(outputs) is not tf.SparseTensor:
                    tf.summary.histogram(self.name + '/outputs', outputs)

            return outputs

    def variable_space_name(self):
        return self.name + '_var'

    def log_vars(self):
        if self.logging_callable:
            for var in tf.trainable_variables(tf.get_variable_scope().name):
                self.logging(self.name + '/vars/' + var, self.vars[var])
        elif self.logging:
            for var in tf.trainable_variables(tf.get_variable_scope().name):
                tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class ModelTf(object):
    def __init__(self, input_ph, target_ph, loss_fun, optimizer, name: str = None, metric_fun=None, input_dim=None, output_dim=None, hidden_dims: Tuple = None, prediction_fun=None):
        self.name = self.__class__.__name__.lower() if not name else name
        self.input_ph = input_ph
        self.target_ph = target_ph
        self.dimensions = [input_dim]
        self.dimensions.extend(hidden_dims)
        self.dimensions.append(output_dim)

        with tf.variable_scope(self.name):
            self.model = self.forward(input_ph)

        x = input_ph
        if type(self.model) is list:
            self.embeddings = []
            for layer in self.model:
                x = layer(x)
                self.embeddings.append(layer.embeddings)
        else:
            x = self.model(x)
            self.embeddings = self.model.embeddings
        self.outputs = x

        self.loss_fun = loss_fun if loss_fun is not None else self.default_loss()
        self.prediction_fun = prediction_fun if prediction_fun is not None else self.default_prediction()
        assert self.loss_fun is not None, "Loss function is not defined."
        self.metric_fun = metric_fun if metric_fun is not None else self.default_metric()

        self.sess = self.watch = None
        self.optimizer = optimizer
        self.opt_op = self.optimizer.minimize(self.loss_fun)

    def forward(self, x):
        raise NotImplementedError

    def _init_sess(self):
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def train(self, batch_data, stop_loss: float = 1e-6, max_iter: int = 1000, verbose=True, print_interval=10, validation_data=None, test_data=None, early_stop_lookback=5):
        if batch_data:
            prev_evals = []

            def _eval_msg(eval_data, eval_name: str):
                if self.metric_fun is None:
                    eval_outputs = self.sess.run([self.loss_fun], feed_dict=eval_data)
                    return "{eval_name} loss: {eval_loss:.5f}".format(eval_name=eval_name, eval_loss=eval_outputs[0]), \
                           check_early_stop(look_back=early_stop_lookback, curr_eval=eval_outputs[0], prev_evals=prev_evals), eval_outputs
                else:
                    eval_outputs = self.sess.run([self.loss_fun, self.metric_fun], feed_dict=eval_data)
                    return "{eval_name} loss: {eval_loss:.5f}, {eval_name} metric: {eval_metric:.5f}".format(eval_name=eval_name, eval_loss=eval_outputs[0], eval_metric=eval_outputs[1]), \
                           check_early_stop(look_back=early_stop_lookback, curr_eval=-eval_outputs[1], prev_evals=prev_evals), eval_outputs

            self._init_sess()
            if self.watch is None:
                self.watch = TicToc(update_interval=print_interval)

            has_validation = validation_data is not None
            for iter_idx in range(max_iter):
                if self.metric_fun is None:
                    train_outputs = self.sess.run([self.opt_op, self.loss_fun], feed_dict=batch_data)
                else:
                    train_outputs = self.sess.run([self.opt_op, self.loss_fun, self.metric_fun], feed_dict=batch_data)

                train_loss = train_outputs[1]
                if train_loss < stop_loss:
                    if verbose:
                        print("Stop loss {stop_loss} reached at iteration {iter_idx}, "
                              "with train loss {train_loss:.5f}.".format(stop_loss=stop_loss,
                                                                         iter_idx=iter_idx,
                                                                         train_loss=train_loss))
                    break

                last_iter = iter_idx == max_iter - 1
                if verbose and (self.watch.toc() or last_iter):
                    train_msg = "iter: {iter:04d}, train loss:{train_loss:.5f}, train metric:{train_metric:.5f}".format(iter=iter_idx, train_loss=train_loss, train_metric=train_outputs[2])
                    time_msg = "recent runtime:{recent_runtime:.5f}, average runtime:{avg_runtime:.5f}".format(recent_runtime=self.watch.recent_runtime,
                                                                                                               avg_runtime=self.watch.avg_runtime)
                    if has_validation:
                        eval_msg, early_stop, _ = _eval_msg(validation_data, 'validation')
                        print(train_msg + ', ' + eval_msg + ', ' + time_msg)
                        if early_stop:
                            print("Early stop activated due to decline of validation set performance.")
                            break
                    else:
                        print(train_msg + ', ' + time_msg)

            train_outputs = (train_outputs[1], train_outputs[2] if self.metric_fun is not None else None)
            if test_data is not None:
                test_msg, _, test_outputs = _eval_msg(test_data, 'test')
                print(test_msg)
                return batch_data, (test_outputs[0], test_outputs[1] if self.metric_fun is not None else None), train_outputs
            else:
                return batch_data, train_outputs

    def predict(self, batch_data, argmax: bool = False):
        self._init_sess()
        if argmax:
            return self.sess.run(tf.argmax(self.prediction_fun, axis=1), feed_dict=batch_data)
        else:
            return self.sess.run(self.prediction_fun, feed_dict=batch_data)

    def default_prediction(self):
        return NotImplementedError

    def default_loss(self):
        return NotImplementedError

    def default_metric(self):
        return None

    def eval_embeddings(self, batch_data):
        return self.sess.run(self.embeddings, feed_dict=batch_data)

    def reset(self):
        if self.sess is not None:
            self.sess.reset()
            self.sess = None

    def save(self, file_path: str):
        if self.sess is None:
            raise AttributeError("No active tensorflow training session.")
        saver = tf.train.Saver()
        print("Model saved in file: '{file_path}'.".format(file_path=saver.save(self.sess, file_path)))

    def load(self, file_path: str):
        if self.sess is None:
            self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, file_path)
        print("Model restored from file: '{file_path}'.".format(file_path=file_path))
