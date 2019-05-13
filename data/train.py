from __future__ import division
from __future__ import print_function

import tensorflow as tf


from data.utils import *
from data.models import GCN, MLP
from data.data_load import *
from data.data_common import *
from util.print_ext import *
from util.path_ext import *
from itertools import product
import time
import numpy as np

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'aminer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 512, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-6, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')


# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data_aminer(FLAGS.dataset)

dblp_folder = path.join(path.dirname(path.abspath(__file__)), 'dblp')
normalize_feature = True
use_one_hot_features = False
use_specified_mask = True

adj, features, labels, train_mask, val_mask, test_mask, mask_file = \
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


print("------------")
print(features.shape)

# Some preprocessing
# 将特征矩阵归一化，并转换成元组
# features = preprocess_features(features)
print("归一化后的特征矩阵")
print(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]   # 图的归一化拉普拉斯矩阵
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features.shape), # 'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features.shape[1], logging=True) #model = model_func(placeholders, input_dim=features[2][1], logging=True)



# Initialize session
sess = tf.Session()

# support ： 图的归一化拉普拉斯矩阵
# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)



# self code
loss_scalar = tf.summary.scalar('model_loss', model.loss)
accuray_scalar = tf.summary.scalar('model_accuray', model.accuracy)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./log", sess.graph)


# Init variables
sess.run(tf.global_variables_initializer())



cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

   # print("-----model.inputs-------")
   # print(model.inputs)


    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, labels, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)

    #self code
    summary, _ = sess.run([merged, model.opt_op], feed_dict=feed_dict)
    # self code
    train_writer.add_summary(summary, global_step=epoch)
    # train_writer.add_summary(outs[1], global_step=epoch)
    # train_writer.add_summary(outs[2], global_step=epoch)
    # train_writer.add_summary(outs[3], global_step=epoch)



    # Validation
    cost, acc, duration = evaluate(features, support, labels, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

   # print("--------------output features----------")
    #features = features.todense()
   # print(features)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
       print("Early stopping...")
       break

train_writer.close()

# self code
# np.savetxt('E:\\gcnVector\\vector_0.0004.csv', outs[3])


# AXW乘积, W为最后一层的输出
features = sp.csr_matrix(features)
AX = np.dot(adj, features)
W = sp.csr_matrix(outs[3].T)
AXW = np.dot(AX, W)


hidden_1 = sess.run(model.activations[-2], feed_dict=feed_dict)
# print(type(hidden_1))
# hidden_1 = sp.csr_matrix(hidden_1).tolil()
# print(hidden_1.shape)
# hidden_1 = preprocess_features_1(hidden_1)
# print(hidden_1)
np.savetxt('./vector_2.csv', hidden_1)

print("Optimization Finished!")

# label_dict = {0:"0.0000000e+00",1:"1.0000000e+00",2:"2.0000000e+00",3:"3.0000000e+00",4:"4.0000000e+00",5:"5.0000000e+00",6:"6.0000000e+00",7:"7.0000000e+00",8:"8.0000000e+00"} # 定义标签颜色字典
# # 写文件
# with open("./embeddings.txt", "w") as fe, open("./labels.txt", 'w') as fl:
#     for i in range(len(outs[3])):
#         fl.write(label_dict[int(list(labels[i]).index(1.))]+"\n")
#         fe.write(" ".join(map(str, outs[3][i]))+"\n")


# Testing
test_cost, test_acc, test_duration = evaluate(features, support, labels, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

