import tensorflow as tf

#mask是一个索引向量，值为1表示该位置的标签在训练数据中是给定的
#loss的shape与mask的shape相同，等于样本的数量（None，）
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask   #向量点乘
    return tf.reduce_mean(loss)

#准确率也是一样，只对在验证集或测试集上的标签位置计算准确率，并且准确率扩大了tf.reduce_mean(mask)的倒数倍
def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
