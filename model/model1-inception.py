from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
#from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
#from tflearn.layers.estimator import regression
import tensorflow as tf
import math
def sigmoid(inputs):
    output=[math.log(1+math.exp(x)) for x in inputs]
    return output
def inception_A(input_data):
    A_1= conv_2d(input_data, 32, 1, activation='relu', name='inception_A')
    A_2 = conv_2d(input_data, 32, 1, activation='relu', name='inception_A_1')
    A_2 = conv_2d(A_2, 32, 3, activation='relu', name='inception_A_1_3')
    A_3 = conv_2d(input_data, 32, 1, activation='relu', name='inception_A_1_3_3')
    A_3 = conv_2d(A_3, 32, 3, activation='relu', name='inception_A_1_3_3_2')
    A_3 = conv_2d(A_3, 32, 3, activation='relu', name='inception_A_1_3_3_3')
    merge = tflearn.layers.merge_ops.merge([A_1, A_2, A_3], mode='concat',
                                           axis=3)
    inception_A_L = conv_2d(merge, 256, 1, activation='Linear', name='inception_A_L')
    # merge inception_A1_1
    # inception_A1=tflearn.layers.merge_ops.merge([network,inception_A1_L],mode='sum',axis=1)
    inception_A = input_data + inception_A_L
    # inception_A1_out
    A_out = tf.nn.relu(inception_A)
    return A_out

def inception_B(input_data):

    B_1 = conv_2d(input_data, 128, 1, activation='relu', name='inception_B')
    B_2 = conv_2d(input_data, 128, 1, activation='relu', name='inception_B_1')
    B_2 = conv_2d(B_2, 128, filter_size=[1,7], activation='relu', name='inception_B_1_7_7')
    B_2 = conv_2d(B_2, 128, filter_size=[7, 1], activation='relu', name='inception_B_1_7_7')
    merge = tflearn.layers.merge_ops.merge([B_1, B_2], mode='concat',axis=3)
    B_L = conv_2d(merge, 896, 1, activation='Linear', name='inception_1_7_L')
    # merge inception_B1
    #inception_B1=tflearn.layers.merge_ops.merge([reduction_A,inception_1_7_L],mode='sum')
    inception_B = input_data + B_L
    B_out = tf.nn.relu(inception_B)
    return B_out


def inception_C(input_data):
    C_1 = conv_2d(input_data, 192, 1, activation='relu', name='inception_C')
    C_2 = conv_2d(input_data, 192, 1, activation='relu', name='inception_c_2')
    C_2 = conv_2d(C_2, 192, filter_size=[1,3], activation='relu', name='inception_c_2_3')
    C_2 = conv_2d(C_2, 192, filter_size=[3,1], activation='relu', name='inception_c_2_3')
    # merge inception_C1
    merge = tflearn.layers.merge_ops.merge([C_1, C_2], mode='concat',axis=3, name='merge')
    C_L = conv_2d(merge, 1792, 1, activation='Linear', name='incption_C_L')
    # inception_c1=tflearn.layers.merge_ops.merge([reduction_B,incption_C1_L],mode='sum')
    inception_c1 = input_data + C_L
    C_out = tf.nn.relu(inception_c1)
    return C_out
def inference(input_data,n_classes):
    network = conv_2d(input_data, 32, 3, strides=2, activation='relu', name='conv1_3_3')
    network = conv_2d(network, 32, 3, activation='relu', name='con2_3_3')
    network = conv_2d(network, 64, 3, padding='SAME', activation='relu', name='conv3_3_3')
    pool = max_pool_2d(network, 3, strides=2)
    network = conv_2d(pool, 80, 1, activation='relu', name='conv4_1_1')
    network = conv_2d(network, 192, 3, strides=2, activation='relu', name='conv5_3_3')
    net = conv_2d(network, 256, 3, strides=2, activation='relu', name='conv6_3_3')

    for i in range(4):
        net=inception_A(net)
    # reduction_A
    reduction_A_pool = max_pool_2d(net, 3, strides=2, name='reduction_A_pool')
    reduction_A_3_3 = conv_2d(net, 384, 3, strides=2, activation='relu', name='reduction_A_3_3')
    reduction_A_1_1 = conv_2d(net, 192, 1, activation='relu', name='reduction_A_1_1')
    reduction_A_1_1_3_3 = conv_2d(reduction_A_1_1, 192, 3, activation='relu', name='reduction_A_1_1_3_3')
    reduction_A_1_1_3_3_3_3 = conv_2d(reduction_A_1_1_3_3, 256, 3, strides=2, activation='relu',
                                          name='recduction_A_1_1_3_3_3_3')
    # merge reduction_A
    net = tflearn.layers.merge_ops.merge([reduction_A_pool, reduction_A_3_3, reduction_A_1_1_3_3_3_3],
                                                     mode='concat', axis=3, name='reduction_A')

    for i in range(7):
        net = inception_B(net)
    # reducetion_B
    reduction_B_pool = max_pool_2d(net, 3, strides=2, name='reduction_B_pool')
    reduction_B_1_1 = conv_2d(net, 256, 1, activation='relu', name='reduction_B_1_1')
    reduction_B_1_1_3_3 = conv_2d(reduction_B_1_1, 384, 3, strides=2, activation='relu', name='reduction_B_1_1_3_3')
    reduction_B_1_1_2 = conv_2d(net, 256, 1, activation='relu', name='reduction_B_1_1_2')
    reduction_B_1_1_2_3_3 = conv_2d(reduction_B_1_1_2, 256, 3, strides=2, activation='relu',
                                        name='reduction_B_1_1_2_3_3')
    reduction_B_1_1_3 = conv_2d(net, 256, 1, activation='relu', name='reduction_B_1_1_3')
    reduction_B_1_1_3_3_3 = conv_2d(reduction_B_1_1_3, 256, 3, activation='relu', name='reduction_B_1_1_3_3_3')
    reduction_B_1_1_3_3_3_3 = conv_2d(reduction_B_1_1_3_3_3, 256, 3, strides=2, activation='relu',
                                          name='reduction_B_1_1_3_3_3_3')
    # merge reduction_B
    net = tflearn.layers.merge_ops.merge(
            [reduction_B_pool, reduction_B_1_1_3_3, reduction_B_1_1_2_3_3, reduction_B_1_1_3_3_3_3], mode='concat',
            axis=3, name='reduction_B')
    for i in range(3):
        net= inception_C(net)
    pool = tflearn.global_avg_pool(net)
    # softmax
    drop= dropout(pool, 0.8)
    soft = fully_connected(drop, n_classes, activation='relu')
    soft=tf.nn.softmax(soft)

    return soft
# 损失函数
def loss(logits, label):
    with tf.variable_scope('loss_function') as scope:
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits)
        #loss = tf.reduce_mean(cross_entropy, name='loss')
        loss = tflearn.objectives.softmax_categorical_crossentropy(logits, label)
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss

        #loss = tflearn.objectives.softmax_categorical_crossentropy(logits, label)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))
        #return loss
#优化函数
def optimizer(loss,learning_rate):
    with tf.name_scope('optimizer_function'):

        op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
        return op
#accuracy
def accuracy(predict, label):
    '''
    with tf.name_scope('accuracy_function'):
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy
    '''
    with tf.variable_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy