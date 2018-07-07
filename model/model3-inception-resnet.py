import tensorflow as tf
import math
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
'''
def conv_2d(inputs, filters, size, stride):
    channels = inputs.get_shape()[3]
    weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[filters]))

    pad_size = size // 2
    pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    inputs_pad = tf.pad(inputs, pad_mat)

    conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID')
    conv_biased = tf.add(conv, biases )

    return conv_biased


# 定义池化层函数

# In[4]:

def pool_2d(inputs, size, stride):
    print ('    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (size, size, stride))
    return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                         )


# 定义全连接层函数 激活函数Relu
# In[5]:

def fc_layer(inputs, hiddens, flat=False, linear=False):
    input_shape = inputs.get_shape().as_list()
    if flat:
        dim = input_shape[1] * input_shape[2] * input_shape[3]
        inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
        inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
    else:
        dim = input_shape[1]
        inputs_processed = inputs

    weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
    ip = tf.add(tf.matmul(inputs_processed, weight), biases)
    return  ip
'''
#inception_resnet_A
def inception_A(input_data):
    A1=conv_2d(input_data,32,1,activation='softplus',name='A1')
    A2=conv_2d(input_data,32,1,activation='softplus',name='A2')
    A2=conv_2d(A2, 32, 3,activation='softplus', name='A2')
    A3=conv_2d(input_data,32,1,activation='softplus',name='A3')
    A3 = conv_2d(A3, 48, 3,activation='softplus', name='A3')
    A3 = conv_2d(A3, 64, 3,activation='softplus', name='A3')
    # print('a1=',A1)
    # print('a2=',A2)
    # print('a3=',A3)
    merge = tflearn.layers.merge_ops.merge([A1, A2, A3], mode='concat',axis=3)#特征图的加和，如57,57,32、57,57,32、57,57,64merge
                                                                                         # 后的结果为57,57,128
    # print('merge=',merge)
    merge=conv_2d(merge,384,1,activation='Linear',name='concat')
    sum=input_data+merge
    output=tf.nn.relu(sum)
    return output
#reduction-A
def reduction_A(input_data):
    RA1=max_pool_2d(input_data,3,strides=2,name='RA1')
    RA2=conv_2d(input_data,384,3,strides=2,activation='softplus',name='RA2')
    RA3=conv_2d(input_data,256,1,activation='softplus',name='RA3')
    RA3=conv_2d(RA3,256,3,activation='softplus',name='RA3')
    RA3 = conv_2d(RA3, 384, 3,strides=2,activation='softplus',name='RA3')
    merge = tflearn.layers.merge_ops.merge([RA1, RA2, RA3], mode='concat', axis=3)
    return merge
#inception_resnet_B
def inception_B(input_data):
    B1=conv_2d(input_data,192,1,activation='softplus',name='B1')
    B2 = conv_2d(input_data, 128, 1, activation='softplus',name='B2')
    B2 = conv_2d(B2, 160, filter_size=[1,7], activation='softplus',name='B2')
    B2 = conv_2d(B2, 160, filter_size=[7, 1], activation='softplus',name='B2')
    merge = tflearn.layers.merge_ops.merge([B1, B2], mode='concat', axis=3)
    merge = conv_2d(merge, 1154, 1, activation='Linear', name='concat')
    return merge
#reducetion_B
def reduction_B(input_data):
    RB1=max_pool_2d(input_data,3,strides=2,name='RB1')
    RB2=conv_2d(input_data,256,1,activation='softplus',name='RB2')
    RB2 = conv_2d(RB2, 384, 3,2, activation='softplus',name='RB2')
    RB3=conv_2d(input_data,256,1,activation='softplus',name='RB3')
    RB3= conv_2d(RB3, 288, 3,2, activation='softplus',name='RB3')
    RB4=conv_2d(input_data,256,1,activation='softplus',name='RB4')
    RB4 = conv_2d(RB4, 288, 3,activation='softplus',name='RB4')
    RB4 = conv_2d(RB4, 320, 3, strides=2,activation='softplus',name='RB4')
    merge = tflearn.layers.merge_ops.merge([RB1, RB2, RB3,RB4], mode='concat', axis=3)
    return merge
#inception_resnet_C
def inception_C(input_data):
    C1=conv_2d(input_data,192,1,activation='softplus',name='C1')
    C2 = conv_2d(input_data, 192, 1, activation='softplus',name='C2')
    C2 = conv_2d(C2, 224, filter_size=[1,3], activation='softplus',name='C2')
    C2 = conv_2d(C2, 256, filter_size=[3, 1], activation='softplus',name='C2')
    merge = tflearn.layers.merge_ops.merge([C1, C2], mode='concat', axis=3)
    merge=conv_2d(merge,2048,1,activation='Linear',name='merge')
    return merge

#network
def inference(input_data,n_classes):
    #stem
    net=conv_2d(input_data,32,3,strides=2,activation='softplus',name='conv1')
    net = conv_2d(net, 32, 3, activation='softplus',name='conv2')
    net = conv_2d(net, 64, 3,activation='softplus', name='conv3')
    net1 = max_pool_2d(net,3, strides=2,name='pool4')
    net2 = conv_2d(net, 96,3, strides=2,activation='softplus', name='conv4')
    merge5=tflearn.layers.merge_ops.merge([net1, net2], mode='concat', axis=3)
    net6_1=conv_2d(merge5,64,1,activation='softplus',name='net6_1')
    net7_1=conv_2d(net6_1,96,3,activation='softplus',name='net6_1')
    net6_2=conv_2d(merge5,64,1,activation='softplus',name='net6_2')
    net7_2 = conv_2d(net6_2, 64, filter_size=[1,7], activation='softplus',name='net7_2')
    net8_2 = conv_2d(net7_2, 64, filter_size=[7, 1], activation='softplus',name='net8_2')
    net10_2 = conv_2d(net8_2, 96, 3, activation='softplus',name='net10_2')
    merge = tflearn.layers.merge_ops.merge([net7_1, net10_2], mode='concat', axis=3)
    net11_1=conv_2d(merge,192,3,activation='softplus',name='net11_1')
    net11_2=max_pool_2d(merge,1,name='net11_2')
    x = tflearn.layers.merge_ops.merge([net11_1, net11_2], mode='concat', axis=3)
    for i in range(3):
        x=inception_A(x)
    x=reduction_A(x)
    for i in range(5):
        x=inception_B(x)
    x=reduction_B(x)
    for i in range(3):
        x=inception_C(x)
    pool = tflearn.global_avg_pool(x)
    # softmax
    drop = tf.nn.dropout(pool, 0.8)
    soft = fully_connected(drop, n_classes, activation=None)
    soft = tf.nn.softmax(soft)
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

