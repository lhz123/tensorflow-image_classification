import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d,global_avg_pool
from tflearn.layers.core import fully_connected
from tflearn.layers.normalization import local_response_normalization,batch_normalization
import math
#network
def spp(bins,data,num_sample):
    n=len(bins)
    s=data.get_shape().as_list()[1]

    for i in range(n):
        strides=math.floor(s/bins[i])
        
        filters=math.ceil(s/bins[i])

        pool_out=tf.nn.max_pool(data,ksize=[1,filters,filters,1],strides=[1,strides,strides,1], padding='SAME')
        #pool_out[i]=tf.reshape(pool_out[i],shape=[None,-1])
        #out=tf.concat(1,[pool_out[0],pool_out[1],pool_out[2]])
        if (i==0):
            spp = tf.reshape(pool_out,[num_sample,-1])
        else:
            spp=tf.concat(axis=1,values=[spp,tf.reshape(pool_out,[num_sample,-1])])
    return spp
bins=[3,2,1]
def inference(input_data,n_classes):

    net = conv_2d(input_data, 96, 11, strides=4, activation='relu', name='conv1')
    max_pool = max_pool_2d(net, 3, strides=2, name='max_pool1')
    net = local_response_normalization(max_pool, name='local_response_normalization')
    net = conv_2d(net, 256, 5, strides=1, activation='relu', name='conv2')
    max_pool = max_pool_2d(net, 3, strides=2, name='max_pool2')
    net = batch_normalization(max_pool, decay=0.9, stddev=0.002, trainable=True, restore=True,
                              name='batch_normalization')
    net = conv_2d(net, 384, 3, strides=1, activation='relu', name='conv3')
    net = conv_2d(net, 384, 3, strides=1, activation='relu', name='conv4')
    net = conv_2d(net, 256, 3, strides=1, activation='relu', name='conv5')
    #max_pool = max_pool_2d(net, 3, strides=2, name='max_pool5')
    #pool = tflearn.global_avg_pool(max_pool, name='global_avg_pool')
    pool=spp(bins,net,net.get_shape().as_list()[0])
    soft = fully_connected(pool, n_classes, activation='softmax', name='fc7')
    '''
    net = conv_2d(input_data, 64, 7, strides=2, activation='relu')
    net = max_pool_2d(net, 3, strides=2)
    shortcut1 = conv_2d(net, 256, 2, strides=2, activation='relu')
    residual = conv_2d(net, 64, 1, activation='relu')
    residual = conv_2d(residual, 64, 3, strides=2, activation='relu')
    residual = conv_2d(residual, 256, 1, activation='relu')
    output1 = residual + shortcut1

    shortcut2 = conv_2d(output1, 512, 2, strides=2, activation='relu')
    residual = conv_2d(output1, 128, 1, activation='relu')
    residual = conv_2d(residual, 128, 3, strides=2, activation='relu')
    residual = conv_2d(residual, 512, 1, activation='relu')
    output2 = residual + shortcut2

    shortcut3 = conv_2d(output2, 1024, 2, strides=2, activation='relu')
    residual = conv_2d(output2, 256, 1, activation='relu')
    residual = conv_2d(residual, 256, 3, strides=2, activation='relu')
    residual = conv_2d(residual, 1024, 1, activation='relu')
    output3 = residual + shortcut3

    shortcut4 = conv_2d(output3, 2048, 2, strides=2, activation='relu')
    residual = conv_2d(output3, 512, 1, activation='relu')
    residual = conv_2d(residual, 512, 3, strides=2, activation='relu')
    residual = conv_2d(residual, 2048, 1, activation='relu')
    output2 = residual + shortcut4

    net = global_avg_pool(output2)
    net = fully_connected(net, n_classes, activation='softmax')
 '''
    return soft


# 损失函数
def loss(logits, label):
    loss = tflearn.objectives.softmax_categorical_crossentropy(logits, label)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))
    return loss
#优化函数
def optimizer(loss,learning_rate):
    op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return op
#accuracy
def accuracy(predict, label):
    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy
