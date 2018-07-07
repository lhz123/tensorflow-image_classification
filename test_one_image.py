import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import read_tfrecord
import mobilenet
import os
import Xception
import model3

image_dir='/home/lhz/alldata/flower_tfrecord/b.jpg'
#images,labels=read_tfrecord.read_and_decode(image_dir)
#ind = np.random.randint(0, 3670)#随机生成一个整数
#print(ind)
#image = images[ind]
#label = labels[ind]
image=Image.open(image_dir)
#plt.imshow(image)
#plt.show()
image=image.resize([227,227])
#
image = np.array(image)
#print(image.shape)

image=tf.cast(image,tf.float32)
image = tf.image.per_image_standardization(image)
image=tf.reshape(image,[1,227,227,3])
x=tf.placeholder(tf.float32,[1,227,227,3])
soft=model3.inference(x,5)
#accuracy=model3.accuracy(soft,label)
saver=tf.train.Saver()
log_dir='/home/lhz/alldata/flower_tfrecord/log'
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        image=sess.run(image)
        softmax=sess.run(soft,feed_dict={x:image})
        print(softmax)
        # accuracy=sess.run(accuracy)
        # print('the accyracy is %f' % accuracy)
    else:
        print('No checkpoint file found')





