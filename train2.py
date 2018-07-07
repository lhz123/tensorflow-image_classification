import tensorflow as tf

import read_tfrecord
import numpy as np

import model3

import os

batch_size=8

num_classes=5

max_step = 10000
learning_rate = 0.0001

# def run_training():
log_dir = g'
train_file=''
test_file=''
tensorboard=''
test_tensorboard=''
train_image,train_label=read_tfrecord.read_and_decode(train_file,batch_size)

train_label=tf.one_hot(train_label,depth=5)

test_image,test_label=read_tfrecord.read_and_decode(test_file,batch_size)

test_label=tf.one_hot(test_label,depth=5)
x=tf.placeholder(tf.float32,[batch_size,227,227,3])
y=tf.placeholder(tf.float32,[batch_size,num_classes])

logits=model3.inference(x,num_classes)

loss=model3.loss(logits,y)
op=model3.optimizer(loss,learning_rate)
acc=model3.accuracy(logits,y)

sess=tf.Session()
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)

summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(tensorboard, sess.graph)
test_writer = tf.summary.FileWriter(test_tensorboard, sess.graph)

try:
    for step in range(max_step):


        tra_image,tra_label=sess.run([train_image,train_label])

        _,tra_loss,tra_acc=sess.run([op,loss,acc],feed_dict={x:tra_image,y:tra_label})
        if  step % 5 ==0:
            print('step %d,train_loss=%.2f,train_acc=%.2f'%(step,tra_loss,tra_acc))
            # summary_str=sess.run(summary_op)
            # train_writer.add_summary(summary_str,step)#出错，暂时无法解决

        if (step+1) %20 ==0 or step+1 == max_step:
            test_image, test_label=sess.run([test_image,test_label])
            val_loss,val_acc=sess.run([loss,acc],feed_dict={x:test_image,y:test_label})
            print('step %d,test_loss=%.2f,test_acc=%.2f'%(step,val_loss,val_acc))
            # summary_str = sess.run(summary_op)
            # test_writer.add_summary(summary_str, step)#出错，暂时无法解决
        if step % 1000 ==max_step:
            checkpoint_path = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
coord.join(threads)
sess.close()




