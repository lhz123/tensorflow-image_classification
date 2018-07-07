
import tensorflow as tf
import read_tfrecord
import mobilenet
import numpy as np
import os
import Xception
import model3
batch_size = 8
n_classes = 5
max_step = 10000
learning_rate = 0.0001

log_dir = ''
file_path=''
tensorboard=''
images, labels = read_tfrecord.read_and_decode(file_path, batch_size)
labels = tf.one_hot(labels, depth=5)
soft = model3.inference(images, n_classes)
cost = model3.loss(soft, labels)
optimizer = model3.optimizer(cost, learning_rate)
summary_op = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(tensorboard, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    for step in np.arange(max_step):
        if coord.should_stop():
           break
        _, tra_loss = sess.run([optimizer, cost])
        if step % 5 == 0:
            print('Step %d, train loss = %.4f' % (step, tra_loss))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        if (step %100) == 0:
            checkpoint_path = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
coord.join(threads)
sess.close()

'''
#train double tfrecord

file_path = ''
tfrecord_name = os.listdir(file_path)#遍历每一个类名
for i,name in enumerate(tfrecord_name):
    tfrecord_path=file_path+'/'+str(name)
    image, labels = read_tfrecord.read_and_decode(tfrecord_path, batch_size)
image = tf.reshape(image, shape=[batch_size, 227, 227, 3])
labels = tf.one_hot(labels, depth=5)
train_logits = model.inference(image, n_classes)
train_loss = model.loss(train_logits, labels)
train_op = model.optimizer(train_loss, learning_rate)
summary_op = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(log_dir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    for step in np.arange(max_step):
        if coord.should_stop():
            break
        _, tra_loss = sess.run([train_op, train_loss])

        if step % 5 == 0:
            print('Step %d, train loss = %.2f' % (step, tra_loss))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)

        if (step + 1) == max_step:
            checkpoint_path = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
coord.join(threads)
sess.close()
'''
