import tensorflow as tf
import read_tfrecord
#import train
import numpy as np
import model2
import Xception
import model3
batch_size = 8
n_classes = 5

log_dir = '/home/lhz/alldata/flower_tfrecord/log'

file_path='/home/lhz/alldata/flower_tfrecord/tfrecord/test_flower.tfrecords'

test_tensorboard='/home/lhz/alldata/flower_tfrecord/test_tensorboard'
test_images,test_labels=read_tfrecord.read_and_decode(file_path,batch_size)
test_labels=tf.one_hot(test_labels,depth=5)
logit=model3.inference(test_images,n_classes)
accuracy=model3.accuracy(logit,test_labels)

saver=tf.train.Saver()
sess=tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
with tf.Session() as sess:
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(test_tensorboard,sess.graph)
    ckpt=tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        acc=sess.run(accuracy)
        print('after %s training step,test accuracy is %f' % (global_step, accuracy))
        summary_str = sess.run(summary_op)
        train_writer.add_summary(summary_str)
    else:
        print('No checkpoint file found')
    coord.request_stop()
coord.join(threads)
sess.close()



# batch_size=16
# n_classes=5
# log_dir='/home/lhz/alldata/flower_tfrecord/log'
# file_path='/home/lhz/alldata/flower_tfrecord/tfrecord/test_flower.tfrecords'
# test_tensorboard='/home/lhz/alldata/flower_tfrecord/test_tensorboard'
# # summary_op = tf.summary.merge_all()
# #
# # train_writer = tf.summary.FileWriter(test_tensorboard, sess.graph)
# def test(file_path):
#
#     test_images,test_labels=read_tfrecord.read_and_decode(file_path,batch_size)
#     x=tf.placeholder(tf.float32,[batch_size,227,227,3])
#     y=tf.placeholder(tf.float32,[batch_size,n_classes])
#
#     soft=model2.inference(x,n_classes)
#     accuracy=model2.accuracy(soft,y)
#     saver=tf.train.Saver()
#     with tf.Session() as sess:
#         summary_op = tf.summary.merge_all()
#         train_writer = tf.summary.FileWriter(test_tensorboard,sess.graph)
#         ckpt=tf.train.get_checkpoint_state(log_dir)
#         if ckpt and ckpt.model_checkpoint_path:
#
#             saver.restore(sess,ckpt.model_checkpoint_path)
#             global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#             print(global_step)
#             #test_images,test_labels=sess.run([test_images,test_labels])
#             accuracy=sess.run(accuracy,feed_dict={x:test_images,y:test_labels})
#             print('after %s training step,test accuracy is %f' % (global_step, accuracy))
#             summary_str = sess.run(summary_op)
#             train_writer.add_summary(summary_str)
#         else:
#             print('No checkpoint file found')
#
# def main(argv=None):
#     test(file_path)
# if __name__ == '__main__':
#     tf.app.run()

