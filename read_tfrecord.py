import tensorflow as tf

def read_and_decode(filename,batch_size):
    
    filename_queue=tf.train.string_input_producer([filename])#生成一个队列
    reader=tf.TFRecordReader()#创建一个reader来读取文件
    _,serialized_example=reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    images=tf.decode_raw(features['img_raw'],tf.uint8)
    labels=tf.cast(features['label'],tf.int32)
    images=tf.reshape(images, [227, 227, 3])
    images=tf.cast(images, tf.float32) * (1. / 255) - 0.5
        min_after_dequeue=100
    capacity=min_after_dequeue+3*batch_size
    image_batch,label_batch=tf.train.batch([images,labels],
                                                   batch_size=batch_size,num_threads=2,
                                                   capacity=capacity,)
    return image_batch,label_batch

'''
# 定义读取函数
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    images = tf.decode_raw(features['img_raw'], tf.uint8)
    images = tf.reshape(images, [227, 227, 3])
    images = tf.cast(images, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return images, label
'''
