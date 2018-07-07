import tensorflow as tf
import os
from PIL import Image

tfrecord_path='/'

#train_cwd=''#图片地址
#test_cwd=''

classes=os.listdir(test_cwd)#每一个类名

writer=tf.python_io.TFRecordWriter(tfrecord_path+'train_flower.tfrecords')#创建一个writer来写tfrecord

for index,name in enumerate(classes):
    classes_path=test_cwd+'/'+name+'/'
    for img_name in os.listdir(classes_path):
        img_path=classes_path+img_name
        images=Image.open(img_path)
        images=images.resize((227,227))
        images=images.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            # value=[index]决定了图片数据的类型label
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())

writer.close()

