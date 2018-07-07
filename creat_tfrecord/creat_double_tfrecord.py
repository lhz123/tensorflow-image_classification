import os
import tensorflow as tf
from PIL import Image

cwd=''#图片地址

tfrecord_dir=''#tfrecord地址

classes=os.listdir(cwd)#遍历每一个文件夹得到每一个类名

best_num=1000#每一个tfrecord中存放的图片个数

n_image=0 #第几个图片

n_record=0 #第几个tfrecord文件

tfrecord_name=('flower.tfrecords-%d'% n_record)#每一个tfrecord名

writer=tf.python_io.TFRecordWriter(tfrecord_dir+tfrecord_name) #创建一个writer来写入tfrecord

for index,name in enumerate(classes):
    print(index)
    print(name)
    classes_path=cwd+'/'+name+'/' #每一个类地址
    for image_name in os.listdir(classes_path):
        n_image=n_image+1
        if n_image>best_num:
            n_image=1
            n_record=n_record+1
            tfrecord_name = ('flower.tfrecords-%d' % n_record)

            writer = tf.python_io.TFRecordWriter(tfrecord_dir + tfrecord_name)

        img_path = classes_path + image_name  # 每一个图片的地址
        img = Image.open(img_path)
        img = img.resize((227, 227))
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(
            features=tf.train.Features(feature={
                # value=[index]决定了图片数据的类型label
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
        # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
