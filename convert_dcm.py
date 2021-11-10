from io import TextIOBase
import pydicom 

from PIL import Image
import numpy as np
import glob
import tensorflow as tf
from skimage.transform import resize
IMG_PX_SIZE = 512
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

train__covid_txt = open("/home/julia/Downloads/dados_covid/val_covid.txt",'r')
train_txt_non_covid = open("/home/julia/Downloads/dados_covid/val_non_covid.txt",'r')
train_txt = open("/home/julia/Downloads/dados_covid/val.txt",'r')

train_covid = []
train_non_covid = []
val_covid = []
val_non_covid = []
test_covid = []
test_non_covid = []
train_filename = []

for line in train__covid_txt:
    train_covid.append(line.strip())
for line_2 in train_txt_non_covid:
    train_non_covid.append(line_2.strip())

for line_train in train_txt:
    train_filename.append(line_train.strip())
print(len(train_filename))

img_path = "/home/julia/Downloads/dados_covid/"
recordPath = "/home/julia/Downloads/dados_covid/tf_record/"
keys = [str(i) for i in list(range(2))]
values = [i for i in list(range(2))]
classes = dict(zip(keys, values))

recordFileNum = 0
recordFileName = ("val.tfrecords-%.3d" % recordFileNum)
writer = tf.io.TFRecordWriter(recordPath + recordFileName)

for filename in train_filename:
    if filename in train_non_covid:
        label = 0
    elif filename in train_covid:
        label = 1
    print(label)
    
    recordFileNum += 1
    ds = pydicom.dcmread(filename)
    image = ds.pixel_array.astype(float)
    image = (np.maximum(image, 0) / image.max()) * 255.0
    image = resize(image, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
    image = image.reshape((image.shape[0], image.shape[1], 1))
    print(image.shape)
    image_raw = image.flatten().tostring()
    print(image.shape)

    example = tf.train.Example(features=tf.train.Features(feature={
    'height': _int64_feature(512),
    'width': _int64_feature(512),
    'depth': _int64_feature(1),
    "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
writer.write(example.SerializeToString())
writer.close()