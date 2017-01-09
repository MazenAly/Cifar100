import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
import tensorflow as tf
import urllib

def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float) 
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images

def onehot_labels(labels):
    return np.eye(100)[labels]

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

data1  = unpickle('../data_batch_1')
data2  = unpickle('../data_batch_2')
data3  = unpickle('../data_batch_3')
data4  = unpickle('../data_batch_4')
data5  = unpickle('../data_batch_5')

X = np.concatenate((get_proper_images(data1['data']), 
                           get_proper_images(data2['data']), 
                           get_proper_images(data3['data']), 
                           get_proper_images(data4['data']), 
                           get_proper_images(data5['data'])))
Y = np.concatenate((onehot_labels(data1['labels']), 
                           onehot_labels(data2['labels']), 
                           onehot_labels(data3['labels']), 
                           onehot_labels(data4['labels']), 
                           onehot_labels(data5['labels'])))

X_test = get_proper_images(unpickle('test')['data'])
Y_test = onehot_labels(unpickle('test')['labels'])

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=15.)

network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, strides=1, padding='same', activation='relu', bias=True, 
                  bias_init='zeros', weights_init='uniform_scaling')
network = max_pool_2d(network, 2 , strides=None, padding='same')
network = conv_2d(network, 64, 3, strides=1, padding='same', activation='relu', bias=True, 
                  bias_init='zeros', weights_init='uniform_scaling')
network = conv_2d(network, 64, 3 , strides=1, padding='same', activation='relu', bias=True, 
                  bias_init='zeros', weights_init='uniform_scaling')
network = max_pool_2d(network, 2 , strides=None, padding='same')
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 100, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
                     
with tf.device('cpu:0'):
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=100 , run_id='aa2')
