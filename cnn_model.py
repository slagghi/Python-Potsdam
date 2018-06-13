#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


# READING PARAM FILES FROM PRE-TRAINED POTSDAM NETWORK
def load_param(name):
    path='mat2numpy/'+name+'.npy'
    out=np.load(path)
    return out
# Layer 1 Parameters (convolutional, 7x7, 5->64)
f1=load_param('l1f')
b1=load_param('l1b')
# Layer 2 Parameters (batch normalization)
m2=load_param('l2m')
b2=load_param('l2b')
x2=load_param('l2x')
# Layer 6 Parameters (convolutional, 5x5, 64->64)
f6=load_param('l6f')
b6=load_param('l6b')
# Layer 7 Parameters (batch normalization)
m7=load_param('l7m')
b7=load_param('l7b')
x7=load_param('l7x')
# Layer 11 Parameters (convolutional, 3x3, 64->128)
f11=load_param('l11f')
b11=load_param('l11b')
# Layer 12 Parameters (batch normalization)
m12=load_param('l12m')
b12=load_param('l12b')
x12=load_param('l12x')
# Layer 16 Parameters (convolutional, 3x3, 128->256)
f16=load_param('l16f')
b16=load_param('l16b')
# Layer 17 Parameters (batch normalization)
m17=load_param('l17m')
b17=load_param('l17b')
x17=load_param('l17x')
# Layer 20 Parameters (deconvolutional, 3x3, 256->512, [2,2] upsampling)
f20=load_param('l20f')
b20=load_param('l20b')
# Layer 21 Parameters (batch normalization)
m21=load_param('l21m')
b21=load_param('l21b')
x21=load_param('l21x')
# Layer 24 Parameters (deconvolutional, 3x3, 512->512, [2,2] upsampling)
f24=load_param('l24f')
b24=load_param('l24b')
# Layer 25 Parameters (batch normalization)
m25=load_param('l25m')
b25=load_param('l25b')
x25=load_param('l25x')
# Layer 28 Parameters (deconvolutional, 3x3, 512->512, [2,2] upsampling)
f28=load_param('l28f')
b28=load_param('l28b')
# Layer 29 Parameters (batch normalization)
m29=load_param('l29m')
b29=load_param('l29b')
x29=load_param('l29x')
# Layer 32 Parameters (convolutional, 1x1, 512->6)
f32=load_param('l32f')
b32=load_param('l32b')


# CNN MODEL
def conv_net(images):
    
    # input image should be reshaped
    
    # I need to know, respectively, the batch size and number of channels of the input
    
    #images = tf.reshape(images, shape=[batch_size, 65, 65, channels])
    
    # 1st layer: convolution 7x7 5->64
    
    conv1=tf.nn.conv2d(images,f1,strides=[1,1,1,1],padding='SAME')
    conv1=tf.nn.bias_add(conv1,b1)
    
    # 2nd layer: batch normalization
    banorm1=tf.nn.batch_normalization(conv1,m2,x2,b2,variance_epsilon=1.0e-04)
    
    # 3rd layer: ReLU
    # UNSURE if leaky_relu preserves the input tensor's shape of [batch,heigth,width,channels]
    relu1=tf.nn.leaky_relu(banorm1,alpha=0.2,name=None)
#
    # 4th layer: pooling
    pool1=tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')
    
#   UNSURE if I should implement the dropout

#    # 5th layer: drop out
#    # Probability to keep each value is 0.75
#    # keep_prob array must have same size as input
#    keep_prob=np.copy(pool1)
#    keep_prob.fill(1)
#    keep_prob=0.75*keep_prob
#    
#    pool1=tf.nn.dropout(pool1,keep_prob)
    
    # 6th layer: convolution 5x5 64->64
    
    conv2=tf.nn.conv2d(pool1,f6,strides=[1,1,1,1],padding='SAME')
    conv2=tf.nn.bias_add(conv2,b6)
    
    # 7th layer: batch normalization
    banorm2=tf.nn.batch_normalization(conv2,m7,x7,b7,variance_epsilon=1.0e-04)
    
    # 8th layer: ReLU
    relu2=tf.nn.leaky_relu(banorm2,alpha=0.1,name=None)
    
    # 9th layer: pooling
    pool2=tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')

    #  UNSURE if I should include the dropout
    # for implementation, see layer 5
    
    # 11th layer: convolution 3x3 64->128
    conv3=tf.nn.conv2d(pool2,f11,strides=[1,1,1,1],padding='SAME')
    conv3=tf.nn.bias_add(conv3,b11)
    
    # 12th layer: batch normalization
    banorm3=tf.nn.batch_normalization(conv3,m12,x12,b12,variance_epsilon=1.0e-04)
    
    # 13th layer: ReLU
    relu3=tf.nn.leaky_relu(banorm3,alpha=0.1,name=None)

    # 14th layer: pooling
    pool3=tf.nn.max_pool(relu3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')

    
    #  UNSURE if I should include the dropout
    # for implementation, see layer 5
    
    # 16th layer: convolution 3x3 128->256
    conv4=tf.nn.conv2d(pool3,f16,strides=[1,1,1,1],padding='SAME')
    conv4=tf.nn.bias_add(conv4,b16)
    
    # 17th layer: batch normalization
    banorm4=tf.nn.batch_normalization(conv4,m17,x17,b17,variance_epsilon=1.0e-04)
    
    # 18th layer: ReLU
    relu4=tf.nn.leaky_relu(banorm4,alpha=0.1,name=None)
    
    # RECAP:
    # at this point I should have 256 feature maps of size 1/8*initial patch size (which is 65x65)-->it should be 9x9
    # (due to the 3 maxpooling layers with stride 2)
    
    #  UNSURE if I should include the dropout
    # for implementation, see layer 5
    
    # 20th layer: deconvolution 3x3 256->512
    deconv_heigth = 2*relu4.shape[1] #since I have an upsampling of factor 2
    deconv_width  = 2*relu4.shape[2]
    # I need to know the batch size, here I set it to -1 as a placeholder
    convt5=tf.nn.conv2d_transpose(relu4,f20,[-1,deconv_heigth,deconv_width,512],strides=[1,1,1,1],padding="SAME")
    convt5=tf.nn.bias_add(convt5,b20)
    
    # 21st layer: batch normalization
    banorm5=tf.nn.batch_normalization(convt5,m21,x21,b21,variance_epsilon=1.0e-04)
    
    # 22nd layer: ReLU
    relu5=tf.nn.leaky_relu(banorm5,alpha=0.1,name=None)
    
    #  UNSURE if I should include the dropout
    # for implementation, see layer 5
    
    # 24th layer: deconvolution 3x3 512->512
    deconv_heigth = 2*deconv_heigth #since I have an upsampling of factor 2
    deconv_width  = 2*deconv_width
    convt6=tf.nn.conv2d_transpose(relu5,f24,[-1,deconv_heigth,deconv_width,512],strides=[1,1,1,1],padding="SAME")
    convt6=tf.nn.bias_add(convt6,b24)
    
    # 25th layer: batch normalization
    banorm6=tf.nn.batch_normalization(convt6,m25,x25,b25,variance_epsilon=1.0e-04)
    
    # 26th layer: ReLU
    relu6=tf.nn.leaky_relu(banorm6,alpha=0.1,name=None)
    
    #  UNSURE if I should include the dropout
    # for implementation, see layer 5
    
    # 28th layer: deconvolution 3x3 512->512
    deconv_heigth = 2*deconv_heigth #since I have an upsampling of factor 2
    deconv_width  = 2*deconv_width
    convt7=tf.nn.conv2d_transpose(relu6,f28,[-1,deconv_heigth,deconv_width,512],strides=[1,1,1,1],padding="SAME")
    convt7=tf.nn.bias_add(convt7,b28)
    
    # 29th layer: batch normalization
    banorm7=tf.nn.batch_normalization(convt7,m29,x29,b29,variance_epsilon=1.0e-04)
    
    #30th layer: ReLU
    relu7=tf.nn.leaky_relu(banorm7,alpha=0.1,name=None)
    
    #  UNSURE if I should include the dropout
    # for implementation, see layer 5
    
    #32nd layer: convolution 1x1 512->6
    conv8=tf.nn.conv2d(relu7,f32,strides=[1,1,1,1],padding='SAME')
    conv8=tf.nn.bias_add(conv8,b32)
    
    # RECAP: I should now have in output a tensor of shape [batch_size,65,65,6]
    # 6 is the number of classes, and the vector represents a 6-dimension matrix of class scores

# I still need to download the dataset in order to give something as input and get the predictions

#logits=conv_net(images)
#prediction=tf.nn.softmax(logits)
    
