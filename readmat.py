#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:07:29 2018

@author: jacopo
"""


import tensorflow as tf
import scipy.io as sio
import numpy as np


# READING PARAM FILES FROM PRE-TRAINED POTSDAM NETWORK

M=sio.loadmat("Potsdam-denseprediction-distribution.mat")

# Layer 1 Parameters (convolutional, 7x7, 5->64)
l1f=sio.loadmat("l1f.mat")
l1b=sio.loadmat("l1b.mat")

f1=l1f['l1f']
b1=l1b['l1b']

# cleanup
del l1f,l1b

# Layer 2 Parameters (batch normalization)
l2m=sio.loadmat("l2m.mat")
l2b=sio.loadmat("l2b.mat")
l2x=sio.loadmat("l2x.mat")

m2=l2m['l2m']
b2=l2b['l2b']
x2=l2x['l2x']

#cleanup
del l2m,l2b,l2x

# Layer 6 Parameters (convolutional, 5x5, 64->64)
l6f=sio.loadmat("l6f.mat")
l6b=sio.loadmat("l6b.mat")

f6=l6f['l6f']
b6=l6b['l6b']

#cleanup
del l6f,l6b

# Layer 7 Parameters (batch normalization)
l7m=sio.loadmat("l7m.mat")
l7b=sio.loadmat("l7b.mat")
l7x=sio.loadmat("l7x.mat")

m7=l7m['l7m']
b7=l7b['l7b']
x7=l7x['l7x']

#cleanup
del l7m,l7b,l7x

# Layer 11 Parameters (convolutional, 3x3, 64->128)
l11f=sio.loadmat("l11f.mat")
l11b=sio.loadmat("l11b.mat")

f11=l11f['l11f']
b11=l11b['l11b']

#cleanup
del l11f,l11b

# Layer 12 Parameters (batch normalization)
l12m=sio.loadmat("l12m.mat")
l12b=sio.loadmat("l12b.mat")
l12x=sio.loadmat("l12x.mat")

m12=l12m['l12m']
b12=l12b['l12b']
x12=l12x['l12x']

#cleanup
del l12m,l12b,l12x

# Layer 16 Parameters (convolutional, 3x3, 128->256)
l16f=sio.loadmat("l16f.mat")
l16b=sio.loadmat("l16b.mat")

f16=l16f['l16f']
b16=l16b['l16b']

#cleanup
del l16f,l16b

# Layer 17 Parameters (batch normalization)
l17m=sio.loadmat("l17m.mat")
l17b=sio.loadmat("l17b.mat")
l17x=sio.loadmat("l17x.mat")

m17=l17m['l17m']
b17=l17b['l17b']
x17=l17x['l17x']

#cleanup
del l17m,l17b,l17x

# Layer 20 Parameters (deconvolutional, 3x3, 256->512, [2,2] upsampling)
l20f=sio.loadmat("l20f.mat")
l20b=sio.loadmat("l20b.mat")

f20=l20f['l20f']
b20=l20b['l20b']

#cleanup
del l20f,l20b

# Layer 21 Parameters (batch normalization)
l21m=sio.loadmat("l21m.mat")
l21b=sio.loadmat("l21b.mat")
l21x=sio.loadmat("l21x.mat")

m21=l21m['l21m']
b21=l21b['l21b']
x21=l21x['l21x']

#cleanup
del l21m,l21b,l21x

# Layer 24 Parameters (deconvolutional, 3x3, 512->512, [2,2] upsampling)
l24f=sio.loadmat("l24f.mat")
l24b=sio.loadmat("l24b.mat")

f24=l24f['l24f']
b24=l24b['l24b']

#cleanup
del l24f,l24b

# Layer 25 Parameters (batch normalization)
l25m=sio.loadmat("l25m.mat")
l25b=sio.loadmat("l25b.mat")
l25x=sio.loadmat("l25x.mat")

m25=l25m['l25m']
b25=l25b['l25b']
x25=l25x['l25x']

#cleanup
del l25m,l25b,l25x

# Layer 28 Parameters (deconvolutional, 3x3, 512->512, [2,2] upsampling)
l28f=sio.loadmat("l28f.mat")
l28b=sio.loadmat("l28b.mat")

f28=l28f['l28f']
b28=l28b['l28b']

#cleanup
del l28f,l28b

# Layer 29 Parameters (batch normalization)
l29m=sio.loadmat("l29m.mat")
l29b=sio.loadmat("l29b.mat")
l29x=sio.loadmat("l29x.mat")

m29=l29m['l29m']
b29=l29b['l29b']
x29=l29x['l29x']

#cleanup
del l29m,l29b,l29x

# Layer 32 Parameters (convolutional, 1x1, 512->6)
l32f=sio.loadmat("l32f.mat")
l32b=sio.loadmat("l32b.mat")

f32=l32f['l32f']
b32=l32b['l32b']

#cleanup
del l32f,l32b



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

#logits=conv_net(images)
#prediction=tf.nn.softmax(logits)
    
