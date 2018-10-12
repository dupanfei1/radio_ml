#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 20:30:54 2017

@author: lab548
"""

import os
#os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import keras.models as models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Reshape
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import  ZeroPadding2D
from keras.regularizers import *
from keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
import h5py
from keras.models import model_from_json
from collections import OrderedDict
from keras.layers.recurrent import LSTM

X1 = cPickle.load(open("bpsknew.dat",'rb'))
X2 = cPickle.load(open("qpsknew.dat",'rb'))
X3 = cPickle.load(open("16qamnew.dat",'rb'))
X4=  cPickle.load(open("4fsknew.dat",'rb'))
X5= cPickle.load(open("2fsknew.dat",'rb'))
X6 = cPickle.load(open("8psknew.dat",'rb'))
#X7 = cPickle.load(open("64qamnew.dat",'rb'))

Xd1=dict(X1.items()+X2.items()+X3.items()+X4.items()+X5.items()+X6.items())
Xd={}
keys=Xd1.keys()
for key in keys:
    x2=np.zeros([1440,2,400],dtype=np.floating)
    xt1=Xd1[key]
    x2[:,0,:]=xt1[:,0,:].reshape((1440,400))
    x2[:,1,:]=xt1[:,1,:].reshape((1440,400))
    #xt2=xt1[0:720,:,:]
    Xd[key]=x2
cPickle.dump( Xd, file("data400.dat", "wb" ) )


snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
classes = mods

dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Reshape(in_shp + [1], input_shape=in_shp)) #[1]+[2,128]= [1, 2, 128]
#model.add(Reshape([2,128,1]))
#==============================================================================
# model.add(ZeroPadding2D((0, 2)))
# model.add(Conv2D(256, (2, 7), kernel_initializer="glorot_normal", name="conv1", activation="relu", padding="valid"))
# model.add(Dropout(dr))
#==============================================================================

model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(128, (2, 8), kernel_initializer="glorot_normal", name="conv2", activation="relu", padding="valid"))
model.add(Dropout(dr))

model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(64, (1, 6), kernel_initializer="glorot_normal", name="conv3", activation="relu", padding="valid"))
model.add(Dropout(dr))

model.add(Flatten())
model.add(Dense(128, kernel_initializer="he_normal", activation="relu", name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), kernel_initializer="he_normal", name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))

#LeakyReLU and PReLU  activation
#adamxmy=optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer='adamax',metrics=['accuracy'])
model.summary()
nb_epoch = 15     # number of epochs to train on
batch_size = 512  # zhuyi!!!!!training batch size, 

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'data400.h5'
history=model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    #show_accuracy=True,
    verbose=1,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
model.load_weights(filepath)

json_string = model.to_json()
open('data400.json','w').write(json_string)

score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print score

plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.figure()
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)

#==============================================================================
# myacc={}
# for cl in range(0,6):
#     
#     acc1=[]
#     #cl=5
#     for i in range(-9,11,1):
#         tt=0  
#         xx=Xd[(classes[cl],i)]
#         yy = model.predict(xx, batch_size=10)
#         for i in range(0,yy.shape[0]):
#             if np.argmax(yy[i,:])==cl:
#                 tt=tt+1
#         acc1.append(float(tt)/yy.shape[0])
#     myacc[classes[cl]]=acc1
# cPickle.dump(myacc, file("cln_800.dat", "wb" ) )
#==============================================================================

myconf={}
acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    myconf[snr]=confnorm
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print "Overall Accuracy: ", cor / (cor+ncor)
    acc[snr] = 1.0*cor/(cor+ncor)

#cPickle.dump(myconf, file("confcln_800.dat", "wb" ) )
#dict2= sorted(acc.items(), key=lambda d:d[0])
dict1= sorted(acc.values())

plt.figure()
plt.title('Training performance')
plt.plot(snrs,dict1)

# Plot accuracy curve
plt.plot(snrs, map(lambda x: acc[x], snrs))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy For SNRs")
