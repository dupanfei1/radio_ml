
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:50:03 2017

@author: lab548
"""

import os,random
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
from scipy.fftpack import fft,ifft


X1= cPickle.load(open("data/bpsk_0530.dat",'rb'))
X2= cPickle.load(open("data/qpsk_0530.dat",'rb'))

Xd=dict(X1.items()+X2.items())
Xnew={}
Xf={}

LL=2880

data1=np.zeros([LL,800],dtype=np.complexfloating)
key=sorted(Xd.keys())

for i in key:
    data=Xd[i]
    for k in range(0,LL):
        for p in range(0,800):
            data1[k,p]=(complex(data[k,0,p],data[k,1,p]))
    Xnew[i]=data1
    data1=np.zeros([LL,800],dtype=np.complexfloating)

y=np.zeros(800,dtype=np.complexfloating)

data3=np.zeros([LL,2,60],dtype=np.floating)
for m in key:
    data2=Xnew[m]
    for l in range(0,LL):
        y=data2[l,:]
        yy=fft(y)
        yy=np.fft.fftshift(yy)
#        yy=yy/max(abs(yy))
        data3[l,0,:] = yy.real[370:430]              
        data3[l,1,:] = yy.imag[370:430]
    Xf[m]=data3
    data3=np.zeros([LL,2,60],dtype=np.floating)  #qingling 00!!!!


Xd=Xf

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  
            lbl.append((mod,snr))
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
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(80, (2, 5), kernel_initializer="he_normal", name="conv1", activation="relu", padding="valid"))
model.add(Dropout(dr))

#model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(64, (1, 5),strides=(1,1), kernel_initializer="he_normal", name="conv2", activation="relu", padding="valid"))
model.add(Dropout(dr))


model.add(Flatten())
model.add(Dense(32, kernel_initializer="he_normal", activation="relu", name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), kernel_initializer="he_normal", activation="softmax",name="dense2"))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
nb_epoch = 120     # number of epochs to train on
batch_size = 256  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'modelfft.h5'
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
open('modelfft.json','w').write(json_string)

score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print score

#==============================================================================
# plt.figure()
# plt.title('Training performance')
# plt.plot(history.epoch, history.history['loss'], label='train loss+error')
# plt.plot(history.epoch, history.history['val_loss'], label='val_error')
#==============================================================================
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
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print "Overall Accuracy: ", cor / (cor+ncor)
    acc[snr] = 1.0*cor/(cor+ncor)
#dict2= sorted(acc.items(), key=lambda d:d[0])
dict1= sorted(acc.values())

plt.figure()
plt.title('Training performance')
plt.plot(snrs,dict1)

# Plot accuracy curve
plt.plot(snrs, map(lambda x: acc[x], snrs))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")


