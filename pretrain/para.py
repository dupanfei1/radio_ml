
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
from keras.layers import Conv2D, MaxPooling2D,concatenate,Input
from keras.layers import Activation, Dropout, Flatten, Dense,Reshape
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import  ZeroPadding2D
from keras.regularizers import *
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
import h5py
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from collections import OrderedDict
from scipy.fftpack import fft,ifft

# Xd= cPickle.load(open("data/data10a-6.dat",'rb'))
Xd= cPickle.load(open("dataset/RML2016.10a_dict.dat",'rb'))

key=sorted(Xd.keys())
for t1 in range(0,201,20):
    for i in range(t1,t1+5,1):
        Xd.pop(key[i])




# for k in range(-6,19,2):
#     Xd.pop(('QAM64',k))
# for k in range(-6,19,2):
#     Xd.pop(('AM-SSB',k))
# for k in range(-6,19,2):
#     Xd.pop(('AM-DSB',k))
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


dr = 0.4 # dropout rate (%)
model = models.Sequential()
model.add(Reshape((1,2,128), input_shape=in_shp))
#model.add(Reshape([2,128,1]))
input=Input(shape=[1,2,128])
cnnup=Conv2D(128, (2,7),strides=(2,1),data_format='channels_first', kernel_initializer="glorot_normal", name="conv1", activation="relu", padding="same")(input)
# cnn1 = Dropout(dr, name='cldnn_conv1_drp')(cnn1)
print cnnup
cnnup=Conv2D(128, (1,7), strides=(1,1),data_format='channels_first',kernel_initializer="glorot_normal", name="conv2", activation="relu", padding="same")(cnnup)
cnnup = Dropout(dr, name='cldnn_conv2_drp')(cnnup)
# cnnup=Conv2D(128, (1,5), strides=(1,1),data_format='channels_first',kernel_initializer="glorot_normal", name="conv3", activation="relu", padding="same")(cnnup)
# cnnup = Dropout(dr, name='cldnn_conv3_drp')(cnnup)

cnndown=Conv2D(128, (2,5), strides=(2,1),data_format='channels_first',kernel_initializer="glorot_normal", name="conv4", activation="relu", padding="same")(input)
# cnndown = Dropout(dr, name='cldnn_conv4_drp')(cnndown)

# cnndown=Conv2D(128, (1,5), strides=(1,1),data_format='channels_first',kernel_initializer="glorot_uniform", name="conv5", activation="relu", padding="same")(cnndown)
# cnndown = Dropout(dr, name='cldnn_conv5_drp')(cnndown)
# cnndown=Conv2D(128, (1,5), strides=(1,1),data_format='channels_first',kernel_initializer="glorot_uniform", name="conv5_2", activation="relu", padding="same")(cnndown)
# cnndown = Dropout(dr, name='cldnn_conv5_2_drp')(cnndown)

cnnout=concatenate([cnndown,cnnup])
print cnndown
cnn_reshape=Reshape((128,256))(cnnout)

fl=Flatten()(cnn_reshape)

output=Dense(128,kernel_initializer="glorot_uniform", activation='relu',name="dense1")(fl)
output = Dropout(dr, name='dense_drp')(output)
output=Dense(len(classes),kernel_initializer="glorot_normal", activation='softmax',name="dense2")(output)

model1=Model(input,output)
model.add(model1)

#LeakyReLU and PReLU  activation
#adamxmy=optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
nb_epoch = 60     # number of epochs to train on
batch_size = 256  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'modelpara.h5'
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

# plot_model(model, to_file='modelpara.png',show_shapes=True)

json_string = model.to_json()
open('modelpara.json','w').write(json_string)

score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print score


plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.figure()

loss={}
loss['train']=history.history['loss']
loss['val']=history.history['val_loss']
loss['trainacc']=history.history['acc']
loss['valacc']=history.history['val_acc']


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Greens, labels=[]):
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
np.save('conf_all.npy', confnorm)

acc = {}
conf_snr=[]
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
    conf_snr.append(confnorm)

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print "Overall Accuracy: ", cor / (cor+ncor)
    acc[snr] = 1.0*cor/(cor+ncor)

dict1= sorted(acc.values())
conf_snr.append(acc)
np.save('conf_snr.npy', conf_snr)


plt.figure()
plt.title('Training performance')
plt.plot(snrs,dict1)

loss['snr'] = dict1
loss['snr1'] = acc
cPickle.dump(loss, file("para10a_64_0626.dat", "wb" ) )
# Plot accuracy curve
plt.plot(snrs, map(lambda x: acc[x], snrs))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN Classification Accuracy")
# plt.show()
