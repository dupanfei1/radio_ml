
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
from keras.layers import Conv2D, MaxPooling2D,concatenate,Input,Bidirectional
from keras.layers import Activation, Dropout, Flatten, Dense,Reshape
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import  ZeroPadding2D
from keras.regularizers import *
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras,time
import h5py
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from collections import OrderedDict
from scipy.fftpack import fft,ifft

Xd= cPickle.load(open("data/data10a-6.dat",'rb'))
for k in range(-6,19,2):
    Xd.pop(('QAM64',k))
for k in range(-6,19,2):
    Xd.pop(('AM-SSB',k))
for k in range(-6,19,2):
    Xd.pop(('AM-DSB',k))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
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
model.add(Reshape((1,2,128), input_shape=in_shp))
#model.add(Reshape([2,128,1]))
input=Input(shape=[1,2,128])
cnn1=Conv2D(164, (2,7),strides=(2,1),data_format='channels_first', kernel_initializer="glorot_normal", name="conv1", activation="relu", padding="same")(input)
cnn1 = Dropout(0.4, name='cldnn_conv2_drp')(cnn1)
print cnn1
#==============================================================================
# #model.add(ZeroPadding2D((0, 2)))
# cnn2=Conv2D(NUM_CNN_OUTPUTS, CNN_KERNEL, strides=(1,1),data_format='channels_first',kernel_initializer="glorot_normal", name="conv2", activation="relu", padding="same")(cnn1)       
# cnn2 = Dropout(dr, name='cldnn_conv3_drp')(cnn2)
#==============================================================================

#model.add(ZeroPadding2D((0, 2)))
cnn3=Conv2D(128, (1,5), strides=(1,1),data_format='channels_first',kernel_initializer="glorot_normal", name="conv3", activation="relu", padding="same")(cnn1)
cnn3 = Dropout(0.5, name='cldnn_conv5_drp')(cnn3)

print cnn3

#==============================================================================
# cnnd1=Conv2D(64, (2,3), strides=(2,1),data_format='channels_first',kernel_initializer="glorot_normal", name="conv5", activation="relu", padding="same")(input)       
# cnndown = Dropout(dr, name='cldnn_conv1_drp')(cnnd1)
#==============================================================================

cnndown2=Conv2D(128, (2,5), strides=(2,1),data_format='channels_first',kernel_initializer="glorot_normal", name="conv4", activation="relu", padding="same")(input)       
cnndown2 = Dropout(0.4, name='cldnn_conv4_drp')(cnndown2)

#cnnout=concatenate([cnndown,cnndown2])
#cnnout=Reshape((128,1,128))(cnnout)
cnnout=concatenate([cnndown2,cnn3])
print cnndown2
cnn_reshape=Reshape((128,256))(cnnout)
lstm2=Bidirectional(LSTM(128,dropout=dr,return_sequences=False,name='cldnn_lstm2'),merge_mode='sum')(cnn_reshape)
#==============================================================================
# lstm1=LSTM(NUM_LSTM_OUTPUTS,dropout=dr,return_sequences=True,name='cldnn_lstm1')(cnn_reshape)
# 
# lstm2=LSTM(NUM_LSTM_OUTPUTS,dropout=dr,return_sequences=False,name='cldnn_lstm2')(lstm1)
#==============================================================================
#==============================================================================
# fl=Flatten()(cnn_reshape)
# 
# output=Dense(128,kernel_initializer="he_normal", activation='relu',name="dense1")(fl)
# output = Dropout(dr, name='dense_drp')(output)
#==============================================================================
output=Dense(len(classes),kernel_initializer="he_normal", activation='softmax',name="dense2")(lstm2)

model1=Model(input,output)
model.add(model1)
#model.add(Reshape([len(classes)]))

#LeakyReLU and PReLU  activation
#adamxmy=optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer='adamax',metrics=['accuracy'])
model.summary()
nb_epoch = 60     # number of epochs to train on
batch_size = 256  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'modelpabi10b.h5'


start = time.time()
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
end = time.time()
print "finish time: %f s" % (end - start)

# plot_model(model, to_file='modelpara.png',show_shapes=True)

json_string = model.to_json()
open('modelpabi10b.json','w').write(json_string)

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
# cPickle.dump(loss, file("pabi10b.dat", "wb" ) )

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
plt.show()




