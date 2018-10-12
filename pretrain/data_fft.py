#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:50:03 2017

@author: lab548
"""

import numpy as np
import cPickle
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn

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

data3=np.zeros([LL,2,800],dtype=np.floating)
for m in key:
    data2=Xnew[m]
    for l in range(0,LL):
        y=data2[l,:]
        yy=fft(y)
#        yy=yy/max(abs(yy))
        data3[l,0,:] = yy.real               
        data3[l,1,:] = yy.imag
    Xf[m]=data3
    data3=np.zeros([LL,2,800],dtype=np.floating)  #qingling 00!!!!



data3=Xnew[('QPSK', 8)]
y=data3[9,:]
#plt.plot(c)
y2=fft(y)
yy=y2/max(abs(y2))
yy=np.fft.fftshift(yy)

plt.figure(1)
e=yy.imag[360:440]
plt.plot(e)
r=yy.real[360:440]
plt.plot(r)
