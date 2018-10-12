#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 23:02:53 2017

@author: lab548
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:45:29 2017

@author: lab548
"""

import sys,time
sys.path.append("..")

#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import cPickle
#from spectrum import *
from numpy import arange
import Q7interface
import Queue
from tx.symbol_tx import *
from tx.FSK_tx import *


if len(sys.argv)<2:
    print 'please input 2 parameters:'
    print ' sample_frequency'
    print ' over_sample_rate'
    exit
else:
    sample_freq = float(sys.argv[1])
    osr = float(sys.argv[2])
    if len(sys.argv) > 3:
        offset = int(sys.argv[3])
        phase = float(sys.argv[4])
    else:
        offset = 0
        phase = 0
    print 'sample_frequency is ', sample_freq
    print 'over_sample_rate is ', osr
    print 'sample offset is', offset
    print 'phase offset is', phase

##q = Queue.Queue()
##tx_q = q
##rx_q = q
#==============================================================================
# sample_freq=1920000
# osr=64
# offset=0
# phase=0
# 
# print 'sample_frequency is ', sample_freq
# print 'over_sample_rate is ', osr
# print 'sample offset is', offset
# print 'phase offset is', phase
#==============================================================================

dataset={}
mod_order=6
#modlist=['bpsk','qpsk','8psk','16qam','64qam','2fsk','4fsk']
modlist=['64qam']
for k in range(1):
    tx_q = Q7interface.tx()
    rx_q = Q7interface.rx()
    
    data_input_q = Queue.Queue()
    
    mod=modlist[k]
    #mod_order=2**k
    send_task = SYMBOL_TX(mod_order,tx_q, data_input_q)
    #send_task = FSK_TX(mod_order,tx_q)
    send_task.start()
    #mod_order = mod_order*2   #global problem
    sym_num_display = 20
    input_sample_len = 115200
    time_step = 1./sample_freq
    sym_idx = np.round(np.arange(sym_num_display)*osr).astype(int)

#7----20db  2---10db
    for i in range(-9,11,1):
        #def input_data():
        data = np.empty(input_sample_len, dtype=complex)
        dataset[(mod,i)]=np.zeros([144,2,800],dtype=np.float32)
        pInput_real = (c_double*input_sample_len)()
        pInput_imag = (c_double*input_sample_len)()
        
        #while True:
        data_buf = rx_q.get(input_sample_len*4)
        ok_separater = C_lib.CSM_data_separater(data_buf,pInput_real,pInput_imag,input_sample_len)
        data.imag = pInput_imag[:]
        data.real = pInput_real[:]
        mean = np.sqrt(np.mean(np.abs(data)**2))     #normalization
        noise_amp = 10**(-i/10.0)
        data = data/mean + (noise_amp*np.random.rand(input_sample_len) + 1j*noise_amp*np.random.rand(input_sample_len))
        #data = data/mean
        data = data*np.exp(1j*phase)

        dataset[(mod,i)][:,0,:]=data.real.reshape((144,800))
        dataset[(mod,i)][:,1,:]=data.imag.reshape((144,800))

        print 'complete'
    #yield data 
    #del tx_q
    #del rx_q
#data1=dataset[('bpsk',2)]
#modpredict(data1)
cPickle.dump(dataset, file("64qam.dat", "wb" ) )
print 'done'


#ani = animation.FuncAnimation(fig, update, input_data)
#plt.show()


