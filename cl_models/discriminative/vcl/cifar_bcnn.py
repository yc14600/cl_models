
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:


import tensorflow as tf
import edward as ed
import matplotlib.pyplot as plt
from IPython import display
import seaborn as sb
import numpy as np
import scipy as sp
import csv
import copy
import six
import importlib
import os
path = os.getcwd()
import sys
sys.path.append(path+'/../')
sys.path.append(path+'/../hsvi/')

# In[3]:
from hsvi.hsvi import hsvi
from utils.model_util import *
from utils.train_util import *
from tensorflow.python.keras.datasets import cifar10,cifar100


# In[4]:



# In[5]:





# In[6]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[7]:


num_class = 10
hidden = [512,512]
batch_size = 500
learning_rate = 0.001
epoch = 100
#print_iter = 100
decay = (2000,0.1)
num_samples = 1
vi_type = 'KLqp_analytic'
scale = 1.

# In[8]:


y_train = one_hot_encoder(y_train.reshape(-1),num_class)
y_test = one_hot_encoder(y_test.reshape(-1),num_class)


# In[9]:

# standardize data
X = np.vstack((X_train,X_test))
X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
X_train = X[:y_train.shape[0],]
X_test = X[y_train.shape[0]:,]

TRAIN_SIZE = X_train.shape[0]
# In[10]:

X_train = X_train.astype(np.float32)
X_test=X_test.astype(np.float32)


# In[11]:

x_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,*X_train.shape[1:]])
y_ph = tf.placeholder(dtype=tf.int32,shape=[num_samples,batch_size,num_class])


# In[12]:
conv_W = []
parm_var_dict = {}
with tf.variable_scope('task'):
    # first layer conv2d
    filter_shape = [3,3,3,32]
    strides = [1,2,2,1]
    h,w,parm_var = build_bayesian_conv_bn_acfn(x_ph,0,filter_shape,strides=strides,local_rp=False)
    conv_W.append(w)
    parm_var_dict.update(parm_var)
    print('L1',h.shape)
    # second layer conv2d
    filter_shape = [3,3,32,32]
    strides = [1,1,1,1]
    h,w,parm_var = build_bayesian_conv_bn_acfn(h,1,filter_shape,strides=strides,local_rp=False)
    conv_W.append(w)
    parm_var_dict.update(parm_var)
    print('L2',h.shape)

    
    # pooling
    #h = tf.reduce_mean(h,axis=0)
    #h = tf.nn.dropout(h,keep_prob=0.25)
    h = tf.nn.max_pool(value=h,ksize=[1,2,2,1],strides=strides,padding='SAME')
    
   
    # thrid layer conv2d
    filter_shape = [3,3,32,64]
    strides = [1,1,1,1]
    h,w,parm_var = build_bayesian_conv_bn_acfn(h,2,filter_shape,strides=strides,local_rp=False)
    conv_W.append(w)
    parm_var_dict.update(parm_var)
    print('L3',h.shape)
   
    # fourth layer conv2d
    filter_shape = [3,3,64,64]
    strides = [1,1,1,1]
    h,w,parm_var = build_bayesian_conv_bn_acfn(h,3,filter_shape,strides=strides,local_rp=False)
    conv_W.append(w)
    parm_var_dict.update(parm_var)
    print('L4',h.shape)
   

    # pooling
    #h = tf.reduce_mean(h,axis=0)
    #h = tf.nn.dropout(h,keep_prob=0.25)
    h = tf.nn.max_pool(value=h,ksize=[1,2,2,1],strides=strides,padding='SAME')

    # flatten
    #h = tf.reduce_mean(h,axis=0)
    h = tf.reshape(h,shape=[batch_size,-1])
    print('flatten',h.shape)
    # dense layer
    net_shape = [h.shape[1].value]+hidden+[num_class]
    print('dense shape',net_shape)
    qW,qB,H,TS,qW_samples,qB_samples,parm_var = build_nets(net_shape,h,bayes=True,num_samples=num_samples,dropout=0.5)
    parm_var_dict.update(parm_var)



    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(learning_rate,
                                                global_step,
                                                decay[0], decay[1], staircase=True)
    global_optimizer = (tf.train.AdamOptimizer(learning_rate),global_step)


# In[30]:
task_var_cfg = gen_posterior_conf(conv_W+qW+qB)
print(len(task_var_cfg))


# In[31]:
inference = slvi.Hierarchy_SLVI(latent_vars={'task':task_var_cfg},data={'task':{H[-1]:y_ph}})
inference.initialize(vi_types={'task':vi_type},scale={H[-1]:scale},optimizer={'task':global_optimizer},train_size=10*TRAIN_SIZE)



# In[32]:


sess = ed.get_session()
tf.global_variables_initializer().run()


# In[ ]:


n_iter = int(np.ceil(X_train.shape[0]/batch_size))
for e in range(epoch):
    shuffle_inds = np.arange(X_train.shape[0])
    np.random.shuffle(shuffle_inds)
    X_train = X_train[shuffle_inds]
    y_train = y_train[shuffle_inds]
    err = 0.
    ii = 0
    for _ in range(n_iter):
        x_batch,y_batch,ii = get_next_batch(X_train,B=batch_size,labels=y_train,ii=ii)
        y_batch = np.expand_dims(y_batch,axis=0)
        y_batch = np.repeat(y_batch,num_samples,axis=0)
        feed_dict = {x_ph:x_batch,y_ph:y_batch}
        info_dict = inference.update(scope='task',feed_dict=feed_dict)
        err = info_dict['loss']
        kl,ll = sess.run([inference.kl,inference.ll],feed_dict=feed_dict)
        #print(e,_,err,kl,-ll) 
    print(e,err,ll)  


# In[ ]:


acc = predict(X_test,y_test,x_ph,H[-1],batch_size,sess)
print(acc)


# In[ ]:




# In[ ]:



