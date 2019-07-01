
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:


import tensorflow as tf
#import edward as ed
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import scipy as sp
import sys
sys.path.append('/Users/ycaamz/code/')
from utils.train_util import gen_permuted_data, get_next_batch
from EWC.EWC_Model import EWC_Model 

# In[3]:


from edward.models import MultivariateNormalFullCovariance, MultivariateNormalDiag, OneHotCategorical
from edward.inferences import KLqp,MAP
from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


DATA_DIR = '../MNIST_data/'
    
data = input_data.read_data_sets(DATA_DIR,one_hot=True) 


# In[5]:


num_tasks = 2
TRAIN_SIZE = 10000
TEST_SIZE =  1000
hidden = [100,100]
batch_size = 200
lamb = 100.
num_iter = 10000
learning_rate = 0.5
print_iter = 1000


# In[6]:


X_TRAIN = data.train.images[:TRAIN_SIZE]
Y_TRAIN = data.train.labels[:TRAIN_SIZE]
X_TEST = data.test.images[:TEST_SIZE]
Y_TEST = data.test.labels[:TEST_SIZE]


# In[7]:


in_dim = X_TRAIN.shape[1]
out_dim = Y_TRAIN.shape[1]
net_shape = [in_dim]+hidden+[out_dim]


# In[9]:

model = EWC_Model(net_shape,lamb=lamb,num_iter=num_iter,learning_rate=learning_rate,batch_size=batch_size,print_iter=print_iter)

test_sets=[]
for t in range(num_tasks):
    x_train_task,x_test_task = gen_permuted_data(t,X_TRAIN,X_TEST)
    test_sets.append(x_test_task)
    model.fit(t,x_train_task,Y_TRAIN)
    print('performance on parameters at task '+str(t))
    for ts in test_sets:
        acc = model.predict(ts,Y_TEST)
        print(acc)



