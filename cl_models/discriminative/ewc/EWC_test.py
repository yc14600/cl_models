
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:


import tensorflow as tf
#import edward as ed
#import matplotlib.pyplot as plt
#import seaborn as sb
import numpy as np
import scipy as sp
import csv


# In[3]:

from tensorflow.examples.tutorials.mnist import input_data


# In[4]:
import os
path = os.getcwd()
import sys
sys.path.append(path+'/../')

from utils.train_util import load_task_data,gen_next_task_data,gen_permuted_data,gen_class_split_data
from EWC_Model import EWC_Model


# In[5]:
args = sys.argv
dargs = {}
for i in range(1,len(args)):
    arg = args[i].split(':')
    dargs[arg[0]] = arg[1]

seed = int(dargs.get('seed',42))
print('seed',seed)
tf.set_random_seed(seed)
np.random.seed(seed)


DATA_DIR = '../datasets/MNIST_data/'
file_path = './mnist_ewc_permuted_sd'+str(seed)

# In[6]:

dataset = 'mnist'
num_tasks = 10
TRAIN_SIZE = 55000
TEST_SIZE =  10000
hidden = [100,100]
batch_size = 100
lamb = 100.
epoch = 10
#num_iter = 10000
learning_rate = 0.5
print_iter = 10
task_name = 'permuted'#'permuted','split_norepeat','split'#
diag_fisher = True


# In[7]:
if task_name == 'permuted':
       out_dim = 10
elif 'split' in task_name:
    if 'cifar' not in dataset:
        cl_cmb = np.arange(10)
        out_dim = 2
        #np.random.shuffle(cl_cmb)
    else:
        cl_cmb = np.arange(100)
        if num_heads == 1:
            out_dim = 100
            
        else:
            out_dim = 10

X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = load_task_data(task_name,DATA_DIR,TRAIN_SIZE=TRAIN_SIZE,TEST_SIZE=TEST_SIZE,dataset=dataset,out_dim=out_dim)


in_dim = X_TRAIN.shape[1]

net_shape = [in_dim]+hidden+[out_dim]


# In[8]:


if task_name=='split' and num_tasks > 5:
    raise TypeError('DO NOT SUPPORT split_norepeat WITH MORE THAN 5 TASKS!')


# In[9]:


model = EWC_Model(net_shape,lamb=lamb,num_epoch=epoch,learning_rate=learning_rate,batch_size=batch_size,print_iter=print_iter,diag_fisher=diag_fisher)


# In[10]:


#plt.imshow(X_TRAIN[:batch_size].reshape(28,28))


# In[11]:


# In[12]:


test_sets=[]
cl_k = 0
acc_records = []
for t in range(num_tasks):
    acc_records.append([])
    if task_name == 'permuted':
        x_train_task,x_test_task = gen_permuted_data(t,X_TRAIN,X_TEST)
        y_train_task = Y_TRAIN
        test_sets.append([x_test_task,Y_TEST])
    elif 'split' in task_name:  
        if task_name == 'split_norepeat':
            x_train_task,y_train_task,x_test_task,y_test_task = gen_class_split_data(t,TRAIN_SIZE,TEST_SIZE,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,cls=cl_cmb[cl_k:cl_k+2])
            cl_k+=2
        else:
            x_train_task,y_train_task,x_test_task,y_test_task = gen_class_split_data(t,TRAIN_SIZE,TEST_SIZE,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST)
        test_sets.append([x_test_task,y_test_task])

    model.fit(t,x_train_task,y_train_task)
    print('performance on parameters at task '+str(t))
    for ts in test_sets:
        acc = model.predict(ts[0],ts[1])
        print(acc)
        acc_records[-1].append(acc)
    print('avg acc',np.mean(acc_records[-1]))

with open(file_path+'_'+'accuracy_record.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(acc_records)):
        writer.writerow(acc_records[t])






