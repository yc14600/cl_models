from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#import matplotlib.pyplot as plt
#import seaborn as sb
import numpy as np
import scipy as sp
import six
import os
import sys

from utils.train_util import get_next_batch,expand_nsamples,shuffle_data
from utils.model_util import *
from utils.distributions import Normal,OneHotCategorical,MultivariateNormalTriL
from hsvi.methods.svgd import SVGD



def update_distance(dists, x_train, current_id):
    for i in range(x_train.shape[0]):
        current_dist = np.linalg.norm(x_train[i,:]-x_train[current_id,:])
        dists[i] = np.minimum(current_dist, dists[i])
    return dists


def gen_random_coreset(x_train,y_train,coreset_size,clss=None):
    if clss is None:
        cln = np.sum(y_train.sum(axis=0)>1)
        clss = range(cln) 
    else:
        cln = len(clss)
    n_c = int(coreset_size/cln)
    r_c = coreset_size - n_c*cln 
    print('cln',cln, 'clss',clss)
        
    core_y, core_x = [],[]
    for c in clss:
        cids = y_train[:,c]==1
        core_y.append(y_train[cids][:n_c])
        core_x.append(x_train[cids][:n_c])
    if r_c > 0:
        core_y.append(y_train[:r_c])
        core_x.append(x_train[:r_c])
    core_y = np.vstack(core_y)
    core_x = np.vstack(core_x)
    core_y,core_x = shuffle_data(core_y,core_x)

    return core_x, core_y


def gen_kcenter_coreset(x_train, coreset_size):
    # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    dists = np.full(x_train.shape[0], np.inf)
    current_id = 0
    dists = update_distance(dists, x_train, current_id)
    idx = [ current_id ]
   
    for i in range(1, coreset_size):
        current_id = np.argmax(dists)
        dists = update_distance(dists, x_train, current_id)
        idx.append(current_id)

    return idx

def gen_rdproj_coreset(x_train, y_train, coreset_size,num_class,cls=None):
    c_size = int(coreset_size/num_class)
    r = coreset_size%num_class
    core_x,core_y = [],[]
    if cls is None:
        cls = range(num_class)
    for c in cls:
        c_n = c_size
        if c < r:
            c_n += 1          
        if len(y_train.shape) == 1:
            class_ids = (y_train==c).reshape(-1)
        else:
            class_ids = (y_train[:,c]==1).reshape(-1)
        #print(c,class_ids.shape)
        # random projection
        proj_mx = np.random.normal(size=(c_n,class_ids.sum()))
        core_x_c = np.matmul(proj_mx, x_train[class_ids])

        core_x.append(core_x_c)
        #print(c_n,y_train[class_ids][:c_n].shape,y_train[class_ids].shape)
        core_y.append(y_train[class_ids][:c_n])

    core_x = np.concatenate(core_x,axis=0)
    core_y = np.concatenate(core_y,axis=0)
    #print(core_x.shape,core_y.shape)
    # shuffle coreset
    rids = np.arange(core_x.shape[0])
    np.random.shuffle(rids)
    core_x = core_x[rids]
    core_y = core_y[rids]
    #print(core_x.shape,core_y.shape)
    return core_x,core_y


def gen_stein_coreset(init,core_y_data,qW,qB,n_samples,ac_fn,conv_W=None,LR=False,noise_std=0.001,sess=None):
    stein_core_x = tf.get_variable('stein_cx',initializer=init.astype(np.float32),dtype=tf.float32)
    print('gen stein coreset')
    if LR:
        stein_core_y = Normal(loc=tf.matmul(stein_core_x,qW)+qB,scale=noise_std)
    elif conv_W is not None:
        ## to do: change to general function ##
        h = forward_cifar_model(stein_core_x,conv_W)
        stein_core_y = forward_nets(qW,qB,h,ac_fn=ac_fn,bayes=True,num_samples=10)[-1]
    else:
        stein_core_y = forward_nets(qW,qB,stein_core_x,ac_fn=ac_fn,bayes=True,num_samples=10)[-1]
    lnp = tf.reduce_mean(stein_core_y.log_prob(core_y_data),axis=0)
    dlnp = tf.gradients(lnp,stein_core_x)
    svgd = SVGD()
    #print('shape check',stein_core_x.shape)
    core_sgrad = svgd.gradients(stein_core_x,dlnp[0])

    return stein_core_x,stein_core_y,core_sgrad

def aggregate_coreset(core_sets,core_y,coreset_type,num_heads,t,n_samples,sess):
    if num_heads == 1:  
        if 'stein' in coreset_type:         
            x_core_sets = np.concatenate(sess.run(core_sets[0]),axis=0)
        else:
            x_core_sets = np.concatenate(core_sets[0],axis=0)
        y_core_sets = np.concatenate(core_sets[1],axis=0)
        x_core_sets,y_core_sets = shuffle_data(x_core_sets,y_core_sets)
        core_y_data = expand_nsamples(y_core_sets,n_samples)
        return x_core_sets,y_core_sets, {core_y:core_y_data}
        #inference.reinitialize(task_id=t+1,coresets={'task':{core_y:core_y_data}})
    else:
        if 'stein' in coreset_type:
            x_core_sets = sess.run(core_sets[0])
        else:
            x_core_sets = core_sets[0]
        y_core_sets = core_sets[1]
        #x_core_sets,y_core_sets = shuffle_data(x_core_sets,y_core_sets)
        c_task = {}
        core_y_data = [None]*(t+1)
        for k in range(t+1):
            core_y_data[k] = expand_nsamples(y_core_sets[k],n_samples)
            c_task.update({core_y[k]:core_y_data[k]})
        return x_core_sets,y_core_sets, c_task

def train_coresets_final(core_sets,core_y,x_ph,y_ph,core_x_ph,coreset_type,num_heads,t,n_samples,batch_size,epoch,sess,inference,print_iter=10):
    x_core_sets,y_core_sets,c_cfg = aggregate_coreset(core_sets,core_y,coreset_type,num_heads,t,n_samples,sess)
    #inference.train_size *= (t+1)  
    if num_heads > 1:
        inference.data['task'] = {}  
        #inference.train_size = (t+1)*x_core_sets[0].shape[0]  
        inference.reinitialize(task_id=t,coresets={'task':c_cfg})
        task_optimizer = inference.optimizer['task']
        sess.run(tf.variables_initializer(task_optimizer[0].variables()))

        feed_dict = {}
        for k in range(t+1):
            feed_dict.update({core_x_ph[k]:x_core_sets[k]})
        
        L = 1 # number of iters
    else:
        #inference.train_size = x_core_sets.shape[0]
        #inference.reinitialize(task_id=t)
        #task_optimizer = inference.optimizer['task']
        #sess.run(tf.variables_initializer(task_optimizer[0].variables()))
        L = int(np.ceil(x_core_sets.shape[0]/batch_size))
    
    for e in range(epoch):
        ii = 0
        for _ in range(L):
            if num_heads == 1:
                x_batch,y_batch,ii = get_next_batch(x_core_sets,batch_size,ii,labels=y_core_sets)
                y_batch = expand_nsamples(y_batch,n_samples)
                feed_dict = {x_ph:x_batch,y_ph:y_batch}            

            info_dict = inference.update(scope='task',feed_dict=feed_dict)
        if (e+1) % print_iter == 0:
            print('coresets final model',e, info_dict['loss'])