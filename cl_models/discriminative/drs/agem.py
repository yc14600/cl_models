
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import csv
import copy
import six
import importlib
import os
import sys

import tensorflow as tf
import edward as ed

from edward.models import Normal
from cl_models import BCL_BNN
from cl_models import VCL
from utils.model_util import *
from utils.train_util import *
from utils.coreset_util import *
from utils.resnet_util import *
from base_models.gans import GAN
from functools import reduce

from .drs_cl import DRS_CL,MLE_Inference



class AGEM(DRS_CL):

    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,dropout=None,initialization=None,\
                    ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,conv_net_shape=None,strides=None,pooling=False,\
                    regularization=False,lambda_reg=0.0001,coreset_mode='online',batch_iter=5,task_type='split',\
                    net_type='dense',fixed_budget=True,mem_batch_size=256,*args,**kargs):

        self.mem_batch_size = mem_batch_size

        super(AGEM,self).__init__(net_shape=net_shape,x_ph=x_ph,y_ph=y_ph,num_heads=1,batch_size=batch_size,\
                    coreset_size=coreset_size,coreset_type=coreset_type,coreset_usage=coreset_usage,vi_type=vi_type,\
                    conv=conv,dropout=dropout,initialization=initialization,ac_fn=ac_fn,n_smaples=n_smaples,local_rpm=local_rpm,\
                    conv_net_shape=conv_net_shape,strides=strides,pooling=pooling,regularization=regularization,\
                    lambda_reg=lambda_reg,discriminant=False,lambda_dis=.001,ER=False,coreset_mode=coreset_mode,\
                    batch_iter=batch_iter,task_type=task_type,net_type=net_type,fixed_budget=fixed_budget)


    def config_inference(self,*args,**kargs):
        self.inference = AGEM_Inference(var_list=self.vars,grads=self.grads,optimizer=self.task_optimizer,memory=self.core_sets)

    
    def train_update_step(self,t,s,sess,feed_dict,err=0.,x_train_task=None,y_train_task=None,local_iter=0,*args,**kargs):
        assert(self.coreset_size > 0)
        
        x_batch, y_batch = feed_dict[self.x_ph], feed_dict[self.y_ph]
        
        if self.coreset_mode == 'ring_buffer':  
            self.update_ring_buffer(t,x_batch,y_batch)
            #print('check core sets', self.core_sets.keys())

        if t>0:   
            if self.task_type == 'split':
                clss_batch = np.sum(y_batch,axis=0) > 0
                clss_batch = np.argsort(clss_batch)[-np.sum(clss_batch):]
                #print('clss batch',clss_batch) 
                clss_mem = set(self.core_sets.keys()) - set(clss_batch)
                #print('clss mem',clss_mem)
                mem_x = np.vstack([self.core_sets[c] for c in clss_mem])
                #mem_y = [self.core_sets[c].shape[0] for c in clss_mem]
                mem_y = []
                for c in clss_mem:
                    tmp = np.zeros([self.core_sets[c].shape[0],self.net_shape[-1]])
                    tmp[:,c] = 1
                    mem_y.append(tmp)
                mem_y = np.vstack(mem_y)

            else:
                mem_x,mem_y = [],[]
                for c in self.core_sets.keys():
                    if c < t:
                        mem_x.append(self.core_sets[c][0])
                        mem_y.append(self.core_sets[c][1])
                mem_x = np.vstack(mem_x)
                mem_y = np.vstack(mem_y)

            mem_x,mem_y = shuffle_data(mem_x,mem_y)
            bids = np.random.choice(mem_x.shape[0],size=self.mem_batch_size)
            ref_feed_dict = {self.x_ph:mem_x[bids],self.y_ph:mem_y[bids]}
            if self.training is not None:
                ref_feed_dict.update({self.training:True})
            sess.run(self.inference.store_ref_grads,ref_feed_dict)

        self.inference.update(t=t,sess=sess,feed_dict=feed_dict)
        




class AGEM_Inference(MLE_Inference):

    def __init__(self,var_list,grads,optimizer=None,memory={},*args,**kargs):
        self.memory = memory
        super(AGEM_Inference,self).__init__(var_list,grads,optimizer)

    
    def config_train(self):

        ####### main code is from https://github.com/facebookresearch/agem ########
        self.ref_grads, self.projected_gradients_list = [],[]
        for v in range(len(self.var_list)):
            self.ref_grads.append(tf.Variable(tf.zeros(self.var_list[v].get_shape()), trainable=False))
            self.projected_gradients_list.append(tf.Variable(tf.zeros(self.var_list[v].get_shape()), trainable=False))

        self.store_ref_grads = [tf.assign(ref, grad) for ref, grad in zip(self.ref_grads, self.grads)]
        flat_ref_grads =  tf.concat([tf.reshape(grad, [-1]) for grad in self.ref_grads], 0)

        flat_task_grads = tf.concat([tf.reshape(grad, [-1]) for grad in self.grads], 0)

        with tf.control_dependencies([flat_task_grads]):
            dotp = tf.reduce_sum(tf.multiply(flat_task_grads, flat_ref_grads))
            ref_mag = tf.reduce_sum(tf.multiply(flat_ref_grads, flat_ref_grads))
            proj = flat_task_grads - ((dotp/ ref_mag) * flat_ref_grads)
            projected_gradients = tf.cond(tf.greater_equal(dotp, 0), lambda: tf.identity(flat_task_grads), lambda: tf.identity(proj))

            offset = 0
            store_proj_grad_ops = []
            for v in self.projected_gradients_list:
                shape = [d.value for d in v.shape]
                v_params = reduce((lambda x,y: x*y),shape)
                store_proj_grad_ops.append(tf.assign(v, tf.reshape(projected_gradients[offset:offset+v_params], shape)))
                offset += v_params
            self.store_proj_grads = tf.group(*store_proj_grad_ops)

            with tf.control_dependencies([self.store_proj_grads]):
                self.train_subseq_tasks = self.optimizer[0].apply_gradients(zip(self.projected_gradients_list, self.var_list),global_step=self.optimizer[1])
        
        grads_and_vars = list(zip(self.grads,self.var_list))
        self.train_first_task = self.optimizer[0].apply_gradients(grads_and_vars,global_step=self.optimizer[1])  


        
    def update(self,t,sess,feed_dict=None,*args,**kargs):
        if t == 0:
            sess.run(self.train_first_task, feed_dict)
        else:
            sess.run(self.train_subseq_tasks,feed_dict)

        return  