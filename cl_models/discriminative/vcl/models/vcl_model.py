
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:
#import matplotlib.pyplot as plt
#from IPython import display
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
# In[3]:

from .bcl_base_bnn import BCL_BNN
from utils.model_util import *
from utils.train_util import *
from utils.coreset_util import *


# In[4]:

from hsvi.methods.svgd import SVGD
from edward.models import Normal,MultivariateNormalTriL
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.datasets import cifar10,cifar100




class VCL(BCL_BNN):
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,dropout=None,initialization=None,\
                    ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,*args,**kargs):

        super(VCL,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,dropout,initialization,ac_fn,n_smaples,local_rpm)

        self.define_model(initialization,dropout)

        self.x_core_sets,self.y_core_sets = None, None
        if self.coreset_size > 0:
            if num_heads > 1:
                self.config_coresets(self.qW_list,self.qB_list,self.conv_W)
            else:
                self.config_coresets(self.qW,self.qB,self.conv_W)

        return

    
    def define_model(self,initialization=None,dropout=None):
        
        super(VCL,self).define_model(initialization,dropout)
        
        if self.conv :
            self.task_var_cfg = gen_posterior_conf(self.conv_W+self.qW+self.qB)

        else:
            self.task_var_cfg = gen_posterior_conf(self.qW+self.qB)

        return


    def train_update_step(self,t,s,sess,feed_dict,kl=0.,ll=0.,err=0.,local_iter=10,*args,**kargs):
        if self.coreset_size > 0:
            if t > 0 and self.coreset_usage != 'final':
                if self.num_heads > 1:
                    for k in range(t):
                        feed_dict.update({self.core_x_ph[k]:self.x_core_sets[k]})
                else:    
                    feed_dict.update({self.core_x_ph:self.x_core_sets})
    
            if self.coreset_type == 'stein' and (s+1)%local_iter==0:
                sess.run(self.stein_train)
        
        info_dict = self.inference.update(scope='task',feed_dict=feed_dict)
    
        _kl,_ll = sess.run([self.inference.kl,self.inference.ll],feed_dict=feed_dict)
        kl += _kl
        ll += -_ll
        err += info_dict['loss']  
        return kl,ll,err


    def config_next_task_parms(self,t,sess,*args,**kargs):
        # update parameter configurations
    
        task_var_cfg = {}
        # update priors
        if self.num_heads > 1:
            self.qW,self.qB = self.qW_list[t+1],self.qB_list[t+1]
            pW,pB = self.qW_list[t],self.qB_list[t]

            for l in range(len(self.qW)-1):
                update_variable_tables(pW[l],self.qW[l],sess,task_var_cfg)         
                update_variable_tables(pB[l],self.qB[l],sess,task_var_cfg)
                
            # configure head layer for new task    
            npw = Normal(loc=tf.zeros_like(self.qW[-1]),scale=tf.ones_like(self.qW[-1]))
            task_var_cfg[npw] = self.qW[-1]
            npb = Normal(loc=tf.zeros_like(self.qB[-1]),scale=tf.ones_like(self.qB[-1]))
            task_var_cfg[npb] = self.qB[-1]

            # configure head layer for all seen tasks
            for k in range(t+1):
                update_variable_tables(self.qW_list[k][-1],self.qW_list[k][-1],sess,task_var_cfg)
                update_variable_tables(self.qB_list[k][-1],self.qB_list[k][-1],sess,task_var_cfg)

        else:
            for l in range(len(self.qW)):
                # update weights prior and trans
                update_variable_tables(self.qW[l],self.qW[l],sess,task_var_cfg)             
                # update bias prior and trans
                update_variable_tables(self.qB[l],self.qB[l],sess,task_var_cfg)

        if self.conv :
            for qw in self.conv_W:
                update_variable_tables(qw,qw,sess,task_var_cfg)

        self.task_var_cfg = task_var_cfg

        return 




    def update_task_data_and_inference(self,sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,\
                                    original_batch_size=500,cl_n=2,cl_k=0,cl_cmb=None,*args,**kargs):    

        if self.coreset_type == 'distill':
            self.data_distill(t,sess,*args,**kargs)

        if self.coreset_size>0 and self.coreset_usage != 'final':
            self.x_core_sets,self.y_core_sets,c_cfg = aggregate_coreset(self.core_sets,self.core_y,self.coreset_type,self.num_heads,t,self.n_samples,sess)
        
        ## re-configure priors ##
        self.config_next_task_parms(t,sess,*args,**kargs)

        # update data and inference for next task         
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = super(VCL,self).update_task_data(sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,original_batch_size,cl_n,cl_k,cl_cmb)

        self.inference.latent_vars['task'] = self.task_var_cfg
        if self.coreset_size>0 and self.coreset_usage != 'final':
            #self.x_core_sets,self.y_core_sets,c_cfg = aggregate_coreset(self.core_sets,self.core_y,self.coreset_type,self.num_heads,t,self.n_samples,sess)
            
            if self.num_heads > 1:
                self.inference.data['task'] = {self.H_list[t+1][-1]:self.y_ph}
                self.inference.reinitialize(task_id=t+1,coresets={'task':c_cfg})
                sess.run(tf.variables_initializer(self.task_optimizer[0].variables()))
            else:
                self.inference.reinitialize(task_id=t+1,coresets={'task':c_cfg})
            
        else:
            
            if self.num_heads > 1:
                self.inference.data['task'] = {self.H_list[t+1][-1]:self.y_ph}
                self.inference.reinitialize(task_id=t+1)
                sess.run(tf.variables_initializer(self.task_optimizer[0].variables()))
            else:
                self.inference.reinitialize(task_id=t+1)

        return x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss

