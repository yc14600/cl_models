
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
from edward.models import RandomVariable

from .bcl_base_bnn import BCL_BNN
from utils.model_util import *
from utils.train_util import *
from utils.coreset_util import *



class VCL(BCL_BNN):
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,dropout=None,initialization=None,\
                    ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,conv_net_shape=None,strides=None,pooling=False,\
                    coreset_mode='offline',B=-1,task_type='split',*args,**kargs):

        super(VCL,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,dropout,initialization,ac_fn,n_smaples,local_rpm,conv_net_shape,\
                    strides,pooling,coreset_mode=coreset_mode,task_type=task_type,*args,**kargs)
        self.B = B
        print('init',initialization)
        self.define_model(initialization,dropout)

        self.x_core_sets,self.y_core_sets = None, None
        if self.coreset_size > 0:
            if num_heads > 1:
                self.config_coresets(self.qW_list,self.qB_list,self.conv_W)
            else:
                self.config_coresets(self.qW,self.qB,self.conv_W)

        return

    
    def define_model(self,initialization=None,dropout=None,*args,**kargs):
        
        super(VCL,self).define_model(initialization,dropout)
        
        if self.conv :
            self.task_var_cfg = gen_posterior_conf(self.conv_W+self.qW+self.qB)

        else:
            self.task_var_cfg = gen_posterior_conf(self.qW+self.qB)

        return


    def train_update_step(self,t,s,sess,feed_dict,kl=0.,ll=0.,err=0.,local_iter=10,*args,**kargs):
        if self.B < 0:
            if self.coreset_size > 0:
                if t > 0 and self.coreset_usage != 'final':
                    if self.num_heads > 1:
                        for k in range(t):
                            feed_dict.update({self.core_x_ph[k]:self.x_core_sets[k]})
                    else:    
                        feed_dict.update({self.core_x_ph:self.x_core_sets})
        
                if self.coreset_type == 'stein' and (s+1)%local_iter==0:
                    sess.run(self.stein_train)
        else:
            if t > 0:
                n = int(self.B/(t+1)) # number of tasks in one particle batch
                r = int(self.B%(t+1))
                
                bids = np.random.choice(len(self.x_core_sets),size=self.batch_size*(n*t+r))

                cids = np.random.choice(self.batch_size,size=self.batch_size*n)
                coreset_x = np.vstack([self.x_core_sets[bids],feed_dict[self.x_ph][cids]])
                #print('check shape',self.y_core_sets.shape,self.y_ph.shape)
                coreset_y = np.vstack([self.y_core_sets[bids],np.squeeze(feed_dict[self.y_ph][cids])])
                feed_dict[self.x_ph] = coreset_x
                feed_dict[self.y_ph] = np.expand_dims(coreset_y,axis=0)

            if self.coreset_type == 'stein' and (s+1)%local_iter==0:
                sess.run(self.stein_train)

            
        info_dict = self.inference.update(scope='task',feed_dict=feed_dict,sess=sess)
    
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


    def update_task_data(self,sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,\
                        original_batch_size=500,cl_n=2,cl_k=0,cl_cmb=None,train_size=-1,\
                        test_size=-1,*args,**kargs): 

        if self.coreset_type == 'distill':
            self.data_distill(t,sess,*args,**kargs)

        if self.coreset_size>0 and self.coreset_usage != 'final':
            if self.coreset_mode == 'offline':
                self.x_core_sets,self.y_core_sets,c_cfg = aggregate_coreset(self.core_sets,self.core_y,self.coreset_type,\
                                                                        self.num_heads,t,self.n_samples,sess)
            else:    
                #cnum = len(self.core_sets.keys()) 
                #for c in self.curr_buf.keys():                  
                #    self.core_sets[c] = self.curr_buf[c]
                #self.core_sets[1].append(self.curr_buf[1])
                c_cfg = None

        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = super(VCL,self).update_task_data(sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,\
                                                                                            out_dim,original_batch_size,cl_n,cl_k,cl_cmb,train_size,test_size)
        
        if self.B > 0:
            c_cfg = {}
        return x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss,c_cfg


    def update_inference(self,sess,t,c_cfg,*args,**kargs):

        ## re-configure priors ##
        self.config_next_task_parms(t,sess,*args,**kargs)
        
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



    def update_task_data_and_inference(self,sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,\
                                    original_batch_size=500,cl_n=2,cl_k=0,cl_cmb=None,train_size=-1,test_size=-1,*args,**kargs):    

        ## update data for next task         
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss,c_cfg = self.update_task_data(sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,\
                                                                                                out_dim,original_batch_size,cl_n,cl_k,cl_cmb,train_size,test_size)
        ## update inference for next task
        self.update_inference(sess,t,c_cfg)

        return x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss


    def get_tasks_vec(self,sess,t,test_sets,test_sample=False):
        vecs,nlls = [],[]
        vars_list = get_vars_by_scope(scope='task')
        for ts in test_sets:
            #print('ts shape',ts[0].shape,ts[1].shape)
            if not test_sample:
                ty = np.expand_dims(ts[1],axis=0)
                ty = np.repeat(ty,self.n_samples,axis=0)
                feed_dict={self.x_ph:ts[0],self.y_ph:ty}
                if self.coreset_size > 0:
                    if t > 0 and self.coreset_usage != 'final':
                        if self.num_heads > 1:
                            for k in range(t):
                                feed_dict.update({self.core_x_ph[k]:self.x_core_sets[k]})
                        else:    
                            feed_dict.update({self.core_x_ph:self.x_core_sets})
                
                grads = tf.gradients(-self.inference.ll, vars_list)
            else:
                ty = np.expand_dims(ts[1],axis=0)
                tx = np.expand_dims(ts[0],axis=0)

                if isinstance(self.H[-1],RandomVariable):
                    ty = np.expand_dims(ty,axis=0)
                    ty = np.repeat(ty,self.n_samples,axis=0)
                    feed_dict={self.x_ph:tx,self.y_ph:ty}
                    nll_i = -self.H[-1].log_prob(self.y_ph)
                    nll = tf.reduce_mean(nll_i)
                else:
                    feed_dict={self.x_ph:tx,self.y_ph:ty}
                    nll_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.H[-1],labels=self.y_ph)
                    nll = tf.reduce_mean(nll_i)
                    
                nlls.append(sess.run(nll,feed_dict))
                
                grads = tf.gradients(nll, vars_list)
                for i,g in enumerate(grads):
                    grads[i] = tf.reshape(g,[-1])
                

            g_vec = np.concatenate(sess.run(grads,feed_dict))

            vecs.append(g_vec)

        return vecs,np.array(nlls)