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
from .vcl_model import VCL
from .coreset import gen_random_coreset

from utils.model_util import *
from utils.train_util import *

from tensorflow.contrib.distributions import Normal


class VCL_KD(VCL):

    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                coreset_usage='regret',vi_type='KLqp_analytic',conv=False,\
                dropout=None,initialization=None,ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,\
                conv_net_shape=None,strides=None,pooling=False,enable_kd_reg=True,enable_vcl_reg=True,*args,**kargs):
        
        
        self.X_hat = None
        self.enable_kd_reg = enable_kd_reg
        self.enable_vcl_reg = enable_vcl_reg

        super(VCL_KD,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,dropout,initialization,ac_fn,n_smaples,local_rpm,\
                    conv_net_shape,strides,pooling)


        return
 


    def data_distill(self,t,sess,x_train_task,y_train_task,lr=0.0001,iters=1000,print_iter=100,rpath='./',clss=None,*args,**kargs):
        ## distill data for whole training set ##
        if t==0:
            self.distill_opt = config_optimizer(lr,'step',scope='distill')

        with tf.variable_scope('distill',reuse=tf.AUTO_REUSE):
            ## to do: change to reuse with assign op ##
            x_hat_t,Y_hat_t = gen_random_coreset(x_train_task,y_train_task,self.coreset_size,clss)
            X_hat_t = tf.get_variable(name='X_hat_'+str(t),dtype=tf.float32,initializer=x_hat_t)
       
        ## only consider single head for now ##
        if self.num_heads > 1:
            raise NotImplementedError('Not support multihead in distillation currently.')
        
        ## to do: add code for conv later ##
        #if self.conv:
        #    raise NotImplementedError('Not support conv=True yet.')
        
        ## prepare input list of each layer ##
        if self.conv:
            H_hat = [self.conv_h] + [tf.squeeze(h) for h in self.H[:-1]]
        else:
            H_hat = [X_hat_t] + [tf.squeeze(h) for h in self.H[:-1]]
        KL = 0.
        x = x_train_task
        for w,b,x_hat in zip(self.qW,self.qB,H_hat):
            w_mu = sess.run(self.parm_var[w][0])
            w_sigma = sess.run(tf.exp(self.parm_var[w][1]))
            b_mu = sess.run(self.parm_var[b][0])
            b_sigma = sess.run(tf.exp(self.parm_var[b][1]))

            a = get_acts_dist(x,w_mu,w_sigma,b_mu,b_sigma)
            x = forward_dense_layer(x,w_mu,b_mu,self.ac_fn)

            a_hat = get_acts_dist(x_hat,w_mu,w_sigma,b_mu,b_sigma)

            log_q_a = calc_log_marginal(a_hat,a_hat.sample())
            log_p_a = calc_log_marginal(a,a_hat.sample())
            KL += tf.reduce_mean(log_q_a) - tf.reduce_mean(log_p_a)

        grads = tf.gradients(KL, X_hat_t)
        
        #print('grads',grads)
        distill_train = self.distill_opt[0].apply_gradients([(grads[0],X_hat_t)], global_step=self.distill_opt[1])
        print('train data distill')  
        reinitialize_scope(['distill'],sess)
        sess.run(tf.variables_initializer(self.distill_opt[0].variables()))
        for _ in range(iters):
            x_hat = sess.run(X_hat_t)
            __,kl = sess.run([distill_train,KL],feed_dict={self.x_ph:x_hat})
            if (_+1)%print_iter==0:
                print('iter {}: KL {}'.format(_+1,kl))
            if kl < 0.01:
                print('early stop satisfied: KL {}'.format(kl))
                break
                
        samples = sess.run(X_hat_t)
    
        self.core_sets[0].append(samples)
        self.core_sets[1].append(Y_hat_t)

        fig = plot(samples.reshape(-1,28,28),shape=(2,int(np.ceil(self.coreset_size/2))))
        fig.savefig(os.path.join(rpath,'data_rep_task'+str(t)+'.png'))

        return 

    
    def config_next_task_parms(self,t,sess,x_train_task,y_train_task,clss,rpath='./',*args,**kargs):
        ## only consider single head for now ##

        if self.enable_kd_reg:
            self.task_var_cfg = {}
            if self.coreset_type == 'stein':
                core_x = np.vstack(sess.run(self.core_sets[0]))
            else:
                core_x = np.vstack(self.core_sets[0])
            core_y = np.vstack(self.core_sets[1])
            y_lables = np.sum(core_y,axis=0)
                
            #print('core x',type(core_x))
            for c,y in enumerate(y_lables):
                if y == 0:
                    continue  
                x_hat = core_x[core_y[:,c]==1]                      
                for w,b in zip(self.qW,self.qB):
                    #print('x_hat {},w {}, b {}'.format(x_hat.shape,w.shape,b.shape))
                    pre_w_mu = sess.run(self.parm_var[w][0])
                    pre_w_sigma = sess.run(tf.exp(self.parm_var[w][1]))
                    pre_b_mu = sess.run(self.parm_var[b][0])
                    pre_b_sigma = sess.run(tf.exp(self.parm_var[b][1]))
                    a_dist = Wrapped_Marginal(get_acts_dist(x_hat,pre_w_mu,pre_w_sigma,pre_b_mu,pre_b_sigma))
                                    
                    w_mu = self.parm_var[w][0]
                    w_sigma = tf.exp(self.parm_var[w][1])
                    b_mu = self.parm_var[b][0]
                    b_sigma = tf.exp(self.parm_var[b][1])

                    qa_dist = Wrapped_Marginal(get_acts_dist(x_hat,w_mu,w_sigma,b_mu,b_sigma))
                    self.task_var_cfg[a_dist] = qa_dist
                    x_hat = forward_dense_layer(x_hat,pre_w_mu,pre_b_mu,self.ac_fn)
            if self.enable_vcl_reg:
                super(VCL_KD,self).config_next_task_parms(t,sess,*args,**kargs)
        else:
            super(VCL_KD,self).config_next_task_parms(t,sess,*args,**kargs)

        return
    

