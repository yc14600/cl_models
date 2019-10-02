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
from utils.model_util import *
from utils.train_util import *
from tensorflow.contrib.distributions import Normal


class VCL_KD(VCL):

    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,conv=False,\
                dropout=None,initialization=None,ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,*args,**kargs):
        
        vi_type='KLqp'
        self.X_hat_size = coreset_size
        self.X_hat = None
        coreset_type='distill'
        coreset_usage='distill'
        coreset_size = 0  ## no use of usual coresets
        super(VCL_KD,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,dropout,initialization,ac_fn,n_smaples,local_rpm)


        return
 


    def data_distill(self, X, sess,t,lr=0.01,iters=100,rpath='./'):
        ## distill data for whole training set ##
        if self.X_hat is None:
            ## first task, init X_hat optimizer ##
            self.distill_opt = config_optimizer(0.001,'step',scope='distill')
            self.init_val = tf.Variable(np.zeros([self.X_hat_size,*X.shape[1:]]),dtype=tf.float32)

        with tf.variable_scope('distill',reuse=tf.AUTO_REUSE):
            rids = np.random.choice(X.shape[0],size=self.X_hat_size,replace=False)
            self.X_hat = tf.get_variable(name='X_hat_'+str(t),dtype=tf.float32,initializer=X[rids])
            #self.X_hat = tf.sigmoid(self.X_hat_var)

        #else:
            ## add X_hat to training set ##
        #    X = np.vstack([X,sess.run(self.X_hat)])       
        ## only consider single head for now ##
        if self.num_heads > 1:
            raise NotImplementedError('Not support multihead in distillation currently.')
        
        ## prepare input list of each layer ##
        ## to do: add code for conv later ##
        if not self.conv:
            ## self.x_ph.shape[0] == None ##
            H = sess.run(self.H[:-1],feed_dict={self.x_ph:X})
        else:
            raise NotImplementedError('Not support conv=True yet.')
        H = [X] + [tf.squeeze(h) for h in H]
        H_hat = [self.X_hat] + [tf.squeeze(h) for h in self.H[:-1]]
        KL = 0.
        A = []
        #print(self.qW,self.qB,self.H)
        #print(self.parm_var)
        for w,b,x,x_hat in zip(self.qW,self.qB,H,H_hat):
            w_mu = sess.run(self.parm_var[w][0])
            w_sigma = sess.run(tf.exp(self.parm_var[w][1]))
            b_mu = sess.run(self.parm_var[b][0])
            b_sigma = sess.run(tf.exp(self.parm_var[b][1]))

            a = get_acts_dist(x,w_mu,w_sigma,b_mu,b_sigma)
            a_hat = get_acts_dist(x_hat,w_mu,w_sigma,b_mu,b_sigma)
            A.append(a_hat)

            log_q_a = calc_log_marginal(a_hat,a_hat.sample())
            log_p_a = calc_log_marginal(a,a_hat.sample())
            KL += tf.reduce_mean(log_q_a) - tf.reduce_mean(log_p_a)

        grads = tf.gradients(KL, self.X_hat)
        
        #print('grads',grads)
        distill_train = self.distill_opt[0].apply_gradients([(grads[0],self.X_hat)], global_step=self.distill_opt[1])
        print('train data distill')  
        reinitialize_scope(['distill'],sess)
        sess.run(tf.variables_initializer(self.distill_opt[0].variables()))
        for _ in range(iters):
            x_hat = sess.run(self.X_hat)
            __,kl = sess.run([distill_train,KL],feed_dict={self.x_ph:x_hat})
            if (_+1)%10==0:
                print('iter {}: KL {}'.format(_+1,kl))

        X_hat = sess.run(self.X_hat)
        fig = plot(X_hat.reshape(-1,28,28),shape=(2,3))
        fig.savefig(os.path.join(rpath,'data_rep_task'+str(t)+'.png'))

        return A

    
    def config_next_task_parms(self,t,sess,x_train_task,rpath='./',*args,**kargs):
        ## only consider single head for now ##
        A_dists = self.data_distill(x_train_task,sess,t,rpath=rpath)
        A_dists = [Wrapped_Marginal(a_dt) for a_dt in A_dists]
        
        H_hat = [self.X_hat] + [tf.squeeze(h) for h in self.H[:-1]]
        self.task_var_cfg = {}
        for w,b,x_hat,a_dist in zip(self.qW[-1:],self.qB[-1:],H_hat[-1:],A_dists[-1:]):
            w_mu = self.parm_var[w][0]
            w_sigma = tf.exp(self.parm_var[w][1])
            b_mu = self.parm_var[b][0]
            b_sigma = tf.exp(self.parm_var[b][1])

            qa_dist = Wrapped_Marginal(get_acts_dist(x_hat,w_mu,w_sigma,b_mu,b_sigma))
            self.task_var_cfg[a_dist] = qa_dist

        return
    

