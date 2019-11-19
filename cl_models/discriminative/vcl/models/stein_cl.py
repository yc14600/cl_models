
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
from .bcl_base_bnn import BCL_BNN
from .vcl_model import VCL
from utils.model_util import *
from utils.train_util import *
from utils.coreset_util import *
from base_models.gans import GAN
from functools import reduce



class Stein_CL(VCL):
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,dropout=None,initialization=None,\
                    ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,conv_net_shape=None,strides=None,pooling=False,\
                    B=3,eta=0.001,K=5,lambda_reg=0.0001,*args,**kargs):
        assert(num_heads==1)
        assert(B>1)
        self.B = B # number of Stein particles
        self.eta = eta # meta learning rate
        self.K = K # number of iterations of meta stein gradients
        self.lambda_reg = lambda_reg # multiplier of regularization
        print('Stein_CL: B {}, K {}, eta {}'.format(B,K,eta))
        super(Stein_CL,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,dropout,initialization,ac_fn,n_smaples,local_rpm,conv_net_shape,\
                    strides,pooling,*args,**kargs)


        return


    def define_model(self,initialization=None,dropout=None,*args,**kargs):
        self.x_b_list, self.y_b_list, self.qH_b_list, self.grads_b_list, self.grad_logtp_list = [], [], [], [], []
        net_shape = [self.conv_net_shape,self.net_shape] if self.conv else self.net_shape
               
        self.qW, self.qB, self.qH = GAN.define_d_net(self.x_ph,net_shape=net_shape,reuse=False,conv=self.conv,ac_fn=self.ac_fn,\
                                scope='task',pooling=self.pooling,strides=self.strides,initialization=initialization)
        self.W_prior = Normal(loc=0., scale=1.)
        if not self.conv:
            self.conv_W,conv_parm_var,self.conv_h = None,None,None
        else:
            raise NotImplementedError('Not support Conv NN yet.')


        for b in range(self.B):
            x_b_ph = tf.placeholder(dtype=tf.float32,shape=self.x_ph.shape,name='x_'+str(b))
            y_b_ph = tf.placeholder(dtype=tf.float32,shape=self.y_ph.shape,name='y_'+str(b))
            self.x_b_list.append(x_b_ph)
            self.y_b_list.append(y_b_ph)
            _, _, qH = GAN.define_d_net(x_b_ph,net_shape=net_shape,reuse=True,conv=self.conv,ac_fn=self.ac_fn,\
                                scope='task',pooling=self.pooling,strides=self.strides,initialization=initialization)
            self.qH_b_list.append(qH)
            #print('qH {},y_b_ph {}'.format(qH,y_b_ph))
            ll = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=qH[-1],labels=y_b_ph))
            reg = 0.
            for w in self.qW+self.qB:
                reg += tf.reduce_sum(self.W_prior.log_prob(w))
            loss_b = ll- self.lambda_reg * reg
            grads_b = tf.gradients(loss_b,self.qW+self.qB)
            self.grads_b_list.append(grads_b)

            #print('b,{},grads,{}'.format(b,self.grads_b_list[-1]))
            W_hat = []
            reg = 0.
            for w,gb in zip(self.qW+self.qB,grads_b):
                w_hat = w - self.eta*gb
                reg += tf.reduce_sum(self.W_prior.log_prob(w_hat))
                W_hat.append(w_hat)
            H_b = forward_nets(W_hat[:len(self.qW)],W_hat[len(self.qW):],self.x_ph,ac_fn=self.ac_fn,bayes_output=False)
            #print('H_b',H_b[-1],'y_ph',self.y_ph)
            ll = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=H_b[-1],labels=self.y_ph))
            self.grad_logtp_list.append(tf.gradients(ll-reg,W_hat))

    
    def init_inference(self,learning_rate,decay=None,grad_type='adam',*args,**kargs):
        self.config_optimizer(starter_learning_rate=learning_rate,decay=decay,grad_type='adam')
        self.config_inference(*args,**kargs)

        return

    
    def config_coresets(self,qW,qB,conv_W=None,core_x_ph=None,core_sets=[[],[]],K=None,bayes=False,bayes_ouput=False,*args,**kargs):
        return super(Stein_CL,self).config_coresets(qW,qB,conv_W,core_x_ph,core_sets,K,bayes,bayes_ouput,*args,**kargs)


    def config_inference(self,*args,**kargs):

        self.inference = Meta_Stein_Inference(var_list=self.qW+self.qB,grads_b_list=self.grads_b_list,grad_logtp_list=self.grad_logtp_list,\
                                            eta=self.eta,optimizer=self.task_optimizer)



    
    def train_update_step(self,t,s,sess,feed_dict,err=0.,x_train_task=None,y_train_task=None,*args,**kargs):
        assert(self.coreset_size > 0)
        #print(self.x_b_list,self.y_b_list)
        if t > 0:
            n = int(self.B/(t+1)) # number of tasks in one particle batch
            r = int(self.B%(t+1))
            #print('n {}, r {}'.format(n,r))
            #if n == 0:
            
            bids = np.random.choice(len(self.x_core_sets),size=self.batch_size*(n*t+r))

            cids = np.random.choice(self.batch_size,size=self.batch_size*n)
            coreset_x = np.vstack([self.x_core_sets[bids],feed_dict[self.x_ph][cids]])
            coreset_y = np.vstack([self.y_core_sets[bids],feed_dict[self.y_ph][cids]])
            #coreset_x,coreset_y = shuffle_data(coreset_x,coreset_y)
            for b in range(self.B):                
                #coresets_x_b, coresets_y_b = self.x_core_sets, self.y_core_sets
                #bids = np.random.choice(len(self),size=self.batch_size)
                feed_dict.update({self.x_b_list[b]:coreset_x[b*self.batch_size:(b+1)*self.batch_size],self.y_b_list[b]:coreset_y[b*self.batch_size:(b+1)*self.batch_size]})
            #
            #print('len core x',len(self.core_sets[0]))
            '''
            b = 0
            x_sets = [*self.core_sets[0][:-1],feed_dict[self.x_ph]]
            y_sets = [*self.core_sets[1][:-1],feed_dict[self.y_ph]]
            for x,y in zip(x_sets,y_sets):
                bids = np.random.choice(len(x),size=self.batch_size*n)
                for i in range(n):
                    #print('i {}, b {}'.format(i,b))
                    feed_dict.update({self.x_b_list[b]:x[bids[i*self.batch_size:(i+1)*self.batch_size]],self.y_b_list[b]:y[bids[i*self.batch_size:(i+1)*self.batch_size]]})
                    b+=1
            '''
            x_sets = np.vstack([self.x_core_sets[:self.batch_size*t],feed_dict[self.x_ph]])
            y_sets = np.vstack([self.y_core_sets[:self.batch_size*t],feed_dict[self.y_ph]])
            x_sets,y_sets = shuffle_data(x_sets,y_sets)
            '''
            if r > 0:
                bids = np.random.choice(len(x_sets),size=self.batch_size*r)
                for i in range(r):
                    
                    feed_dict.update({self.x_b_list[b]:x_sets[bids[i*self.batch_size:(i+1)*self.batch_size]],self.y_b_list[b]:y_sets[bids[i*self.batch_size:(i+1)*self.batch_size]]})
                    b+=1            
            '''
            #n = int(self.batch_size/(t+1)) 
            #r = int(self.batch_size%(t+1))
            #x_sets = np.vstack([x[:n] for x in self.core_sets[0][:-1]]+[feed_dict[self.x_ph][:n+r]])
            #y_sets = np.vstack([y[:n] for y in self.core_sets[1][:-1]]+[feed_dict[self.y_ph][:n+r]])
            feed_dict[self.x_ph] = x_sets
            feed_dict[self.y_ph] = y_sets
            
            
            #feed_dict.update({self.x_b_list[-1]:feed_dict[self.x_ph][bids],self.y_b_list[-1]:feed_dict[self.y_ph][bids]})

        else:
            bids = np.random.choice(self.batch_size,size=self.batch_size*self.B)
            for b in range(self.B):                  
                feed_dict.update({self.x_b_list[b]:feed_dict[self.x_ph][bids[b*self.batch_size:(b+1)*self.batch_size]],self.y_b_list[b]:feed_dict[self.y_ph][bids[b*self.batch_size:(b+1)*self.batch_size]]})

        #print('feed dict',feed_dict.keys())
        self.inference.update(sess=sess,K=self.K,feed_dict=feed_dict)
        ## todo: update err
        return err


    def train_task(self,sess,t,x_train_task,y_train_task,epoch,print_iter=5,local_iter=10,\
                    tfb_merged=None,tfb_writer=None,tfb_avg_losses=None,*args,**kargs):

        # training for current task
        num_iter = int(np.ceil(x_train_task.shape[0]/self.batch_size))
        #sess.run(self.task_optimizer[1].initializer)
        print('num iter',num_iter)
        for e in range(epoch):
            shuffle_inds = np.arange(x_train_task.shape[0])
            np.random.shuffle(shuffle_inds)
            x_train_task = x_train_task[shuffle_inds]
            y_train_task = y_train_task[shuffle_inds]
            err = 0.
            ii = 0
            for _ in range(num_iter):
                x_batch,y_batch,ii = get_next_batch(x_train_task,self.batch_size,ii,labels=y_train_task)
                #y_batch = np.repeat(y_batch,self.n_samples,axis=0)
                #print('y_batch',y_batch[:2])
                feed_dict = {self.x_ph:x_batch,self.y_ph:y_batch}

                err = self.train_update_step(t,_,sess,feed_dict,err,x_train_task,y_train_task,*args,**kargs)

            if (e+1)%print_iter==0:
                print('epoch',e+1,'avg loss',err/num_iter)
        
        return


    def update_inference(self,sess,*args,**kargs):
        self.inference.reinitialization(sess)
        return

    def test_all_tasks(self,t,test_sets,sess,epoch=10,saver=None,file_path=None,*args,**kargs):
        acc_record, pred_probs = [], []
        for ts in test_sets: 
            acc, y_probs = predict(ts[0],ts[1],self.x_ph,self.qH[-1],self.batch_size,sess,regression=False)
            print('accuracy',acc)
            acc_record.append(acc)
            pred_probs.append(y_probs)
        print('avg accuracy',np.mean(acc_record))
        #print('pred prob',sess.run(tf.nn.softmax(y_probs))[:3])
        return acc_record,pred_probs
        #return super(Stein_CL,self).test_all_tasks(t,test_sets,sess=sess,epoch=epoch,saver=saver,file_path=file_path,bayes=False,bayes_output=False)



class Meta_Stein_Inference:
    def __init__(self,var_list,grads_b_list,grad_logtp_list,eta,optimizer=None,*args,**kargs):
        self.var_list = var_list
        self.grads_b_list = grads_b_list
        self.grad_logtp_list = grad_logtp_list
        self.B = len(grads_b_list)
        self.eta = eta
        self.optimizer = optimizer

        self.config_train()


    def reinitialization(self,sess,scope='task',*args,**kargs):
        reinitialize_scope(scope=scope,sess=sess)
        return

    
    def get_stein_grads(self,*args,**kargs):
        Kb_mat = 0.
        for gb in zip(*self.grads_b_list):
            #diag_mask = 1.-tf.eye(len(gb)) #remove diagnal elements for \sum_b'
            gb = [tf.reshape(gw,[1,-1]) for gw in gb]
            gb = tf.concat(gb,axis=0)
            Kb = tf.matmul(gb,tf.transpose(gb)) #* diag_mask
            Kb_mat += Kb

        Kb_mat /= self.B #average
        stein_grads_list = []
        for b in range(self.B):
            Kb = Kb_mat[b]
            sgd = self.grads_b_list[b] #[0.]*(len(self.qW)+len(self.qB))
            for b in range(self.B):
                ltp = self.grad_logtp_list[b]                             
                for i,g in enumerate(ltp):
                    #print('g_logtp',g)
                    sgd[i] += Kb[b]*g
            
            stein_grads_list.append(sgd)  

        return stein_grads_list 


    def get_w_grads(self,*args,**kargs):
        w_grads = []
        for i in range(len(self.grads_b_list[0])):
            sum_i = reduce((lambda x, y: x+y),[gb[i] for gb in self.grads_b_list])
            sum_i /= self.B
            w_grads.append(sum_i)
        return w_grads

    
    def config_train(self,*args,**kargs):
        self.grads = self.get_w_grads()
        grads_and_vars = list(zip(self.grads,self.var_list))
        self.train = self.optimizer[0].apply_gradients(grads_and_vars,global_step=self.optimizer[1])

        return


    def update_meta_grads(self,*args,**kargs):
        stein_grads_list = self.get_stein_grads()
        #update_step = [[tf.assign_add(gbi,self.eta*dgbi) for gbi,dgbi in zip(gb,dgb)] for gb,dgb in zip(self.grads_b_list,stein_grads_list)]      
        #sess.run(update_step,feed_dict=feed_dict)
        for gb,dgb in zip(self.grads_b_list,stein_grads_list):
            for i,(gbi,dgbi) in enumerate(zip(gb,dgb)):
                gbi += self.eta*dgbi 
                gb[i] = gbi

        return

    
    def update(self,sess,K=1,feed_dict=None,*args,**kargs):
        #print('before',sess.run(self.grads_b_list[0][0][0][0],feed_dict))
        for _ in range(K):
            self.update_meta_grads()
        #print('after',sess.run(self.grads_b_list[0][0][0][0],feed_dict))
        sess.run(self.train, feed_dict)

        return


