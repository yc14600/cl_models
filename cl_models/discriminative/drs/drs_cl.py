
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
from scipy.special import softmax


class DRS_CL(VCL):
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,dropout=None,initialization=None,\
                    ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,conv_net_shape=None,strides=None,pooling=False,\
                    B=3,eta=0.001,K=5,regularization=False,lambda_reg=0.0001,discriminant=False,lambda_dis=.001,\
                    WEM=False,coreset_mode='offline',batch_iter=1,task_type='split',net_type='dense',fixed_budget=True,*args,**kargs):
        assert(num_heads==1)
        #assert(B>1)
        self.B = B # training batch size
        #print('B',self.B)
        self.eta = eta 
        self.K = K 
        self.regularization = regularization
        self.lambda_reg = lambda_reg # multiplier of regularization
        self.discriminant =discriminant
        self.lambda_dis = lambda_dis
        self.WEM = WEM # Weighted Episodic Memory
        self.batch_iter = batch_iter
        self.net_type = net_type
        self.fixed_budget = fixed_budget # fixed memory budget or not

        print('DRS_CL: B {}, K {}, eta {}, dis {}, batch iter {}'.format(B,K,eta,discriminant,batch_iter))
        super(DRS_CL,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,dropout,initialization,ac_fn,n_smaples,local_rpm,conv_net_shape,\
                    strides,pooling,coreset_mode=coreset_mode,B=B,task_type=task_type,*args,**kargs)


        return


    def define_model(self,initialization=None,dropout=None,*args,**kargs):

        if self.net_type == 'dense':

            net_shape = [self.conv_net_shape,self.net_shape] if self.conv else self.net_shape
                
            self.qW, self.qB, self.H = GAN.define_d_net(self.x_ph,net_shape=net_shape,reuse=False,conv=self.conv,ac_fn=self.ac_fn,\
                                    scope='task',pooling=self.pooling,strides=self.strides,initialization=initialization)
            self.W_prior = Normal(loc=0., scale=1.)
            self.vars = self.qW+self.qB

        elif self.net_type == 'resnet18':
            # Same resnet-18 as used in GEM paper
            self.training = tf.placeholder(tf.bool, name='train_phase')
            kernels = [3, 3, 3, 3, 3]
            filters = [20, 20, 40, 80, 160]
            strides = [1, 0, 2, 2, 2]
            self.H, self.vars = resnet18_conv_feedforward(self.x_ph,kernels=kernels,filters=filters,strides=strides,out_dim=self.net_shape[-1],train_phase=self.training)
            self.qW, self.qB = [],[]
        if not self.conv:
            self.conv_W,conv_parm_var,self.conv_h = None,None,None
        else:
            raise NotImplementedError('Not support Conv NN yet.')


        loss,self.ll,self.kl,self.dis = self.config_loss(self.x_ph,self.y_ph,self.vars,self.H,regularization=self.regularization,discriminant=self.discriminant)
        self.grads = tf.gradients(loss,self.vars)

        
    
    def init_inference(self,learning_rate,decay=None,grad_type='adam',*args,**kargs):
        self.config_optimizer(starter_learning_rate=learning_rate,decay=decay,grad_type=grad_type)
        self.config_inference(*args,**kargs)

        return

    
    def config_coresets(self,qW,qB,conv_W=None,core_x_ph=None,core_sets=[[],[]],K=None,bayes=False,bayes_ouput=False,*args,**kargs):
        if self.net_type != 'resnet18':
            return super(DRS_CL,self).config_coresets(qW,qB,conv_W,core_x_ph,core_sets,K,bayes,bayes_ouput,*args,**kargs)
        else:
            #### todo: complete resnet code for offline coresets ####
            self.core_sets = {}



    def config_inference(self,*args,**kargs):

        self.inference = MAP_Inference(var_list=self.vars,grads=self.grads,optimizer=self.task_optimizer,ll=self.ll,kl=self.kl)

    
    
    def config_loss(self,x,y,var_list,H,regularization=True,discriminant=True,likelihood=True,*args,**kargs):
        loss,ll,reg, dis = 0.,0.,0.,0.
        
        if likelihood:
            ll = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=H[-1],labels=y))
            loss += ll

        if regularization:
                   
            for w in var_list:
                reg += tf.reduce_sum(self.W_prior.log_prob(w))
            loss -= self.lambda_reg * reg
        #print('config loss: discriminant {}'.format(discriminant))
        if discriminant:
            yids = tf.matmul(y, tf.transpose(y))
            mask = tf.eye(self.B)
            #print('y',y,'yids',yids)
            for h in H[:]:
                #h = H[0]
                if len(h.shape) > 2:
                    h = tf.reshape(h,[self.B,-1])
                
                sim = tf.matmul(h,tf.transpose(h))
                dis += 0.5*tf.reduce_mean(sim*(1.-yids)-0.5*sim*(mask-yids))
            loss += self.lambda_dis * dis


        return loss,ll,reg,dis
   

    def train_update_step(self,t,s,sess,feed_dict,err=0.,x_train_task=None,y_train_task=None,local_iter=0,*args,**kargs):
        assert(self.coreset_size > 0)

        x_batch, y_batch = feed_dict[self.x_ph], feed_dict[self.y_ph]
        buffer_size = self.B 
        cx, cy = x_batch,y_batch 

        if self.coreset_mode == 'ring_buffer':            
            if self.task_type == 'split':
                y_mask = np.sum(y_batch,axis=0) > 0
                nc_batch = np.sum(y_mask)                
                cls_batch = np.argsort(y_mask)[-nc_batch:]
                #print('cls batch',cls_batch,'nc mem',nc_mem)

                for c in cls_batch:
                    cx = self.core_sets.get(c,None)
                    self.core_sets[c] = x_batch[y_batch[:,c]==1] if cx is None else np.vstack([cx,x_batch[y_batch[:,c]==1]])
                
            
            else:
                cxy = self.core_sets.get(t,None)
                cx = x_batch if cxy is None else np.vstack([cxy[0],x_batch])
                cy = y_batch if cxy is None else np.vstack([cxy[1],y_batch])
                self.core_sets[t] = (cx,cy)
                
            self.online_update_coresets(self.coreset_size,self.fixed_budget,t)
          
            
        if t > 0:
            
            num_cl = len(self.core_sets)
            per_cl_size = int(buffer_size/num_cl)  
            rd = buffer_size % num_cl   
            clss = set(np.random.choice(list(self.core_sets.keys()),size=rd,replace=False))  

            #print('num cl',num_cl,'per cl size',per_cl_size,'rd',rd,'clss',clss)    

            coreset_x, coreset_y = [], []
            #print('num cl',per_cl_size)
            if self.WEM:
                if self.task_type == 'split':
                    #print('coreset',self.core_sets.keys())
                    tmp_d = [(x, one_hot_encoder(np.ones(x.shape[0])*y,H=self.net_shape[-1])) for y,x in self.core_sets.items()]
                    tmp_x = np.vstack([d[0] for d in tmp_d])
                    tmp_y = np.vstack([d[1] for d in tmp_d])
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.H[-1],labels=self.y_ph),axis=1)
                    loss = sess.run(loss,feed_dict={self.x_ph:tmp_x,self.y_ph:tmp_y})

                    ptr = 0

                    for i, cx in self.core_sets.items(): 
                        tsize = per_cl_size+1 if rd>0 and i in clss else per_cl_size                      
                        p = softmax(loss[ptr:ptr+len(cx)])
                        ids = np.random.choice(len(cx),size=tsize,p=p)
                        ptr += len(cx)

                        tmp_y = np.zeros([tsize,self.net_shape[-1]])
                        tmp_y[:,i] = 1
                        tmp_x = cx[ids]

                        coreset_x.append(tmp_x)
                        coreset_y.append(tmp_y)
                else:
                    tmp_d = [(t,x) for t,x in self.core_sets.items()]
                    tmp_x = np.vstack([d[1][0] for d in tmp_d])
                    tmp_y = np.vstack([d[1][1] for d in tmp_d])
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.H[-1],labels=self.y_ph),axis=1)
                    loss = sess.run(loss,feed_dict={self.x_ph:tmp_x,self.y_ph:tmp_y})

                    ids = np.argsort(loss)[-buffer_size:]#np.random.choice(len(tmp_x),size=buffer_size,p=softmax(loss))
                    coreset_x = tmp_x[ids]
                    coreset_y = tmp_y[ids]

            else:
                if self.task_type == 'split':
                    for i, cx in self.core_sets.items(): 
                        tsize = per_cl_size+1 if rd>0 and i in clss else per_cl_size                      
                        ids = np.random.choice(len(cx),size=tsize)
                        tmp_y = np.zeros([tsize,self.net_shape[-1]])
                        tmp_y[:,i] = 1
                        tmp_x = cx[ids]
                        coreset_x.append(tmp_x)
                        coreset_y.append(tmp_y)
                else:
                    
                    for i, cx in self.core_sets.items():
                        tsize = per_cl_size+1 if rd>0 and i in clss else per_cl_size                      
                        num_cl = len(self.core_sets[i][0])
                        ids = np.random.choice(num_cl,size=tsize)
                        tmp_x = self.core_sets[i][0][ids]
                        tmp_y = self.core_sets[i][1][ids]
                        coreset_x.append(tmp_x)
                        coreset_y.append(tmp_y)
                    
            if isinstance(coreset_x,list):
                coreset_x, coreset_y = np.vstack(coreset_x), np.vstack(coreset_y)#np.vstack([*coreset_x,cx]), np.vstack([*coreset_y,cy])
            feed_dict.update({self.x_ph:coreset_x,self.y_ph:coreset_y})
                
        ### empty memory ###              
        else:
            if self.task_type == 'split':
                cx, cy = [], []
                for c in self.core_sets.keys():
                    cx.append(self.core_sets[c])
                    tmp_y = np.zeros([cx[-1].shape[0],self.net_shape[-1]])
                    tmp_y[:,c] = 1
                    cy.append(tmp_y)

                cx = np.vstack(cx)
                cy = np.vstack(cy)
                cx, cy = shuffle_data(cx,cy)

            bids = np.random.choice(len(cx),size=buffer_size) 
            feed_dict.update({self.x_ph:cx[bids],self.y_ph:cy[bids]})

        #print('feed dict',feed_dict[self.x_ph].shape)
        self.inference.update(sess=sess,K=self.K,feed_dict=feed_dict)

        ## todo: update err
        return err


    def train_task(self,sess,t,x_train_task,y_train_task,epoch,print_iter=5,local_iter=10,\
                    tfb_merged=None,tfb_writer=None,tfb_avg_losses=None,*args,**kargs):

        # training for current task
        num_iter = int(np.ceil(x_train_task.shape[0]/self.batch_size))
        #sess.run(self.task_optimizer[1].initializer)
        print('training set',x_train_task.shape[0],'num iter',num_iter)
        
        for e in range(epoch):
            shuffle_inds = np.arange(x_train_task.shape[0])
            np.random.shuffle(shuffle_inds)
            x_train_task = x_train_task[shuffle_inds]
            y_train_task = y_train_task[shuffle_inds]
            err = 0.
            ii = 0
            for _ in range(num_iter):
                #print('{} iter'.format(_))
                x_batch,y_batch,ii = get_next_batch(x_train_task,self.batch_size,ii,labels=y_train_task)

                for __ in range(self.batch_iter):
                    feed_dict = {self.x_ph:x_batch,self.y_ph:y_batch}
                    if self.net_type == 'resnet18':
                        feed_dict.update({self.training:True})

                    err = self.train_update_step(t,_,sess,feed_dict,err,x_train_task,y_train_task,local_iter=local_iter,*args,**kargs)
                
                if self.coreset_type == 'stein' and (_+1)%local_iter==0:
                    sess.run(self.stein_train)

            if (e+1)%print_iter==0:
                if self.discriminant:
                    ll,kl,dis = sess.run([self.ll,self.kl,self.dis],feed_dict=feed_dict)
                    print('epoch',e+1,'ll',ll,'kl',kl,'dis',dis)
                else:
                    ll,kl = sess.run([self.ll,self.kl],feed_dict=feed_dict)
                    print('epoch',e+1,'ll',ll,'kl',kl)
        #print('curr buf',self.curr_buf[0].shape)
        return


    def update_inference(self,sess,*args,**kargs):
        self.inference.reinitialization(sess)
        return


    def test_all_tasks(self,t,test_sets,sess,epoch=10,saver=None,file_path=None,confusion=False,*args,**kargs):
        acc_record, pred_probs = [], []
        dim = test_sets[0][1].shape[1]
        cfmtx = np.zeros([dim,dim])
        #print('ts len',len(test_sets))
        for t,ts in enumerate(test_sets): 
            #print('{} test set'.format(t))
            
            feed_dict = {self.training:False} if self.net_type=='resnet18' else {}
            acc, y_probs,cfm = predict(ts[0],ts[1],self.x_ph,self.H[-1],self.batch_size,sess,regression=False,confusion=confusion,feed_dict=feed_dict)
            print('accuracy',acc)
            #print('cfm',cfm)
            acc_record.append(acc)
            pred_probs.append(y_probs)
            cfmtx += cfm
        print('avg accuracy',np.mean(acc_record))
        #print('pred prob',sess.run(tf.nn.softmax(y_probs))[:3])
        return acc_record,pred_probs,cfmtx
        #return super(Stein_CL,self).test_all_tasks(t,test_sets,sess=sess,epoch=epoch,saver=saver,file_path=file_path,bayes=False,bayes_output=False)


class MAP_Inference:
    def __init__(self,var_list,grads,optimizer=None,ll=0.,kl=0.,*args,**kargs):
        self.var_list = var_list
        self.grads = grads
        self.optimizer = optimizer
        self.ll = ll
        self.kl = kl
        self.config_train()

    def reinitialization(self,sess=None,scope='task',warm_start=True,*args,**kargs):
        if not warm_start:
            reinitialize_scope(scope=scope,sess=sess)
        return

    
    def config_train(self,*args,**kargs):
        
        grads_and_vars = list(zip(self.grads,self.var_list))
        self.train = self.optimizer[0].apply_gradients(grads_and_vars,global_step=self.optimizer[1])

        return

    
    def update(self,sess,feed_dict=None,*args,**kargs):

        sess.run(self.train, feed_dict)

        return
    


