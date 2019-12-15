
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
from base_models.gans import GAN
from functools import reduce
from scipy.special import softmax


class DRS_CL(VCL):
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,dropout=None,initialization=None,\
                    ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,conv_net_shape=None,strides=None,pooling=False,\
                    B=3,eta=0.001,K=5,regularization=True,lambda_reg=0.0001,discriminant=False,lambda_dis=.001,\
                    WEM=False,coreset_mode='offline',batch_iter=1,task_type='split',*args,**kargs):
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

        print('DRS_CL: B {}, K {}, eta {}, dis {}, batch iter {}'.format(B,K,eta,discriminant,batch_iter))
        super(DRS_CL,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,dropout,initialization,ac_fn,n_smaples,local_rpm,conv_net_shape,\
                    strides,pooling,coreset_mode=coreset_mode,B=B,task_type=task_type,*args,**kargs)


        return


    def define_model(self,initialization=None,dropout=None,*args,**kargs):
        self.x_b_list, self.y_b_list, self.H_b_list, self.grads_b_list, self.grad_logtp_list = [], [], [], [], []
        net_shape = [self.conv_net_shape,self.net_shape] if self.conv else self.net_shape
               
        self.qW, self.qB, self.H = GAN.define_d_net(self.x_ph,net_shape=net_shape,reuse=False,conv=self.conv,ac_fn=self.ac_fn,\
                                scope='task',pooling=self.pooling,strides=self.strides,initialization=initialization)
        self.W_prior = Normal(loc=0., scale=1.)
        if not self.conv:
            self.conv_W,conv_parm_var,self.conv_h = None,None,None
        else:
            raise NotImplementedError('Not support Conv NN yet.')


        loss,self.ll,self.kl,self.dis = self.config_loss(self.x_ph,self.y_ph,self.qW,self.qB,self.H,regularization=self.regularization,discriminant=self.discriminant)
        self.grads = tf.gradients(loss,self.qW+self.qB)
        #self.grads_b_list = [grads]
        #self.x_b_list, self.y_b_list = [self.x_ph], [self.y_ph]
        
    
    def init_inference(self,learning_rate,decay=None,grad_type='adam',*args,**kargs):
        self.config_optimizer(starter_learning_rate=learning_rate,decay=decay,grad_type=grad_type)
        self.config_inference(*args,**kargs)

        return

    
    def config_coresets(self,qW,qB,conv_W=None,core_x_ph=None,core_sets=[[],[]],K=None,bayes=False,bayes_ouput=False,*args,**kargs):
 
        return super(DRS_CL,self).config_coresets(qW,qB,conv_W,core_x_ph,core_sets,K,bayes,bayes_ouput,*args,**kargs)




    def config_inference(self,*args,**kargs):

        #self.inference = Meta_Stein_Inference(var_list=self.qW+self.qB,grads_b_list=self.grads_b_list,grad_logtp_list=self.grad_logtp_list,\
        #                                    eta=self.eta,optimizer=self.task_optimizer,ll=self.ll,kl=self.kl)
        self.inference = MAP_Inference(var_list=self.qW+self.qB,grads=self.grads,optimizer=self.task_optimizer,ll=self.ll,kl=self.kl)

    
    
    def config_loss(self,x,y,W,B,H,regularization=True,discriminant=True,likelihood=True,*args,**kargs):
        loss,ll,reg, dis = 0.,0.,0.,0.
        
        if likelihood:
            ll = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=H[-1],labels=y))
            loss += ll

        if regularization:
                   
            for w in W+B:
                reg += tf.reduce_sum(self.W_prior.log_prob(w))
            loss -= self.lambda_reg * reg
        #print('config loss: discriminant {}'.format(discriminant))
        if discriminant:
            yids = tf.matmul(y, tf.transpose(y))
            mask = tf.eye(self.B)
            #print('y',y,'yids',yids)
            for h in H:
                #h = H[0]
                sim = tf.matmul(h,tf.transpose(h))
                dis += 0.5*tf.reduce_mean(sim*(1.-yids)-0.5*sim*(mask-yids))
            loss += self.lambda_dis * dis


        return loss,ll,reg,dis
    '''
    def train_update_step(self,t,s,sess,feed_dict,err=0.,x_train_task=None,y_train_task=None,local_iter=0,*args,**kargs):
        assert(self.coreset_size > 0)
        #print(self.x_b_list,self.y_b_list)
        #empty_mem = len(self.core_sets[0])==0
        x_batch, y_batch = feed_dict[self.x_ph], feed_dict[self.y_ph]
        buffer_size = self.B
        
        if self.coreset_mode == 'ring_buffer':
            if len(self.curr_buf[0])==0:
                self.curr_buf[0] = x_batch
                self.curr_buf[1] = y_batch
            else:
                num_per_task = int(self.coreset_size/(t+1))

                if num_per_task >= len(self.curr_buf[0])+len(x_batch):
                    self.curr_buf[0] = np.vstack([self.curr_buf[0],x_batch])
                    self.curr_buf[1] = np.vstack([self.curr_buf[1],y_batch])
                else:
                    self.curr_buf[0] = np.vstack([self.curr_buf[0],x_batch])[-num_per_task:]
                    self.curr_buf[1] = np.vstack([self.curr_buf[1],y_batch])[-num_per_task:]
        cx, cy = (x_batch,y_batch) if self.coreset_mode=='offline' else (self.curr_buf[0],self.curr_buf[1])
        if len(cx) > buffer_size:
            cx, cy = cx[-buffer_size:], cy[-buffer_size:]
            
        if t > 0:
            mem_x, mem_y = self.x_core_sets, self.y_core_sets
            nc_mem = np.sum(np.sum(mem_y,axis=0) > 0) # number of classes in episodic memory
            nc_batch = np.sum(np.sum(self.curr_buf[1],axis=0) > 0) # number of classes in current batch
            #print('nc mem',nc_mem,'nc batch',nc_batch)
            per_cl_size = int(buffer_size/(nc_mem+nc_batch))              
            msize = per_cl_size * nc_mem
            csize = buffer_size - msize

            ### sampling from memory and current batch ###
            if self.WEM:

                # update sampling probability every K iters
                if s % self.K == 0:
                    x_core_sets, y_core_sets = np.vstack([mem_x,cx]), np.vstack([mem_y,cy])

                    hx = sess.run(self.H,feed_dict={self.x_ph:x_core_sets,self.y_ph:y_core_sets})
                    hx = np.hstack(hx)
                    sims = calc_similarity(hx,sess=sess)
                    yids = 1.- np.matmul(y_core_sets,y_core_sets.transpose())
                    scores = (np.sum(sims*yids,axis=1)*0.5) - (np.sum(sims*(1.-yids),axis=1)*0.5)
                    #scores = scores[:-len(x_batch)]
                    #print('scores',scores.shape)
                    self.p_samples = []
                    for i in range(nc_mem):
                        ids = y_core_sets[:,i]==1
                        self.p_samples.append(softmax(scores[ids]))                   
                    #print('p samples',self.p_samples)

                coreset_x, coreset_y = [], []
                for i in range(nc_mem): 
                    ids = np.random.choice(len(self.p_samples[i]),size=per_cl_size,p=self.p_samples[i]) #np.argsort(self.p_samples)[:self.batch_size*self.B]
                    coreset_x.append(mem_x[mem_y[:,i]==1][ids])
                    coreset_y.append(mem_y[mem_y[:,i]==1][ids])
                cids = np.random.choice(len(cx),size=csize,p=softmax(scores[-len(cx):]))
            
            else:
                
                coreset_x, coreset_y = [], []
                for i in range(nc_mem): 
                    num_cl = np.sum(mem_y[:,i]==1)
                    ids = np.random.choice(num_cl,size=per_cl_size) #np.argsort(self.p_samples)[:self.batch_size*self.B]
                    coreset_x.append(mem_x[mem_y[:,i]==1][ids])
                    coreset_y.append(mem_y[mem_y[:,i]==1][ids])

                cids = np.random.choice(len(cx),size=csize)

            cx, cy = cx[cids], cy[cids]#np.vstack([cx[cids],x_batch]), np.vstack([cy[cids],y_batch])
            coreset_x, coreset_y = np.vstack([*coreset_x,cx]), np.vstack([*coreset_y,cy])
            #print('mem',coreset_x.shape,coreset_y.shape)
            #print('mem size',mem_x.shape,mem_y.shape)
            feed_dict.update({self.x_ph:coreset_x,self.y_ph:coreset_y})
                
        ### empty memory ###              
        else:
            bids = np.random.choice(len(cx),size=buffer_size)                  
            feed_dict.update({self.x_ph:cx[bids],self.y_ph:cy[bids]})

        #print('feed dict',feed_dict[self.x_ph].shape)
        self.inference.update(sess=sess,K=self.K,feed_dict=feed_dict)




        ## todo: update err
        return err
        '''

    def train_update_step(self,t,s,sess,feed_dict,err=0.,x_train_task=None,y_train_task=None,local_iter=0,*args,**kargs):
        assert(self.coreset_size > 0)
        #print(self.x_b_list,self.y_b_list)
        #empty_mem = len(self.core_sets[0])==0
        x_batch, y_batch = feed_dict[self.x_ph], feed_dict[self.y_ph]
        buffer_size = self.B
        cx, cy = x_batch,y_batch 

        if self.coreset_mode == 'ring_buffer':
            #nc_mem = len(self.core_sets)
            if self.task_type == 'split':
                y_mask = np.sum(y_batch,axis=0) > 0
                nc_batch = np.sum(y_mask)                
                cls_batch = np.argsort(y_mask)[-nc_batch:]
                #print('cls batch',cls_batch,'nc mem',nc_mem)

                for c in cls_batch:
                    cx = self.core_sets.get(c,None)
                    self.core_sets[c] = x_batch[y_batch[:,c]==1] if cx is None else np.vstack([cx,x_batch[y_batch[:,c]==1]])
                
                #num_per_clss = int(self.coreset_size/len(self.core_sets))
                #print('curr buf',self.curr_buf.keys(),'num cls',num_per_clss)
                '''
                self.curr_buf_size = 0
                for c in self.curr_buf.keys():   
                    #print('c',c) 
                    if num_per_clss < len(self.curr_buf[c]):
                        self.curr_buf[c] = self.curr_buf[c][-num_per_clss:]

                    self.curr_buf_size += len(self.curr_buf[c])
                '''
            
            else:
                #num_per_task = int(self.coreset_size/(len(self.core_sets)+1))
                cxy = self.core_sets.get(t,None)
                cx = x_batch if cxy is None else np.vstack([cxy[0],x_batch])
                cy = y_batch if cxy is None else np.vstack([cxy[1],y_batch])
                #if num_per_task < len(cx):
                #    cx, cy = cx[-num_per_task:], cy[-num_per_task:]
                self.core_sets[t] = (cx,cy)
                #self.curr_buf_size = len(cx)
                
            self.online_update_coresets(self.coreset_size)
          
            
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
                    '''
                    ptr = 0
                    for i, cx in self.core_sets.items(): 
                        tsize = per_cl_size+1 if rd>0 and i in clss else per_cl_size                      
                        p = softmax(loss[ptr:ptr+len(cx[0])])
                        ids = np.random.choice(len(cx[0]),size=tsize,p=p)
                        ptr += len(cx[0])

                        tmp_x = cx[0][ids]
                        tmp_y = cx[1][ids]                        

                        coreset_x.append(tmp_x)
                        coreset_y.append(tmp_y)
                    '''

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
                #y_batch = np.repeat(y_batch,self.n_samples,axis=0)
                #print('y_batch',y_batch[:2])
                for __ in range(self.batch_iter):
                    feed_dict = {self.x_ph:x_batch,self.y_ph:y_batch}

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
            acc, y_probs,cfm = predict(ts[0],ts[1],self.x_ph,self.H[-1],self.batch_size,sess,regression=False,confusion=confusion)
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

    def reinitialization(self,sess,scope='task',*args,**kargs):
        #reinitialize_scope(scope=scope,sess=sess)
        return

    
    def config_train(self,*args,**kargs):
        
        grads_and_vars = list(zip(self.grads,self.var_list))
        self.train = self.optimizer[0].apply_gradients(grads_and_vars,global_step=self.optimizer[1])

        return

    
    def update(self,sess,feed_dict=None,*args,**kargs):

        sess.run(self.train, feed_dict)

        return
    


