
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


from utils.distributions import Normal
from .bcl_base_bnn import BCL_BNN
from .vcl_model import VCL
from .coreset import *
from utils.model_util import *
from utils.train_util import *
from base_models.gans import GAN
from functools import reduce
from scipy.special import softmax


class Stein_CL(VCL):
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=500,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,dropout=None,initialization=None,\
                    ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,conv_net_shape=None,strides=None,pooling=False,\
                    B=3,eta=0.001,K=5,regularization=True,lambda_reg=0.0001,discriminant=False,lambda_dis=.001,\
                    WEM=True,*args,**kargs):
        assert(num_heads==1)
        #assert(B>1)
        self.B = B # number of Stein particles
        #print('B',self.B)
        self.eta = eta # meta learning rate
        self.K = K # number of iterations of meta stein gradients
        self.regularization = regularization
        self.lambda_reg = lambda_reg # multiplier of regularization
        self.discriminant =discriminant
        self.lambda_dis = lambda_dis
        self.WEM = WEM # Weighted Episodic Memory
        print('Stein_CL: B {}, K {}, eta {}, dis {}'.format(B,K,eta,discriminant))
        super(Stein_CL,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,dropout,initialization,ac_fn,n_smaples,local_rpm,conv_net_shape,\
                    strides,pooling,B=B,*args,**kargs)


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
        return super(Stein_CL,self).config_coresets(qW,qB,conv_W,core_x_ph,core_sets,K,bayes,bayes_ouput,*args,**kargs)


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
            mask = tf.eye(self.batch_size*self.B)
            #print('y',y,'yids',yids)
            for h in H:
                #h = H[0]
                sim = tf.matmul(h,tf.transpose(h))
                dis += 0.5*tf.reduce_mean(sim*(1.-yids)-0.5*sim*(mask-yids))
            loss += self.lambda_dis * dis


        return loss,ll,reg,dis


    def train_update_step(self,t,s,sess,feed_dict,err=0.,x_train_task=None,y_train_task=None,local_iter=0,*args,**kargs):
        assert(self.coreset_size > 0)
        #print(self.x_b_list,self.y_b_list)
        empty_mem = len(self.core_sets[0])==0
        x_batch, y_batch = feed_dict[self.x_ph], feed_dict[self.y_ph]
        buffer_size = self.batch_size * self.B
        if t>0:
            mem_x, mem_y = self.x_core_sets, self.y_core_sets#np.vstack(self.core_sets[0]), np.vstack(self.core_sets[1])
            nc_mem = np.sum(np.sum(mem_y,axis=0) > 0) # number of classes in episodic memory
            nc_batch = np.sum(np.sum(y_batch,axis=0) > 0) # number of classes in current task
            
            per_cl_size = int(buffer_size/(nc_mem+nc_batch))              
            msize = per_cl_size * nc_mem
            csize = buffer_size - msize
            #print('nc_mem',nc_mem,nc_batch)
            ### sampling from memory and current batch ###
            if self.WEM:

                # update sampling probability every K iters
                if s % self.K == 0:
                    x_core_sets, y_core_sets = np.vstack([mem_x,x_batch]), np.vstack([mem_y,y_batch])

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
            
            else:
                
                coreset_x, coreset_y = [], []
                for i in range(nc_mem): 
                    num_cl = np.sum(mem_y[:,i]==1)
                    ids = np.random.choice(num_cl,size=per_cl_size) #np.argsort(self.p_samples)[:self.batch_size*self.B]
                    coreset_x.append(mem_x[mem_y[:,i]==1][ids])
                    coreset_y.append(mem_y[mem_y[:,i]==1][ids])

            cids = np.random.choice(self.batch_size,size=csize)
            x_batch, y_batch = x_batch[cids], y_batch[cids]
            #print('batch shape',x_batch.shape,y_batch.shape)
            #print('coreset shape',coreset_x[0].shape,coreset_y[0].shape)
            coreset_x, coreset_y = np.vstack([*coreset_x,x_batch]), np.vstack([*coreset_y,y_batch])
            #print('mem',coreset_x.shape,coreset_y.shape)
            #print('mem size',mem_x.shape,mem_y.shape)
            feed_dict.update({self.x_ph:coreset_x,self.y_ph:coreset_y})
                
        ### empty memory ###              
        else:
            bids = np.random.choice(self.batch_size,size=buffer_size)                  
            feed_dict.update({self.x_ph:x_batch[bids],self.y_ph:y_batch[bids]})

        #print('feed dict',feed_dict[self.x_ph].shape)
        self.inference.update(sess=sess,K=self.K,feed_dict=feed_dict)


        ## todo: update err
        return err

    '''
    def train_update_step(self,t,s,sess,feed_dict,err=0.,x_train_task=None,y_train_task=None,local_iter=0,*args,**kargs):
        assert(self.coreset_size > 0)
        #print(self.x_b_list,self.y_b_list)

        if t > 0:
           
            #bids = np.random.choice(len(self.x_core_sets),size=self.batch_size)

            if self.WEM:
                buffer_size = self.batch_size*self.B
                msize = int(buffer_size/(2*(t+1)))
                csize = buffer_size - msize*(2*t)
                
                cids = np.random.choice(self.batch_size,size=csize)
                batch_x, batch_y = feed_dict[self.x_ph][cids], feed_dict[self.y_ph][cids]
                #coreset_x = np.vstack([self.x_core_sets,feed_dict[self.x_ph]])
                #coreset_y = np.vstack([self.y_core_sets,feed_dict[self.y_ph]])
                if s % self.K == 0:
                    x_core_sets, y_core_sets = np.vstack([self.x_core_sets,batch_x]), np.vstack([self.y_core_sets,batch_y])
                    #g_vecs,nlls = self.get_tasks_vec(sess,t,zip(x_core_sets,y_core_sets),test_sample=True)
                    #g_vecs = np.vstack(g_vecs)
                    hx = sess.run(self.H,feed_dict={self.x_ph:x_core_sets,self.y_ph:y_core_sets})
                    hx = np.hstack(hx)
                    sims = calc_similarity(hx,sess=sess)
                    #sims = -(np.sum(sims,axis=1) - 1.)
                    yids = 1.- np.matmul(y_core_sets,y_core_sets.transpose())
                    scores = (np.sum(sims*yids,axis=1)*0.5) - (np.sum(sims*(1.-yids),axis=1)*0.5)
                    self.p_samples = []
                    for i in range(2*t):
                        ids = y_core_sets[:,i]==1
                        self.p_samples.append(softmax(scores[ids]))
                   
                    print('p samples',self.p_samples)
                coreset_x, coreset_y = [], []
                for i in range(2*t):  
                    ids = np.random.choice(len(self.p_samples[i]),size=msize,p=self.p_samples[i]) #np.argsort(self.p_samples)[:self.batch_size*self.B]
                    coreset_x.append(self.x_core_sets[self.y_core_sets[:,i]==1][ids])
                    coreset_y.append(self.y_core_sets[self.y_core_sets[:,i]==1][ids])
                coreset_x, coreset_y = np.vstack([*coreset_x,batch_x]), np.vstack([*coreset_y,batch_y])
                #print('mem',coreset_x.shape,coreset_y.shape)
                ##
                mem_x,mem_y = [],[]
                rsize = self.batch_size*self.B
                #print('p samples',np.sum(p_samples),p_samples)
                
                while rsize > 0:
                    print('rsize',rsize)
                    ids = np.random.choice(len(coreset_x),size=rsize)
                    u_probs = np.random.uniform(size=len(ids))
                    accept = p_samples[ids] > u_probs 
                    mem_x.append(coreset_x[ids[accept]])
                    mem_y.append(coreset_y[ids[accept]])
                    rsize -= np.sum(accept)
                mem_x, mem_y = np.vstack(mem_x), np.vstack(mem_y)
                ##
                #print('mem size',mem_x.shape,mem_y.shape)
                feed_dict.update({self.x_ph:coreset_x,self.y_ph:coreset_y})
                

            else:
                cids = np.random.choice(self.batch_size,size=self.coreset_size)

                coreset_x = np.vstack([self.x_core_sets,feed_dict[self.x_ph][cids]])
                coreset_y = np.vstack([self.y_core_sets,feed_dict[self.y_ph][cids]])
                #coreset_x = np.vstack([self.x_core_sets[bids],feed_dict[self.x_ph]])
                #coreset_y = np.vstack([self.y_core_sets[bids],feed_dict[self.y_ph]])

                coreset_x,coreset_y = shuffle_data(coreset_x,coreset_y)
                bids = np.random.choice(len(coreset_x),size=self.batch_size*self.B)
                coreset_x,coreset_y = coreset_x[bids], coreset_y[bids]
                feed_dict.update({self.x_ph:coreset_x,self.y_ph:coreset_y})
                       
        else:
            bids = np.random.choice(self.batch_size,size=self.batch_size*self.B)
            #for b in range(len(self.x_b_list)):                  
            feed_dict.update({self.x_ph:feed_dict[self.x_ph][bids],self.y_ph:feed_dict[self.y_ph][bids]})

        #print('feed dict',feed_dict[self.x_ph].shape)
        self.inference.update(sess=sess,K=self.K,feed_dict=feed_dict)


        ## todo: update err
        return err
    '''

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
        reinitialize_scope(scope=scope,sess=sess)
        return

    
    def config_train(self,*args,**kargs):
        
        grads_and_vars = list(zip(self.grads,self.var_list))
        self.train = self.optimizer[0].apply_gradients(grads_and_vars,global_step=self.optimizer[1])

        return

    
    def update(self,sess,feed_dict=None,*args,**kargs):

        sess.run(self.train, feed_dict)

        return
    


class Meta_Stein_Inference(MAP_Inference):
    def __init__(self,var_list,grads_b_list,grad_logtp_list,eta,optimizer=None,ll=0.,kl=0.,*args,**kargs):
        self.var_list = var_list
        self.grads_b_list = grads_b_list
        self.grad_logtp_list = grad_logtp_list
        self.B = len(grads_b_list)
        self.eta = eta
        self.optimizer = optimizer
        print(*args,**kargs)
        if self.B == 1:
            self.kl = kl
            self.ll = ll

        self.config_train()


    def config_train(self,*args,**kargs):
        
        self.grads = self.get_w_grads()
        super(Meta_Stein_Inference,self).config_train()

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
            sum_i /= len(self.grads_b_list)
            w_grads.append(sum_i)
        return w_grads

    



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


