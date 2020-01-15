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
path = os.getcwd()
import sys
sys.path.append(path+'/../')

import tensorflow as tf
# In[3]:
from abc import ABC, abstractmethod
from .bcl_base_model import BCL_BASE_MODEL
from utils.model_util import *
from utils.train_util import *
from utils.coreset_util import *

from hsvi.hsvi import Hierarchy_SVI
from hsvi.methods.svgd import SVGD
from edward.models import RandomVariable

class BCL_BNN(BCL_BASE_MODEL):

    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=512,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,dropout=None,initialization=None,\
                    ac_fn=tf.nn.relu,n_smaples=1,local_rpm=False,conv_net_shape=None,strides=None,pooling=False,\
                    coreset_mode='offline',task_type='split',*args,**kargs):
        
        super(BCL_BNN,self).__init__(net_shape,x_ph,y_ph,num_heads,batch_size,coreset_size,coreset_type,\
                    coreset_usage,vi_type,conv,ac_fn)

        self.n_samples = n_smaples
        self.local_rpm = local_rpm
        self.conv_net_shape = conv_net_shape
        self.strides = strides
        self.pooling = pooling
        print('pooling',pooling)
        self.coreset_mode = coreset_mode
        self.task_type = task_type
        print('coreset mode',self.coreset_mode,'task type',task_type)

        return


    @abstractmethod
    def define_model(self,initialization=None,dropout=None,*args,**kargs):
        with tf.variable_scope('task'):
            if self.conv:
                self.conv_W,conv_parm_var,self.conv_h = build_bayes_conv_net(self.x_ph,self.batch_size,self.conv_net_shape,self.strides,pooling=self.pooling,local_rpm=self.local_rpm,initialization=initialization)
                in_x = self.conv_h
                self.net_shape[0] = in_x.shape[1].value
                
            else:
                in_x = self.x_ph   
                self.conv_W,conv_parm_var,self.conv_h = None,None,None

        
            if self.num_heads==1:
                print('single head net')
                self.qW,self.qB,self.H,self.TS,self.qW_samples,self.qB_samples,self.parm_var = build_nets(self.net_shape,in_x,bayes=True,ac_fn=self.ac_fn,share='isotropic',\
                                                                                                        initialization=initialization,dropout=dropout,num_samples=self.n_samples,\
                                                                                                        gaussian_type='logvar',local_rpm=self.local_rpm)
            else:
                print('multi-head net')
                self.qW_list,self.qB_list,self.H_list,self.TS,self.qW_list_samples,self.qB_list_samples,self.parm_var = build_nets(self.net_shape,in_x,bayes=True,ac_fn=self.ac_fn,share='isotropic',\
                                                                                                            initialization=initialization,dropout=dropout,num_samples=self.n_samples,gaussian_type='logvar',\
                                                                                                            num_heads=self.num_heads,local_rpm=self.local_rpm)
                self.qW,self.qB,self.H = self.qW_list[0],self.qB_list[0],self.H_list[0]
                self.qW_samples,self.qB_samples = self.qW_list_samples[0],self.qB_list_samples[0]
                #print('last layer',self.qW_list[-1][-1].shape)
            if self.conv:
                self.parm_var.update(conv_parm_var)

        return 


    def config_coresets(self,qW,qB,conv_W=None,core_x_ph=None,core_sets=[[],[]],K=None,bayes=True,bayes_ouput=True,*args,**kargs):
        print('check mode',self.coreset_mode)
        if self.coreset_mode != 'offline':            
            self.core_sets = {}
            return
       
        if K is None:
            K = self.num_heads

        if self.num_heads > 1:
            if core_x_ph is None:
                core_x_ph = [tf.placeholder(dtype=tf.float32,shape=self.x_ph.shape) for i in range(self.num_heads)]
            core_y = []
            if conv_W is None:
                for k in range(K):  
                    core_Hk = forward_nets(qW[k],qB[k],core_x_ph[k],ac_fn=self.ac_fn,bayes=bayes,num_samples=self.n_samples,bayes_output=bayes_ouput)
                    core_y.append(core_Hk[-1])

            else:
                for k in range(K):
                    h_k = forward_bayes_conv_net(core_x_ph[k],conv_W,self.strides,pooling=self.pooling) 
                    core_Hk = forward_nets(qW[k],qB[k],h_k,ac_fn=self.ac_fn,bayes=bayes,num_samples=self.n_samples,bayes_output=bayes_ouput)
                    core_y.append(core_Hk[-1])
        else:
            if core_x_ph is None:
                core_x_ph = tf.placeholder(dtype=tf.float32,shape=self.x_ph.shape)
            
            if conv_W is not None:
                h = forward_bayes_conv_net(core_x_ph,conv_W,self.strides,pooling=self.pooling) 
            else:
                h = core_x_ph

            core_y = forward_nets(qW,qB,h,ac_fn=self.ac_fn,bayes=bayes,num_samples=self.n_samples,bayes_output=bayes_ouput)[-1]
    
        self.core_x_ph = core_x_ph  
        self.core_y = core_y  
        self.core_sets = core_sets
        
        return        

    def gen_task_coreset(self,t,x_train_task,y_train_task,task_name,sess,cl_n=0,clss=None,*args,**kargs):
        if 'kcenter' in self.coreset_type :
            idx = gen_kcenter_coreset(x_train_task,self.coreset_size)
            core_x_set = x_train_task[idx]
            core_y_set = y_train_task[idx]
        
        elif 'random' in self.coreset_type or self.coreset_type == 'stein':
            # default initialization of stein is random samples
            '''
            while True:
                idx = np.random.choice(x_train_task.shape[0],self.coreset_size)
                core_y_set = y_train_task[idx]
                if np.sum(core_y_set.sum(axis=0)>1) == cl_n:
                    break
            core_x_set = x_train_task[idx]
            '''
            core_x_set,core_y_set = gen_random_coreset(x_train_task,y_train_task,self.coreset_size,clss)
            #print('core x {}'.format(core_x_set.shape))
            #print('core y {}, {}'.format(core_y_set.shape,core_y_set[:3]))
        elif 'rdproj' in self.coreset_type:
            if 'split' in task_name:
                core_x_set,core_y_set = gen_rdproj_coreset(x_train_task,y_train_task,self.coreset_size,cl_n,clss)
            else:
                core_x_set,core_y_set = gen_rdproj_coreset(x_train_task,y_train_task,self.coreset_size,cl_n)

        elif 'distill' in self.coreset_type:
            return x_train_task,y_train_task
        
        self.core_sets[1].append(core_y_set)
        curnt_core_y_data = expand_nsamples(self.core_sets[1][-1],self.n_samples)
        if self.coreset_usage == 'final' and 'rdproj' not in self.coreset_type:
            # remove coreset from the training set
            x_train_task = np.delete(x_train_task,idx,axis=0)
            y_train_task = np.delete(y_train_task,idx,axis=0)
        
        
        if 'stein' not in self.coreset_type:
            self.core_sets[0].append(core_x_set)
        
        else:  # define stein samples
            if isinstance(self.qW[0],RandomVariable):
                qW = [w.loc for w in self.qW]
                qB = [b.loc for b in self.qB]
                if self.conv:
                    cW = [w.loc for w in self.conv_W]
            else:
                qW = self.qW
                qB = self.qB
                if self.conv:
                    cW = self.conv_W
            
            with tf.variable_scope('stein_task'+str(t)):
                if self.conv :
                    self.stein_core_x,self.stein_core_y,core_sgrad = gen_stein_coreset(core_x_set,curnt_core_y_data,qW,qB,self.n_samples,self.ac_fn,conv_W=cW)
                else:
                    self.stein_core_x,self.stein_core_y,core_sgrad = gen_stein_coreset(core_x_set,curnt_core_y_data,qW,qB,self.n_samples,self.ac_fn)

            self.core_sets[0].append(self.stein_core_x)
            self.stein_train = self.stein_optimizer[0].apply_gradients([(core_sgrad,self.stein_core_x)],global_step=self.stein_optimizer[1])
            tf.variables_initializer(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="stein_task"+str(t))).run()
            sess.run(tf.variables_initializer(self.stein_optimizer[0].variables()))

        self.curnt_core_y_data = curnt_core_y_data

        return x_train_task,y_train_task


    def update_ring_buffer(self,t,x_batch,y_batch):
           
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

        return

    def online_update_coresets(self,coreset_size,fixed_budget,t):
    
        if self.coreset_mode == 'ring_buffer':
            if fixed_budget:
                clen = [(c,len(cx)) for c,cx in self.core_sets.items()] if self.task_type=='split' else [(c,len(cx[0])) for c,cx in self.core_sets.items()]
                lens = [it[1] for it in clen]
                R = np.sum(lens) - coreset_size
                while R > 0:
                    c = clen[np.argmax(lens)][0]
                    if self.task_type == 'split':
                        self.core_sets[c] = self.core_sets[c][1:]
                    else:
                        self.core_sets[c] = (self.core_sets[c][0][1:],self.core_sets[c][1][1:])
                    R -= 1
                    clen = [(c,len(cx)) for c,cx in self.core_sets.items()] if self.task_type=='split' else [(c,len(cx[0])) for c,cx in self.core_sets.items()]
                    lens = [it[1] for it in clen]
                #print('{} samples in mem'.format(lens))
                
            else:
                if self.task_type == 'split':
                    for i in self.core_sets.keys():
                        cx = self.core_sets[i]  
                        if coreset_size < len(cx):                                            
                            cx = cx[-coreset_size:]
                            self.core_sets[i] = cx
                            #print('after online update',i,len(cx))
                else:
                    ## permuted task ##
                    cx = self.core_sets[t][0]
                    cy = self.core_sets[t][1]
                    num_per_cls = int(coreset_size/cy.shape[1])
                    num_cls = np.sum(cy,axis=0).astype(int)
                    
                    clss = num_cls > num_per_cls
                    #print('clss',clss,clss.sum())
                    tot = clss.sum()
                    if tot > 0:
                        clss = np.argsort(clss)[-tot:]
                        #print('num cls',num_cls,clss)
                        for c in clss:
                            #print('check online update',c,num_cls[c],num_per_cls)
                            cids = cy[:,c]==1
                            rids = np.argsort(cids)[-num_cls[c]:-num_per_cls]
                            cids = np.ones(len(cx))
                            cids[rids] = 0
                            cx = cx[cids.astype(bool)]
                            #print('after update',c,len(cx))
                            cy = cy[cids.astype(bool)]
                        self.core_sets[t] = (cx,cy)
                







    def config_inference(self,TRAIN_SIZE,scale=1.,shrink=1.,*args,**kargs):
        
        inference = Hierarchy_SVI(latent_vars={'task':self.task_var_cfg},data={'task':{self.H[-1]:self.y_ph}})

        if 'KLqp' in self.vi_type or 'MLE' in self.vi_type:
            if 'NG' in self.vi_type:
                inference.initialize(vi_types={'task':self.vi_type},scale={self.H[-1]:scale},optimizer={'task':self.task_optimizer},train_size=TRAIN_SIZE*shrink,trans_parm={'task':self.parm_var})
            else:
                inference.initialize(vi_types={'task':self.vi_type},scale={self.H[-1]:scale},optimizer={'task':self.task_optimizer},train_size=TRAIN_SIZE*shrink)
        else:
            raise TypeError('NOT supported VI type!')

        self.inference = inference

        return


    def train_task(self,sess,t,x_train_task,y_train_task,epoch,print_iter=5,local_iter=10,\
                    tfb_merged=None,tfb_writer=None,tfb_avg_losses=None,*args,**kargs):
        #print('args',args,'kargs',kargs)
        # training for current task
        num_iter = int(np.ceil(x_train_task.shape[0]/self.batch_size))
        #sess.run(self.task_optimizer[1].initializer)
        print('num iter',num_iter)
        for e in range(epoch):
            shuffle_inds = np.arange(x_train_task.shape[0])
            np.random.shuffle(shuffle_inds)
            x_train_task = x_train_task[shuffle_inds]
            y_train_task = y_train_task[shuffle_inds]
            err,kl,ll = 0.,0.,0.
            ii = 0
            for _ in range(num_iter):
                x_batch,y_batch,ii = get_next_batch(x_train_task,self.batch_size,ii,labels=y_train_task)
                y_batch = np.expand_dims(y_batch,axis=0)
                y_batch = np.repeat(y_batch,self.n_samples,axis=0)
                
                feed_dict = {self.x_ph:x_batch,self.y_ph:y_batch}

                ll,kl,err = self.train_update_step(t,_,sess,feed_dict,kl,ll,err,local_iter,*args,**kargs)

            if tfb_merged is not None:
                summary = sess.run(tfb_merged,feed_dict={self.x_ph:x_batch,self.y_ph:y_batch,tfb_avg_losses[0]:err/num_iter,tfb_avg_losses[1]:kl/num_iter,tfb_avg_losses[2]:ll/num_iter})
                tfb_writer.add_summary(summary, e+t*epoch)
            if (e+1)%print_iter==0:
                print('epoch',e+1,'avg loss',err/num_iter)
        
        return

    def save_parm(self,t,file_path,sess,*args,**kargs):
        w_stds, w_means, b_stds,b_means = [],[],[],[]
        for l in range(len(self.qW)):           
            w_stds.append(sess.run(self.qW[l].scale))          
            w_means.append(sess.run(self.qW[l].loc))
            b_stds.append(sess.run(self.qB[l].scale))
            b_means.append(sess.run(self.qB[l].loc))
        
        np.save(file_path+'task'+str(t)+'_parms_std',w_stds+b_stds)
        np.save(file_path+'task'+str(t)+'_parms_mean',w_means+b_means)    
    
        return

    def test_all_tasks(self,t,test_sets,sess,epoch=10,saver=None,file_path=None,bayes=True,bayes_output=True,*args,**kargs):
        # test on all tasks 
        #if acc_record is None:
        #    acc_record = []
        if t > 0 and self.coreset_size >  0 and self.coreset_usage == 'final':
            saver.save(sess, file_path+"_model.ckpt")
            train_coresets_final(self.core_sets,self.core_y,self.x_ph,self.y_ph,self.core_x_ph,self.coreset_type,self.num_heads,t,self.n_samples,\
                                        self.batch_size,epoch,sess,self.inference)

        if self.num_heads > 1:  
            accs, probs,cfm = test_tasks(t,test_sets,self.qW_list,self.qB_list,self.num_heads,self.x_ph,self.ac_fn,self.batch_size,sess,conv_h=self.conv_h,bayes=bayes,bayes_output=bayes_output,*args,**kargs)       
        else:
            accs, probs,cfm = test_tasks(t,test_sets,self.qW,self.qB,self.num_heads,self.x_ph,self.ac_fn,self.batch_size,sess,conv_h=self.conv_h,bayes=bayes,bayes_output=bayes_output,*args,**kargs)        
        
        #acc_record.append(accs)  
        # reset variables
        if t > 0 and self.coreset_size > 0 and self.coreset_usage == 'final':
            saver.restore(sess, file_path+"_model.ckpt")

        return accs, probs,cfm



    def update_task_data(self,sess,t,task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,\
                        original_batch_size=500,cl_n=2,cl_k=0,cl_cmb=None,train_size=-1,test_size=-1,*args,**kargs):    
        # update data and inference for next task 
        print('train size {}, test_size {}'.format(train_size,test_size))
        if 'permuted' in task_name:
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,sd=t+1,train_size=train_size,test_size=test_size)
        
        elif 'cross_split' in task_name:
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,sd=t+1,cl_k=cl_k,out_dim=out_dim,train_size=train_size,test_size=test_size)
        

        elif 'split' in task_name:
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,cl_n=cl_n,out_dim=out_dim,num_heads=self.num_heads,cl_cmb=cl_cmb,cl_k=cl_k,train_size=train_size,test_size=test_size)
        
            TRAIN_SIZE = x_train_task.shape[0]    
            #TEST_SIZE = x_test_task.shape[0]
            if original_batch_size > TRAIN_SIZE:
                self.batch_size = TRAIN_SIZE  
            else:
                self.batch_size = original_batch_size
            self.inference.train_size = TRAIN_SIZE

            print('train size',TRAIN_SIZE,'batch size',self.batch_size)
        

        return x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss


