
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import edward as ed
#import matplotlib.pyplot as plt
#import seaborn as sb
import numpy as np
import scipy as sp
import sys
#sys.path.append('/Users/ycaamz/code/')
#sys.path.append('/home/ubuntu/code/')
from utils.train_util import get_next_batch

from tensorflow.examples.tutorials.mnist import input_data

class EWC_Model:
    def __init__(self,net_shape,lamb=1.,learning_rate=0.001,num_epoch=50,batch_size=256,print_iter=10,diag_fisher=True):
        self.x_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,net_shape[0]])
        self.y_ph = tf.placeholder(dtype=tf.int32,shape=[batch_size,net_shape[-1]])
        self.net_shape = net_shape
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.print_iter = print_iter
        
        self.build_nets()
        
        self.prev_parm = []
        self.parm = self.W+self.B
        self.acc_fisher = []
        self.acc_rl = None
        self.diag_fisher = diag_fisher

        #self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.99,)
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        
        self.sess = ed.get_session()#tf.get_default_session()
        tf.global_variables_initializer().run()
        
    def build_nets(self):
        W = []
        B = []
        h = self.x_ph
        net_shape = self.net_shape
        H =[]
        for l in range(len(net_shape)-1):
            d1 = net_shape[l]
            d2 = net_shape[l+1]
            w = tf.Variable(tf.random_normal([d1,d2],stddev=1./np.sqrt(d1)))
            b = tf.Variable(tf.zeros([d2]))
            if l == len(net_shape) - 2:
                h = tf.nn.softmax(tf.add(tf.matmul(h,w),b))
            else:
                h = tf.nn.relu(tf.add(tf.matmul(h,w),b))
            W.append(w)
            B.append(b)
            H.append(h)
        self.W = W
        self.B = B
        self.y = h
        self.H = H
        return 
    
    def build_task_loss(self, task_id):
        #y = OneHotCategorical(probs=y_pred)
        #ll = tf.reduce_sum(y.log_prob(y_ph))
        ll = -tf.reduce_sum(tf.losses.log_loss(labels=self.y_ph,predictions=self.y))
        rl = tf.zeros([])
        #print(ll.shape)
        if task_id == 0:
            loss = -ll
        else:
            if self.diag_fisher:
                #g1 = tf.gradients(-ll,self.parm)
                scale = 1./self.batch_size
                if self.acc_rl is None:
                    self.acc_rl = [0.]*len(self.parm)
                for i in range(len(self.parm)):
                    #ratio = self.acc_fisher[i]/(scale*tf.square(g1[i]))
                    #self.acc_rl[i] += tf.nn.relu(tf.multiply(scale*self.acc_fisher[i],self.parm[i]-self.prev_parm[i])) + \
                    #                    tf.multiply(scale*tf.square(self.acc_fisher[i]),tf.square(self.parm[i]-self.prev_parm[i]))
                    rl += tf.reduce_sum(tf.multiply(self.acc_fisher[i],tf.square(self.parm[i]-self.prev_parm[i])))
                    #rl += tf.reduce_sum(self.acc_rl[i])#*self.learning_rate
            else:
                scale = 1./self.batch_size
                
                # iterate each task
                for k in range(len(self.acc_fisher)):
                    # iterate each sample
                    for s in range(len(self.acc_fisher[k])):
                        gparm = 0.
                        # iterate parameters of each layer
                        for i in range(len(self.parm)):
                            gparm += tf.reduce_sum((self.parm[i]-self.prev_parm[i])*self.acc_fisher[k][s][i])
                        rl+=scale*tf.square(gparm)


            loss = -ll+0.5*self.lamb*rl#*tf.exp(-0.2*task_id) 

        self.rl = rl
        self.loss = loss
        self.ll = ll
        return 
    
    def update_acc_fisher(self):
        scale = 1./self.batch_size
        lli=tf.reduce_sum(tf.losses.log_loss(labels=self.y_ph,predictions=self.y,reduction=tf.losses.Reduction.NONE),axis=1)
        parm = self.parm
        if self.diag_fisher:
            
            #acc_fisher = self.acc_fisher
            #grad1 = tf.gradients(ll,parm)
            #if len(acc_fisher) == 0:
            acc_fisher = [0.]*len(parm)
            for s in range(lli.shape[0]):
                g1 = tf.gradients(lli[s],parm)
                
                for i,g in enumerate(g1):
                    g = tf.clip_by_value(g,-1.,1.)
                    acc_fisher[i] += scale*tf.square(g)
                    #acc_fisher[i] = scale*g
        else:
            #ll = self.ll           
            acc_fisher = []
            for s in range(lli.shape[0]):
                g1 = tf.gradients(lli[s],parm)
                sample_acc_fisher = [0.]*len(parm)
                for i,g in enumerate(g1):
                    sample_acc_fisher[i] = g   
                acc_fisher.append(sample_acc_fisher)
        return acc_fisher
    
    def save_parm(self):
        self.prev_parm = self.sess.run(self.parm)
        
        
    def update_train_step(self,task_id):
        self.build_task_loss(task_id)
        self.train_step = self.opt.minimize(self.loss)
                
        
    def fit(self,task_id,x_train,y_train):
        self.update_train_step(task_id)
        if task_id==0:
            tf.global_variables_initializer().run()
        num_iter = int(np.ceil(x_train.shape[0]/self.batch_size))
        print('num inter',num_iter)
        for e in range(self.num_epoch):
            ii = 0           
            shuffle_ids = np.arange(x_train.shape[0])
            np.random.shuffle(shuffle_ids)
            x_train = x_train[shuffle_ids] 
            y_train = y_train[shuffle_ids]          
            for _ in range(num_iter):
                x_batch,y_batch,ii = get_next_batch(x_train,self.batch_size,ii,labels=y_train)
                feed_dict = {self.x_ph:x_batch,self.y_ph:y_batch}
                rl,loss,__ = self.sess.run([self.rl,self.loss,self.train_step], feed_dict=feed_dict)
            if (e+1)%self.print_iter==0:
                print('loss',loss,'rl',rl)
                
        
        self.save_parm()
        if self.diag_fisher:
            self.acc_fisher = self.sess.run(self.update_acc_fisher(),feed_dict=feed_dict)
            
            mff = [mf.mean() for mf in self.acc_fisher]
            #print('fisher mean for parms of recent task')
            for m in mff:
                print(m)
        
        else:
            self.acc_fisher.append(self.sess.run(self.update_acc_fisher(),feed_dict=feed_dict))
            #print('acc fisher len',len(self.acc_fisher))
            #mff = [mf.mean() for mf in self.acc_fisher[-1]]
        
        
        
            
    def predict(self,x_test,y_test):
        n = int(x_test.shape[0]/self.batch_size)
        correct = 0.
        ii = 0
        for i in range(n):
            x_batch,y_batch,ii = get_next_batch(x_test,self.batch_size,ii,labels=y_test)
            feed_dict = {self.x_ph:x_batch}
            y_pred_prob = self.sess.run(self.y,feed_dict=feed_dict)
            y_pred = np.argmax(y_pred_prob,axis=1)
            #print(y_batch[range(self.batch_size),y_pred]==1)
            #print(y_pred_prob[0],y_batch[0],y_pred[0])
            correct += np.sum(np.argmax(y_batch,axis=1)==y_pred)

        accuracy = correct/y_test.shape[0]
        return accuracy