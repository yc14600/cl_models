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

from abc import ABC, abstractmethod
from utils.model_util import *
from utils.train_util import *
from .coreset import *

class BCL_BASE_MODEL(ABC):
    
    def __init__(self,net_shape,x_ph,y_ph,num_heads=1,batch_size=512,coreset_size=0,coreset_type='random',\
                    coreset_usage='regret',vi_type='KLqp_analytic',conv=False,\
                    ac_fn=tf.nn.relu,*args,**kargs):

        self.net_shape = net_shape
        print('net shape',self.net_shape)
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.coreset_size = coreset_size
        self.coreset_type = coreset_type
        self.coreset_usage = coreset_usage
        self.vi_type = vi_type
        self.ac_fn = ac_fn
        self.x_ph = x_ph
        self.y_ph = y_ph
        self.conv = conv
        return


    def init_inference(self,learning_rate,train_size,decay=None,grad_type='adam',*args,**kargs):
        self.config_optimizer(starter_learning_rate=learning_rate,decay=decay,grad_type=grad_type)
        self.config_inference(train_size,*args,**kargs)

        return

    
    @abstractmethod
    def define_model(self,*args,**kargs):
        pass

    
    @abstractmethod
    def config_coresets(self,*args,**kargs):   
        pass
    
    
    @abstractmethod
    def gen_task_coreset(self,*args,**kargs): 
        pass

    
    def config_optimizer(self,starter_learning_rate,decay=None, grad_type='adam',*agrs,**kargs):
        #with tf.variable_scope('task'):
            #if self.conv:
            #    self.task_optimizer = config_optimizer(starter_learning_rate,'task_step','adam',decay=decay)
            #else:
        self.task_optimizer = config_optimizer(starter_learning_rate,'task_step',grad_type,scope='task')

        if self.coreset_size > 0 and 'stein' in self.coreset_type:
            #with tf.variable_scope('stein'):
            self.stein_optimizer = config_optimizer(starter_learning_rate,'stein_step',grad_type,decay=decay,scope='stein')

        return


    @abstractmethod
    def config_inference(self,*args,**kargs):
        pass

    
    @abstractmethod
    def train_update_step(self,t,s,sess,feed_dict,kl=0.,ll=0.,err=0.,local_iter=10,*args,**kargs):
        return ll,kl,err

    
    @abstractmethod
    def train_task(self,*args,**kargs):
        pass

    @abstractmethod
    def test_all_tasks(self,*args,**kargs):
        pass

    @abstractmethod
    def config_next_task_parms(self,*args,**kargs):
        pass

    @abstractmethod
    def update_task_data_and_inference(self,*args,**kargs):
        pass

    @abstractmethod
    def update_task_data(self,*args,**kargs):
        pass