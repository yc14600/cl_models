# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:
#import matplotlib.pyplot as plt
#from IPython import display
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
import argparse


from utils.model_util import *
from utils.train_util import *
from utils.coreset_util import *
from utils.test_util import *


from hsvi import hsvi
from hsvi.methods.svgd import SVGD
from utils.data_util import save_samples
from utils.train_util import shuffle_data
from models.vcl_model import VCL
from models.vcl_kd import VCL_KD
from edward.models import Normal,MultivariateNormalTriL
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.datasets import cifar10,cifar100

# In[5]:
parser = argparse.ArgumentParser()

parser.add_argument('-sd','--seed', default=42, type=int, help='random seed')
parser.add_argument('-ds','--dataset', default='mnist', type=str, help='specify datasets')
parser.add_argument('-rp','--result_path',default='./results/',type=str,help='the path for saving results')
parser.add_argument('-ttp','--task_type', default='split', type=str, help='task type can be split, permuted, cross split, batch')
parser.add_argument('-vtp','--vi_type', default='KLqp_analytic', type=str, help='type of variational inference')
parser.add_argument('-e','--epoch', default=50, type=int, help='number of epochs')
parser.add_argument('-csz','--coreset_size', default=0, type=int, help='size of each class in a coreset')
parser.add_argument('-ctp','--coreset_type', default='random', type=str, help='type of coresets, can be random,stein,kcenter')
parser.add_argument('-cus','--coreset_usage', default='regret', type=str, help='usage type of coresets, can be regret, final')
parser.add_argument('-gtp','--grad_type', default='adam', type=str, help='type of gradients optimizer')
parser.add_argument('-bsz','--batch_size', default=500, type=int, help='batch size')
parser.add_argument('-trsz','--train_size', default=50000, type=int, help='size of training set')
parser.add_argument('-tesz','--test_size', default=10000, type=int, help='size of testing set')
parser.add_argument('-nts','--num_tasks', default=10, type=int, help='number of tasks')
parser.add_argument('-nli','--local_iter', default=50, type=int, help='number of local iterations for stein coreset')
parser.add_argument('-mh','--multihead', default=True, type=str2bool, help='multihead model')
parser.add_argument('-svp','--save_parm', default=False, type=str2bool, help='if save parameters')
parser.add_argument('-gi','--ginit', default=3, type=int, help='power of global initialization of variance, 3 -> e-3')
parser.add_argument('-ns','--num_samples', default=1, type=int, help='number of parameter samples')
parser.add_argument('-lrp','--local_rpm', default=False, type=str2bool, help='if use local reparameterzation')
parser.add_argument('-lr','--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('-af','--ac_fn', default='relu', type=str, help='activation function of hidden layers')
parser.add_argument('-irt','--irt', default=False, type=str2bool, help='generate responses for IRT modelling')
parser.add_argument('-tb','--tensorboard', default=False, type=str2bool, help='enable tensorboard')
parser.add_argument('-mtp','--model_type', default='continual', type=str,help='model type can be continual,single')
parser.add_argument('-fim','--save_FIM', default=False,type=str2bool,help='save Fisher Info Matrix of the model')
parser.add_argument('-vcltp','--vcl_type', default='vanilla', type=str,help='vcl type can be vanilla,kd')
parser.add_argument('-hdn','--hidden',default=[100,100],type=str2ilist,help='hidden units of each layer of the network')
parser.add_argument('-kdr','--kd_reg',default=False,type=str2bool,help='if enable kd regularizer for vcl kd')


args = parser.parse_args()
print(args)

seed = args.seed
print('seed',seed)
tf.set_random_seed(seed)
np.random.seed(seed)

ac_fn = set_ac_fn(args.ac_fn)
# In[6]:

dataset = args.dataset

if dataset == 'mnist':
    DATA_DIR = '../../../../data/mnist/'
elif dataset == 'fashion':
    DATA_DIR = '/home/yu/gits/data/fashion/'
elif dataset == 'notmnist':
    DATA_DIR = '/home/yu/gits/data/not-mnist/'
elif dataset == 'not-notmnist':
    args.task_type = 'cross_split'
    DATA_DIR = ['/home/yu/gits/data/mnist/','/home/yu/gits/data/not-mnist/']
elif dataset == 'quickdraw':
    DATA_DIR = '/home/yu/gits/data/quickdraw/full/numpy_bitmap/'
print(dataset)


hidden = args.hidden #[256,256]
scale = 1.#TRAIN_SIZE/batch_size#weights of likelihood
shrink = 1. #shrink train_size, smaller gives larger weights of KL-term


decay=(1000,0.9)
print_iter = 10

share_type = 'isotropic'
prior_type = 'normal'

#n_samples = 20
gaussian_type = 'logvar'

conv = False

# In[8]:

print(args.task_type)

if 'split' in args.task_type:
    if dataset in ['fashion','mnist','notmnist']:
        num_tasks = 5
    elif dataset in ['not-notmnist','cifar']:
        num_tasks = 10
    elif dataset == 'quickdraw':
        num_tasks = 8
elif 'batch' in args.task_type:
    num_tasks = 1
else:
    num_tasks = args.num_tasks

if not args.multihead:
    num_heads = 1
else:
    num_heads = num_tasks

if 'cifar' in dataset:
    conv = True

print('heads',num_heads)
# In[9]:

result_path = args.result_path



# load data for different task

if  args.task_type in ['permuted', 'batch']:
    data = input_data.read_data_sets(DATA_DIR,one_hot=True) 
    shuffle_ids = np.arange(data.train.images.shape[0])
    X_TRAIN = data.train.images[shuffle_ids][:args.train_size]
    Y_TRAIN = data.train.labels[shuffle_ids][:args.train_size]
    X_TEST = data.test.images[:args.test_size]
    Y_TEST = data.test.labels[:args.test_size]
    out_dim = Y_TRAIN.shape[1]
    cl_n = out_dim # number of classes in each task
    cl_cmb = None
    # generate data for first task
    if 'permuted' in args.task_type:
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST)
    else:
        x_train_task,y_train_task,x_test_task,y_test_task = X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
        clss = None

elif 'cross_split' in args.task_type:
    if dataset == 'not-notmnist':
        data1 = input_data.read_data_sets(DATA_DIR[0])
        data2 = input_data.read_data_sets(DATA_DIR[1])
        X_TRAIN = [data1.train.images,data2.train.images]
        Y_TRAIN = [data1.train.labels,data2.train.labels]
        X_TEST = [data1.test.images,data2.test.images]
        Y_TEST = [data1.test.labels,data2.test.labels]
        out_dim = 2
        cl_n = out_dim
        cl_cmb = None
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST) 
    
    elif dataset == 'quickdraw':
        X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = load_mini_quick_draw(DATA_DIR)
        out_dim = len(X_TRAIN)
        cl_n = out_dim
        cl_cmb = None
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim=out_dim) 


elif 'split' in args.task_type:
    if dataset == 'cifar':
        hidden = [512,512]
        (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar10.load_data() 
        # standardize data
        X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)
        print('data shape',X_TRAIN.shape)
        
        if num_heads > 1:
            out_dim = 10
        else:
            out_dim = 100

        Y_TRAIN = one_hot_encoder(Y_TRAIN.reshape(-1),out_dim)
        Y_TEST = one_hot_encoder(Y_TEST.reshape(-1),out_dim)

        # first task use all cifar10 data
        x_train_task,y_train_task,x_test_task,y_test_task = X_TRAIN,Y_TRAIN,X_TEST,Y_TEST
        # load cifar 100
        (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar100.load_data() 
        X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)
        cl_cmb = np.arange(100)
        cl_k = 0
        cl_n = 10
        clss = cl_cmb[cl_k:cl_k+cl_n]
    else:
        data = input_data.read_data_sets(DATA_DIR) 
        X_TRAIN = np.concatenate([data.train.images,data.validation.images],axis=0)
        Y_TRAIN = np.concatenate([data.train.labels,data.validation.labels],axis=0)
        X_TEST = data.test.images
        Y_TEST = data.test.labels
        
        if num_heads > 1:
            out_dim = 2
        else:
            out_dim = 2 * num_tasks

        cl_cmb = np.arange(10)
        cl_k = 0
        cl_n = 2
        
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,\
                                                                    cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads)


TRAIN_SIZE = x_train_task.shape[0]
TEST_SIZE = x_test_task.shape[0]
original_batch_size = args.batch_size
batch_size = TRAIN_SIZE if args.batch_size > args.train_size else args.batch_size
print('batch size',batch_size)

# set results path and file name
if not os.path.exists(result_path):
    os.mkdir(result_path)
head = 'multi' if args.multihead else 'single'
file_name = dataset+'_'+args.vi_type+'_tsize'+str(TRAIN_SIZE)+'_cset'+str(args.coreset_size)+args.coreset_type+'_'+args.coreset_usage+'_nsample'+str(args.num_samples)+'_bsize'+str(batch_size)+'_init'+str(int(args.ginit))\
            +'_e'+str(args.epoch)+'_lit'+str(args.local_iter)+'_'+args.task_type+'_lrpm'+str(args.local_rpm)+'_'+args.grad_type+'_'+head+'_'+args.model_type+'_'+args.vcl_type+'_sd'+str(seed)

file_path = result_path+file_name
file_path = config_result_path(file_path)
with open(file_path+'configures.txt','w') as f:
    f.write(str(args))

# In[16]:
# Initialize model

if args.ginit > 0:
    # set init value of variance
    initialization={'w_s':-1.*args.ginit,'b_s':-1.*args.ginit,'cw_s':-1*args.ginit}
else:
    initialization=None

if conv:
    x_ph = tf.placeholder(dtype=tf.float32,shape=[None]+list(x_train_task.shape[1:]))
    in_dim = None
    dropout = 0.5
else:
    x_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_train_task.shape[1]])
    in_dim = x_train_task.shape[1]
    dropout = None

y_ph = tf.placeholder(dtype=tf.int32,shape=[args.num_samples,None,out_dim]) 

net_shape = [in_dim]+hidden+[out_dim]



if args.vcl_type=='vanilla':
    Model = VCL(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,args.coreset_usage,\
                args.vi_type,conv,dropout,initialization=initialization,ac_fn=ac_fn,n_samples=args.num_samples,local_rpm=args.local_rpm)
    scale = 1.
elif args.vcl_type=='kd':
    args.model_type = 'continual' # can only be continual for vcl_kd
    Model = VCL_KD(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,conv=conv,dropout=dropout,vi_type=args.vi_type,\
                initialization=initialization,ac_fn=ac_fn,n_samples=args.num_samples,local_rpm=args.local_rpm,enable_kd_reg=args.kd_reg)
    scale = 1.
else:
    raise TypeError('Wrong type of VCL')


Model.init_inference(learning_rate=args.learning_rate,train_size=TRAIN_SIZE,grad_type=args.grad_type,scale=scale)

sess = ed.get_session() 

# choose parameters for tensorboard record
if args.tensorboard:
    for gi in Model.inference.grads['task']:
        if gi[1] == tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task/layer0w_mean")[0]:
            g = gi
        if conv and gi[1] == tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task/convlayer0w_var")[0]:
            cg = gi

    i,j = 689,8
    tf.summary.histogram('weight_exsample1_loc', tf.gather(tf.gather(Model.qW[0].loc,i),j))
    tf.summary.scalar('weight_exsample1_loc', tf.gather(tf.gather(Model.qW[0].loc,i),j))
    tf.summary.scalar('weight_exsample1_logvar', tf.log(tf.square(tf.gather(tf.gather(Model.qW[0].scale,i),j))))
    tf.summary.scalar('weight_exsample1_grads', tf.gather(tf.gather(g[0],i),j))

    i,j = 392,4
    tf.summary.histogram('weight_exsample2_loc', tf.gather(tf.gather(Model.qW[0].loc,i),j))
    tf.summary.scalar('weight_exsample2_loc', tf.gather(tf.gather(Model.qW[0].loc,i),j))
    tf.summary.scalar('weight_exsample2_logvar', tf.log(tf.square(tf.gather(tf.gather(Model.qW[0].scale,i),j))))
    tf.summary.scalar('weight_exsample2_grads', tf.gather(tf.gather(g[0],i),j))

    # conv layer variance
    if conv:    
        tf.summary.scalar('conv_weight_example_logvar', tf.log(tf.square(tf.gather(tf.gather(tf.gather(tf.gather(Model.conv_W[0].scale,0),0),0),0))))
        tf.summary.scalar('conv_weight_example_grads', tf.gather(tf.gather(tf.gather(tf.gather(cg[0],0),0),0),0))

    '''
    if local_rpm:
        # hidden units
        tf.summary.scalar('layer0_hidden_mean', tf.reduce_mean(Model.H[0].loc))
        tf.summary.scalar('layer0_hidden_maxv', tf.reduce_max(Model.H[0].scale))
        tf.summary.scalar('layer0_hidden_minv', tf.reduce_min(Model.H[0].scale))
    '''
    # writer for tensorboard
    avg_err = tf.placeholder(shape=[],dtype=tf.float32)
    avg_kl = tf.placeholder(shape=[],dtype=tf.float32)
    avg_ll = tf.placeholder(shape=[],dtype=tf.float32)
    tf.summary.scalar('Negative_ELBO',avg_err)
    tf.summary.scalar('KL_term',avg_kl)
    tf.summary.scalar('Negative_LogLikelihood', avg_ll)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./tfb_summary/'+file_name,
                                        sess.graph)


# Start training tasks
test_sets = []
acc_record, probs_record = [], []
pre_parms = {}
saver = tf.train.Saver()
tf.global_variables_initializer().run()
print('num tasks',num_tasks)
for t in range(num_tasks):
    # get test data
    test_sets.append((x_test_task,y_test_task))
    if Model.coreset_size > 0:
        x_train_task,y_train_task = Model.gen_task_coreset(t,x_train_task,y_train_task,args.task_type,sess,cl_n,clss)
    
    if args.tensorboard:
        Model.train_task(sess,t,x_train_task,y_train_task,args.epoch,print_iter,args.local_iter,\
                        tfb_merged=merged,tfb_writer=train_writer,tfb_avg_losses=[avg_err,avg_kl,avg_ll])
    else:
        Model.train_task(sess,t,x_train_task,y_train_task,args.epoch,print_iter,args.local_iter)

    if args.save_parm:
        Model.save_parm(t,file_path,sess)
    
    if args.save_FIM:
        FIM = get_model_FIM(Model,sess=sess)
        np.save(file_path+'model_FIM_task'+str(t)+'.npy',FIM)

    accs, probs = Model.test_all_tasks(t,test_sets,sess,args.epoch,saver=saver,file_path=file_path)
    acc_record.append(accs)
    if args.irt:
        #print(probs[0].shape,test_sets[0][1].shape)
        probs = [prb[np.arange(len(prb)),np.argmax(ts[1],axis=1)] for prb,ts in zip(probs,test_sets)]
        if num_heads  > 1:
            labels = [np.argmax(ts[1],axis=1)+t*out_dim for ts in test_sets]
        else:
            labels = [np.argmax(ts[1],axis=1) for ts in test_sets]
        #samples = [ts[0] for ts in test_sets]
        #samples = np.vstack(samples)
        probs = np.concatenate(probs)
        #print(type(labels[0][0]))
        labels = np.concatenate(labels).astype(np.uint8)
        #samples,probs,labels = shuffle_data(samples,probs,labels)
        save_samples(file_path,[probs,labels],['test_resps_t'+str(t), 'test_labels_t'+str(t)])

    if t < num_tasks-1:

        '''
        if t==4 and dataset=='notmnist' and  args.task_type=='split':
            DATA_DIR = '../datasets/MNIST_data/'
            X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = load_task_data(args.task_type,DATA_DIR)
            cl_k = 0
        '''
        if args.model_type == 'continual':
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = Model.update_task_data_and_inference(sess,t,args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,\
                                                                                                        original_batch_size=batch_size,cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,clss=clss,\
                                                                                                        x_train_task=x_train_task,y_train_task=y_train_task,rpath=file_path)
        elif args.model_type == 'single':
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = Model.update_task_data(sess,t,args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,original_batch_size=batch_size,cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb)
            if Model.coreset_size>0 and Model.coreset_usage != 'final':
                Model.x_core_sets,Model.y_core_sets,c_cfg = aggregate_coreset(Model.core_sets,Model.core_y,Model.coreset_type,Model.num_heads,t,Model.n_samples,sess)
            tf.global_variables_initializer().run()
            Model.inference.reinitialize(task_id=t+1,coresets={'task':c_cfg})

with open(file_path+'accuracy_record.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(acc_record)):
        writer.writerow(acc_record[t])

if not args.save_parm and args.coreset_usage=='final':
    os.system('rm '+file_path+"ssmodel.ckpt*")
