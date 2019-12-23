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
from utils.model_util import mean_list
from models.vcl_model import VCL
from models.vcl_kd import VCL_KD
from models.stein_cl import Stein_CL
from cl_models.discriminative.drs.drs_cl import DRS_CL
from edward.models import Normal,MultivariateNormalTriL
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.datasets import cifar10,cifar100
import matplotlib.pyplot as plt
import seaborn as sn
# In[5]:
parser = argparse.ArgumentParser()

parser.add_argument('-sd','--seed', default=42, type=int, help='random seed')
parser.add_argument('-ds','--dataset', default='mnist', type=str, help='specify datasets')
parser.add_argument('-dp','--data_path',default='/home/yu/gits/data/',type=str,help='path to dataset')
parser.add_argument('-rp','--result_path',default='./results/',type=str,help='the path for saving results')
parser.add_argument('-ttp','--task_type', default='split', type=str, help='task type can be split, permuted, cross split, batch')
parser.add_argument('-vtp','--vi_type', default='KLqp_analytic', type=str, help='type of variational inference')
parser.add_argument('-e','--epoch', default=50, type=int, help='number of epochs')
parser.add_argument('-pe','--print_epoch', default=10, type=int, help='number of epochs of printing loss')
parser.add_argument('-csz','--coreset_size', default=0, type=int, help='size of each class in a coreset')
parser.add_argument('-ctp','--coreset_type', default='random', type=str, help='type of coresets, can be random,stein,kcenter')
parser.add_argument('-cus','--coreset_usage', default='regret', type=str, help='usage type of coresets, can be regret, final')
parser.add_argument('-cmod','--coreset_mode', default='offline', type=str, help='construction mode of coresets, can be offline, ring_buffer')
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
parser.add_argument('-dcy','--decay', default=[1000,0.9], type=str2flist, help='decay of learning rate')
parser.add_argument('-af','--ac_fn', default='relu', type=str, help='activation function of hidden layers')
parser.add_argument('-irt','--irt', default=False, type=str2bool, help='generate responses for IRT modelling')
parser.add_argument('-tb','--tensorboard', default=False, type=str2bool, help='enable tensorboard')
parser.add_argument('-mtp','--model_type', default='continual', type=str,help='model type can be continual,single')
parser.add_argument('-fim','--save_FIM', default=False,type=str2bool,help='save Fisher Info Matrix of the model')
parser.add_argument('-vcltp','--vcl_type', default='vanilla', type=str,help='vcl type can be vanilla,kd,stein')
parser.add_argument('-hdn','--hidden',default=[100,100],type=str2ilist,help='hidden units of each layer of the network')
parser.add_argument('-kdr','--kd_reg',default=False,type=str2bool,help='if enable kd regularizer for vcl kd')
parser.add_argument('-kdvr','--kd_vcl_reg',default=True,type=str2bool,help='if enable vcl and kd regularizers together for vcl kd')
parser.add_argument('-tdt','--task_dst',default=False,type=str2bool,help='if calc task distance')
parser.add_argument('-cv','--conv',default=False,type=str2bool,help='if use CNN on top')
parser.add_argument('-B','--B',default=3,type=int,help='training batch size')
parser.add_argument('-K','--K',default=10,type=int,help='every K iters update sampling probabilities in DRS weighted EM')
parser.add_argument('-eta','--eta',default=0.001,type=float,help='learning rate of meta Stein gradients')
parser.add_argument('-disc','--discriminant',default=False,type=str2bool,help='enable discriminant in drs cl')
parser.add_argument('-lam_dis','--lambda_disc',default=0.001,type=float,help='lambda discriminant')
parser.add_argument('-lam_reg','--lambda_reg',default=0.0001,type=float,help='lambda regularization')
parser.add_argument('-wem','--WEM',default=False,type=str2bool,help='enable weighted EM in drs cl')
parser.add_argument('-bit','--batch_iter',default=1,type=int,help='iterations on one batch')
parser.add_argument('-ntp','--net_type',default='dense',type=str,help='network type, can be dense, conv, resnet18')
parser.add_argument('-fxbt','--fixed_budget',default=True,type=str2bool,help='if budget of episodic memory is fixed or not')
parser.add_argument('-ptp','--pretrained_path',default='',type=str,help='path to pretrained resnet18 on cifar10')



args = parser.parse_args()
print(args)

seed = args.seed
print('seed',seed)
tf.set_random_seed(seed)
np.random.seed(seed)

ac_fn = set_ac_fn(args.ac_fn)
# In[6]:

dataset = args.dataset

if dataset in ['mnist','fashion','notmnist']:
    DATA_DIR = os.path.join(args.data_path,dataset)
elif dataset == 'not-notmnist':
    args.task_type = 'cross_split'
    DATA_DIR = [os.path.join(args.data_path,d) for d in ['mnist','notmnist']]
elif dataset == 'quickdraw':
    DATA_DIR = os.path.join(args.data_path,'/quickdraw/full/numpy_bitmap/')
print(dataset)


hidden = args.hidden #[256,256]
scale = 1.#TRAIN_SIZE/batch_size#weights of likelihood
shrink = 1. #shrink train_size, smaller gives larger weights of KL-term

if args.decay is not None:  
    decay = (int(args.decay[0]),args.decay[1])
else:
    decay = None



share_type = 'isotropic'
prior_type = 'normal'

#n_samples = 20
gaussian_type = 'logvar'

conv = args.conv

# In[8]:

print(args.task_type)

if 'split' in args.task_type:
    if dataset in ['fashion','mnist','notmnist','cifar10']:
        num_tasks = 5
    elif dataset in ['not-notmnist']:
        num_tasks = 10
    elif dataset == 'quickdraw':
        num_tasks = 8
    else:
        num_tasks = args.num_tasks
elif 'batch' in args.task_type:
    num_tasks = 1
else:
    num_tasks = args.num_tasks

if not args.multihead:
    num_heads = 1
else:
    num_heads = num_tasks

#if 'cifar' in dataset:
#    conv = True

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
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,\
                                                                                        train_size=args.train_size,test_size=args.test_size)
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
    if 'cifar' in dataset:
        
        if args.net_type == 'resnet18':
            conv = False
            hidden = []
            args.vcl_type = 'drs'
        else:
            conv =True
            hidden = [512,512]
        
        if dataset  == 'cifar10':

            (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar10.load_data() 
            Y_TRAIN,Y_TEST = Y_TRAIN.reshape(-1), Y_TEST.reshape(-1)
            # standardize data
            X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)
            print('data shape',X_TRAIN.shape)
            #num_tasks = 5
            if num_heads > 1:
                out_dim = 2
            else:
                out_dim = 10

            #Y_TRAIN = one_hot_encoder(Y_TRAIN.reshape(-1),out_dim)
            #Y_TEST = one_hot_encoder(Y_TEST.reshape(-1),out_dim)
            cl_cmb = np.arange(10)
            cl_k = 0
            cl_n = 2
            # first task use all cifar10 data
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,train_size=args.train_size,test_size=args.test_size,\
                                                                        cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads) #X_TRAIN,Y_TRAIN,X_TEST,Y_TEST
            # load cifar 100
            #(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar100.load_data() 
            #X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)

            #clss = cl_cmb[cl_k:cl_k+cl_n]
        elif dataset == 'cifar100':
            (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar100.load_data() 
            Y_TRAIN,Y_TEST = Y_TRAIN.reshape(-1), Y_TEST.reshape(-1)
            # standardize data
            X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)
            print('data shape',X_TRAIN.shape)
           
            if num_heads > 1:
                out_dim = int(100/num_tasks)
            else:
                out_dim = 100

            cl_cmb = np.arange(100)
            cl_k = 0
            cl_n = int(100/num_tasks)
            
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,train_size=args.train_size,test_size=args.test_size,\
                                                                        cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads) #X_TRAIN,Y_TRAIN,X_TEST,Y_TEST



    else:
        data = input_data.read_data_sets(DATA_DIR) 
        X_TRAIN = np.concatenate([data.train.images,data.validation.images],axis=0)
        Y_TRAIN = np.concatenate([data.train.labels,data.validation.labels],axis=0)
        X_TEST = data.test.images
        Y_TEST = data.test.labels
        if conv:
            X_TRAIN,X_TEST = X_TRAIN.reshape(-1,28,28,1),X_TEST.reshape(-1,28,28,1)
        
        if num_heads > 1:
            out_dim = 2
        else:
            out_dim = 2 * num_tasks

        cl_cmb = np.arange(10)
        cl_k = 0
        cl_n = 2
        
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,train_size=args.train_size,\
                                                                        test_size=args.test_size,cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads)


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
            +'_e'+str(args.epoch)+'_lit'+str(args.local_iter)+'_fxb'+str(args.fixed_budget)+'_'+args.task_type+'_disc'+str(args.discriminant)+'_'+args.grad_type+'_'+head+'_'+args.model_type+'_'+args.vcl_type+'_sd'+str(seed)

file_path = os.path.join(result_path,file_name)
file_path = config_result_path(file_path)
with open(file_path+'configures.txt','w') as f:
    f.write(str(args))

# In[16]:
# Initialize model

if args.ginit > 0 and args.vcl_type in ['vanilla','kd']:
    # set init value of variance
    initialization={'w_s':-1.*args.ginit,'b_s':-1.*args.ginit,'cw_s':-1*args.ginit}
else:
    initialization=None


if args.net_type == 'resnet18':
    x_ph = tf.placeholder(dtype=tf.float32,shape=[None,*x_train_task.shape[1:]])
    in_dim = None
    dropout = None
    conv_net_shape,strides = None, None
    pooling = False

elif conv:
    x_ph = tf.placeholder(dtype=tf.float32,shape=[None,*x_train_task.shape[1:]])
    in_dim = None
    dropout = 0.5
    if dataset == 'cifar':
        conv_net_shape = [[3,3,3,32],[3,3,32,32],[3,3,32,64],[3,3,64,64]]
        strides = [[1,2,2,1],[1,2,2,1],[1,1,1,1],[1,1,1,1]]
        hidden = [512,256]
    else:
        conv_net_shape = [[4,4,1,32],[4,4,32,32]]
        strides = [[1,2,2,1],[1,1,1,1]]
    
    pooling = True

else:
    x_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_train_task.shape[1]])
    in_dim = x_train_task.shape[1]
    dropout = None
    conv_net_shape,strides = None, None
    pooling = False

if args.vcl_type in ['vanilla','kd']:
    y_ph = tf.placeholder(dtype=tf.int32,shape=[args.num_samples,None,out_dim]) 
else:
    y_ph = tf.placeholder(dtype=tf.float32,shape=[None,out_dim]) 

net_shape = [in_dim]+hidden+[out_dim]



if args.vcl_type=='vanilla':
    Model = VCL(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,args.coreset_usage,\
                args.vi_type,conv,dropout,initialization=initialization,ac_fn=ac_fn,n_samples=args.num_samples,\
                local_rpm=args.local_rpm,conv_net_shape=conv_net_shape,strides=strides,pooling=pooling,B=args.B,task_type=args.task_type)
    scale = 1.
elif args.vcl_type=='kd':
    args.model_type = 'continual' # can only be continual for vcl_kd
    Model = VCL_KD(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,args.coreset_usage,\
                conv=conv,dropout=dropout,vi_type=args.vi_type,initialization=initialization,ac_fn=ac_fn,n_samples=args.num_samples,\
                local_rpm=args.local_rpm,enable_kd_reg=args.kd_reg,enable_vcl_reg=args.kd_vcl_reg)
    scale = 1.
elif args.vcl_type=='stein':
    Model = Stein_CL(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,args.coreset_usage,\
                conv=conv,dropout=dropout,vi_type=args.vi_type,initialization=initialization,ac_fn=ac_fn,n_samples=args.num_samples,\
                local_rpm=args.local_rpm,enable_kd_reg=args.kd_reg,enable_vcl_reg=args.kd_vcl_reg,B=args.B,eta=args.eta,K=args.K,\
                discriminant=args.discriminant,lambda_dis=args.lambda_disc,WEM=args.WEM)

elif args.vcl_type=='drs':
    Model = DRS_CL(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,args.coreset_usage,\
            conv=conv,dropout=dropout,vi_type=args.vi_type,initialization=initialization,ac_fn=ac_fn,n_samples=args.num_samples,\
            local_rpm=args.local_rpm,enable_kd_reg=args.kd_reg,enable_vcl_reg=args.kd_vcl_reg,B=args.B,eta=args.eta,K=args.K,\
            discriminant=args.discriminant,lambda_dis=args.lambda_disc,WEM=args.WEM,coreset_mode=args.coreset_mode,\
            task_type=args.task_type,batch_iter=args.batch_iter,lambda_reg=args.lambda_reg,net_type=args.net_type,fixed_budget=args.fixed_budget)

else:
    raise TypeError('Wrong type of VCL')


Model.init_inference(learning_rate=args.learning_rate,decay=decay,train_size=TRAIN_SIZE,grad_type=args.grad_type,scale=scale)

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
avg_accs ,acc_record, probs_record, task_dsts,task_sims,t2m_sims = [], [], [], [],[],[]

pre_parms = {}
saver = tf.train.Saver()
tf.global_variables_initializer().run()
print('num tasks',num_tasks)
for t in range(num_tasks):
    # get test data
    test_sets.append((x_test_task,y_test_task))

    if Model.coreset_size > 0:
        if args.coreset_mode=='offline':
            x_train_task,y_train_task = Model.gen_task_coreset(t,x_train_task,y_train_task,args.task_type,sess,cl_n,clss)

    '''
    if args.task_dst:
        #g_vecs = Model.get_tasks_vec(sess,t,list(zip(Model.core_sets[0],Model.core_sets[1]))+[(x_train_task,y_train_task)])
        g_vecs = Model.get_tasks_vec(sess,t,zip(x_train_task,y_train_task),test_sample=True)
        print('g vecs len',len(g_vecs))
        #g_vecs = Model.get_tasks_vec(sess,t,test_sets[:-1]+[(x_train_task,y_train_task)])
        dsts_t,dsts_v = [], []
        for i in range(len(g_vecs)-1):
            for j in range(i+1,len(g_vecs)):
                dsts_t.append(calc_similarity(g_vecs[i],g_vecs[j],sess=sess))
                dsts_v.append(calc_similarity(x_train_task[i],x_train_task[j],sess=sess))
        #task_dsts.append(dsts_t)
        dsts_t = np.concatenate(dsts_t)
        dsts_v = np.concatenate(dsts_v)
        plt.plot(dsts_t,dsts_v)
        plt.savefig(file_path+'grads_corr'+str(t)+'.pdf')
        
        print('task sim',dsts_t)
        np.savez(file_path+'task_grads_t'+str(t),g_vecs)
        
        mean_gvec = mean_list(g_vecs)
        m_dst = []
        for gv in g_vecs:
            m_dst.append(calc_similarity(mean_gvec,gv,sess=sess))
        print('distance to mean gvec',m_dst)
        task_sims.append(np.sum(m_dst))
        t2m_sims.append(m_dst)
        print('task {} similarity: {}'.format(t+1,task_sims[-1]))
        
        if args.coreset_size > 0:
            #print('coreset shape',Model.core_sets[0][-1].shape,Model.core_sets[1][-1].shape)
            cg_vecs = Model.get_tasks_vec(sess,t,zip(Model.core_sets[0],Model.core_sets[1]))
            tg_vec = Model.get_tasks_vec(sess,t,[(x_train_task,y_train_task)])
            np.savez(file_path+'task_coresets_grads_t'+str(t),cg_vecs,tg_vec) 
            cdst = []
            for i in range(len(cg_vecs)):     
                cdst.append(calc_similarity(cg_vecs[i],tg_vec[0],sess=sess))
            task_coreset_dst.append(cdst)
            print('coresets sim',task_coreset_dst[-1])
        '''
    #if args.task_dst and args.coreset_type=='stein':
    #    px = sess.run(Model.core_sets[0][-1])


    if args.tensorboard:
        Model.train_task(sess,t,x_train_task,y_train_task,args.epoch,args.print_epoch,args.local_iter,\
                        tfb_merged=merged,tfb_writer=train_writer,tfb_avg_losses=[avg_err,avg_kl,avg_ll])
    else:
        Model.train_task(sess,t,x_train_task,y_train_task,args.epoch,args.print_epoch,args.local_iter)

    if args.save_parm:
        Model.save_parm(t,file_path,sess)
    
    if args.save_FIM:
        FIM = get_model_FIM(Model,sess=sess)
        np.save(file_path+'model_FIM_task'+str(t)+'.npy',FIM)

    if args.task_dst:
        #for i in range(t+1):
        '''
        cx = np.vstack(Model.core_sets[0]) if Model.coreset_type!='stein' else np.vstack(sess.run(Model.core_sets[0]))
        cy = np.vstack(Model.core_sets[1])
        cx,cy = shuffle_data(cx,cy)
        cx,cy = cx[:min(len(cx),200)],cy[:min(len(cy),200)]
        '''
        x = np.vstack([tx[0] for tx in test_sets])
        y = np.vstack([tx[1] for tx in test_sets])
        x,y = shuffle_data(x,y)
        x,y = x[:min(len(x),100)],y[:min(len(y),100)]
        m_vec,m_label = [],[]
        for j in range(2*t):
            ids = Model.y_core_sets[:,j]==1
            m_vec.append(Model.x_core_sets[ids].mean(axis=0))
            m_label.append(Model.y_core_sets[ids][0])
        for j in range(2*t,2*(t+1)):
            ids = y_train_task[:,j]==1
            m_vec.append(x_train_task[ids].mean(axis=0))
            m_label.append(y_train_task[ids][0])
        m_y = np.vstack(m_label)


        #g_vecs = Model.get_tasks_vec(sess,t,list(zip(Model.core_sets[0],Model.core_sets[1]))+[(x_train_task,y_train_task)])
        #g_vecs = Model.get_tasks_vec(sess,t,zip(cx,cy),test_sample=True)
        #print('g vecs len',len(g_vecs),g_vecs[0].shape)
        #g_vecs = Model.get_tasks_vec(sess,t,test_sets[:-1]+[(x_train_task,y_train_task)])
        #dsts_t,dsts_v = [], []
        #dsts_t = calc_similarity(np.vstack(g_vecs),sess=sess)
        #dsts_v = calc_similarity(x,sess=sess) 
        #print(dsts_t.shape,dsts_v.shape)

        #cmplx = (np.sum(dsts_t)-dsts_t.shape[0])/2.
        #print('complex',cmplx)
        #np.savetxt(file_path+'dsts_cx_t'+str(t)+'.csv',dsts_t,delimiter=',')
        #np.savetxt(file_path+'dsts_v'+str(t)+'.csv',dsts_v,delimiter=',')
        #np.savetxt(file_path+'dsts_t'+str(t)+'_i'+str(i)+'.csv',dsts_t,delimiter=',')
        #np.savetxt(file_path+'dsts_v'+str(t)+'_i'+str(i)+'.csv',dsts_v,delimiter=',')
        #task_dsts.append(dsts_t)
        #dsts_t = np.concatenate(dsts_t)
        #dsts_v = np.concatenate(dsts_v)
        #dsts_t = dsts_t.reshape(-1)
        #plt.plot(dsts_t,dsts_v,'o')
        #plt.savefig(file_path+'grads_corr_t'+str(t)+'.pdf')
        #plt.savefig(file_path+'grads_corr_t'+str(t)+'_i'+str(i)+'.pdf')
        #plt.close()
        '''
        yids = np.matmul(cy,cy.transpose()).reshape(-1)
        sn.distplot(dsts_t[yids==0])
        sn.distplot(dsts_t[yids==1])
        plt.legend(['diff class','same class'])
        #plt.plot(yids.reshape(-1),dsts_t.reshape(-1),'o')
        #plt.plot(yids.reshape(-1),dsts_v.reshape(-1),'*')
        plt.savefig(file_path+'grads_class_corr_cx_t'+str(t)+'.pdf')
        #plt.savefig(file_path+'grads_class_corr_t'+str(t)+'_i'+str(i)+'.pdf')
        plt.close()
        '''
        g_vecs,nlls = Model.get_tasks_vec(sess,t,zip(x,y),test_sample=True)
        m_g_vecs,m_nlls = Model.get_tasks_vec(sess,t,zip(m_vec,m_y),test_sample=True)
        #print('m g vec',np.vstack(m_g_vecs).shape)
        m_dsts_t = calc_similarity(np.vstack(g_vecs),np.vstack(m_g_vecs),sess=sess)
        m_dsts_t = np.squeeze(m_dsts_t,axis=1)
        
        #dsts_t = calc_similarity(np.vstack(g_vecs),sess=sess)
        yids = 1.- np.matmul(y,m_y.transpose())
        #print(m_dsts_t.shape,yids.shape)
        #rank_t = -(np.sum(dsts_t*yids,axis=0))*0.5/np.sum(yids,axis=0) \
        #            + ((np.sum(dsts_t*(1.-yids),axis=0)-1.)*0.5)/(np.sum(1.-yids,axis=0)-1.)
        rank_t = -(np.sum(m_dsts_t*yids,axis=1)*0.5)/np.sum(yids,axis=1) \
                    + (np.sum(m_dsts_t*(1.-yids),axis=1)*0.5)/np.sum(1.-yids,axis=1)
        
        print('rank_t \n {}'.format(rank_t.shape))
        #rank_l = np.argsort(-nlls)
        #print('rank_l \n {}'.format((rank_l)))
        sn.regplot(rank_t,nlls)
        #sn.scatterplot(rank_t[yids==1],nlls[yids==1])
        plt.savefig(file_path+'rank_corr'+str(t)+'.pdf')
        plt.close()
        #np.savetxt(file_path+'dsts_tx_t'+str(t)+'.csv',dsts_t,delimiter=',')
        #print('h shape',Model.H[0].shape)
        
        hx = sess.run(Model.H,feed_dict={Model.x_ph:x,Model.y_ph:y})
        hx = np.hstack(hx)
        yids = 1.- np.matmul(y,y.transpose())
        print('hx',hx.shape)
        
        dsts_v = calc_similarity(hx,sess=sess)
        np.savetxt(file_path+'dsts_hx_v'+str(t)+'.csv',dsts_v,delimiter=',')
        rank_v = (np.sum(dsts_v*yids,axis=1)*0.5) \
                    - (np.sum(dsts_v*(1.-yids),axis=1)*0.5)
        sn.regplot(rank_v,nlls)
        #sn.scatterplot(rank_t[yids==1],nlls[yids==1])
        plt.savefig(file_path+'hx_rank_corr'+str(t)+'.pdf')
        plt.close()

        '''
        for i in range(2*(t+1)):
            sn.distplot(m_dsts_t[:,i][y[:,i]==0])
            plt.savefig(file_path+'grads_class_corr_tx_t'+str(t)+'_c'+str(i)+'.pdf')
            plt.close()

            
            for j in range(i+1,2*(t+1)):
                ids_ij = (y[:,i]==1) | (y[:,j]==1)
                dsts_t_ij = dsts_t[ids_ij][:,ids_ij].reshape(-1)
                yids = np.matmul(y[ids_ij],y[ids_ij].transpose()).reshape(-1)
                sn.distplot(dsts_t_ij[yids==0])
                sn.distplot(dsts_t_ij[yids==1])
                plt.legend(['diff class','same class'])
                plt.savefig(file_path+'grads_class_corr_tx_t'+str(t)+'_c'+str(i)+str(j)+'.pdf')
                plt.close()

                
                dsts_v_ij = dsts_v[ids_ij][:,ids_ij].reshape(-1)
                sn.scatterplot(x=dsts_t_ij[yids==0],y=dsts_v_ij[yids==0])
                sn.scatterplot(x=dsts_t_ij[yids==1],y=dsts_v_ij[yids==1])
                plt.legend(['diff class','same class'])
                plt.savefig(file_path+'grads_class_euc_corr_hx_t'+str(t)+'_c'+str(i)+str(j)+'.pdf') 
                plt.close()
        
        if args.coreset_type=='stein':
            
            g_vecs = Model.get_tasks_vec(sess,t,zip(px,y),test_sample=True)
            dsts_t = calc_similarity(np.vstack(g_vecs),sess=sess)
            cmplx = (np.sum(dsts_t)-dsts_t.shape[0])/2.
            print('prev complex',cmplx)
        '''
    accs, probs, cfm = Model.test_all_tasks(t,test_sets,sess,args.epoch,saver=saver,file_path=file_path,confusion=True)
    print('confusion matrix \n',cfm.astype(int))
    np.savetxt(file_path+'cfmtx'+str(t)+'.csv',cfm.astype(int),delimiter=',')

    acc_record.append(accs)
    avg_accs.append(np.mean(accs))
    if args.irt:
        #print(probs[0].shape,test_sets[0][1].shape)
        probs = [prb[np.arange(len(prb)),np.argmax(ts[1],axis=1)] for prb,ts in zip(probs,test_sets)]
        if num_heads  > 1:
            labels = [np.argmax(ts[1],axis=1)+t*out_dim for ts in test_sets]
        else:
            labels = [np.argmax(ts[1],axis=1) for ts in test_sets]

        probs = np.concatenate(probs)
        #print(type(labels[0][0]))
        labels = np.concatenate(labels).astype(np.uint8)
        save_samples(file_path,[probs,labels],['test_resps_t'+str(t), 'test_labels_t'+str(t)])

    if t < num_tasks-1:

        '''
        if t==4 and dataset=='notmnist' and  args.task_type=='split':
            DATA_DIR = '../datasets/MNIST_data/'
            X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = load_task_data(args.task_type,DATA_DIR)
            cl_k = 0
        '''
        if args.model_type == 'continual':
            #print('coresets',Model.core_sets)
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = Model.update_task_data_and_inference(sess,t,args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,\
                                                                                                        original_batch_size=batch_size,cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,clss=clss,\
                                                                                                        x_train_task=x_train_task,y_train_task=y_train_task,rpath=file_path,\
                                                                                                        train_size=args.train_size,test_size=args.test_size)
        elif args.model_type == 'single':
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = Model.update_task_data(sess,t,args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim,original_batch_size=batch_size,cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb)
            if Model.coreset_size>0 and Model.coreset_usage != 'final':
                Model.x_core_sets,Model.y_core_sets,c_cfg = aggregate_coreset(Model.core_sets,Model.core_y,Model.coreset_type,Model.num_heads,t,Model.n_samples,sess)
            tf.global_variables_initializer().run()
            if args.coreset_size > 0:
                Model.inference.reinitialize(task_id=t+1,coresets={'task':c_cfg})
            else:
                Model.inference.reinitialize(task_id=t+1)


with open(file_path+'accuracy_record.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(acc_record)):
        writer.writerow(acc_record[t])

with open(file_path+'avg_accuracy.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow(avg_accs)
    writer.writerow(task_sims)


with open(file_path+'task_distances.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(task_dsts)):
        writer.writerow(task_dsts[t])

with open(file_path+'t2m_distances.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(t2m_sims)):
        writer.writerow(t2m_sims[t])
'''
if args.irt:
    for t,ts in enumerate(test_sets):
        save_samples(file_path,[*ts],file_name=['test_samples_t'+str(t),'test_labels_t'+str(t)])
'''

if not args.save_parm and args.coreset_usage=='final':
    os.system('rm '+file_path+"ssmodel.ckpt*")
