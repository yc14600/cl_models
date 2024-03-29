# coding: utf-8

# In[1]:


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
import time

import tensorflow as tf
import edward as ed
import argparse


from utils.model_util import *
from utils.train_util import *
from utils.test_util import *


from hsvi import hsvi
from hsvi.methods.svgd import SVGD
from utils.data_util import save_samples
from utils.train_util import shuffle_data
from utils.model_util import mean_list
from models.coreset import *
from models.vcl_model import VCL
from models.vcl_kd import VCL_KD
from models.stein_cl import Stein_CL
from cl_models.discriminative.drs.drs_cl import DRS_CL
from cl_models.discriminative.drs.agem import AGEM
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
parser.add_argument('-mh','--multihead', default=False, type=str2bool, help='multihead model')
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
parser.add_argument('-lam_reg','--lambda_reg',default=0.,type=float,help='lambda regularization')
parser.add_argument('-er','--ER',default=False,type=str2bool,help='test experience replay')
parser.add_argument('-bit','--batch_iter',default=1,type=int,help='iterations on one batch')
parser.add_argument('-ntp','--net_type',default='dense',type=str,help='network type, can be dense, conv, resnet18')
parser.add_argument('-fxbt','--fixed_budget',default=True,type=str2bool,help='if budget of episodic memory is fixed or not')
parser.add_argument('-mbs','--mem_bsize',default=256,type=int,help='memory batch size used in AGEM')
parser.add_argument('-ptp','--pretrained_path',default='',type=str,help='path to pretrained resnet18 on cifar10')
parser.add_argument('-irt_bi','--irt_binary_prob',default=True,type=str2bool,help='save irt response as binary')




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
    DATA_DIR = os.path.join(args.data_path,'quickdraw/full/numpy_bitmap/')
print(dataset)


hidden = args.hidden #[256,256]
scale = 1.   #TRAIN_SIZE/batch_size#weights of likelihood
shrink = 1.  #shrink train_size, smaller gives larger weights of KL-term

if len(args.decay) == 2:  
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


print('heads',num_heads)

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
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,train_size=args.train_size,test_size=args.test_size,out_dim=out_dim) 


elif 'split' in args.task_type:
    if 'cifar' in dataset:
        
        if args.net_type == 'resnet18':
            conv = False
            hidden = []
            #args.vcl_type = 'drs'
            assert(args.vcl_type=='drs' or args.vcl_type=='agem')
        else:
            conv =True
            hidden = [512,512]
        
        if dataset  == 'cifar10':

            (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar10.load_data() 
            Y_TRAIN,Y_TEST = Y_TRAIN.reshape(-1), Y_TEST.reshape(-1)
            # standardize data
            X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)
            print('data shape',X_TRAIN.shape)

            if num_heads > 1:
                out_dim = 2
            else:
                out_dim = 10

            #Y_TRAIN = one_hot_encoder(Y_TRAIN.reshape(-1),out_dim)
            #Y_TEST = one_hot_encoder(Y_TEST.reshape(-1),out_dim)
            cl_cmb = np.arange(10)
            cl_k = 0    # the start class index
            cl_n = 2    # the number of classes of each task

            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss = gen_next_task_data(args.task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,train_size=args.train_size,test_size=args.test_size,\
                                                                        cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads) #X_TRAIN,Y_TRAIN,X_TEST,Y_TEST

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
            +'_e'+str(args.epoch)+'_lit'+str(args.local_iter)+'_fxb'+str(args.fixed_budget)+'_'+args.task_type+'_disc'+str(args.discriminant)+'_ER'+str(args.ER)+'_'+args.grad_type+'_'+head+'_'+args.model_type+'_'+args.vcl_type+'_sd'+str(seed)

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
    if 'cifar' in dataset:
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
            local_rpm=args.local_rpm,enable_kd_reg=args.kd_reg,enable_vcl_reg=args.kd_vcl_reg,B=args.B,\
            discriminant=args.discriminant,lambda_dis=args.lambda_disc,ER=args.ER,coreset_mode=args.coreset_mode,\
            task_type=args.task_type,batch_iter=args.batch_iter,lambda_reg=args.lambda_reg,net_type=args.net_type,fixed_budget=args.fixed_budget)

elif args.vcl_type=='agem':
    Model = AGEM(net_shape,x_ph,y_ph,num_heads,batch_size,args.coreset_size,args.coreset_type,args.coreset_usage,\
            conv=conv,dropout=dropout,vi_type=args.vi_type,initialization=initialization,ac_fn=ac_fn,n_samples=args.num_samples,\
            local_rpm=args.local_rpm,enable_kd_reg=args.kd_reg,enable_vcl_reg=args.kd_vcl_reg,B=args.B,\
            coreset_mode=args.coreset_mode,task_type=args.task_type,batch_iter=args.batch_iter,lambda_reg=args.lambda_reg,\
            net_type=args.net_type,fixed_budget=args.fixed_budget,mem_batch_size=args.mem_bsize)

else:
    raise TypeError('Wrong type of model')


Model.init_inference(learning_rate=args.learning_rate,decay=decay,train_size=TRAIN_SIZE,grad_type=args.grad_type,scale=scale)


sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) 

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
tf.global_variables_initializer().run(session=sess)
print('num tasks',args.num_tasks)

time_count = 0.
for t in range(args.num_tasks):
    # get test data
    test_sets.append((x_test_task,y_test_task))

    if Model.coreset_size > 0:
        if args.coreset_mode=='offline':
            print('generate offline coresets')
            if args.multihead:
                x_train_task,y_train_task = Model.gen_task_coreset(t,x_train_task,y_train_task,args.task_type,sess,cl_n)
            else:
                x_train_task,y_train_task = Model.gen_task_coreset(t,x_train_task,y_train_task,args.task_type,sess,cl_n,clss)

    start = time.time()

    if args.tensorboard:
        Model.train_task(sess,t,x_train_task,y_train_task,args.epoch,args.print_epoch,args.local_iter,\
                        tfb_merged=merged,tfb_writer=train_writer,tfb_avg_losses=[avg_err,avg_kl,avg_ll])
    else:
        Model.train_task(sess,t,x_train_task,y_train_task,args.epoch,args.print_epoch,args.local_iter)
    end = time.time()
    time_count += end-start
    print('training time',time_count)
    if args.save_parm:
        Model.save_parm(t,file_path,sess)
    
    if args.save_FIM:
        FIM = get_model_FIM(Model,sess=sess)
        np.save(file_path+'model_FIM_task'+str(t)+'.npy',FIM)

    if args.task_dst:
        
        print('test len',len(test_sets))
        x = np.vstack([tx[0] for tx in test_sets])
        y = np.vstack([tx[1] for tx in test_sets])
        x,y = shuffle_data(x,y)
        x,y = x[:min(len(x),100)],y[:min(len(y),100)]
        
        g_vecs,_ = Model.get_tasks_vec(sess,t,zip(x,y),test_sample=True)
        print('g vecs len',len(g_vecs),g_vecs[0].shape)

        dsts_t = calc_similarity(np.array(g_vecs),sess=sess)
        dsts_v = calc_similarity(x,sess=sess) 
        
        hx = sess.run(Model.H,feed_dict={Model.x_ph:x,Model.y_ph:y})
        hx = np.hstack(hx)
        yids = 1- np.matmul(y,y.transpose())
        mask = np.eye(y.shape[0])
        yids_s = mask - np.matmul(y,y.transpose())
        yids = yids.reshape(-1).astype(bool)
        yids_s = yids_s.reshape(-1).astype(bool)
        print('hx',hx.shape,'y_s',np.sum(yids_s))
        
        dsts_h = calc_similarity(hx,sess=sess)
        sn.scatterplot(dsts_t.reshape(-1)[yids],dsts_h.reshape(-1)[yids])
        print('rep diff corrcoef',np.corrcoef(dsts_t.reshape(-1)[yids],dsts_h.reshape(-1)[yids]))
        sn.scatterplot(dsts_t.reshape(-1)[yids_s],dsts_h.reshape(-1)[yids_s])
        print('rep same corrcoef',np.corrcoef(dsts_t.reshape(-1)[yids_s],dsts_h.reshape(-1)[yids_s]))
        #sn.scatterplot(dsts_t[1-yids].reshape(-1),dsts_v[1-yids].reshape(-1))
        plt.legend(['diff class','same class'],fontsize=13)
        plt.xlabel('Similarity of gradients',fontsize=13)
        plt.ylabel('Similarity of representations',fontsize=13)
        plt.savefig(file_path+'grads_class_euc_corr_hx_t'+str(t)+'.pdf')
        plt.close()
        sn.scatterplot(dsts_t.reshape(-1)[yids],dsts_v.reshape(-1)[yids])
        print('feature diff corrcoef',np.corrcoef(dsts_t.reshape(-1)[yids],dsts_v.reshape(-1)[yids]))
        sn.scatterplot(dsts_t.reshape(-1)[yids_s],dsts_v.reshape(-1)[yids_s])
        print('feature same corrcoef',np.corrcoef(dsts_t.reshape(-1)[yids_s],dsts_v.reshape(-1)[yids_s]))
        #sn.scatterplot(dsts_t[1-yids].reshape(-1),dsts_v[1-yids].reshape(-1))
        plt.legend(['diff class','same class'],fontsize=13)
        plt.xlabel('Similarity of gradients',fontsize=13)
        plt.ylabel('Similarity of features',fontsize=13)
        plt.savefig(file_path+'grads_class_euc_corr_fx_t'+str(t)+'.pdf')
        plt.close()
        np.savetxt(file_path+'dsts_hx_diff_'+str(t)+'.csv',dsts_h.reshape(-1)[yids],delimiter=',')
        np.savetxt(file_path+'dsts_hx_same_'+str(t)+'.csv',dsts_h.reshape(-1)[yids_s],delimiter=',')
        np.savetxt(file_path+'dsts_gx_diff_'+str(t)+'.csv',dsts_t.reshape(-1)[yids],delimiter=',')
        np.savetxt(file_path+'dsts_gx_same_'+str(t)+'.csv',dsts_t.reshape(-1)[yids_s],delimiter=',')

    accs, probs, cfm = Model.test_all_tasks(t,test_sets,sess,args.epoch,saver=saver,file_path=file_path,confusion=True)


    acc_record.append(accs)
    avg_accs.append(np.mean(accs))
    if args.irt:
        if args.irt_binary_prob:
            probs = [np.argmax(prb,axis=1)==np.argmax(ts[1],axis=1) for prb,ts in zip(probs,test_sets)]
        else:
            probs = [prb[np.arange(len(prb)),np.argmax(ts[1],axis=1)] for prb,ts in zip(probs,test_sets)]
        if num_heads  > 1:
            labels = [np.argmax(ts[1],axis=1)+t*out_dim for ts in test_sets]
        else:
            labels = [np.argmax(ts[1],axis=1) for ts in test_sets]

        probs = np.concatenate(probs)
        if args.irt_binary_prob:
            probs = probs.astype(np.uint8)
        labels = np.concatenate(labels).astype(np.uint8)
        save_samples(file_path,[probs,labels],['test_resps_t'+str(t), 'test_labels_t'+str(t)])
    if t < num_tasks-1:
        if args.model_type == 'continual':
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

with open(file_path+'eplapsed_time.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow([time_count])
    


with open(file_path+'task_distances.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(task_dsts)):
        writer.writerow(task_dsts[t])

with open(file_path+'t2m_distances.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(t2m_sims)):
        writer.writerow(t2m_sims[t])


if not args.save_parm and args.coreset_usage=='final':
    os.system('rm '+file_path+"ssmodel.ckpt*")
