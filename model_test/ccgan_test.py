#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import edward as ed
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import argparse


# In[2]:


import os
path = os.getcwd()
import sys
sys.path.append(path+'/../')


# In[3]:


from base_models.gans import fGAN,GAN,CGAN,CWGAN,CfGAN
from cl_models import Continual_CGAN,Continual_Ratio_fGAN,Continual_VAE
from base_models.vae import VAE
from cdre import Cond_Continual_LogLinear_Estimator
from utils.model_util import define_dense_layer
from utils.train_util import *
from utils.data_util import extract_data,extract_labels
from utils.test_util import *


# In[4]:


from tensorflow.examples.tutorials.mnist import input_data


# In[ ]:
def set_tblog(ccgan):
    g_grads = tf.gradients(ccgan.model.g_loss, ccgan.model.g_var_list)
    for g in g_grads:
        if len(g.shape)==1:
            g_grad = g[0]
            break
            
    #g_grad = g_grads[0][0]
    d_grads = tf.gradients(ccgan.model.d_loss, ccgan.model.d_var_list)
    for g in d_grads:
        if len(g.shape)==1:
            d_grad = g[0]
            break
    #d_grad = d_grads[0][0]
    print('grad shape',g_grad,d_grad)
    tf.summary.scalar('g_weight_exsample_grad', g_grad)
    tf.summary.scalar('d_weight_exsample_grad', d_grad)
    merged = tf.summary.merge_all()
    return merged
'''
def plot(samples,MNIST=True,shape=None):
    if shape is None:
        rows = 4
        cols = 4
    else:
        rows = shape[0]
        cols = shape[1]
        
    fig = plt.figure(figsize=(rows, cols))
    gs = gridspec.GridSpec(rows, cols)    
    #gs.update(wspace=0.01, hspace=0.01)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.set_aspect('equal')
        if MNIST:
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        else:
            plt.plot(sample)

    return fig
'''
class CDRE_CFG:
    def __init__(self,args):
        self.sample_size = args.cdre_sample_size
        self.test_sample_size = args.cdre_test_sample_size
        self.batch_size = args.cdre_batch_size
        self.epoch = args.cdre_epoch
        self.min_epoch = args.cdre_min_epoch
        self.print_e = args.cdre_print_e
        self.learning_rate = args.cdre_learning_rate
        self.constr = args.cdre_constr
        self.lambda_constr = args.cdre_lambda_constr
        self.divergence = args.cdre_divergence
        self.hidden_layers = args.cdre_hidden_layers
        self.dim_reduction = args.cdre_dim_reduction
        self.z_dim = args.cdre_z_dim
        self.vae_loadpath = args.cdre_vae_loadpath
        self.vae_savepath = args.cdre_vae_savepath
        self.early_stop = True
        self.filter_min = args.cdre_filter_min



# In[ ]:


# In[ ]:
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../../data/mnist/', type=str, help='specify data directory')
parser.add_argument('--T', default=10, type=int, help='number of tasks')
parser.add_argument('--z_dim', default=64, type=int, help='dimension of epsilon')
parser.add_argument('--conv', default=True, type=str2bool, help='if True, use conv nets')
parser.add_argument('--batch_norm', default=True, type=str2bool, help='if True, use batch normalization')
parser.add_argument('--pooling', default=False, type=str2bool, help='if True, use max pooling after conv layers')
parser.add_argument('--model_type', default='fgan', type=str, help='type of conditional GAN,can be fgan, rfgan, wgan,wgan-gp')
parser.add_argument('--divergence', default='Jensen_Shannon', type=str, help='divergence type of fGAN')
#parser.add_argument('--train_size', default=50000, type=int, help='number of training samples')
parser.add_argument('--test_size', default=10000, type=int, help='number of test samples')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--epoch', default=15, type=int, help='number of epochs')
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate')
parser.add_argument('--result_path', default='../vis_results/ccfgan/', type=str, help='directory of saving results')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--reg', default=None, type=str, help='regularizer, l1 or l2')
parser.add_argument('--train_type',default='continual',type=str,help='training data is from last model or original data')
parser.add_argument('--clip', default=None, type=str2flist, help='add clip for gradients')
parser.add_argument('--save_model', default=False, type=str2bool, help='if True, save the model after first task')
parser.add_argument('--save_samples', default=True, type=str2bool, help='if True, save the model samples after each task')
parser.add_argument('--warm_start', default='', type=str, help='specify the file path to load a trained model for feature extraction')
parser.add_argument('--tblog', default=False, type=str2bool, help='if enable tensorboard log')
parser.add_argument('--lamb_constr', default=0., type=float, help='lambda of ratio constraint in fGAN')
parser.add_argument('--disc_reg', default=False, type=str2bool, help='if enable discriminator regularization')
parser.add_argument('--gamma0', default=1., type=float, help='init gamma of disc reg')
parser.add_argument('--alpha', default=0.01, type=float, help='if enable annealing of discriminator regularization,the value of decay')
parser.add_argument('--memrplay', default=0., type=float, help='enable memory replay if larger than 0')
parser.add_argument('--multihead', default=False, type=str2bool, help='use multi-dims output for discriminator')
parser.add_argument('--cdre', default=False, type=str2bool, help='if use CDRE during continual training')
parser.add_argument('--cdre_sample_size', default=20000, type=int, help='number of samples of each task')
parser.add_argument('--cdre_test_sample_size', default=5000, type=int, help='number of test samples of each task')
parser.add_argument('--cdre_batch_size', default=2000, type=int, help='batch size')
parser.add_argument('--cdre_epoch', default=1000, type=int, help='number of epochs')
parser.add_argument('--cdre_min_epoch', default=100, type=int, help='minimum number of epochs before early stop')
parser.add_argument('--cdre_print_e', default=100, type=int, help='number of epochs for printing message')
parser.add_argument('--cdre_learning_rate', default=0.00001, type=float, help='learning rate')
parser.add_argument('--cdre_constr', default=True, type=str2bool, help='if add continual constraints to the objective')
parser.add_argument('--cdre_lambda_constr', default=1., type=float, help='Lagrange multiplier of continual constraint')
parser.add_argument('--cdre_divergence', default='KL', type=str, help='the divergence used to optimize the ratio model, one of [KL, Chi]')
parser.add_argument('--cdre_hidden_layers', default=[256,256], type=str2ilist, help='size of hidden layers, no space between characters')
parser.add_argument('--cdre_dim_reduction', default=None, type=str, help='reduce dimension before ratio estimation,could be vae or cvae')
parser.add_argument('--cdre_z_dim',default=64,type=int,help='dimension of latent feature space')
parser.add_argument('--cdre_vae_loadpath', default='', type=str, help='specify the file path to load a pre-trained VAE for feature extraction')
parser.add_argument('--cdre_vae_savepath', default='', type=str, help='specify the file path to save a pre-trained VAE for feature extraction')
parser.add_argument('--cdre_filter_min', default=0., type=float, help='minimum ratio after filtering by cdre')


args = parser.parse_args()

tf.set_random_seed(args.seed)
np.random.seed(args.seed)



if args.model_type in ['rfgan','fgan']:
    file_name = args.train_type + '_' + args.model_type + '_' + args.divergence 
else:
    file_name =  args.train_type + '_' + args.model_type 
result_path = args.result_path + file_name + '/'

c_dim = args.T

# In[ ]:


data = input_data.read_data_sets(args.data_dir,one_hot=False) 

X_TRAIN = np.vstack((data.train.images,data.validation.images))
Y_TRAIN = np.concatenate((data.train.labels,data.validation.labels))
X_TEST,Y_TEST = None, None
if args.cdre:
    ### data for cdre validation ###
    X_TEST = data.test.images
    Y_TEST = data.test.labels



# In[ ]:

x_dim = X_TRAIN.shape[-1]
if args.multihead:
    out_dim = c_dim
else:
    out_dim = 1

if args.conv:
    X_TRAIN = X_TRAIN.reshape(-1,28,28,1)
    if X_TEST is not None:
        X_TEST = X_TEST.reshape(-1,28,28,1)
    g_net_shape = [[args.z_dim+c_dim,1024,128*7*7],[[7,7,128],[4,4,64,128],[4,4,1,64]]]
    d_net_shape = [[[4,4,c_dim+1,64],[4,4,64,128]],[128*7*7,1024,out_dim]]
else:
    g_net_shape = [args.z_dim+c_dim,1024,512,512,x_dim] ##
    d_net_shape = [x_dim+c_dim,512,256,out_dim] ##



if args.conv:
    x_ph = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,28,28,c_dim+1])
else:
    x_ph = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,x_dim+c_dim])

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

if args.cdre:
    print('CONFIG CDRE')
    cdre_cfg = CDRE_CFG(args)
    in_dim = x_dim if not cdre_cfg.dim_reduction else cdre_cfg.z_dim
    if not isinstance(in_dim,int):
        in_dim = np.prod(in_dim).astype(int)
    print('in dim for cdre',in_dim)
    net_shape = [in_dim+args.T] + cdre_cfg.hidden_layers + [args.T]

    nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,in_dim+args.T],name='nu_ph')
    de_ph = tf.placeholder(dtype=tf.float32,shape=[None,in_dim+args.T],name='de_ph')
    c_ph = tf.placeholder(dtype=tf.float32,shape=[None,args.T],name='c_ph')

    if cdre_cfg.dim_reduction == 'cvae':
        prev_nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,in_dim+args.T],name='prev_nu_ph')
        prev_de_ph = tf.placeholder(dtype=tf.float32,shape=[None,in_dim+args.T],name='prev_de_ph')        

    else:
        prev_nu_ph, prev_de_ph = None, None

    if cdre_cfg.dim_reduction == 'cvae':
        args.cdre_dim_reduction = Continual_VAE(x_dim,cdre_cfg.z_dim,batch_size=200,e_net_shape=[512,512],d_net_shape=[256,256],sess=sess,epochs=100,reg='l2',noise_std=0.05)
    elif cdre_cfg.dim_reduction == 'vae':
        args.cdre_dim_reduction = VAE(x_dim,cdre_cfg.z_dim,batch_size=200,e_net_shape=[512,512],d_net_shape=[256,256],sess=sess,epochs=100, reg='l2')
                    
    if cdre_cfg.dim_reduction and cdre_cfg.vae_loadpath:
        saver = tf.train.Saver()
        saver.restore(sess,cdre_cfg.vae_loadpath)

    args.cdre = Cond_Continual_LogLinear_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,prev_nu_ph=prev_nu_ph,\
                                               prev_de_ph=prev_de_ph,c_ph=c_ph,conv=False,reg=None,cl_constr=cdre_cfg.constr,\
                                                div_type=cdre_cfg.divergence,lambda_constr=cdre_cfg.lambda_constr,c_dim=args.T)

    args.cdre.estimator.config_train(learning_rate=cdre_cfg.learning_rate)

else:
    cdre_cfg = None    

if args.model_type == 'rfgan':
    ccgan = Continual_Ratio_fGAN(x_ph,g_net_shape,d_net_shape,args.batch_size,sess=sess,ac_fn=tf.nn.leaky_relu,\
                                batch_norm=args.batch_norm,learning_rate=args.learning_rate,conv=args.conv,\
                                divergence=args.divergence,c_dim=c_dim,reg=args.reg,clip=args.clip,lamb_constr=args.lamb_constr,\
                                g_penalty=args.disc_reg,gamma0=args.gamma0,alpha=args.alpha,mem_replay=args.memrplay,pooling=args.pooling,\
                                cdre=args.cdre,cdre_feature_extractor=args.cdre_dim_reduction,cdre_config=cdre_cfg)
else:
    ccgan = Continual_CGAN(x_ph,g_net_shape,d_net_shape,args.batch_size,sess=sess,ac_fn=tf.nn.leaky_relu,\
                        batch_norm=args.batch_norm,learning_rate=args.learning_rate,conv=args.conv,\
                        model_type=args.model_type,divergence=args.divergence,c_dim=c_dim,reg=args.reg,\
                        clip=args.clip,lamb_constr=args.lamb_constr,g_penalty=args.disc_reg,gamma0=args.gamma0,\
                        alpha=args.alpha,mem_replay=args.memrplay,pooling=args.pooling,\
                        cdre=args.cdre,cdre_feature_extractor=args.cdre_dim_reduction,cdre_config=cdre_cfg)

if args.tblog:
    merged = set_tblog(ccgan)
    train_writer = tf.summary.FileWriter('./tfb_summary/'+file_name,
                                        ccgan.model.sess.graph)
else:
    merged, train_writer = None, None


warm_start = False
for t in range(args.T):
    spath = result_path+'task'+str(t)+'/'
    config_result_path(spath)
    if t > 0:
        warm_start = True   
    if args.train_type == 'continual':
        cls = [t]
        sample_size = cdre_cfg.sample_size if args.cdre else args.test_size
        test_sample_size = cdre_cfg.test_sample_size if args.cdre else 0
        x_train_task,y_train_task,x_test_task,y_test_task = gen_class_split_data(args.seed,sample_size,test_sample_size,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,cls,one_hot=False,C=1)
        y_train_task = one_hot_encoder(y_train_task,c_dim)
        if y_test_task is not None:
            y_test_task = one_hot_encoder(y_test_task,c_dim)
        old_c = np.arange(t)
        X, Y = ccgan.merge_train_data(x_train_task,y_train_task,old_c,c_dim,save_samples=False,sample_size=sample_size)
        print('X,Y,x_train_task,y_train_task',X.shape,Y.shape,x_train_task.shape,y_train_task.shape)
        if args.cdre:
            X_test,Y_test = ccgan.merge_train_data(x_test_task,y_test_task,old_c,c_dim,save_samples=False,sample_size=cdre_cfg.test_sample_size)
       
    elif args.train_type == 'truedata':
        cls = np.arange(t+1)
        X,Y,_,_ = gen_class_split_data(args.seed,None,None,X_TRAIN,Y_TRAIN,None,None,cls,one_hot=False,C=t+1)
        Y = one_hot_encoder(Y,c_dim)

    X_train, Y_train = shuffle_data(X,Y) 

    saver = tf.train.Saver()
    if t==0 and len(args.warm_start)>0:        
        saver.restore(ccgan.model.sess,args.warm_start)
    else:
        #tf.variables_initializer(self.model.d_var_list).run() 
        reinitialize_scope(['generator','discriminator'],sess)  
        ccgan.model.training(X_train,Y_train,args.batch_size,disc_iter=1,epoch=args.epoch,vis=True,result_path=spath,warm_start=warm_start,
                            merged=merged,train_writer=train_writer)

    if args.save_model and t==0:         
         saver.save(ccgan.model.sess,spath+'model')

    #if args.model_type == 'rfgan':
    #    ccgan.optimize_disc(X,Y,args.batch_size,epoch=5)
    test_size = cdre_cfg.sample_size if args.cdre else args.test_size
    ### samples for training cdre model, no filtering here ###
    samples,labels = ccgan.gen_samples(np.arange(t+1),X_TRAIN[:test_size].shape,c_dim=c_dim)

    if args.save_samples:
        samples[samples<0.1]=0.
        samples[samples>0.9]=1.
        ccgan.save_samples(spath,samples,labels)

    if args.cdre:
        if t > 0:
            ccgan.cdre.update_estimator(sess,t)
            reinitialize_scope('ratio',sess)
            if isinstance(ccgan.cdre_feature_extractor, Continual_VAE):
                ccgan.cdre_feature_extractor.update_inference()
        ### prepare data ###
        test_samples,test_labels = ccgan.gen_samples(np.arange(t+1),X_TRAIN[:cdre_cfg.test_sample_size].shape,c_dim=c_dim)
        if np.sum(labels!=Y)>0 or np.sum(test_labels!=Y_test)>0  :
            assert('label not aligned!')
        samples,labels,X,Y = shuffle_data(samples,labels,X,Y)
        test_samples,test_labels,X_test,Y_test = shuffle_data(test_samples,test_labels,X_test,Y_test)
        ### fit cdre model and save sample ratios ###
        estimated_ratios = ccgan.fit_cdre_model(t,samples,labels,test_samples,test_labels,prev_samples=X,prev_labels=Y,\
                                prev_test_samples=X_test,prev_test_labels=Y_test)

        sample_ratios = pd.DataFrame()
        sample_ratios['estimated_original_ratio'] = estimated_ratios.sum(axis=1)
        sample_ratios['sample_c'] = np.argmax(labels,axis=1)
        #print('check c',test_samples_c[:3],sample_ratios.sample_c[:3])
        sample_ratios.to_csv(spath+'sample_ratios_t'+str(t+1)+'.csv',index=False)

        ### display worst samples ###
        wids = np.argsort(sample_ratios['estimated_original_ratio'].values)[:64]
        print('worst sample ratios',sample_ratios['estimated_original_ratio'].values[wids])
        fig = plot(samples[wids],shape=[8,8])
        fig.savefig(os.path.join(spath,'task'+str(t)+'worstsamples.pdf'))
        plt.close()
        ### display best samples ###
        bids = np.argsort(sample_ratios['estimated_original_ratio'].values)[-64:]
        print('best sample ratios',sample_ratios['estimated_original_ratio'].values[bids])
        fig = plot(samples[bids],shape=[8,8])
        fig.savefig(os.path.join(spath,'task'+str(t)+'bestsamples.pdf'))
        plt.close()
        ### display selected worst samples ###
        ratios = sample_ratios['estimated_original_ratio'].values
        selected = ratios[ratios>=cdre_cfg.filter_min]
        wids = np.argsort(selected)[:64]
        fig = plot(samples[ratios>=cdre_cfg.filter_min][wids],shape=[8,8])
        fig.savefig(os.path.join(spath,'task'+str(t)+'selected_worstsamples.pdf'))
        plt.close()
    

    if t < args.T-1: # and args.model_type == 'rfgan' 
        ccgan.update_model(t+1)
        if args.tblog:
            merged = set_tblog(ccgan)




