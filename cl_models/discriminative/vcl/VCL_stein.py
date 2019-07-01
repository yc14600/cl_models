
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
path = os.getcwd()
import sys
sys.path.append(path+'/../')
sys.path.append(path+'/../slvi/')
sys.path.append(path+'/../../edward/')


import tensorflow as tf
import edward as ed
# In[3]:


from utils.model_util import *
from utils.train_util import *
from utils.coreset_util import *


# In[4]:


from slvi.slvi import slvi
from slvi.methods.svgd import SVGD
from edward.models import Normal,MultivariateNormalTriL
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.datasets import cifar10,cifar100

# In[5]:
args = sys.argv
dargs = {}
for i in range(1,len(args)):
    arg = args[i].split(':')
    dargs[arg[0]] = arg[1]

seed = int(dargs.get('seed',42))
print('seed',seed)

#seed = 1
tf.set_random_seed(seed)
np.random.seed(seed)


# In[6]:

dataset = dargs.get('dataset','mnist')

if dataset == 'mnist':
    DATA_DIR = '../datasets/MNIST_data/'
elif dataset == 'fashion':
    DATA_DIR = '../datasets/fashion-mnist/'
elif dataset == 'notmnist':
    DATA_DIR = '../datasets/not-mnist/'
elif dataset == 'not-notmnist':
    task_name = 'cross_split'
    DATA_DIR = ['../datasets/MNIST_data/','../datasets/not-mnist/']
elif dataset == 'quickdraw':
    DATA_DIR = '../datasets/quickdraw/full/numpy_bitmap/'
print(dataset)
# In[7]:

task_name = dargs.get('task_name','permuted')
vi_type = dargs.get('vi_type','KLqp_analytic')  #'KLqp','KLqp_analytic','KLqp_gaussian_NG','cumulative_KLqp','IWAE'
epoch = int(dargs.get('epoch',50))
coreset_size = int(dargs.get('coreset_size',40))
print('coreset size',coreset_size)
coreset_type = dargs.get('coreset_type','random')
coreset_usage = dargs.get('coreset_usage','regret')
grad_type = dargs.get('grad_type','adam')
print('coreset type',coreset_type)
batch_size = int(dargs.get('batch_size',500))
TRAIN_SIZE = int(dargs.get('train_size',55000)) 
TEST_SIZE =  int(dargs.get('test_size',10000))
num_tasks = int(dargs.get('num_tasks',10))
local_iter = int(dargs.get('local_iter',50))
head_type = dargs.get('head_type','single')
save_parm = bool(dargs.get('save_parm',False))
ginit = float(dargs.get('ginit',3))
n_samples = int(dargs.get('num_samples',20))
print('num samples',n_samples)
hidden = [100,100]
scale = 1.#TRAIN_SIZE/batch_size#weights of likelihood
shrink = 1. #shrink train_size, smaller gives larger weights of KL-term

starter_learning_rate = 0.001
decay=(1000,0.9)
print_iter = 5

share_type = 'isotropic'
prior_type = 'normal'

#n_samples = 20
gaussian_type = 'logvar'
ac_fn = tf.nn.relu
ac_name = 'relu'

conv = False

# In[8]:

print(task_name)

if 'split' in task_name:
    if dataset in ['fashion','mnist']:
        num_tasks = 5
    elif dataset in ['notmnist','not-notmnist']:
        num_tasks = 10
    elif dataset == 'quickdraw':
        num_tasks = 8
    elif dataset in ['cifar']:
        num_tasks = 10
        conv = True

if head_type == 'single':
    num_heads = 1
else:
    num_heads = num_tasks

print('heads',num_heads)
# In[9]:


result_path = './results/'


# In[10]:


if 'permuted' in task_name:
    data = input_data.read_data_sets(DATA_DIR,one_hot=True) 
    shuffle_ids = np.arange(data.train.images.shape[0])
    X_TRAIN = data.train.images[shuffle_ids][:TRAIN_SIZE]
    Y_TRAIN = data.train.labels[shuffle_ids][:TRAIN_SIZE]
    X_TEST = data.test.images[:TEST_SIZE]
    Y_TEST = data.test.labels[:TEST_SIZE]
    out_dim = Y_TRAIN.shape[1]
    cl_n = out_dim # number of classes in each task
    # generate data for first task
    x_train_task,y_train_task,x_test_task,y_test_task,_ck,_cs = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST)

elif 'cross_split' in task_name:
    if dataset == 'not-notmnist':
        data1 = input_data.read_data_sets(DATA_DIR[0])
        data2 = input_data.read_data_sets(DATA_DIR[1])
        X_TRAIN = [data1.train.images,data2.train.images]
        Y_TRAIN = [data1.train.labels,data2.train.labels]
        X_TEST = [data1.test.images,data2.test.images]
        Y_TEST = [data1.test.labels,data2.test.labels]
        out_dim = 2
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,_cs = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST) 
    
    elif dataset == 'quickdraw':
        X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = load_mini_quick_draw(DATA_DIR)
        out_dim = len(X_TRAIN)
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,_cs = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,out_dim=out_dim) 


elif 'split' in task_name:
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
        cls = cl_cmb[cl_k:cl_k+cl_n]
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
        
        x_train_task,y_train_task,x_test_task,y_test_task,cl_k,cls = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,\
                                                                    cl_n=cl_n,cl_k=cl_k,cl_cmb=cl_cmb,out_dim=out_dim,num_heads=num_heads)


TRAIN_SIZE = x_train_task.shape[0]
TEST_SIZE = x_test_task.shape[0]
original_batch_size = batch_size
if batch_size > TRAIN_SIZE:
        batch_size = TRAIN_SIZE
print('batch size',batch_size)
# In[12]:
   



# In[13]:

# In[16]:
if ginit > 0:
    # set init value of variance
    initialization={'w_s':-1.*ginit,'b_s':-1.*ginit,'cw_s':-1*ginit}
else:
    initialization=None

with tf.variable_scope('task'):
    #in_x = x_ph
    if conv:
        x_ph = tf.placeholder(dtype=tf.float32,shape=[None]+list(X_TRAIN.shape[1:]))
        conv_W,conv_parm_var,conv_h = cifar_model(x_ph,batch_size,initialization=initialization)
        dropout = 0.5
        in_x = conv_h
        in_dim = in_x.shape[1].value
        print('cifar indim',in_dim)
    else:
        x_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_train_task.shape[1]])
        dropout = None
        in_x = x_ph
        in_dim = x_train_task.shape[1]
        
    net_shape = [in_dim]+hidden+[out_dim]    
    print('net shape',net_shape)
    y_ph = tf.placeholder(dtype=tf.int32,shape=[n_samples,None,out_dim]) 

    if num_heads==1:
        print('single head net')
        qW,qB,H,TS,qW_samples,qB_samples,parm_var = build_nets(net_shape,in_x,bayes=True,ac_fn=ac_fn,share=share_type,initialization=initialization,dropout=dropout,num_samples=n_samples,gaussian_type=gaussian_type)
    else:
        print('multi-head net')
        qW_list,qB_list,H_list,TS,qW_list_samples,qB_list_samples,parm_var = build_nets(net_shape,in_x,bayes=True,ac_fn=ac_fn,share=share_type,initialization=initialization,dropout=dropout,num_samples=n_samples,gaussian_type=gaussian_type,num_heads=num_heads)

    if conv :
        parm_var.update(conv_parm_var)
# In[17]:


if coreset_size > 0:
    if not conv :
        if num_heads > 1:
            core_x_ph = [tf.placeholder(dtype=tf.float32,shape=[None,x_train_task.shape[1]]) for i in range(num_heads)]
            core_y = []
            for k in range(num_heads):  
                core_yk = forward_nets(qW_list[k],qB_list[k],core_x_ph[k],ac_fn=ac_fn,bayes=True,num_samples=n_samples)
                core_y.append(core_yk)
        else:
            core_x_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_train_task.shape[1]])
            core_y = forward_nets(qW,qB,core_x_ph,ac_fn=ac_fn,bayes=True,num_samples=n_samples)
    else:
        # only support multihead for cifar task    
        core_x_ph = [tf.placeholder(dtype=tf.float32,shape=[None]+list(X_TRAIN.shape[1:])) for i in range(num_heads)]
        core_y = []
        for k in range(num_heads):
            h_k = forward_cifar_model(core_x_ph[k],conv_W,batch_size)
            core_yk = forward_nets(qW_list[k],qB_list[k],h_k,ac_fn=ac_fn,bayes=True,num_samples=n_samples)
            core_y.append(core_yk)
        

# In[18]:


# configure prior for latent variables

if num_heads > 1:
    qW = qW_list[0]
    qB = qB_list[0]
    H = H_list[0]

if conv :
    task_var_cfg = gen_posterior_conf(conv_W+qW+qB)
else:
    task_var_cfg = gen_posterior_conf(qW+qB)
# In[19]:

with tf.variable_scope('task'):
    task_optimizer = config_optimizer(starter_learning_rate,'task_step','adam')

if 'stein' in coreset_type:
    with tf.variable_scope('stein'):
        stein_optimizer = config_optimizer(starter_learning_rate,'stein_step','adam')


# In[21]:    

inference = slvi.Hierarchy_SLVI(latent_vars={'task':task_var_cfg},data={'task':{H[-1]:y_ph}})

if 'KLqp' in vi_type or 'MLE' in vi_type:
    if 'NG' in vi_type:
        inference.initialize(vi_types={'task':vi_type},scale={H[-1]:scale},optimizer={'task':task_optimizer},train_size=TRAIN_SIZE*shrink,trans_parm={'task':parm_var})
    else:
        inference.initialize(vi_types={'task':vi_type},scale={H[-1]:scale},optimizer={'task':task_optimizer},train_size=TRAIN_SIZE*shrink)

# choose parameters for tensorboad record

for g in inference.grads['task']:
    if g[1] == tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task/layer0w_mean")[0]:
        break 

i,j = 689,8
tf.summary.histogram('weight_exsample1_loc', tf.gather(tf.gather(qW[0].loc,i),j))
tf.summary.scalar('weight_exsample1_loc', tf.gather(tf.gather(qW[0].loc,i),j))
tf.summary.scalar('weight_exsample1_logvar', tf.log(tf.square(tf.gather(tf.gather(qW[0].scale,i),j))))
tf.summary.scalar('weight_exsample1_grads', tf.gather(tf.gather(g[0],i),j))

i,j = 392,4
tf.summary.histogram('weight_exsample2_loc', tf.gather(tf.gather(qW[0].loc,i),j))
tf.summary.scalar('weight_exsample2_loc', tf.gather(tf.gather(qW[0].loc,i),j))
tf.summary.scalar('weight_exsample2_logvar', tf.log(tf.square(tf.gather(tf.gather(qW[0].scale,i),j))))
tf.summary.scalar('weight_exsample2_grads', tf.gather(tf.gather(g[0],i),j))


# In[22]:


if not os.path.exists(result_path):
    os.mkdir(result_path)
file_name = dataset+'_'+vi_type+'_'+prior_type+'_'+share_type+'_'+gaussian_type+'_'+ac_name+'_tsize'+str(TRAIN_SIZE)+'_cset'+str(coreset_size)+coreset_type+'_'+coreset_usage+'_nsample'+str(n_samples)+'_bsize'+str(batch_size)+'_init'+str(int(ginit))\
            +'_e'+str(epoch)+'_liter'+str(local_iter)+'_'+task_name+'_sd'+str(seed)

file_path = result_path+file_name


# In[26]:
sess = ed.get_session() 

# In[27]:
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
saver = tf.train.Saver()
# In[28]:

test_sets = []
core_sets = [[],[]]
acc_record = []
pre_parms = {}

tf.global_variables_initializer().run()
for t in range(num_tasks):   

    # get test data
    test_sets.append((x_test_task,y_test_task))
    # generate coresets
    if coreset_size > 0:
        if 'kcenter' in coreset_type :
            idx = gen_kcenter_coreset(x_train_task,coreset_size)
            core_x_set = x_train_task[idx]
            core_y_set = y_train_task[idx]
        
        elif 'random' in coreset_type or coreset_type == 'stein':
            # default initialization of stein is random samples
            idx = np.random.choice(x_train_task.shape[0],coreset_size)
            core_x_set = x_train_task[idx]
            core_y_set = y_train_task[idx]

        elif 'rdproj' in coreset_type:
            if 'split' in task_name and num_heads == 1 :
                core_x_set,core_y_set = gen_rdproj_coreset(x_train_task,y_train_task,coreset_size,cl_n,cls)
            else:
                core_x_set,core_y_set = gen_rdproj_coreset(x_train_task,y_train_task,coreset_size,cl_n)


        else:
            raise TypeError('Non-supported coreset type!') 
        
        core_sets[1].append(core_y_set)
        curnt_core_y_data = expand_nsamples(core_sets[1][-1],n_samples)
        
        
        if 'stein' not in coreset_type:
            core_sets[0].append(core_x_set)
        
        else:  # define stein samples
            
            with tf.variable_scope('stein_task'+str(t)):
                if conv :
                    stein_core_x,stein_core_y,core_sgrad = gen_stein_coreset(core_x_set,curnt_core_y_data,qW,qB,n_samples,ac_fn,conv_W=conv_W)
                else:
                    stein_core_x,stein_core_y,core_sgrad = gen_stein_coreset(core_x_set,curnt_core_y_data,qW,qB,n_samples,ac_fn)

            core_sets[0].append(stein_core_x)
            stein_train = stein_optimizer[0].apply_gradients([(core_sgrad,stein_core_x)],global_step=stein_optimizer[1])
            tf.variables_initializer(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="stein_task"+str(t))).run()
            sess.run(tf.variables_initializer(stein_optimizer[0].variables()))

        if coreset_usage == 'final' and 'rdproj' not in coreset_type:
            # remove coreset from the training set
            x_train_task = np.delete(x_train_task,idx,axis=0)
            y_train_task = np.delete(y_train_task,idx,axis=0)
    
    # training for current task
    num_iter = int(np.ceil(x_train_task.shape[0]/batch_size))
    #sess.run(task_step.initializer)
    print('num iter',num_iter)
    for e in range(epoch):
        shuffle_inds = np.arange(x_train_task.shape[0])
        np.random.shuffle(shuffle_inds)
        x_train_task = x_train_task[shuffle_inds]
        y_train_task = y_train_task[shuffle_inds]
        err,kl,ll = 0.,0.,0.
        ii = 0
        for _ in range(num_iter):
            x_batch,y_batch,ii = get_next_batch(x_train_task,batch_size,ii,labels=y_train_task)
            y_batch = np.expand_dims(y_batch,axis=0)
            y_batch = np.repeat(y_batch,n_samples,axis=0)
            
            feed_dict = {x_ph:x_batch,y_ph:y_batch}
            if coreset_size > 0:
                if t > 0 and coreset_usage != 'final':
                    if num_heads > 1:
                        for k in range(t):
                            feed_dict.update({core_x_ph[k]:x_core_sets[k]})
                    else:    
                        feed_dict.update({core_x_ph:x_core_sets})
           
                if 'stein' in coreset_type and (_+1)%local_iter==0:
                    # update stein samples
                    sess.run(stein_train)
             
            info_dict = inference.update(scope='task',feed_dict=feed_dict)
           
            _kl,_ll = sess.run([inference.kl,inference.ll],feed_dict=feed_dict)
            kl += _kl
            ll += -_ll
            err += info_dict['loss']  
          
        summary = sess.run(merged,feed_dict={x_ph:x_batch,y_ph:y_batch,avg_err:err/num_iter,avg_kl:kl/num_iter,avg_ll:ll/num_iter})
        train_writer.add_summary(summary, e+t*epoch)
        if (e+1)%print_iter==0:
            print('epoch',e+1,'avg loss',err/num_iter)
            #if coreset_type == 'stein' and coreset_size > 0:
                #stein_samples = sess.run(stein_core_x).mean()
                #print('stein mean',stein_samples)
            #stein_fig.savefig(file_path+'_t'+str(t)+'_steinsamples'+str(e)+'.pdf')
    # save params of each task
    if save_parm:
        for l in range(len(qW)):
            if share_type == 'isotropic':
                cov=sess.run(qW[l].scale)
            else:
                cov=sess.run(qW[l].covariance())
            np.save(file_path+'_task'+str(t)+'_layer'+str(l)+'_weights_cov',cov)

            mean=sess.run(qW[l].loc)
            np.save(file_path+'_task'+str(t)+'_layer'+str(l)+'_weights_mean',mean)    
            np.save(file_path+'_task'+str(t)+'_layer'+str(l)+'_bias_cov',sess.run(qB[l].scale))
            np.save(file_path+'_task'+str(t)+'_layer'+str(l)+'_bias_mean',sess.run(qB[l].loc))

    # test on all tasks   
    if conv :
        ch = conv_h
    else:
        ch = None

    if t > 0 and coreset_size >  0 and coreset_usage == 'final':
        save_path = saver.save(sess, file_path+"_model.ckpt")
        train_coresets_final(core_sets,core_y,x_ph,y_ph,core_x_ph,coreset_type,num_heads,t,n_samples,\
                                    batch_size,epoch,sess,inference)

    if num_heads > 1:  
        accs = test_tasks(t,test_sets,qW_list,qB_list,num_heads,x_ph,ac_fn,batch_size,sess,conv_h=ch)       
    else:
        accs = test_tasks(t,test_sets,qW,qB,num_heads,x_ph,ac_fn,batch_size,sess,conv_h=ch)        
    
    acc_record.append(accs)  
    # reset variables
    if t > 0 and coreset_size > 0 and coreset_usage == 'final':
        saver.restore(sess, file_path+"_model.ckpt")
        
    # update parameter configurations
    if t < num_tasks-1:
        task_var_cfg = {}
        # update priors
        if num_heads > 1:
            qW,qB = qW_list[t+1],qB_list[t+1]
            pW,pB = qW_list[t],qB_list[t]

            for l in range(len(qW)-1):
                update_variable_tables(pW[l],qW[l],sess,task_var_cfg)         
                update_variable_tables(pB[l],qB[l],sess,task_var_cfg)
                
            # configure head layer for new task    
            npw = Normal(loc=tf.zeros_like(qW[-1]),scale=tf.ones_like(qW[-1]))
            task_var_cfg[npw] = qW[-1]
            npb = Normal(loc=tf.zeros_like(qB[-1]),scale=tf.ones_like(qB[-1]))
            task_var_cfg[npb] = qB[-1]

            # configure head layer for all seen tasks
            for k in range(t+1):
                update_variable_tables(qW_list[k][-1],qW_list[k][-1],sess,task_var_cfg)
                update_variable_tables(qB_list[k][-1],qB_list[k][-1],sess,task_var_cfg)

        else:
            for l in range(len(qW)):
                # update weights prior and trans
                update_variable_tables(qW[l],qW[l],sess,task_var_cfg)             
                # update bias prior and trans
                update_variable_tables(qB[l],qB[l],sess,task_var_cfg)

        if conv :
            for qw in conv_W:
                update_variable_tables(qw,qw,sess,task_var_cfg)

        # update data and inference for next task 
        
        if 'permuted' in task_name:
            x_train_task,y_train_task,x_test_task,y_test_task,_ck,_cs = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,sd=t+1)
        
        elif 'cross_split' in task_name:
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,_cs = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,sd=t+1,cl_k=cl_k,out_dim=out_dim)
        
        elif 'split' in task_name:
            if t==4 and dataset=='notmnist':
                DATA_DIR = '../datasets/MNIST_data/'
                X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = load_task_data(task_name,DATA_DIR)
                cl_k = 0
            x_train_task,y_train_task,x_test_task,y_test_task,cl_k,cls = gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,cl_n=cl_n,out_dim=out_dim,num_heads=num_heads,cl_cmb=cl_cmb,cl_k=cl_k)
        
            TRAIN_SIZE = x_train_task.shape[0]    
            TEST_SIZE = x_test_task.shape[0]
            if original_batch_size > TRAIN_SIZE:
                batch_size = TRAIN_SIZE  
            else:
                batch_size = original_batch_size
            inference.train_size = TRAIN_SIZE

        print('train size',TRAIN_SIZE,'batch size',batch_size)
        inference.latent_vars['task'] = task_var_cfg 
        
        if coreset_size>0 and coreset_usage != 'final':
            x_core_sets,y_core_sets,c_cfg = config_coreset(core_sets,core_y,coreset_type,num_heads,t,n_samples,sess)
            
            if num_heads > 1:
                inference.data['task'] = {H_list[t+1][-1]:y_ph}
                inference.reinitialize(task_id=t+1,coresets={'task':c_cfg})
                sess.run(tf.variables_initializer(task_optimizer[0].variables()))
            else:
                inference.reinitialize(task_id=t+1,coresets={'task':c_cfg})
            
        else:
            
            if num_heads > 1:
                inference.data['task'] = {H_list[t+1][-1]:y_ph}
                inference.reinitialize(task_id=t+1)
                sess.run(tf.variables_initializer(task_optimizer[0].variables()))
            else:
                inference.reinitialize(task_id=t+1)
# In[29]:

with open(file_path+'_'+'accuracy_record.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(acc_record)):
        writer.writerow(acc_record[t])

if not save_parm and coreset_usage=='final':
    os.system('rm '+file_path+"_model.ckpt*")