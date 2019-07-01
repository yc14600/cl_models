
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:


import tensorflow as tf
import edward as ed
import matplotlib.pyplot as plt
import seaborn as sb
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
sys.path.append(path+'/../hsvi/')


# In[3]:


from utils.model_util import build_nets,fit_model,predict,forward_nets
from utils.train_util import gen_permuted_data,gen_class_split_data,get_next_batch,gen_core_set


# In[4]:


from hsvi.hsvi import hsvi
from edward.models import Normal,MultivariateNormalTriL
from tensorflow.examples.tutorials.mnist import input_data


# In[5]:


seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)


# In[6]:


DATA_DIR = '../MNIST_data/'


# In[7]:


num_tasks = 10
TRAIN_SIZE = 5000
TEST_SIZE =  1000
hidden = [100,100]
batch_size = 100
scale = 1. #TRAIN_SIZE/batch_size#weights of likelihood
shrink = 1. #shrink train_size, smaller gives larger weights of KL-term
epoch = 1
coreset_size =50
coreset_usage = 'regret' #'regret','extra_train'
starter_learning_rate = 0.001
decay=(1000,0.9)
print_iter = 5

task_name = 'permuted'
multi_task_num = 3
vi_type = 'KLqp_analytic' #'KLqp','KLqp_analytic','cumulative_KLqp','IWAE'
ralpha = 2. # only for renyi divergence
prior_type = 'normal' #equivalent to cumulative_KLqp, but can be used in other vi_types
share_type = 'isotropic'
ginit = False
n_samples = 20
gaussian_type = 'logvar'
ac_fn = tf.nn.relu
ac_name = 'relu'

if task_name == 'permuted':
    head = 'unified'
elif task_name == 'split':
    head = 'multi'


# In[8]:


result_path = './results/'
save_parm = False


# In[9]:


if 'permuted' in task_name:
    data = input_data.read_data_sets(DATA_DIR,one_hot=True) 
    shuffle_ids = np.arange(data.train.images.shape[0])
    X_TRAIN = data.train.images[shuffle_ids][:TRAIN_SIZE]
    Y_TRAIN = data.train.labels[shuffle_ids][:TRAIN_SIZE]
    X_TEST = data.test.images[:TEST_SIZE]
    Y_TEST = data.test.labels[:TEST_SIZE]
    out_dim = Y_TRAIN.shape[1]
elif 'split' in task_name:
    data = input_data.read_data_sets(DATA_DIR) 
    X_TRAIN = data.train.images
    Y_TRAIN = data.train.labels
    X_TEST = data.test.images
    Y_TEST = data.test.labels
    out_dim = 2
    cl_k = 0
    cl_cmb = np.arange(10)


# In[10]:


in_dim = X_TRAIN.shape[1]    
net_shape = [in_dim]+hidden+[out_dim]


# In[11]:


x_ph = tf.placeholder(dtype=tf.float32,shape=[batch_size,net_shape[0]])
y_ph = tf.placeholder(dtype=tf.int32,shape=[n_samples,batch_size,net_shape[-1]])  
if ginit:
    y0_ph = tf.placeholder(dtype=tf.int32,shape=[batch_size,net_shape[-1]]) 
if coreset_size > 0:
    x0_ph = tf.placeholder(dtype=tf.float32,shape=[None,net_shape[0]])


# In[12]:


if ginit:
    # generate initialization
    if task_name == 'permuted':
        x_train_task,x_test_task = gen_permuted_data(0,X_TRAIN,X_TEST)
        y_train_task = Y_TRAIN
        y_test_task = Y_TEST
    elif task_name == 'split':
        x_train_task,y_train_task,x_test_task,y_test_task = gen_class_split_data(0,TRAIN_SIZE,TEST_SIZE,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,cls=[0,1])

    qW0,qB0,H0,TS0,_ws,_bs = build_nets(net_shape,x_ph,ac_fn=ac_fn)
    loss = tf.reduce_sum(tf.losses.log_loss(labels=y0_ph,predictions=H0[-1]))

    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    train_step = opt.minimize(loss)
    sess = ed.get_session()
    tf.global_variables_initializer().run()
    L = epoch*int(np.ceil(TRAIN_SIZE/batch_size))
    fit_model(L,x_train_task,y_train_task,x_ph,y0_ph,batch_size,train_step,loss,sess)
    print(predict(x_test_task,y_test_task,x_ph,H0[-1],batch_size,sess))


# In[13]:


if ginit:
    init = {'w':sess.run(qW0),'b':sess.run(qB0)}        
else:
    init = None


# In[14]:


tmp=tf.ones([20,100,784])
tmp=tf.reshape(tmp,[-1,784,100])
#tmp.shape


# In[15]:


with tf.variable_scope('task'):
    qW,qB,H,TS,qW_samples,qB_samples = build_nets(net_shape,x_ph,bayes=True,ac_fn=ac_fn,share=share_type,initialization=init,num_samples=n_samples,gaussian_type=gaussian_type)


# In[16]:


if coreset_size > 0:
    core_y = forward_nets(qW,qB,x0_ph,ac_fn=ac_fn,bayes=True,num_samples=n_samples)

# configure prior for latent variables

task_var_cfg = {}

if ginit:
    for i in range(len(qW)):          
        pW_i = Normal(loc=tf.zeros_like(qW[i].loc),scale=tf.ones_like(qW[i].scale)*1.)  
        #pW_i = Normal(loc=sess.run(qW0[i]),scale=tf.ones_like(qW[i])*1.)
        task_var_cfg[pW_i] = qW[i]

        #pB_i = Normal(loc=sess.run(qB0[i]),scale=tf.ones_like(qB[i])*1.)
        pB_i = Normal(loc=tf.zeros_like(qB[i].loc),scale=tf.ones_like(qB[i].scale)*1.)
        task_var_cfg[pB_i] = qB[i]

else:   
    for i in range(len(qW)):           

        pW_i = Normal(loc=tf.zeros_like(qW[i]),scale=tf.ones_like(qW[i])*1.)
        task_var_cfg[pW_i] = qW[i]                

        pB_i = Normal(loc=tf.zeros_like(qB[i]),scale=tf.ones_like(qB[i])*1.)           
        task_var_cfg[pB_i] = qB[i]          

init_task_var_cfg = dict(task_var_cfg)


# In[18]:


# configure samples of latent variables for computing importance weights
if vi_type in ['IWAE','Renyi','IWAE_ll']:
    var_sample_map = {}
    for i in range(len(qW)):
        var_sample_map[qW[i]] = qW_samples[i]
        var_sample_map[qB[i]] = qB_samples[i]
    


# In[19]:


with tf.variable_scope('task'):
    task_step = tf.Variable(0, trainable=False, name="task_step")

    #learning_rate = tf.train.exponential_decay(starter_learning_rate,
    #                                        task_step,
    #                                        decay[0], decay[1], staircase=True)
    task_optimizer = (tf.train.AdamOptimizer(starter_learning_rate),task_step)
    #task_optimizer = (tf.train.GradientDescentOptimizer(learning_rate),task_step)

# In[21]:

inference = slvi.Hierarchy_SLVI(latent_vars={'task':task_var_cfg},data={'task':{H[-1]:y_ph}})

if 'KLqp' in vi_type:
    inference.initialize(vi_types={'task':vi_type},scale={H[-1]:scale},optimizer={'task':task_optimizer},train_size=TRAIN_SIZE*shrink)
elif 'IWAE' in vi_type:
    inference.initialize(vi_types={'task':vi_type},scale={H[-1]:scale},optimizer={'task':task_optimizer},train_size=TRAIN_SIZE*shrink,samples=var_sample_map)
elif vi_type == 'Renyi':
    inference.initialize(vi_types={'task':vi_type},scale={H[-1]:scale},optimizer={'task':task_optimizer},train_size=TRAIN_SIZE*shrink,samples=var_sample_map,renyi_alpha=ralpha)

#print(inference.var_pair)

if share_type == 'isotropic':

    for g in inference.grads['task']:
        if g[1] == tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task/layer0w_mean")[0]:
            break 

    i,j = 689,87
    tf.summary.histogram('weight_exsample1_loc', tf.gather(tf.gather(qW[0].loc,i),j))
    if gaussian_type == 'rpm':
        tf.summary.scalar('weight_exsample1_logvar', tf.gather(tf.gather(qW[0].logvar,i),j)) 
    else:
        tf.summary.scalar('weight_exsample1_logvar', tf.log(tf.square(tf.gather(tf.gather(qW[0].scale,i),j))))
    tf.summary.scalar('weight_exsample1_grads', tf.gather(tf.gather(g[0],i),j))

    i,j = 392,45
    tf.summary.histogram('weight_exsample2_loc', tf.gather(tf.gather(qW[0].loc,i),j))
    if gaussian_type == 'rpm':
        tf.summary.scalar('weight_exsample2_logvar', tf.gather(tf.gather(qW[0].logvar,i),j)) 
    else:
        tf.summary.scalar('weight_exsample2_logvar', tf.log(tf.square(tf.gather(tf.gather(qW[0].scale,i),j))))
    tf.summary.scalar('weight_exsample2_grads', tf.gather(tf.gather(g[0],i),j))


# In[22]:


if not os.path.exists(result_path):
    os.mkdir(result_path)
file_name = vi_type+'_'+prior_type+'_'+share_type+'_'+head+'_'+gaussian_type+'_'+ac_name+'_tsize'+str(TRAIN_SIZE)+'_cset'+str(coreset_size)+coreset_usage+'_nsample'+str(n_samples)+'_bsize'+str(batch_size)+'_init'+str(ginit)+'_e'+str(epoch)+'_'+task_name+'_sd'+str(seed)
if vi_type == 'Renyi':
    file_name += '_alpha'+str(ralpha)
file_path = result_path+file_name
# In[23]:


test_sets = []
core_sets = [[],[]]
acc_record = []
entropies = [] 
xentropies = []
pre_parms = {}
avg_xentropies = []

sess = ed.get_session()    
tf.global_variables_initializer().run()

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

for t in range(num_tasks):   
    acc_record.append([])
    
    # generate training and test data
    if task_name == 'permuted':
        if not ginit or t > 0:
            x_train_task,x_test_task = gen_permuted_data(t,X_TRAIN,X_TEST)
            y_train_task = Y_TRAIN
            y_test_task = Y_TEST
    elif task_name == 'multitask_permuted':
        x_train_task,x_test_task = [],[]
        y_train_task,y_test_task = [],[]
        for _ in range(multi_task_num):
            x_train_task_t,x_test_task_t = gen_permuted_data(_,X_TRAIN,X_TEST)
            x_train_task.append(x_train_task_t)
            x_test_task.append(x_test_task_t)
            y_train_task.append(Y_TRAIN)
            y_test_task.append(Y_TEST)
            
        y_train_task = np.concatenate(y_train_task,axis=0)
        y_test_task = np.concatenate(y_test_task,axis=0)
            
        x_train_task = np.concatenate(x_train_task,axis=0)
        x_test_task = np.concatenate(x_test_task,axis=0)               
            
    elif task_name == 'split':
        if not ginit or t > 0:
            x_train_task,y_train_task,x_test_task,y_test_task = gen_class_split_data(t,TRAIN_SIZE,TEST_SIZE,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,cls=cl_cmb[cl_k:cl_k+2])
            cl_k+=2
    
    # generate random coresets
    if coreset_size > 0:
        idx = np.random.choice(x_train_task.shape[0],coreset_size)
        core_sets[0].append(x_train_task[idx])
        core_sets[1].append(y_train_task[idx])
        if coreset_usage == 'extra_train':
            x_train_task = np.delete(x_train_task,idx,axis=0)
            y_train_task = np.delete(y_train_task,idx,axis=0)
    
    test_sets.append((x_test_task,y_test_task))
    #core_sets.append(gen_core_set(x_train_task,y_train_task,method='mean'))
    
    # training for current task
    num_iter = int(x_train_task.shape[0]/batch_size)
    sess.run(task_step.initializer)
    
    for e in range(epoch):
        shuffle_inds = np.arange(x_train_task.shape[0])
        np.random.shuffle(shuffle_inds)
        x_train_task = x_train_task[shuffle_inds]
        y_train_task = y_train_task[shuffle_inds]
        err = 0.
        kl = 0.
        ll = 0.
        ii = 0
        for _ in range(num_iter):
            x_batch,y_batch,ii = get_next_batch(x_train_task,batch_size,ii,labels=y_train_task)
            y_batch = np.expand_dims(y_batch,axis=0)
            y_batch = np.repeat(y_batch,n_samples,axis=0)
            if coreset_size > 0 and t > 0 and coreset_usage!='extra_train':
                feed_dict = {x_ph:x_batch,y_ph:y_batch,x0_ph:x_core_sets}
            else:
                feed_dict = {x_ph:x_batch,y_ph:y_batch}
            info_dict = inference.update(scope='task',feed_dict=feed_dict)
           
            _kl,_ll = sess.run([inference.kl,inference.ll],feed_dict=feed_dict)
            kl += _kl
            ll += -_ll
            err += info_dict['loss']  
            #summary = sess.run(merged,feed_dict={x_ph:x_batch,y_ph:y_batch,avg_err:err/(_+1),avg_kl:kl/(_+1),avg_ll:ll/(_+1)})
            #train_writer.add_summary(summary, _+num_iter*(e+t*epoch))
               
        summary = sess.run(merged,feed_dict={x_ph:x_batch,y_ph:y_batch,avg_err:err/num_iter,avg_kl:kl/num_iter,avg_ll:ll/num_iter})
        train_writer.add_summary(summary, e+t*epoch)
        if (e+1)%print_iter==0:
            print('epoch',e+1,'avg loss',err/num_iter)
            #kl,ll = sess.run([inference.kl,inference.ll],feed_dict=feed_dict)
            #print('KL',kl,'LL',ll)

    # test on all tasks
    if coreset_size <= 0 or coreset_usage!='extra_train':
        mW,mB = [],[]
        for l in range(len(qW)):
            if share_type == 'row_covariance':
                mW.append(sess.run(qW[l].loc).transpose())
            else:
                mW.append(sess.run(qW[l].loc))      
            mB.append(sess.run(qB[l].loc))
        
        my = forward_nets(mW,mB,x_ph,ac_fn=ac_fn)
                
        for ts in test_sets:   
            acc = predict(ts[0],ts[1],x_ph,my,batch_size,sess)  #predict(ts[0],ts[1],x_ph,H[-1],batch_size,sess)
            print('accuracy',acc)
            acc_record[-1].append(acc)
    
    # record model entropy for every task
    model_entropy = 0.
    for i,q in enumerate(qW):
        qe = sess.run(q.entropy()).reshape(-1)        
        model_entropy += qe.sum()
        sb.distplot(qe)
        print('entropy layer'+str(i),qe.sum())
    plt.legend(['layer'+str(i) for i in range(len(qW))])
    plt.savefig(file_path+'_task'+str(t+1)+'_entropy.pdf',bbox_inches='tight')
    entropies.append(model_entropy)

    model_xentropy = 0.
    for p,q in six.iteritems(task_var_cfg):
        model_xentropy += sess.run(tf.reduce_sum(q.cross_entropy(p)))
    xentropies.append(model_xentropy)

    model_avg_xentropy = 0.
    # update priors
    p_list = list(task_var_cfg.keys())
    for pw_i in p_list:
        qw_i = task_var_cfg.pop(pw_i)
        mu = sess.run(qw_i.loc)
       
        if isinstance(qw_i,MultivariateNormalTriL):
            scale_tril = sess.run(TS[qw_i])
            npw_i = MultivariateNormalTriL(loc=mu,scale_tril=scale_tril)
        else:
            if prior_type=='kalman' and t > 1:
                mu0 = sess.run(pw_i.loc)
                var0 = np.square(sess.run(pw_i.scale))
                var1 = np.square(sess.run(qw_i.scale))
                mu = (mu*var0+mu0*var1)/(var0+var1)
                sigma = np.sqrt((var0*var1)/(var0+var1))
            elif prior_type == 'min_scale':
                std0 = sess.run(pw_i.scale)
                std1 = sess.run(qw_i.scale)
                sigma = (std0<=std1)*std0+(std0>std1)*std1
            else:
                sigma = sess.run(qw_i.scale)
            npw_i = Normal(loc=mu,scale=sigma)
            
        task_var_cfg[npw_i] = qw_i

        # save parameters of every task and 
        # compute average cross entropy
        pre_parms_t = pre_parms.get(qw_i,None)
        if pre_parms_t is None:
            pre_parms_t = []
        else:
            for p in pre_parms_t:
                model_avg_xentropy += sess.run(tf.reduce_sum(qw_i.cross_entropy(p)))
        pre_parms_t.append(Normal(loc=sess.run(qw_i.loc),scale=sess.run(qw_i.scale)))
        pre_parms[qw_i] = pre_parms_t
    model_avg_xentropy /= (t+1)
    avg_xentropies.append(model_avg_xentropy)

    # update inference for next task   
    inference.latent_vars['task'] = task_var_cfg 
    if coreset_size>0:
        x_core_sets = np.concatenate(core_sets[0],axis=0)
        y_core_sets = np.concatenate(core_sets[1],axis=0)
        
        if coreset_usage=='regret':
            core_y_data = np.expand_dims(y_core_sets,axis=0)
            core_y_data = np.repeat(core_y_data,n_samples,axis=0)
            inference.scale.update({core_y:batch_size/coreset_size})
            inference.reinitialize(task_id=t+1,coresets={'task':{core_y:core_y_data}})

        elif coreset_usage=='extra_train':           
            inference.reinitialize(task_id=t+1)
            save_path = saver.save(sess, result_path+"model.ckpt")
            print('start extra training by coresets')
            ii = 0
            L = epoch*int(np.ceil(x_core_sets.shape[0]/batch_size))
            for _ in range(L):
                x_batch,y_batch,ii = get_next_batch(x_core_sets,batch_size,ii,labels=y_core_sets)
                y_batch = np.expand_dims(y_batch,axis=0)
                y_batch = np.repeat(y_batch,n_samples,axis=0)
                #print(x_batch.shape,y_batch.shape,x_core_sets.shape,y_core_sets.shape)
                feed_dict = {x_ph:x_batch,y_ph:y_batch}
                info_dict = inference.update(scope='task',feed_dict=feed_dict)

            # test on all tasks
            mW,mB = [],[]
            for l in range(len(qW)):
                if share_type == 'row_covariance':
                    mW.append(sess.run(qW[l].loc).transpose())
                else:
                    mW.append(sess.run(qW[l].loc))      
                mB.append(sess.run(qB[l].loc))
            
            my = forward_nets(mW,mB,x_ph,ac_fn=ac_fn)
                    
            for ts in test_sets:   
                acc = predict(ts[0],ts[1],x_ph,my,batch_size,sess)  #predict(ts[0],ts[1],x_ph,H[-1],batch_size,sess)
                print('accuracy',acc)
                acc_record[-1].append(acc)
            
            # reset variables
            saver.restore(sess, result_path+"model.ckpt")

    else:
        inference.reinitialize(task_id=t+1)
    


# In[25]:

with open(file_path+'_entropies.csv','w') as f:
    writer = csv.writer(f,delimiter=',')    
    writer.writerow(entropies)

with open(file_path+'_xentropies.csv','w') as f:
    writer = csv.writer(f,delimiter=',')    
    writer.writerow(xentropies)

with open(file_path+'_avg_xentropies.csv','w') as f:
    writer = csv.writer(f,delimiter=',')    
    writer.writerow(avg_xentropies)

with open(file_path+'_'+'accuracy_record.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for t in range(len(acc_record)):
        writer.writerow(acc_record[t])


# In[26]:


if save_parm:
    for l in range(len(qW)):
        if share_type == 'isotropic':
            cov=sess.run(qW[l].scale)
        else:
            cov=sess.run(qW[l].covariance())
        np.save(file_path+'_layer'+str(l)+'_weights_cov',cov)

        mean=sess.run(qW[l].loc)
        np.save(file_path+'_layer'+str(l)+'_weights_mean',mean)    
        np.save(file_path+'_layer'+str(l)+'_bias_cov',sess.run(qB[l].scale))
        np.save(file_path+'_layer'+str(l)+'_bias_mean',sess.run(qB[l].loc))
    




