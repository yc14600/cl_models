import numpy as np
import tensorflow as tf
import gzip
import os
import matplotlib.pyplot as plt

from collections.abc import Iterable
from base_models.gans import GAN,CfGAN,CWGAN,CWGAN_GP
from base_models.vae import VAE
from .cvae import Continual_VAE
from utils.train_util import shuffle_data,concat_cond_data,one_hot_encoder,get_next_batch,plot
from utils.test_util import save_samples

class Continual_CGAN(object):
    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                    batch_norm=False,learning_rate=0.001,op_type='adam',clip=None,reg=None,c_dim=10,\
                    model_type='fgan',divergence='Jensen_Shannon',lamb_constr=0.,g_penalty=False,\
                    gamma0=1.,alpha=0.01,mem_replay=0.,cdre=None,cdre_feature_extractor=None,cdre_config=None,*args,**kargs):
       
        self.model = self.define_model(x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,batch_norm,\
                                        learning_rate,op_type,clip,reg,c_dim,model_type,divergence,lamb_constr,\
                                        g_penalty,gamma0,alpha,*args,**kargs)
        self.conv = conv
        self.batch_size = batch_size
        self.g_net_shape = g_net_shape
        self.d_net_shape = d_net_shape
        self.mem_replay = mem_replay
        self.cdre = cdre
        self.cdre_feature_extractor = cdre_feature_extractor
        self.cdre_config = cdre_config
        if cdre:
            from cdre import Cond_Continual_LogLinear_Estimator
            assert(isinstance(self.cdre,Cond_Continual_LogLinear_Estimator))
            if self.cdre_feature_extractor:
                assert(isinstance(self.cdre_feature_extractor,VAE) or isinstance(self.cdre_feature_extractor,Continual_VAE))
            
        return 


    
    def define_model(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                        batch_norm=False,learning_rate=0.001,op_type='adam',clip=None,reg=None,c_dim=10,\
                        model_type='fgan',divergence='Jensen_Shannon',lamb_constr=0.,g_penalty=False,\
                        gamma0=1.,alpha=0.01,*args,**kargs):
        print('model type',model_type)
        if model_type == 'fgan':
            return CfGAN(x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,batch_norm,learning_rate,op_type,\
                            clip,reg,divergence,c_dim,lamb_constr,g_penalty=g_penalty,gamma0=gamma0,alpha=alpha,*args,**kargs)
        elif model_type == 'wgan':
            return CWGAN(x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,batch_norm,learning_rate,op_type,\
                            clip,reg,c_dim,*args,**kargs)
        elif model_type == 'wgan-gp':
            return CWGAN_GP(x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,batch_norm,learning_rate,op_type,\
                            clip,reg,c_dim,*args,**kargs)
        else:
            raise NotImplementedError('not supported model type.')

    def save_d_params(self):
        conv_L = len(self.d_net_shape[0]) if self.conv else 0
        print('d conv_L',conv_L)
        self.prev_d_W,self.prev_d_B = self.model.sess.run([self.model.d_W,self.model.d_B])
        self.prev_d_H = GAN.restore_d_net(self.model.x_ph,self.prev_d_W,self.prev_d_B,conv_L)
        self.prev_d_fake_H = GAN.restore_d_net(self.model.fake_x,self.prev_d_W,self.prev_d_B,conv_L)

        return 

    def save_g_params(self):
        conv_L = len(self.g_net_shape[-1]) - 1 if self.conv else 0
        print('g conv_L',conv_L)
        self.prev_g_W,self.prev_g_B = self.model.sess.run([self.model.g_W,self.model.g_B])
        self.prev_g_H = GAN.restore_g_net(self.model.e_ph,self.prev_g_W,self.prev_g_B,conv_L)

    
    def save_params(self,save_d=True,save_g=True):
        if save_d:
            self.save_d_params()
        if save_g:
            self.save_g_params()

    
    def mem_replay_loss(self,t):
        #mask = np.ones([self.model.c_dim],dtype=np.float32)
        #mask[t:] = 0.
        mask = tf.reshape(1.-self.model.c_ph[:,t],[-1,1])
        pv = tf.reshape(self.prev_g_H[-1],[self.batch_size,-1])
        cu = tf.reshape(self.model.g_H[-1],[self.batch_size,-1])
        err = pv-cu
        #err = tf.reshape(err,[err.shape[0].value,1])
        print('mask',mask,'err',err)
        return tf.nn.l2_loss(mask*err)

    
    def update_train(self,update_d=True,update_g=True):
        if update_g:
            g_grads = tf.gradients(self.model.g_loss,self.model.g_var_list)
            g_grads_and_vars = list(zip(g_grads,self.model.g_var_list))
            self.model.g_train = self.model.g_opt[0].apply_gradients(g_grads_and_vars,global_step=self.model.g_opt[1])
        
        if update_d:
            d_grads = tf.gradients(self.model.d_loss,self.model.d_var_list)
            d_grads_and_vars = list(zip(d_grads,self.model.d_var_list))
            self.model.d_train = self.model.d_opt[0].apply_gradients(d_grads_and_vars,global_step=self.model.d_opt[1])

        return


    def update_model(self,t,g_penalty=False):
        update_g = True if self.mem_replay > 0 else False
        self.save_params(save_d=False,save_g=update_g)
        self.model.g_penalty = g_penalty
        self.model.alpha = 0.
        self.model.gamma = 0.01
        self.model.g_loss,self.model.d_loss = self.model.set_loss()
        if update_g:            
            self.model.g_loss += self.mem_replay * self.mem_replay_loss(t)
        self.update_train(update_g=update_g)
        #if self.cdre:
        #    self.cdre.update_estimator(self.model.sess,t)


    def fit_cdre_model(self,t,samples,labels,test_samples,test_labels,prev_samples,prev_labels,\
                            prev_test_samples=None,prev_test_labels=None):
        
        prev_nu_samples,prev_de_samples,t_prev_nu_samples,t_prev_de_samples = None,None,None,None 
        prev_samples, samples = prev_samples.reshape(prev_samples.shape[0],-1), samples.reshape(samples.shape[0],-1)
        prev_test_samples, test_samples = prev_test_samples.reshape(prev_test_samples.shape[0],-1),test_samples.reshape(test_samples.shape[0],-1)

        sess = self.model.sess   
        if self.cdre_feature_extractor:           
            vae = self.cdre_feature_extractor
            vae.train(prev_samples)
            if isinstance(vae, Continual_VAE) and t>0:                
                # encode input for previous estimator
                prev_nu_samples = vae.prev_encode(prev_samples)
                prev_de_samples = vae.prev_encode(samples)
                if prev_test_samples is not None:
                    t_prev_nu_samples = vae.prev_encode(prev_test_samples)
                    t_prev_de_samples = vae.prev_encode(test_samples)
                                
            nu_samples = vae.encode(prev_samples)
            de_samples = vae.encode(samples)
            if prev_test_samples is not None:
                t_nu_samples = vae.encode(prev_test_samples)
                t_de_samples = vae.encode(test_samples)

            #if isinstance(vae, Continual_VAE):
            #    vae.update_inference()
        else:
            nu_samples, de_samples = prev_samples, samples
            t_nu_samples,t_de_samples = prev_test_samples,test_samples

                    
        cfg = self.cdre_config
        #check labels and prev_labels alignment#
        _ = self.cdre.learning(sess,nu_samples,de_samples,labels,t_nu_samples,t_de_samples,\
                            test_labels,batch_size=cfg.batch_size,epoch=cfg.epoch,print_e=cfg.print_e,\
                            early_stop=cfg.early_stop,min_epoch=cfg.min_epoch,\
                            prev_nu_samples=prev_nu_samples,prev_de_samples=prev_de_samples,\
                            t_prev_nu_samples=t_prev_nu_samples,t_prev_de_samples=t_prev_de_samples)

        if t > 0:
            estimated_original_ratio = self.cdre.original_log_ratio(sess,de_samples,de_samples,labels)

        else:
            estimated_original_ratio = self.cdre.estimator.log_ratio(sess,de_samples,de_samples,labels)

        return estimated_original_ratio


    def _gen_samples_by_batches(self,t,size,c_dim,x_shape):
        e = np.random.uniform(low=-1.,size=(size,self.model.g_W[0].shape[0].value-c_dim)).astype(np.float32)
        c = one_hot_encoder(np.ones(size)*t,c_dim)
        e = concat_cond_data(e,c,one_hot=False,dim=c_dim)
        x  = np.zeros([size,*x_shape]).astype(np.float32)
        ii = 0   
        iters = int(np.ceil(size/self.batch_size))
         
        for _ in range(iters):
            start = ii
            e_batch,_,ii = get_next_batch(e,self.batch_size,ii,None,repeat=True)
            end = ii if ii < size and ii > start else size
            #print('start',start,'end',end)
            x[start:end] = self.model.generator(e_batch)[:end-start] 
        return x,c

    
    def gen_samples(self,conds,x_shape,c_dim=10,filter=False):
        samples = []
        labels = []
        print('conds',conds)
        if not isinstance(conds,Iterable):
            conds = range(conds)
        c_sample_size = x_shape[0]
        #print('c_sample_size',c_sample_size,'iters',iters)
        t = len(conds)-1
        for i in conds:
            print('cond',i)
            
            if filter and self.cdre:
                ret_size = c_sample_size
                x,c = [],[]
                while ret_size > 0:
                    x_i,c_i = self._gen_samples_by_batches(i,ret_size,c_dim,x_shape[1:])
                    x_batch = x_i[:64]
                    fig = plot(x_batch[:8*8],shape=[8,8])
                    fig.savefig(os.path.join('../results/','x_batch.pdf'))
                    plt.close()
                    x_i,c_i = self.filter_samples(t,x_i,c_i)
                    x.append(x_i)
                    c.append(c_i)
                    ret_size -= len(x_i)
                    print('size after filtering',ret_size)
                    assert(len(x_i)>0)
                x = np.vstack(x)
                c = np.vstack(c)
                print('x {}, c {}'.format(x.shape,c.shape))
                
            else:
                x,c = self._gen_samples_by_batches(i,c_sample_size,c_dim,x_shape[1:])
                    
            samples.append(x)
            labels.append(c)
        samples = np.vstack(samples)
        labels = np.vstack(labels)
        

        return samples,labels

    
    def filter_samples(self,t,samples,labels):
        assert(self.cdre)
        if not self.cdre.estimator.conv:
            old_shape = samples.shape         
            samples = samples.reshape(samples.shape[0],-1)
            #print('filter check shape',samples.shape,labels.shape)
        if self.cdre_feature_extractor:
            test_samples = self.cdre_feature_extractor.encode(samples)
        else:
            test_samples = samples
        if t>0:
            estimated_ratio = self.cdre.original_log_ratio(self.model.sess,test_samples,test_samples,labels).sum(axis=1)
        else:
            estimated_ratio = self.cdre.estimator.log_ratio(self.model.sess,test_samples,test_samples,labels).sum(axis=1)
        print('check estimated ratio nan',np.sum(np.isnan(estimated_ratio)))
        print('estimated ratio stat',np.mean(estimated_ratio),np.std(estimated_ratio))
        select = (estimated_ratio>=self.cdre_config.filter[0]) & (estimated_ratio<=self.cdre_config.filter[1])
        if not self.cdre.estimator.conv:
            samples = samples.reshape(*old_shape)
        return samples[select], labels[select]

    
    def merge_train_data(self,new_X,new_cond,conds,c_dim=10,save_samples=True,sample_size=None,path='./'):
        if sample_size:
            cids = np.random.choice(new_X.shape[0],size=sample_size)
            new_X = new_X[cids]

        if not isinstance(new_cond,np.ndarray):            
            new_cond = one_hot_encoder(tf.ones(new_X.shape[0])*new_cond,c_dim)
        print(len(conds))
        if len(conds) > 0:
            if not isinstance(conds,Iterable):
                conds = range(conds) 
            #sample_size = new_X.shape[0] * len(conds)
            px,py = self.gen_samples(conds,x_shape=new_X.shape,c_dim=c_dim,filter=True)
            if save_samples:
                self.save_samples(path,px,py)
            print('px {}, new X {}'.format(px.shape,new_X.shape))
            new_X = np.vstack([px,new_X])
            new_cond = np.vstack([py,new_cond])

        return new_X,new_cond

    def save_samples(self,path,samples,labels):
        return save_samples(path,[samples,labels])




class Continual_Ratio_fGAN(Continual_CGAN):
    '''
    def __init__(self,x_ph,g_net_shape,d_net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                    batch_norm=False,learning_rate=0.001,op_type='adam',clip=None,reg=None,c_dim=10,\
                    divergence='Jensen_Shannon',lamb_constr=0.,g_penalty=False,gamma0=1.,alpha=0.01,\
                    *args,**kargs):
        print('crfgan lamb constr',lamb_constr,'g penalty',g_penalty)
        super(Continual_Ratio_fGAN,self).__init__(x_ph,g_net_shape,d_net_shape,batch_size,conv,sess,ac_fn,\
            batch_norm,learning_rate,op_type,clip,reg,c_dim,model_type='fgan',divergence=divergence,lamb_constr=lamb_constr,\
            g_penalty=g_penalty,gamma0=gamma0,alpha=alpha,*args,**kargs)

        #self.prev_x_ph = tf.placeholder(dtype=tf.float32,shape=x_ph.shape,name='prev_x_ph')
        
        return
    '''

    def update_model(self,t,g_penalty=False):
        update_g = True if self.mem_replay > 0 else False
        self.save_params(save_g=update_g)

        c_mask = np.ones([self.model.c_dim,1],dtype=np.float32)
        c_mask[t] = 0.
        c_mask = tf.matmul(self.model.c_ph, c_mask)
        c_mask2 = 1. - c_mask
        #print('check update',c_mask.shape,self.prev_de_r.shape)

        if self.model.divergence == 'KL':
            d_h = (self.model.d_H[-1] - self.prev_d_H[-1] + 1.)*c_mask + self.model.d_H[-1]*c_mask2
            d_fakeh = (self.model.d_fake_H[-1] - self.prev_d_fake_H[-1] + 1.)*c_mask + self.model.d_fake_H[-1]*c_mask2
        elif self.model.divergence == 'Pearson': 
            #nr = tf.clip_by_value((self.model.nu_H[-1]+2.)/(self.prev_nu_r+2.),-1e15,1e15)
            #dr = tf.clip_by_value((self.model.de_H[-1]+2.)/(self.prev_de_r+2.),-1e15,1e15)
            d_h = (self.model.d_H[-1]+2.)/(self.prev_d_H[-1]+2.)
            d_fakeh = (self.model.d_fake_H[-1]+2.)/(self.prev_d_fake_H[-1]+2.)
            d_h = (2.*(d_h - 1.))*c_mask + self.model.d_H[-1]*c_mask2
            d_fakeh = (2.*(d_fakeh - 1.))*c_mask + self.model.d_fake_H[-1]*c_mask2
        else:
            d_h = self.model.d_H[-1] - self.prev_d_H[-1]*c_mask 
            d_fakeh = self.model.d_fake_H[-1] - self.prev_d_fake_H[-1]*c_mask
            #self.model.nu_r = tf.clip_by_value(self.model.nu_r,-30.,30.)
            #self.model.de_r = tf.clip_by_value(self.model.de_r,-30.,30.)
        self.model.g_penalty = g_penalty
        self.model.alpha = 0.
        self.model.gamma = 0.01
        self.model.g_loss,self.model.d_loss = self.model.set_loss(d_h,d_fakeh)
        if update_g:
            self.model.g_loss += self.mem_replay * self.mem_replay_loss(t)

        self.update_train(update_g=update_g)

        #if self.cdre:
        #    self.cdre.update_estimator(self.model.sess,t)

        return
 

    def optimize_disc(self,X,Y,batch_size,epoch,d_obj=None):
        sess = self.model.sess  
        if d_obj is None:
            d_obj = self.model.d_train

        with sess.as_default():    
            #tf.variables_initializer(self.model.d_var_list).run()        
            num_iters = int(np.ceil(X.shape[0]/batch_size))
            for e in range(epoch):
                ii = 0
                for i in range(num_iters):
                    feed_dict,ii = self.model.update_feed_dict(X,Y,ii,batch_size)
                    _,loss = sess.run([d_obj,self.model.d_loss],feed_dict=feed_dict)
                    #if loss < -1.:
                    #    print('early stop satisfied',loss)
                    #    return 
                print('epoch {0} loss {1}'.format(e,loss)) 



