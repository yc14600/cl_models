from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from edward.models import Normal

from utils.train_util import *
from utils.model_util import *
from base_models.vae import VAE,Discriminant_VAE


ds = tf.contrib.distributions

class Continual_VAE(VAE):

    def update_inference(self):
        self.save_params()
        
        if self.bayes:
            pqz = Normal(loc=0.,scale=self.prior_std)
            self.inference.latent_vars[self.scope] = {pqz:self.qz}
            for w,b,qw,qb in zip(self.prev_eW,self.prev_eB,self.eW,self.eB):
                self.inference.latent_vars[self.scope].update({w:qw,b:qb})

            for w,b,qw,qb in zip(self.prev_dW,self.prev_dB,self.dW,self.dB):
                self.inference.latent_vars[self.scope].update({w:qw,b:qb})

            self.inference.latent_vars[self.scope].update({self.prev_sigma_w:self.sigma_w,self.prev_sigma_b:self.sigma_b})

        else:
            pqz = Normal(loc=self.prev_eH[-1],scale=tf.maximum(self.prev_z_sigma,1e-4))
            self.inference.latent_vars[self.scope] = {pqz:self.qz}
        self.inference.reinitialize(self)


class Continual_DVAE(Discriminant_VAE, Continual_VAE):

    def update_inference(self):
        Continual_VAE.update_inference(self)