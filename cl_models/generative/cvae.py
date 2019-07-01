from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import scipy as sp
import numpy as np
import pandas as pd
import tensorflow as tf
import six
import sys
import os
import time

from edward.models import Normal,TransformedDistribution,Gamma

from utils.train_util import *
from utils.model_util import *
from slvi.slvi.slvi import Hierarchy_SLVI
from base_models.vae import VAE,Discriminant_VAE


ds = tf.contrib.distributions

class Continual_VAE(VAE):

    def update_inference(self):
        self.save_params()
        pqz = Normal(loc=self.prev_eH[-1],scale=tf.maximum(self.prev_z_sigma,1e-4))
        self.inference.latent_vars[self.scope] = {pqz:self.qz}
        self.inference.reinitialize(self)


class Continual_DVAE(Discriminant_VAE, Continual_VAE):

    def update_inference(self):
        Continual_VAE.update_inference(self)