#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/yang-song/score_sde/blob/main/Score_SDE_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Preparation
# 
# 1. `git clone https://github.com/yang-song/score_sde.git`
# 
# 2. Install [required packages](https://github.com/yang-song/score_sde/blob/main/requirements.txt)
# 
# 3. `cd` into folder `score_sde`, launch a local jupyter server and connect to colab following [these instructions](https://research.google.com/colaboratory/local-runtimes.html)
# 
# 4. Download pre-trained [checkpoints](https://drive.google.com/drive/folders/1RAG8qpOTURkrqXKwdAR1d6cU9rwoQYnH?usp=sharing) and save them in the `exp` folder.

# In[1]:


#@title Autoload all modules
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import time
import jax.random as random

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import inspect
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import run_lib

import sampling
import losses as losses_lib
import utils
import evaluation
from models import up_or_down_sampling as stylegan_layers
import datasets
from models import wideresnet_noise_conditional
import sde_lib
import likelihood
import controllable_generation

from sampling import *
from sde_lib import *
from scipy import integrate

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'


# In[ ]:





# In[5]:


# @title Load the score-based model
time = 
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import cifar10_ncsnpp_continuous as configs
  ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24"
  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sampling_eps = 1e-5
elif sde.lower() == 'vpsde':
  from configs.vp import cifar10_ddpmpp_continuous as configs
  ckpt_filename = "exp/vp/cifar10_ddpmpp_continuous/checkpoint_8"  
  config = configs.get_config()
  sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3
elif sde.lower() == 'subvpsde':
  from configs.subvp import cifar10_ddpmpp_continuous as configs
  ckpt_filename = "exp/subvp/cifar10_ddpmpp_continuous/checkpoint_26"
  config = configs.get_config()
  sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3

batch_size =   64#@param {"type":"integer"}
local_batch_size = batch_size // jax.local_device_count()
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}
rng = jax.random.PRNGKey(random_seed)
rng, run_rng = jax.random.split(rng)
rng, model_rng = jax.random.split(rng)
score_model, init_model_state, initial_params = mutils.init_model(run_rng, config)
optimizer = losses_lib.get_optimizer(config).create(initial_params)

state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                      model_state=init_model_state,
                      ema_rate=config.model.ema_rate,
                      params_ema=initial_params,
                      rng=rng)  # pytype: disable=wrong-keyword-args
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
state = utils.load_training_state(ckpt_filename, state)


# In[6]:


#@title Visualization code

def image_grid(x):
  size = config.data.image_size
  channels = config.data.num_channels
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def show_samples(x):
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()


# # Predictor-Corrector sampling
# 
# Default settings:
# 
#  | dataset | SDE | predictor | corrector | snr | n_steps |
# |:----:|:----:|:----------:|:--------:|:---:|:----:|
# |CIFAR-10 | VE | ReverseDiffusionPredictor | LangevinCorrector | 0.16| 1|
# |CIFAR-10 | VP | EulerMaruyamaPredictor | None | - | - |
# |CIFAR-10 | subVP| EulerMaruyamaPredictor | None | - | - |
# | LSUN/CelebA-HQ/FFHQ 256px | VE | ReverseDiffusionPredictor | LangevinCorrector | 0.075 | 1 |
# | FFHQ 1024px | VE | ReverseDiffusionPredictor | LangevinCorrector | 0.15| 1 |
# 
# Check `probability_flow` to run PC sampling based on discretizing the probability flow ODE.

# In[8]:


#@title PC sampling
random_seed = 0 #@param {"type": "integer"}
rng = jax.random.PRNGKey(random_seed)
img_size = config.data.image_size
channels = config.data.num_channels
shape = (local_batch_size, img_size, img_size, channels)
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling.get_pc_sampler(sde, score_model, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps)


rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
step_rng = jnp.asarray(step_rng)
pstate = flax.jax_utils.replicate(state)

elapse = time.time()
x, n = sampling_fn(step_rng, pstate)
print(time.time() - elapse)

show_samples(x)


# In[14]:


sampling_fn(step_rng, pstate)


# # Probability flow ODE
# 
# With black-box ODE solvers, we can produce samples, compute likelihoods, and obtain a uniquely identifiable encoding of any data point.

# In[ ]:


#@title ODE sampling
random_seed = 0 #@param {"type": "integer"}
rng = jax.random.PRNGKey(random_seed)

shape = (local_batch_size, 32, 32, 3)
sampling_fn = sampling.get_ode_sampler(sde, 
                                      score_model, 
                                      shape, 
                                      inverse_scaler,                                       
                                      denoise=True, 
                                      eps=sampling_eps)
rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
step_rng = jnp.asarray(step_rng)
pstate = flax.jax_utils.replicate(state)
x, nfe = sampling_fn(step_rng, pstate)
show_samples(x)


# In[ ]:


#@title Likelihood computation

train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=True, evaluation=True)
eval_iter = iter(eval_ds)
bpds = []
likelihood_fn = likelihood.get_likelihood_fn(sde, 
                                             score_model, 
                                             inverse_scaler,                                             
                                             eps=1e-5)
pstate = flax.jax_utils.replicate(state)
random_seed = 0 #@param {"type": "integer"}
rng = jax.random.PRNGKey(random_seed)
for batch in eval_iter:
  img = batch['image']._numpy()
  rng, step_rng = jax.random.split(rng)    
  img = scaler(img)
  rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
  step_rng = jnp.asarray(step_rng)
  bpd, z, nfe = likelihood_fn(step_rng, pstate, img)
  bpds.extend(bpd)
  print(f"average bpd: {jnp.array(bpds).mean()}, NFE: {nfe}")


# In[ ]:


#@title Representations
train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)
eval_batch = next(iter(eval_ds))
eval_images = eval_batch['image']._numpy()
random_seed = 0 #@param {'type': 'integer'}
rng = jax.random.PRNGKey(random_seed)
shape = (local_batch_size, 32, 32, 3)

pstate = flax.jax_utils.replicate(state)
likelihood_fn = likelihood.get_likelihood_fn(sde, score_model, inverse_scaler, eps=1e-5)
sampling_fn = sampling.get_ode_sampler(sde, score_model, shape, inverse_scaler,
                                       denoise=True, eps=sampling_eps)

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.imshow(image_grid(eval_images))
plt.title('Original images')

rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
step_rng = jnp.asarray(step_rng)
_, latent_z, _ = likelihood_fn(step_rng, pstate, scaler(eval_images))

rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
step_rng = jnp.asarray(step_rng)
x, nfe = sampling_fn(step_rng, pstate, latent_z)

plt.subplot(1, 2, 2)
plt.axis('off')
plt.imshow(image_grid(x))
plt.title('Reconstructed images')


# # Controllable generation
# 
# Several demonstrations on how to solve inverse problems with our SDE framework.
# 
# Default settings
# 
#  | dataset | SDE | predictor | corrector | snr | n_steps |
# |:----:|:----:|:----------:|:--------:|:---:|:----:|
# |CIFAR-10 | VE | ReverseDiffusionPredictor | LangevinCorrector | 0.16| 1|
# |CIFAR-10 | VP | EulerMaruyamaPredictor | None | - | - |
# |CIFAR-10 | subVP| EulerMaruyamaPredictor | None | - | - |
# | LSUN/CelebA-HQ/FFHQ 256px | VE | ReverseDiffusionPredictor | LangevinCorrector | 0.075 | 1 |
# | FFHQ 1024px | VE | ReverseDiffusionPredictor | LangevinCorrector | 0.15| 1 |

# In[ ]:


#@title PC inpainting
train_ds, eval_ds, _ = datasets.get_dataset(config)
eval_iter = iter(eval_ds)
bpds = []

random_seed = 0 #@param {"type": "integer"}
rng = jax.random.PRNGKey(random_seed)

predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = AnnealedLangevinDynamics #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}

pc_inpainter = controllable_generation.get_pc_inpainter(sde, score_model,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        denoise=True)
batch = next(eval_iter)
img = batch['image']._numpy()
show_samples(img)
rng, step_rng = jax.random.split(rng)  
img = scaler(img)
mask = np.ones(img.shape)
mask[..., :, 16:, :] = 0.
mask = jnp.asarray(mask)
show_samples(inverse_scaler(img * mask))

rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
step_rng = jnp.asarray(step_rng)
pstate = flax.jax_utils.replicate(state)
x = pc_inpainter(step_rng, pstate, img, mask)
show_samples(x)


# In[ ]:


#@title PC colorizer
train_ds, eval_ds, _ = datasets.get_dataset(config)
eval_iter = iter(eval_ds)
bpds = []

random_seed = 0 #@param {"type": "integer"}
rng = jax.random.PRNGKey(random_seed)

predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}

batch = next(eval_iter)
img = batch['image']._numpy()
show_samples(img)
rng, step_rng = jax.random.split(rng)  
gray_scale_img = np.repeat(np.mean(img, axis=-1, keepdims=True), 3, -1)
show_samples(gray_scale_img)
gray_scale_img = scaler(gray_scale_img)
pc_colorizer = controllable_generation.get_pc_colorizer(
    sde, score_model, predictor, corrector, inverse_scaler,
    snr=snr, n_steps=n_steps, probability_flow=probability_flow,
    continuous=config.training.continuous, denoise=True
)
rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
step_rng = jnp.asarray(step_rng)
pstate = flax.jax_utils.replicate(state)
x = pc_colorizer(step_rng, pstate, gray_scale_img)

show_samples(x)


# ## Class-conditional generation
# 
# The classifier checkpoint can be downloaded from this [Google drive](https://drive.google.com/drive/folders/1FFPJMjENYSN-1CX_Ma-o1MTRz7gZ85N5?usp=sharing). Save it to folder `exp/classifier/wideresnet_noise_conditional`.

# In[ ]:


#@title Load the noise-conditional classifier

ckpt_path = "exp/classifier/wideresnet_noise_conditional/" #@param {"type": "string"}
classifier, classifier_params = mutils.create_classifier(rng, batch_size, ckpt_path)


# In[ ]:


#@title PC conditional sampling
random_seed = 0 #@param {"type": "integer"}
rng = jax.random.PRNGKey(random_seed)
shape = (local_batch_size, 32, 32, 3)
label = 1 #@param {"type":"integer"}
labels = jnp.ones((jax.local_device_count(), shape[0]), dtype=jnp.int32) * label
predictor = ReverseDiffusionPredictor #@param ["NonePredictor", "EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor"] {"type": "raw"}
corrector = LangevinCorrector #@param ["NoneCorrector", "LangevinCorrector", "AnnealedLangevinDynamics"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps =  2#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
pc_conditional_sampler = controllable_generation.get_pc_conditional_sampler(
                           sde, score_model, classifier, classifier_params,
                           shape, predictor, corrector, inverse_scaler, 
                           snr, n_steps=n_steps, 
                           probability_flow=probability_flow, 
                           continuous=config.training.continuous)
rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
step_rng = jnp.asarray(step_rng)
pstate = flax.jax_utils.replicate(state)
x = pc_conditional_sampler(step_rng, pstate, labels)
show_samples(x)

