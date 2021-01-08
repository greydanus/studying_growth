# Studying Growth | 2021
# Sam Greydanus

import numpy as np
import torch
import random
import pickle
import matplotlib.pyplot as plt
from celluloid import Camera


### Generic training & IO tools

def set_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def to_pickle(thing, path):  # save something
  with open(path, 'wb') as handle:
      pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path):  # load something
  thing = None
  with open(path, 'rb') as handle:
      thing = pickle.load(handle)
  return thing

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


### Plotting & visualization tools

def plot_img(img, fig=None, k=0, dpi=100):
  if fig is None:
    fig = plt.figure(dpi=dpi)
  if k > 0:
    img = img[k:-k,k:-k]
  plt.imshow(img) ; plt.axis('off')
  plt.tight_layout()
  return fig

def zoom(x, k=15):
  return x[...,k:-k,k:-k] if k>0 else x

def to_rgb(x):
  rgb, a = x[..., :3,:,:], x[..., 3:4,:,:].clip(0,1)
  return (1.0-a+rgb).clip(0,1)  # assume rgb premultiplied by alpha

def make_video(frames, path, interval=60, **kwargs):
  fig = plt.figure(figsize=[3.25, 3.25], dpi=100)
  camera = Camera(fig)
  for i in range(len(frames)):
    fig = plot_img(frames[i], fig=fig, dpi=80)
    camera.snap()
  anim = camera.animate(blit=True, interval=interval, **kwargs)
  anim.save(path)
  plt.close()