# Studying Growth | 2021
# Sam Greydanus

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import time, PIL.Image, io, requests, copy

from .utils import set_seed, to_pickle, from_pickle, ObjectView, make_circle_masks


# Step 1: Load some data

def get_dataset(image_name, k):
  r = requests.get('https://raw.githubusercontent.com/greydanus/greydanus.github.io'\
                   '/master/files/nca_templates/{}.png'.format(image_name))
  img = PIL.Image.open(io.BytesIO(r.content))  # get image
  img = np.float32(img)/255.0                  # convert image to NumPy array
  img *= img[..., 3:]                          # premultiply RGB by alpha
  img = img.transpose(2,0,1)[None,...]         # axes are [N, C, H, W]
  return {'y': np.pad(img, ((0,0),(0,0),(k,k),(k,k)))}  # pad image


# Step 2: Implement Neural Cellular Automata as a PyTorch module
class CA(nn.Module):
  def __init__(self, state_dim=16, hidden_dim=128, dropout=0):
    super(CA, self).__init__()
    self.state_dim = state_dim
    self.dropout = dropout
    self.update = nn.Sequential(
                      nn.Conv2d(state_dim, 3*state_dim, 3,padding=1, groups=state_dim, bias=False), # perceive
                      nn.Conv2d(3*state_dim, hidden_dim, 1),  # process perceptual inputs
                      nn.ReLU(),                              # nonlinearity
                      nn.Conv2d(hidden_dim, state_dim, 1)     # output a residual update
                    )
    self.update[-1].weight.data *= 0  # initial residual updates should be close to zero
    
    # First conv layer will use fixed Sobel filters to perceive neighbors
    identity = np.outer([0, 1, 0], [0, 1, 0])        # identity filter
    dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0       # Sobel x filter
    kernel = np.stack([identity, dx, dx.T], axis=0)  # stack (identity, dx, dy) filters
    kernel = np.tile(kernel, [state_dim,1,1])        # tile over channel dimension
    self.update[0].weight.data[...] = torch.Tensor(kernel)[:,None,:,:]
    self.update[0].weight.requires_grad = False
  
  def forward(self, x, num_steps, seed_loc=None):
    alive_mask = lambda alpha: nn.functional.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1
    frames = []
    for i in range(num_steps):
      alive_mask_pre = alive_mask(alpha=x[:,3:4])
      update_mask = torch.rand(*x.shape, device=x.device) > self.dropout  # drop some updates, make asynchronous
      x = x + update_mask * self.update(x)                       # state update!
      x = x * alive_mask_pre * alive_mask(alpha=x[:,3:4])        # a cell is either living or dead
      if seed_loc is not None:
        x[..., 3, seed_loc[0], seed_loc[1]] = 1.  # this keeps the original seed from dying (very important!)
      frames.append(x)
    return torch.stack(frames) # axes: [N, B, C, H, W] where N is # of steps


# Step 3: write a simple training loop that handles the three training modalities described in
#         "Growing NCA" (distill.pub/2020/growing-ca/): 1) grow 2) persist 3) regenerate

def get_seed_location(target_img, image_name, padding=10):
  side = target_img.shape[-2] - 2 * padding
  seed_locs = {'rose': (0.6,0.8), 'daffodil': (0.78, 0.8), 'crocus': (0.42,0.83),
               'marigold': (0.49, 0.83), 'sworm': (0.5,0.5), 'multiclass': (0.5, 0.75)}
  loc = seed_locs[image_name]
  return (padding + int(loc[1]*side), padding + int(loc[0]*side))  # set location of seed

def normalize_grads(model):  # makes training more stable, especially early on
  for p in model.parameters():
      p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad

def train(model, args, data):
  model = model.to(args.device)  # put the model on GPU
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)
  scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

  target_rgba = torch.Tensor(data['y']).to(args.device)  # put the target image on GPU
  init_state = torch.zeros(args.batch_size, args.state_dim, *target_rgba.shape[-2:]).to(args.device)
  init_state[...,3:, args.seed_loc[0], args.seed_loc[1]] = 1  # initially, there is just one cell
  pool = init_state[:1].repeat(args.pool_size,1,1,1).cpu()
  
  results = {'loss':[], 'tprev': [time.time()]}
  for step in range(args.total_steps+1):

    # prepare batch and run forward pass
    if args.pool_size > 0:  # draw CAs from pool (if we have one)
      pool_ixs = np.random.randint(args.pool_size, size=[args.batch_size])
      input_states = pool[pool_ixs].to(args.device)
    else:
      input_states = init_state
    if args.perturb_n > 0:  # perturb CAs (if desired)
      perturb = make_circle_masks(args.perturb_n, *init_state.shape[-2:])[:, None, ...]
      input_states[-args.perturb_n:] *= perturb.to(args.device)

    states = model(input_states, np.random.randint(*args.num_steps), args.seed_loc)  # forward pass
    final_rgba = states[-1,:, :4]  # grab rgba channels of last frame

    # compute loss and run backward pass
    mses = (target_rgba.unsqueeze(0)-final_rgba).pow(2)
    batch_mses = mses.view(args.batch_size,-1).mean(-1)
    loss = batch_mses.mean() ; loss.backward() ; normalize_grads(model)
    optimizer.step() ; optimizer.zero_grad() ; scheduler.step()

    # update the pool (if we have one)
    if args.pool_size > 0:
      ixs_to_replace = torch.argsort(batch_mses, descending=True)[:int(.15*args.batch_size)]
      # ixs_to_replace = np.random.randint(args.batch_size, size=int(.15*args.batch_size))
      final_states = states[-1].detach()
      final_states[ixs_to_replace] = init_state[:1]
      pool[pool_ixs] = final_states.cpu()
      del batch_mses

    # bookkeeping and logging
    results['loss'].append(loss.item())
    if step % args.print_every == 0:
      print('step {}, dt {:.3f}s, loss {:.2e}, log10(loss) {:.2f}'\
          .format(step, time.time()-results['tprev'][-1], loss.item(), np.log10(loss.item())))
      results['tprev'].append(time.time())

  results['final_model'] = copy.deepcopy(model.cpu())
  return results


# Step 4: organize hyperparameters & keep them in a single location

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d
def get_args(as_dict=False):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  arg_dict = {'state_dim': 32,         # first 4 are rgba, rest are latent
              'hidden_dim': 128,
              'num_steps': [64, 108],
              'pool_size': 1000,       # pool of persistent CAs (defaults are 0 and 1000)
              'perturb_n': 0,
              'batch_size': 8,
              'learning_rate': 2e-3,
              'milestones': [3000, 6000, 9000],   # lr scheduler milestones
              'gamma': 0.2,           # lr scheduler gamma
              'decay': 3e-5,
              'dropout': 0.2,          # fraction of communications that are dropped
              'print_every': 200,
              'total_steps': 12000,
              'device': device,        # options are {"cpu", "cuda"}
              'padding': 10,
              'image_name': 'rose',
              'project_dir': './',
              'seed': 42}              # the meaning of life (for these little cells, at least)
  print("Using: ", device)
  return arg_dict if as_dict else ObjectView(arg_dict)


# Step 5: optionally, train the model from scratch

if __name__ == '__main__':
  '''Has to be run on a GPU, otherwise it's super slow.'''
  args = get_args() ; set_seed(args.seed)                    # instantiate args & make reproducible
  model = CA(args.state_dim, args.hidden_dim, args.dropout)  # instantiate the NCA model
  data = get_dataset(args.image_name, args.padding)
  args.seed_loc = get_seed_location(data['y'], args.image_name, args.padding)  # not ideal, we but have to do this
  results = train(model, args, data)                         # train model

  to_pickle(results, path=args.project_dir + '{}.pkl'.format(args.image_name))


