# Studying Growth | 2021
# Sam Greydanus

import autograd
import autograd.numpy as np
import numpy as npo

import time
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt

from .utils import set_seed, to_pickle, from_pickle, ObjectView


# Step 1: Load some data

PADDING = 10
def get_dataset(image_name, k=PADDING):
  r = requests.get('https://raw.githubusercontent.com/greydanus/greydanus.github.io'\
                   '/master/files/nca_templates/{}.png'.format(image_name))
  img = PIL.Image.open(io.BytesIO(r.content))  # get image
  img = np.float32(img)/255.0                  # convert image to NumPy array
  img *= img[..., 3:]                          # premultiply RGB by alpha
  img = img.transpose(2,0,1)[None,...]         # axes are [N, C, H, W]
  return {'y': np.pad(img, ((0,0),(0,0),(k,k),(k,k)))}  # pad image


# Step 2: Implement Neural Cellular Automata as a PyTorch module
class CA(nn.Module):
  def __init__(self, state_dim=16, hidden_dim=128):
    super(CA, self).__init__()
    self.state_dim = state_dim
    self.update = nn.Sequential(
                      nn.Conv2d(state_dim, 3*state_dim, 3,padding=1, groups=state_dim, bias=False), # perceive
                      nn.Conv2d(3*state_dim, hidden_dim, 1),  # process perceptual inputs
                      nn.ReLU(),                              # nonlinearity
                      nn.Conv2d(hidden_dim, state_dim, 1)     # output a residual update
                    )
    self.update[-1].weight.data *= 0  # initial residual updates should be close to zero
    
    # First conv layer will use fixed Sobel filters to perceive neighbors
    identity = np.outer([0, 1, 0], [0, 1, 0])       # identity filter
    dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0      # Sobel x filter
    kernel = np.stack([identity, dx, dx.T], axis=0) # stack (identity, dx, dy) filters
    kernel = np.tile(kernel, [state_dim,1,1])       # tile over channel dimension
    self.update[0].weight.data[...] = torch.Tensor(kernel)[:,None,:,:]
    self.update[0].weight.requires_grad = False
  
  def forward(self, x, num_steps):
    alive_mask = lambda alpha: nn.functional.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1
    frames = []
    for i in range(num_steps):
      alive_mask_pre = alive_mask(alpha=x[:,3:4])
      update_mask = torch.rand(*x.shape, device=x.device) > args.dropout  # drop some updates to make asynchronous
      x = x + update_mask * self.update(x)                       # state update!
      x = x * alive_mask_pre * alive_mask(alpha=x[:,3:4])        # a cell is either living or dead
      x[..., 3, SEED_Y, SEED_X] = 1.  # this keeps the original seed from ever dying
      frames.append(x)
    return torch.stack(frames) # axes: [N, B, C, H, W] where N is # of steps


# Step 3: write a simple training loop that handles the three training modalities described in
#         "Growing NCA" (distill.pub/2020/growing-ca/): 1) grow 2) persist 3) regenerate

def normalize_grads(model):  # makes training more stable, especially early on
  for p in model.parameters():
      p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad

def train(model, args, data):
  model = model.to(args.device)  # put the model on GPU
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)
  scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

  target_rgba = torch.Tensor(data['y']).to(args.device)  # put the target image on GPU
  init_state = torch.zeros(args.batch_size, args.state_dim, *target_rgba.shape[-2:]).to(args.device)
  init_state[..., SEED_Y, SEED_X] = 1  # initially, there is just one cell
  pool = init_state[:1].repeat(args.pool_size,1,1,1).cpu()
  
  results = {'loss':[], 'tprev': [time.time()]}
  for step in range(args.total_steps+1):

    # prepare batch and run forward pass
    if args.pool_size > 0:  # draw CAs from pool (if we have one)
      pool_ixs = np.random.randint(args.pool_size, size=[args.batch_size])
      input_states = pool[pool_ixs].to(args.device)
    else:
      input_states = init_state

    states = model(input_states, np.random.randint(*args.num_steps))  # forward pass
    final_rgba = states[-1,:, :4]  # grab rgba channels of last frame

    # compute loss and run backward pass
    mses = (target_rgba.unsqueeze(0)-final_rgba).pow(2)
    batch_mses = mses.view(args.batch_size,-1).mean(-1)
    loss = batch_mses.mean()
    loss.backward() ; normalize_grads(model)
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
  arg_dict = {'state_dim': 32,         # first 4 are rgba, rest are latent
              'hidden_dim': 128,
              'num_steps': [64, 108],
              'pool_size': 1000,       # pool of persistent CAs (defaults are 0 and 1000)
              'batch_size': 8,
              'learning_rate': 2e-3,
              'milestones': [2500, 5000, 7500],   # lr scheduler milestones
              'gamma': 0.2,            # lr scheduler gamma
              'decay': 3e-5,
              'dropout': 0.2,          # fraction of communications that are dropped
              'print_every': 200,
              'total_steps': 10000,
              'device': 'cuda',        # options are {"cpu", "cuda"}
              'image_name': 'rose',
              'seed': 42}              # the meaning of life (for these little cells, at least)
  return arg_dict if as_dict else ObjectView(arg_dict)


# Step 5: optionally, train the model from scratch

if __name__ == '__main__':
  args = get_args()    # instantiate args
  set_seed(args.seed)  # make reproducible
  model = CA(args.state_dim, args.hidden_dim)  # instantiate the NCA model
  results = train(model, args, data=get_dataset(args.image_name))  # train model

  run_tag = '{}.pkl'.format(image_name)
  to_pickle(results, path=project_dir + run_tag)

