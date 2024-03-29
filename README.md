Studying Growth with Neural Cellular Automata
=======

We train simulated cells to grow into organisms by communicating with their neighbors (see [Neural Cellular Automata](https://distill.pub/2020/growing-ca/)). Then we use them to study patterns of growth found in nature.

Blog post: [greydanus.github.io/2022/05/24/studying-growth/](https://greydanus.github.io/2022/05/24/studying-growth/)

![overview.png](static/flowers.png)

Run in your browser:
--------
### [**Minimalist**](https://colab.research.google.com/drive/13wCM9OV2JR004zFvh7zPgUxrga8sU4d1)
A self-contained Neural Cellular Automata Implementation (150 lines of PyTorch). Reimplements all the methods described in [distill.pub/2020/growing-ca/](https://distill.pub/2020/growing-ca/) using the same hyperparameters. Written in PyTorch instead of TensorFlow.

![grow_gecko.png](static/grow_gecko.png)

### [**HD Flowers**](https://colab.research.google.com/drive/1TgGN5qjjH6MrMrTcStEkdHO-giEJ4bZr#scrollTo=k-2PCTfGI-pq)
Grow a 64x64 flower using the code in this GitHub repo. Scales up to 70x70 and hundreds of timesteps, which is nearly double the size of the model published in Distill. Flower options include `rose`, `marigold`, and `crocus` as shown in the lead image of this README.

![grow_rose.png](static/grow_rose.png)

### [**Nautilus**](https://colab.research.google.com/drive/1DUFL5glyej725r8VAYDZIFrWvpR6a6-0)
Grow a Nautilus shell. The neural CA learns to implement a fractal growth pattern which is mostly rotation and scale invariant. The technical term for this pattern is _[gnomonic growth](https://www.geogebra.org/m/waR6eVCQ)_.

![grow_nautilus.png](static/grow_nautilus.gif)

### [**Newt**](https://colab.research.google.com/drive/1fbakmrgkk1y-ZXamH1mKbN1tvkogNrWq)
We grow an image of a newt and then graft its eye onto its belly during development. We do this in homage to [Hans Spemann](https://en.wikipedia.org/wiki/Hans_Spemann) and his student Hilde, who won a Nobel Prize in 1935 for doing something similar with real newts.

![newt_graft.png](static/graft_newt.png)

### [**Bone**](https://colab.research.google.com/drive/1qQcztNsqyMLLMB00CVRxc0Pm7ipca0ww?usp=sharing)
In this experiment we simulate bone growth. Bone growth is interesting because it uses apoptosis (programmed cell death) in order to produce a hollow area in the center of the bone. We see something analogous happen in our model, with a circular tan frontier that gradually expands outwards until it reaches the size of the target image.

![grow_bone.png](static/grow_bone.png)

### [**Multiclass**](https://colab.research.google.com/drive/1vG7yjOHxejdk_YfvKhASanNs0YvKDO5-)
Train a neural CA that can grow from a seed pixel into one of three different flowers depending on initial value of the seed. From a dynamical systems perspective, we are training a model that has three different basins of attraction, one for each flower. The initial seed determines which basin the system ultimately converges to. The initial seed vs. the shared attractor dynamics are analogous to the DNA of a specific flower vs. the shared cellular dynamics across related flower species.

![grow_multiclass.png](static/grow_multiclass.gif)


Dependencies
--------
 * NumPy
 * SciPy
 * PyTorch