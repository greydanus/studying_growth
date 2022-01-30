Studying Growth with Neural Cellular Automata
=======

Blog: [greydanus.github.io/2021/05/07/studying-growth/](https://greydanus.github.io/2021/05/07/studying-growth/)

![overview.png](static/flowers.png)

Run in your browser:
--------
### [**Minimalist**](https://colab.research.google.com/drive/13wCM9OV2JR004zFvh7zPgUxrga8sU4d1)
A self-contained Neural Cellular Automata Implementation (150 lines of PyTorch)
 ![grow_gecko.png](static/grow_gecko.png)

### [**Rose**](https://colab.research.google.com/drive/1TgGN5qjjH6MrMrTcStEkdHO-giEJ4bZr#scrollTo=k-2PCTfGI-pq)
Grow a 64x64 rose
 ![grow_rose.png](static/grow_rose.png)

### [**Multiclass**](https://colab.research.google.com/drive/1vG7yjOHxejdk_YfvKhASanNs0YvKDO5-)
Grow different flowers with a single model, depending on the initial seed.
 ![grow_multiclass.png](static/grow_multiclass.gif)

### [**Nautilus**](https://colab.research.google.com/drive/1DUFL5glyej725r8VAYDZIFrWvpR6a6-0)
Grow a Nautilus shell. This is a fractal growth pattern with rotation and scale invariance.
 ![grow_nautilus.png](static/grow_nautilus.gif)

### [**Newt**](https://colab.research.google.com/drive/1fbakmrgkk1y-ZXamH1mKbN1tvkogNrWq)
Graft a newt's eye onto its belly in homage to [Lazzaro Spallanzani](https://en.wikipedia.org/wiki/Lazzaro_Spallanzani)
 ![newt_graft.png](static/newt_graft.png)

### [**Bone**](https://colab.research.google.com/drive/1qQcztNsqyMLLMB00CVRxc0Pm7ipca0ww?usp=sharing)
Grow a bone. Apoptosis (programmed cell death) occurs along the inside of the hollow bone, just as it does in real bones.
 ![grow_bone.png](static/grow_bone.png)

### [**Worm v1**](https://colab.research.google.com/drive/1wg-PKNwPA5yNzcuyBomZ6IT3Fx2xrewp) [Worm v2](https://colab.research.google.com/drive/1hE8Vxqsf_PZhSitQP1dSg-K022T3jOkK)
Grow a worm.
 ![grow_worm.png](static/grow_worm.png)

Overview
--------

We rewrite code from [Growing CA Distill](https://distill.pub/2020/growing-ca/) PyTorch and modify it in a few ways so as to model a few different patterns of growth seen in nature.


Dependencies
--------
 * NumPy
 * SciPy
 * PyTorch