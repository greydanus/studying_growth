Studying Growth with Neural CA
=======

Blog: [greydanus.github.io/2021/05/07/studying-growth/](https://greydanus.github.io/2021/05/07/studying-growth/)

![overview.png](static/flowers.png)

Run in your browser
--------
* Options
  * [[runs] **Minimalist**: a self-contained Neural Cellular Automata Implementation (150 lines of PyTorch)](https://colab.research.google.com/drive/13wCM9OV2JR004zFvh7zPgUxrga8sU4d1)

  ![grow_gecko.png](static/grow_gecko.png)
  * [[looks ok] **Rose**: use code from this repository to grow a rose](https://colab.research.google.com/drive/1TgGN5qjjH6MrMrTcStEkdHO-giEJ4bZr#scrollTo=k-2PCTfGI-pq)

  ![grow_rose.png](static/grow_rose.png)
  * [[looks ok] **Multiclass**: train a single model to grow different flowers (depending on the initial seed)](https://colab.research.google.com/drive/1vG7yjOHxejdk_YfvKhASanNs0YvKDO5-)

  ![grow_multiclass.png](static/grow_multiclass.gif)
  * [**Nautilus**: Growing a Nautilus shell (a gnomon growth pattern)](https://colab.research.google.com/drive/1DUFL5glyej725r8VAYDZIFrWvpR6a6-0)

  ![grow_nautilus.png](static/grow_nautilus.gif)
  * **Worms**: Growing a worm [(moving reference frame)](https://colab.research.google.com/drive/1wg-PKNwPA5yNzcuyBomZ6IT3Fx2xrewp) [(fixed reference frame)](https://colab.research.google.com/drive/1hE8Vxqsf_PZhSitQP1dSg-K022T3jOkK)

  ![grow_worm.png](static/grow_worm.png)
  * [[looks ok] **Newt**: Grafting a newt's eye onto its belly](https://colab.research.google.com/drive/1fbakmrgkk1y-ZXamH1mKbN1tvkogNrWq)
  ![newt_graft.png](static/newt_graft.png)
  * **Bone**: Osteoblasts and osteoclasts

  ![grow_bone.png](static/grow_bone.png)

Overview
--------

We rewrite code from [Growing CA Distill](https://distill.pub/2020/growing-ca/) PyTorch and modify it in a few ways so as to model a few different patterns of growth seen in nature.


Dependencies
--------
 * NumPy
 * SciPy
 * PyTorch