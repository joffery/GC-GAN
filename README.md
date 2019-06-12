# GC-GAN

code for paper [Geometry-Contrastive GAN for Facial Expression Transfer](https://arxiv.org/abs/1802.01822)

## Overview
In this paper, we propose a Geometry-Contrastive Generative Adversarial Network (GC-GAN) for transferring continuous emotions across different subjects. Given an input face with certain emotion and a target facial expression from another subject, GC-GAN can generate an identity-preserving face with the target expression. Geometry information is introduced into cGANs as continuous conditions to guide the generation of facial expressions. In order to handle the misalignment across different subjects or emotions, contrastive learning is used to transform geometry manifold into an embedded semantic manifold of facial expressions. Therefore, the embedded geometry is injected into the latent space of GANs and control the emotion generation effectively. Experimental results demonstrate that our proposed method can be applied in facial expression transfer even there exist big differences in facial shapes and expressions between different subjects. 
<p align="center"><img src="imgs/fig_arch.pdf" width="600"></p>
### Files

``data_process.py``: create training and test pairs

``vaegan.py``: build model

``main.py``: parameters setting and train/test model

``ops.py``: some general funtions

``utils.py``: some specific functions

## Prerequisites

Python 3.6, Tensorflow 1.3.0

## Citation

If you find this code useful in your research, please consider citing:
```
@article{qiao2018geometry,
  title={Geometry-Contrastive GAN for Facial Expression Transfer},
  author={Qiao, Fengchun and Yao, Naiming and Jiao, Zirui and Li, Zhihao and Chen, Hui and Wang, Hongan},
  journal={arXiv preprint arXiv:1802.01822},
  year={2018}
}
```
```
@article{qiao2018emotional,
  title={Emotional facial expression transfer from a single image via generative adversarial nets},
  author={Qiao, Fengchun and Yao, Naiming and Jiao, Zirui and Li, Zhihao and Chen, Hui and Wang, Hongan},
  journal={Computer Animation and Virtual Worlds},
  volume={29},
  number={3-4},
  pages={e1819},
  year={2018}
}
```
