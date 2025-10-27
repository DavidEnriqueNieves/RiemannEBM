# Follow the Energy, Find the Path: Riemannian Metrics from Energy-Based Models

This repository contains the code to reproduce the NeuRIPS 2025 article: [Follow the Energy, Find the Path: Riemannian Metrics from Energy-Based Models](https://arxiv.org/pdf/2505.18230).



## 1. Tutorial
The easiest way to get familiar with the idea of the paper is to play with the tutorial (i.e. the notebooks in the /tutorial folders). 
### 1.1 Training EMBs
The notebooks [trainEBM_CD.ipynb](./tutorial/trainEBM_CD.ipynb) and [trainEBM_DSM.ipynb](./tutorial/trainEBM_DSM.ipynb), explain how to train de EBM, on the Circular Mixture of Gaussians dataset with Contrastive Divergence and Denoising Score Mathching, resptively. Both methods could be used (and works similarly well with good hyperparameters). To ensure reproducibility, we included the network checkpoints (for the EBM trained with DSM).

## 2.2 Learning Geodesics
Once the EBM are trained, we can use them to derive a Riemannian Metric that defines the data manifold. In the notebook [geodesics.ipynb](./tutorial/geodesics.ipynb), we show i) how to learn the geodesics in such an EBM-derived Riemannian manifold, ii) we benchmarked the obtained geodesics with alternative Riemannian metric. This notebook requires the EBMs weights that are provided in the checkpoints (i.e [EBM_mixture1.pth](./tutorial/EBM_mixture1.pth) and [EBM_mixture2.pth](./tutorial/EBM_mixture1.pth)). Note that, if you wnat to test EBM trained with other method you need to change the checkpoints paths. This notebook reproduce the Figure 2 of the article:

