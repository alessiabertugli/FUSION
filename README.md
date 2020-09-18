# Few-Shot Unsupervised Continual Learning through Meta-Examples
This repository contains the PyTorch code for paper:

**<a href="https://arxiv.org/abs/...">Few-Shot Unsupervised Continual Learning through Meta-Examples</a>**  
*<a href="https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=110">Alessia Bertugli</a>,
<a href="https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=111">Stefano Vincenzi</a>,
<a href="https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=38">Simone Calderara</a>,
<a href="http://disi.unitn.it/~passerini/">Andrea Passerini</a>*  

## Model architecture

![fusion-me - overview](images/model.pdf)

## Prerequisites

* Python >= 3.8
* PyTorch >= 1.5
* CUDA 10.0


## Datasets

* *<a href="https://github.com/brendenlake/omniglot">Omniglot*
* *<a href="http://www.image-net.org">Mini-ImageNet*
* *<a href="https://zenodo.org/record/3672132#.X2R9ay2w3pA">SlimageNet64*
* *<a href="https://www.cs.toronto.edu/~kriz/cifar.html">Cifar100*
* *<a href="http://www.vision.caltech.edu/visipedia/CUB-200.html">Cub*

## Embeddings
You can generate embeddings for Mini-ImageNet and SlimageNet64 using the code of *<a href="https://github.com/facebookresearch/deepcluster">DeepCluster</a>*
and for Omniglot the code of *<a href="https://github.com/brain-research/acai">ACAI* or download them from

## Best models

Available soon.

## Credits
* *<a href="https://github.com/kylehkhsu/cactus-maml">CACTUs-MAML: Clustering to Automatically Generate Tasks for Unsupervised Model-Agnostic Meta-Learning*
* *<a href="https://github.com/khurramjaved96/mrcl">Meta-Learning Representations for Continual Learning*
* *<a href="https://github.com/facebookresearch/deepcluster">Deep Clustering for Unsupervised Learning of Visual Features*


## Cite
If you have any questions,  please contact [alessia.bertugli@unitn.it](mailto:alessia.bertugli@unitn.it)  or [stefano.vincenzi@unimore.it](mailto:alessia.bertugli@unimore.it), or open an issue on this repo. 

If you find this repository useful for your research, please cite the following paper:
```bibtex
@article{Bertugli2020fusion-me,
  title={Few-Shot Unsupervised Continual Learning through Meta-Examples},
  author={Alessia Bertugli and Stefano Vincenzi and Simone Calderara and Andrea Passerini},
  journal={ArXiv},
  year={2020},
  volume={abs/2009.08107}
}
```
