<br />
<p align="center">
  <h3 align="center"><strong>Unsupervised Video Domain Adaptation with<br>Masked Pre-Training and Collaborative Self-Training</strong></h2>
</p>

<div align="center">

[![](https://img.shields.io/badge/CVPR%202024%20PDF-blue)](https://openaccess.thecvf.com/content/CVPR2024/papers/Reddy_Unsupervised_Video_Domain_Adaptation_with_Masked_Pre-Training_and_Collaborative_Self-Training_CVPR_2024_paper.pdf)
[![](https://img.shields.io/badge/Supplementary-7DCBFF)](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Reddy_Unsupervised_Video_Domain_CVPR_2024_supplemental.pdf)
[![](https://img.shields.io/badge/arXiv-b31b1b)](https://arxiv.org/abs/2312.02914)
[![](https://img.shields.io/badge/Video-ff0000)](https://www.youtube.com/watch?v=dDjCVnkuhGg)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/reddyav1/unite/blob/main/LICENSE)
[![](https://img.shields.io/badge/Bibtex-CB8CEA)](#citation)


</div>


## Getting Started

### Data Preparation

### Download Checkpoints

The student model in UNITE is initialized from the Unmasked Teacher (UMT) checkpoint pre-trained on Kinetics-710 (ViT-B/16). You can find a link to this checkpoint in the [UMT repository](https://github.com/OpenGVLab/unmasked_teacher/blob/main/single_modality/MODEL_ZOO.md), or can directly download it from [here](https://www.cis.jhu.edu/~areddy/unite_cvpr24/checkpoints/b16_ptk710_f8_res224.pth).

## Running UNITE

Each of the three stages in UNITE is separated into its own Python file. We provide bash scripts that will launch distributed training for each stage (`stage<X>.sh`).

## Acknowledgement

This repository was built based on [Unmasked Teacher](https://github.com/OpenGVLab/unmasked_teacher).

<a name="citation"></a>
## Citation
```
@inproceedings{reddy2024unite,
  title={Unsupervised Video Domain Adaptation with Masked Pre-Training and Collaborative Self-Training},
  author={Reddy, Arun and Paul, William and Rivera, Corban and Shah, Ketul and de Melo, Celso M and Chellappa, Rama},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={18919--18929},
  year={2024}
}
```
