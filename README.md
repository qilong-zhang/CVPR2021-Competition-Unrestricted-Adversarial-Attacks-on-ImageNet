# Our 6th solution for [CVPR-2021 AIC-VI: Unrestricted Adversarial Attacks on ImageNet]

- Our method (Transfer-based attacks):https://arxiv.org/pdf/1904.02884.pdf
  - Dong et al. (TI-BIM) [[1]](#reference)
  - Zou et al. (DEM) [[3]](#reference), i.e., the variant of Xie et al. (DI-FGSM) [[2]](#reference)
  - Wang et al. (Pre-gradient I-FGSM) [[4]](#reference)
  - Ours works
    - PI-FGSM [[5]](#reference) & Our official [repo](https://github.com/qilong-zhang/Patch-wise-iterative-attack)
    - PI-FGSM++ [[6]](#reference) & Our official [repo](https://github.com/qilong-zhang/Targeted_Patch-wise-plusplus_iterative_attack)
    - 🚀**Staircase sign method (SSM)** [[7]](#reference) & Our official [repo]
      - By replacing the Sign method of DI-TI-FGSM with our SSM <a href="https://www.codecogs.com/eqnedit.php?latex=(\ell_{\infty}=8)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\ell_{\infty}=8)" title="(\ell_{\infty}=8)" /></a>, we significantly increase the score from ~8 to ~22 (full score is 100)  
  - Pre-process tricks: 
    - smooth the images firstly
    - resize to 256 instead of 500 before fed into DNNs at each iteration

- Setting
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\ell_{\infty}=20" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ell_{\infty}=20" title="\ell_{\infty}=20" /></a>
  - Iteration T=40
  - Step size: ~ 0.6
  - Batch size = 2
- Substitute models:
  - [DenseNet-121](https://arxiv.org/abs/1608.06993), [DenseNet-169](https://arxiv.org/abs/1608.06993), [ResNet-50](https://arxiv.org/abs/1512.03385), [ResNet-101](https://arxiv.org/abs/1512.03385), [VGG-19](https://arxiv.org/abs/1409.1556), [EfficientNet-b5](https://arxiv.org/abs/1905.11946)
- The cost of GPU memory: 12G on Titan xp (12G)
- Training time: ~40 hours (If you have more GPU, the training time can be significantly reduced)

## Implementation
- Requirement
  - Python 3.7
  - Pytorch 1.8.0
  - torchvision 0.9.0
  - pandas 1.1.3
  - matplotlib 3.3.4
  - scipy 1.5.4
  - timm 0.4.5

- Download the dataset from [here](https://tianchi.aliyun.com/competition/entrance/531853/information) (or select one of the following datasets)

  - Round 1: https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531853/imagenet_round1_210122.zip
  - Round 2: https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531853/imagenet_round2_210325.zip

- Unzip the downloaded dataset zip, and put 5K images into `"input_dir/images/"` file.

- Change the path into `code/`

  ```python
  cd code/
  ```

- Then run the code

  ```python
  python run.py
  ```

- Finally, the adversarial examples will be saved at `"output_dir/"`

## Result

![result2](https://github.com/qilong-zhang/CVPR2021-Competition-Unrestricted-Adversarial-Attacks-on-ImageNet/blob/main/result.png)

## Reference

[1] Yinpeng Dong, Tianyu Pang, Hang Su and Jun Zhu: [Evading defenses to transferable adversarial examples by translation-invariant attacks](https://arxiv.org/pdf/1904.02884.pdf), CVPR 2019

[2] Cihang Xie, Zhishuai Zhang, Yuyin Zhou, Song Bai, Jianyu Wang, Zhou Ren and Alan L Yuille: [Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/abs/1803.06978) CVPR 2019

[3] Junhua Zou, Zhisong Pan, Junyang Qiu, Xin Liu, Ting Rui and Wei Lin: [Improving the Transferability of Adversarial Examples with Resized-Diverse-Inputs,Diversity-Ensemble and Region Fitting](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670562.pdf), ECCV 2020

[4] Xiaosen Wang, Jiadong Lin, Han Hu, Jingdong Wang and Kun He: [Boosting Adversarial Transferability through Enhanced Momentum](https://arxiv.org/pdf/2103.10609.pdf), ArXiv 2021

[5] Lianli Gao, Qilong Zhang, Jingkuan Song, Xianglong Liu and Heng Tao Shen: [Patch-wise attack for fooling deep neural network](https://arxiv.org/abs/2007.06765), ECCV 2020

[6] Lianli Gao, Qilong Zhang, Jingkuan Song and Heng Tao Shen: [Patch-wise++ Perturbation for Adversarial Targeted Attacks](https://arxiv.org/abs/2012.15503), ArXiv 2020

[7] Lianli Gao, Qilong Zhang, Xiaosu Zhu, Jingkuan Song and Heng Tao Shen: [Staircase Sign Method for Boosting Adversarial Attacks](https://arxiv.org/abs/2012.15503), ArXiv 2021



