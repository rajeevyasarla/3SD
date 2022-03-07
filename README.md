# 3SD
Self-Supervised Saliency Detection method

We present a conceptually simple  self-supervised method for saliency detection. Our method generates and uses pseudo-ground truth labels for training. The generated pseudo-GT labels don't require any kind of human annotations (\emph{e.g.}, pixel-wise labels or weak labels like scribbles).
Recent works show that features extracted from classification tasks provide important saliency cues like structure and semantic information of salient objects in the image. Our method, called 3SD, exploits this idea by adding a branch for a self-supervised classification task in parallel with salient object detection, to obtain class activation maps (CAM maps). These CAM maps along with the edges of the input image are used to generate the pseudo-GT saliency maps to train our 3SD network. Specifically, we propose a contrastive learning-based training on multiple image patches for the classification task. We show the multi-patch classification with contrastive loss improves the quality of the CAM maps compared to naive classification on the entire image. Experiments on six benchmark datasets demonstrate that without any labels, our 3SD method outperforms all existing weakly supervised and unsupervised methods, and its performance is on par with the fully-supervised methods.

## Prerequisites:
1. Linux
2. Python 2 or 3
3. Pytorch version >=1.0
4. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 9.0)

## Dataset structure
1. download the dataset DUTS (http://saliencydetection.net/duts/)
2. download test datasets (https://drive.google.com/open?id=11rPRBzqxdRz0zHYax995uvzQsZmTR4A7)
```
   .
    ├── data 
    |   ├── train # Training  
    |   |   ├── train_img                 
    |   |   |   ├── Image1              
    |   |   |   ├── Image2           
    |   |   |   └── ......
    |   |   ├── train_edge          
    
    |   |   |   ├── Image1              
    |   |   |   ├── Image2           
    |   |   |   └── ......
    |   |   ├── train_label         
    |   |   |   ├── Image1              
    |   |   |   ├── Image2           
    |   |   |   └── ......
    |   └── test  # Testing
    |   |   ├── test_img                 
    |   |   |   ├── Image1              
    |   |   |   ├── Image2           
    |   |   |   └── ......
```

## BN

1. either run u2net_train.py (Training code will be released soon)
```
modify line1 in model/__init__.py as "from .u2net_transformer import U2NET"
For training:
python u2net_train.py
For testing:
python u2net_test.py
```
2. or run following commands to train and test (Training code will be released soon)
```
For training:
python u2net_train_fullysup_final.py
For testing:
python u2net_test_pseudo_dino_final.py
```
Note we used step 3 while reporting the numbers in the paper and presentation. corresponding pretrained model can be founded in the folder "saved_models/trans_syn_u2net"
## To train and test 3SD in self-supervised way:
1. command for training  (Training code will be released soon)
```
    python u2net_train_pseudo_dino_final.py
``` 
2. command for testing
```
    python u2net_test_pseudo_dino_final.py
```
3. download pretrained models for self-supervised 3sd [Dropbox](https://www.dropbox.com/sh/so5um1rfut30f03/AACSfTYBkJlExWjQ29Ovv7LAa?dl=0)

3SD results are available at [Google Drive](https://drive.google.com/file/d/1cVTZQmPitovx142pDMl3_l3KUAXNLjkB/view?usp=sharing)
## Evaluation
For computing metrics run the following command
```angular2html
python compute_metrics.py
```


## Acknowledgements
Thanks to Amir Sadeghian, Renliang Weng and Jinghao Shi, for having insightful and knowledgable discussions throught my internship

Thanks to authors of U2Net and DINO for sharing their code. Most of the code is borrowed from the U2Net and DINO methods
```
https://github.com/xuebinqin/U-2-Net
https://github.com/facebookresearch/dino
https://github.com/lucidrains/vit-pytorch/tree/main/vit_pytorch
```
