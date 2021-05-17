

## Joint Learning Mechanism to Compress and Classify Radiological Images

[![Conference](http://img.shields.io/badge/ICVGIP-2018-4b44ce.svg)](https://cvit.iiit.ac.in/icvgip18/) [![Paper](http://img.shields.io/badge/paper-dl.acm.10.1145/3293353.3293408-B31B1B.svg)](https://dl.acm.org/doi/abs/10.1145/3293353.3293408) [![Slides](http://img.shields.io/badge/slides-pdf-orange.svg)](https://github.com/ekagra-ranjan/AE-CNN/raw/master/Paper55.pptx) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/jointly-learning-convolutional/pneumonia-detection-on-chestx-ray14)](https://paperswithcode.com/sota/pneumonia-detection-on-chestx-ray14?p=jointly-learning-convolutional)

Source code for [ICVGIP 2018](https://cvit.iiit.ac.in/icvgip18/) paper: [**Jointly Learning Convolutional Representations to Compress Radiological Images and Classify Thoracic Diseases in the Compressed Domain**](https://dl.acm.org/doi/abs/10.1145/3293353.3293408)

<br>
<br>
<br>

![](https://github.com/ekagra-ranjan/AE-CNN/blob/master/ae-cnn-final.png)


**Overview of AE-CNN:** *Our proposed framework consists of three main blocks namely encoder, decoder, and classifier. The figure shows the autoencoder based convolutional neural network (AE-CNN) model for disease classification. Here, autoencoder reduces the spatial dimension of the imput image of size 1024 × 1024. The encoder produces a latent code tensor of size 224 × 224 and decoder reconstructs back the image. This latent code tensor is passed through a CNN classifier for classifying the chest x-rays. The final loss is the weighted sum of the resconstruction loss by decoder and classification loss by the CNN classifier.*

## Results

![Table](https://github.com/ekagra-ranjan/AE-CNN/blob/master/AE-CNN-comparison-table.png)


## Citation
Please cite the following paper if you found it useful in your work:
```
@inproceedings{10.1145/3293353.3293408,
author = {Ranjan, Ekagra and Paul, Soumava and Kapoor, Siddharth and Kar, Aupendu and Sethuraman, Ramanathan and Sheet, Debdoot},
title = {Jointly Learning Convolutional Representations to Compress Radiological Images and Classify Thoracic Diseases in the Compressed Domain},
year = {2018},
isbn = {9781450366151},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3293353.3293408},
doi = {10.1145/3293353.3293408},
booktitle = {Proceedings of the 11th Indian Conference on Computer Vision, Graphics and Image Processing},
articleno = {55},
numpages = {8},
keywords = {compression, X-Ray classification, Convolutional autoencoder},
location = {Hyderabad, India},
series = {ICVGIP 2018}
}
```


## Acknowledgement: 
We would like to thank [zoozog](https://github.com/zoogzog/) and [arnoweng](https://github.com/arnoweng/) for open-sourcing their repos which served as the starting point for our work. Their repos can be found [here](https://github.com/zoogzog/chexnet) and [here](https://github.com/arnoweng/CheXNet) respectively. 
