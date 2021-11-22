[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luuleitner/deepMTJ/blob/master/mtj_tracking/predict/mtj_tracking.ipynb)
![GitHub release (release name instead of tag name)](https://img.shields.io/github/v/release/luuleitner/deepMTJ?include_prereleases)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses)
<br>
[![GitHub stars](https://img.shields.io/github/stars/luuleitner/deepMTJ?label=Stars&style=social)](https://github.com/luuleitner/deepMTJ)
[![GitHub forks](https://img.shields.io/github/forks/luuleitner/deepMTJ?label=Fork&style=social)](https://github.com/luuleitner/deepMTJ)
[![Twitter Follow](https://img.shields.io/twitter/follow/luuleitner?label=Follow&style=social)](https://twitter.com/luuleitner)
<br>
[![GitHub contributors](https://img.shields.io/badge/Contributions-Welcome-brightgreen)](https://github.com/luuleitner/deepMTJ)

<p align="center">
<img src="https://github.com/luuleitner/deepMTJ/blob/master/data/v1.0_ieee_embc_2020/deepMTJprediction_small.gif" height="220">
</p>

# deepMTJ: Muscle-Tendon Junction Tracking in Ultrasound Images

`deepMTJ` is a machine learning approach for automatic tracking of muscle tendon junctions (MTJ) in ultrasound images. Our approach is based on a convolutional neural network trained to infer MTJ positions across a variety of ultrasound systems from different vendors, collected in independent laboratories from diverse observers, on distinct muscles and movements. We built `deepMTJ` to support clinical biomechanists and locomotion researchers with an open-source tool for gait analyses.

<p align="center">
<img src="https://github.com/luuleitner/deepMTJ/blob/master/data/v2.0_ieee_tbme_2021/3dvolume.jpg" width="40%">
</p>

## Introduction
This repository contains the full python source code of `deepMTJ` including:

- a google colab notebook to make inferences online
- input/output utilities to load data and save predictions
- model backbone and trained model weights to PREDICT, TRANSFER and LEARN
- a test dataset containing 1344 ultrasound images of muscle tendon junctions to benchmark future models
- a labeling tool to annotate ultrasound images (discontinued after v1.2)

## Predict muscle tendon junctions
For online predictions visit [deepmtj.org](https://deepmtj.org/) (running in beta with 200MB datasize limit) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luuleitner/deepMTJ/blob/master/mtj_tracking/predict/mtj_tracking.ipynb) (multiple and large file predictions)


## Publications
```
[1] @article{deepmtj2021a,
      title={A Human-Centered Machine-Learning Approach for Muscle-Tendon Junction Tracking in Ultrasound Images},
      year={2021}  
      author={Christoph Leitner and Robert Jarolim and Bernhard Englmair and Annika Kruse and Karen Andrea Lara Hernandez and Andreas Konrad and Eric Su and Jörg Schröttner and       Luke A. Kelly and Glen A. Lichtwark and  and Markus Tilp and Christian Baumgartner},
      journal = {IEEE Transactions on Biomedical Engineering},
      publisher={IEEE},
    }

[2] @misc{deepmtj2021b,
      title = {{deepMTJ test-set data}},
      doi = {10.6084/m9.figshare.16822978.v2}
    }
 
[3] @inproceedings{deepmtj2020, 
      title={Automatic Tracking of the Muscle Tendon Junction in Healthy and Impaired Subjects using Deep Learning*},   
      year={2020},  
      author={Leitner, Christoph and Jarolim, Robert and Konrad, Andreas and Kruse, Annika and Tilp, Markus and Schröttner, Jörg and Baumgartner, Christian},  
      booktitle={2020 42nd Annual International Conference of the IEEE Engineering in Medicine   Biology Society (EMBC)},   
      publisher={IEEE},
      pages={4770-4774},  
      doi={10.1109/EMBC44109.2020.9176145}
    }
```

## Repository Structure and Data

### 1. ANNOTATE your video data
`mtj_tracking/label` folder contains the video annotation tool (discontinued after v1.2). We have used the online labeling tool [Labelbox](https://labelbox.com/) in our recent publication.

### 2. TRAIN your network with our backbones or TRANSFER our trained network to your application
`mtj_tracking/train` folder contains the model backbone, the network training and evaluation

#### Download trained networks
Trained networks (375 MB) can be downloaded from our [google cloud storage](https://storage.googleapis.com/deepmtj/IEEEtbme_model_2021/2021_Unet_deepmtj_ieeetbme_model.tf). The provided dataset (`2021_Unet_deepmtj_ieeetbme_model.tf`) is licensed under a [Creative Commons Attribution 4.0 International License](https://github.com/luuleitner/deepMTJ/blob/master/license_dataset).

#### Download test dataset
To benchmark your model you can download the test dataset containing 1344 ultrasound images 

[![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

### 3. PREDICT muscle tendon junctions in your own video data with our trained networks
For multiple and large file predictions: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luuleitner/deepMTJ/blob/master/mtj_tracking/predict/mtj_tracking.ipynb)

Cloud based prediction services accessible via [deepmtj.org](https://deepmtj.org/) (running in beta with 200MB datasize limit)

### Add-on's
- `data` folder with additional result plots and figures in high resolution.

## Getting started with the code

This is one possible way to run `#deepMTJ` source code on your computer. 

1. Install [Anaconda](https://www.anaconda.com/) for Python v3.7 (on prompt choose to include python in your path)
2. Download the trained model from [google cloud storage](https://storage.googleapis.com/deepmtj/IEEEtbme_model_2021/2021_Unet_deepmtj_ieeetbme_model.tf).
3. Clone this GitHub repository to your local machine using [https://github.com/luuleitner/deepMTJ](https://github.com/luuleitner/deepMTJ).
4. Open the terminal and navigate to the downloaded repository (``cd <<repository path>>>``).
5. Create the `deepMTJ` virtual environment in your Anaconda terminal and install all necessary libraries (listed in the `requirements.txt` file) using the following code in the terminal window:

```
conda create --name deepMTJ python=3.7 --file requirements.txt
conda activate deepMTJ
```
6. Run the model:
```
python -m mtj_tracking.predict.main <<downloaded model path>> <<input path>> <<output path>>
```

# Acknowledgment

The experimental works and cloud deployments of the present study were supported by [Google Cloud](https://cloud.google.com/) infrastructure. 

# License

This program is free software and licensed under the GNU General Public License v3.0 - see the [LICENSE](https://github.com/luuleitner/deepMTJ/blob/master/LICENSE) file for details.

