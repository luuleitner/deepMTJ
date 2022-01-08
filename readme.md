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

`deepMTJ` is a machine learning approach for automatically tracking of muscle-tendon junctions (MTJ) in ultrasound images. Our method is based on a convolutional neural network trained to infer MTJ positions across various ultrasound systems from different vendors, collected in independent laboratories from diverse observers, on distinct muscles and movements. We built `deepMTJ` to support clinical biomechanists and locomotion researchers with an open-source tool for gait analyses.

<p align="center">
<img src="https://github.com/luuleitner/deepMTJ/blob/master/data/v2.0_ieee_tbme_2021/3dvolume.jpg" width="40%">
</p>

## Introduction
This repository contains the full python source code of `deepMTJ` including:

- a google colab notebook to make inferences online
- input/output utilities to load data and save predictions
- the model backbone and trained model weights to **PREDICT**, **TRANSFER** and **LEARN**
- a diverse test dataset annotated by 4 specialists (2-10y experience) and containing 1344 ultrasound images of muscle tendon junctions to benchmark future models
- a labeling tool to annotate ultrasound images (discontinued after v1.2)

## Predict muscle tendon junctions
- For multiple and large file predictions [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luuleitner/deepMTJ/blob/master/mtj_tracking/predict/mtj_tracking.ipynb)

- Cloud based predictions are accessible via [deepmtj.org](https://deepmtj.org/). These services run in beta and have a datasize limitation of 200 MB.

- Or use the python code to predict MTJs. Have a look at the [nitty-gritty guide](#nittygritty) to the repository. 


## <a name="citation_1"></a> Publications
```
[1] @article{deepmtj2021a,
      title={A Human-Centered Machine-Learning Approach for Muscle-Tendon Junction Tracking in Ultrasound Images},
      year={2021}  
      author={Leitner, Christoph and Jarolim, Robert and Englmair, Bernhard and Kruse, Annika and Hernandez, Karen Andrea Lara and Konrad, Andreas and Su, Eric and Schröttner,
      Jörg and Kelly, Luke A. and Lichtwark, Glen A. and Tilp, Markus and Baumgartner, Christian},
      journal = {IEEE Transactions on Biomedical Engineering},
      publisher={IEEE},
      doi={10.1109/TBME.2021.3130548}
    }

[2] @misc{deepmtj2021b,
      title = {{deepMTJ test-set data}},
      year={2021},  
      author={Leitner, Christoph and Jarolim, Robert and Englmair, Bernhard and Kruse, Annika and Hernandez, Karen Andrea Lara and Konrad, Andreas and Su, Eric and Schröttner, 
      Jörg and Kelly, Luke A. and Lichtwark, Glen A. and Tilp, Markus and Baumgartner, Christian},
      doi = {10.6084/m9.figshare.16822978.v2}
    }
 
[3] @inproceedings{deepmtj2020, 
      title={Automatic Tracking of the Muscle Tendon Junction in Healthy and Impaired Subjects using Deep Learning*},   
      year={2020},  
      author={Leitner, Christoph and Jarolim, Robert and Konrad, Andreas and Kruse, Annika and Tilp, Markus and Schröttner, Jörg and Baumgartner, Christian},  
      booktitle={2020 42nd Annual International Conference of the IEEE Engineering in Medicine Biology Society (EMBC)},   
      publisher={IEEE},
      pages={4770-4774},  
      doi={10.1109/EMBC44109.2020.9176145}
    }
```

## Repository Structure and Data

### 1. ANNOTATE your video data
`mtj_tracking/label` folder contains the video annotation tool (discontinued after v1.2). We have used the online labeling tool [Labelbox](https://labelbox.com/) in our recent publication [Leitner *et al.* 2021a](#citation_1).

### 2. TRAIN your network with our backbones or TRANSFER our trained network to your application
`mtj_tracking/train` folder contains the model backbone, the network training and evaluation

#### Download trained networks
This repository comprises the [deepMTJ tensorflow model](https://osf.io/wgy4d/). This U-Net model with attention mechanism was trained on a total of 66864 annotated ultrasound images of the muscle-tendon junction. The training dataset, covers 3 functional movements (isometric maximum voluntary contractions, passive torque movements, running), 2 muscles (Lateral Gastrocnemius, Medial Gastrocnemius), collected on 123 healthy and 38 impaired subjects with 3 different ultrasound systems (Aixplorer V6, Esaote MyLab60, Telemed ArtUs).

#### Download test dataset
You can [download the test dataset](https://osf.io/wgy4d/) [[2]](#citation_1) (464 MB) used in [Leitner *et al.* 2021a](#citation_1) (e.g., to benchmark your own model,...). This dataset comprises 1344 images of muscle-tendon junctions recorded with 3 ultrasound imaging systems (Aixplorer V6, Esaote MyLab60, Telemed ArtUs), on 2 muscles (Lateral Gastrocnemius, Medial Gastrocnemius), and 2 movements (isometric maximum voluntary contractions, passive torque movements). We have included the ground truth labels for each image. These reference labels are the computed mean from 4 specialist labels. Specialist annotators had 2-10 years of experience in biomechanical and clinical research investigating muscles and tendons in 2-9 ultrasound studies in the past 2 years.

The provided dataset and models are licensed under a [Creative Commons Attribution 4.0 International License](https://github.com/luuleitner/deepMTJ/blob/master/license_datasets).

[![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

### 3. PREDICT muscle tendon junctions in your own video data with our trained networks
- For multiple and large file predictions [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luuleitner/deepMTJ/blob/master/mtj_tracking/predict/mtj_tracking.ipynb)

- Cloud based predictions are accessible via [deepmtj.org](https://deepmtj.org/). These services run in beta and have a datasize limitation of 200 MB.

### Add-on's
- `data` folder with additional result plots and figures in high resolution.

## <a name="nittygritty"></a> Getting started with the code

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

