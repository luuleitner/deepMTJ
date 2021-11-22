[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luuleitner/deepMTJ/blob/master/mtj_tracking/predict/mtj_tracking.ipynb)
![GitHub release (release name instead of tag name)](https://img.shields.io/github/v/release/luuleitner/deepMTJ?include_prereleases)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses)
[![GitHub contributors](https://img.shields.io/badge/Contributions-Welcome-brightgreen)](https://github.com/luuleitner/deepMTJ)
<br>
[![GitHub stars](https://img.shields.io/github/stars/luuleitner/deepMTJ?label=Stars&style=social)](https://github.com/luuleitner/deepMTJ)
[![GitHub forks](https://img.shields.io/github/forks/luuleitner/deepMTJ?label=Fork&style=social)](https://github.com/luuleitner/deepMTJ)
[![Twitter Follow](https://img.shields.io/twitter/follow/luuleitner?label=Follow&style=social)](https://twitter.com/luuleitner)


<p align="center">
<img src="https://github.com/luuleitner/deepMTJ/blob/master/Examples/deepMTJprediction_small.gif" height="220">
</p>

## deepMTJ 
### Machine-Learning Approach for Muscle-Tendon Junction Tracking in Ultrasound Images

`#deepMTJ` is a tool based on deep learning for automatic tracking of the muscle tendon junction (MTJ) in ultrasound images. We built `#deepMTJ` to support clinical biomechanists and locomotion researchers with an open-source tool for gait analysis.

<p align="center">
<img src="https://github.com/luuleitner/deepMTJ/blob/master/Examples/deepMTJ_Summary.png" width="80%">
</p>

We employ convolutional neural networks with an attention mechanism. The provided networks were trained on a large (training 6400 frames/validation 1600 frames/test 1147 frames) and diverse dataset of healthy and impaired subjects performing full range of motion and maximum contractions.

This repository provides the complete `deepMTJ` python source code for annotation, training and prediction. With `#deepMTJ` you can: 
- train your own networks from scratch 
- use our trained networks to track the muscle tendon junction in your ultrasound video files
- employ our trained networks for transfer learning tasks


### Publications
```
@article{deepmtj2021,
   title={A Human-Centered Machine-Learning Approach for Muscle-Tendon Junction Tracking in Ultrasound Images},
   author={Christoph Leitner and Robert Jarolim and Bernhard Englmair and Annika Kruse and Karen Andrea Lara Hernandez and Andreas Konrad and Eric Su and Jörg Schröttner and        Luke A. Kelly and Glen A. Lichtwark and  and Markus Tilp and Christian Baumgartner},
   booktitle={IEEE Transactions on Biomedical Engineering},
   publisher={IEEE},
   year={2021}  
}

@inproceedings{deepmtj2020,
   title={Automatic Tracking of the Muscle Tendon Junction in Healthy and Impaired Subjects using Deep Learning},
   author={Christoph Leitner and Robert Jarolim and Andreas Konrad and Annika Kruse and Markus Tilp and Christian Baumgartner},
   booktitle={in Proc. 42nd Conferences of the IEEE Engineering in Medicine and Biology Society},
   venue={Montreal,Canada},
   publisher={IEEE},
   month=07,
   year={2020}  
}
```

# Repository Structure

The `#deepMTJ` repository contains:

### 1. ANNOTATE your video data
`mtj_tracking/label` folder contains the video annotation tool: start `main.py`

### 2. TRAIN your own network with our backbones
`mtj_tracking/train` folder contains the network training and evaluation: start `train.py` for `VGG-Attention-3` model and `train_resnet.py` for the `ResNet50` model.

### 3. PREDICT muscle tendon junctions in your own video data with our trained networks
The `mtj_tracking/predict` folder contains an easy to use prediction script (minimal Python knowledge needed to get it running). G to `main.py`, add your data paths and start your predictions...

This script reads your provided AVI-Video Files and returns the annotated frames (downscaled AVI-videos) as well as the X,Y-coordinates of the muscle tendon junction (csv-File). 

#### Trained networks
Trained networks (540 MB) can be downloaded from: [deepmtj.org](https://drive.google.com/file/d/11aTDxaINoAnsefEURpZQ1aZzhz6ikS5Z/view?usp=sharing). The provided datasets (`ResNet-50.hdf5`, `VGG-16.hdf5`, `VGG-Attention-2.hdf5`, `VGG-Attention-3.hdf5`) are licensed under a [Creative Commons Attribution 4.0 International License](https://github.com/luuleitner/deepMTJ/blob/master/LICENSE_Datasets).

[![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

### Add-On's
- `examples` folder with all result plots and figures of the IEEE-EMBC 2020 publication in high resolution.

# Getting Started

This is one possible way to run `#deepMTJ` on your computer. 

1. Install [Anaconda](https://www.anaconda.com/) for Python v3.7 (on prompt choose to include python in your path)
2. Download the trained [VGG-Attention-3 model](https://drive.google.com/file/d/11aTDxaINoAnsefEURpZQ1aZzhz6ikS5Z/view?usp=sharing).
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


# License

This program is free software and licensed under the GNU General Public License v3.0 - see the [LICENSE](https://github.com/luuleitner/deepMTJ/blob/master/LICENSE) file for details.

