[![GitHub stars](https://img.shields.io/github/stars/luuleitner/deepMTJ?label=Stars&style=social)](https://github.com/luuleitner/deepMTJ)
[![GitHub forks](https://img.shields.io/github/forks/luuleitner/deepMTJ?label=Fork&style=social)](https://github.com/luuleitner/deepMTJ)
<br>
[![License: GPLv3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses)
[![GitHub contributors](https://img.shields.io/badge/Contributions-Welcome-brightgreen)](https://github.com/luuleitner/deepMTJ)
[![Twitter Follow](https://img.shields.io/twitter/follow/luuleitner?label=Follow&style=social)](https://twitter.com/luuleitner)

<p align="center">
<img src="https://github.com/luuleitner/deepMTJ/blob/master/Examples/deepMTJprediction_small.gif" height="220">
</p>

## #deepMTJ 
### Automatic muscle tendon junction tracking using deep learning

`#deepMTJ` is a time efficient tool (>7x faster than previous algorithms) for automatic tracking of the muscle tendon junction in ultrasound images based on deep learning. We built `#deepMTJ` to support clinical biomechanists and locomotion researchers with a reliable open-source tool for gait analysis.

<p align="center">
<img src="https://github.com/luuleitner/deepMTJ/blob/master/Examples/deepMTJ_Summary.png" width="80%">
</p>

For robust and precise predictions (i.e. match human labelling accuracy) of muscle tendon junctions in ultrasound images we employ convolutional neural networks with an attention mechanism. The provided networks were trained on a very large (training 6400 frames/validation 1600 frames/test 1147 frames) and highly diverse dataset of healthy and impaired subjects performing full range of motion and maximum contractions. Due to the clear separation of individual subjects into a training and test set, we demonstrate that our approach is capable of tracking the MTJ on previously unseen subjects.

This repository provides the complete `#deepMTJ` Python source code for annotation, training and prediction. With `#deepMTJ` you can: 
- train your own networks from scratch 
- use our trained networks to track the muscle tendon junction in your ultrasound video files
- to employ our trained networks for transfer learning tasks
- or to contribute to the `#deepMTJ` open-source initiative (get in touch via GitHub or [E-mail](mailto:christoph.leitner@tugraz.at?subject=[GitHub]#deepMTJ))



***"do, ut des" (lat.) ...if we could assist you with this code please cite our work:***
```
@inproceedings{LeitnerJarolim2020,
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
- `mtj_tracking/label` folder contains the video annotation tool: start `main.py`

### 2. TRAIN your own network with our backbones
- `mtj_tracking/train` folder contains the network training and evaluation: start `train.py` for `VGG-Attention-3` model and `train_resnet.py` for the `ResNet50` model.

### 3. PREDICT muscle tendon junctions in your own video data with our trained networks
- Trained networks (540 MB) can be downloaded from: [chriskross.org](https://drive.google.com/file/d/18-tX4SX1xabOf3J77Qt8BTNt412rYiK5/view?usp=sharing)

### Add-On's
- `examples` folder with all result plots and figures of the IEEE-EMBC 2020 publication in high resolution.

# Getting Started

This is one possible way to run `#deepMTJ` on your computer. We have used the [Anaconda](https://www.anaconda.com/) package manager and the [Pycharm](https://www.jetbrains.com/pycharm/) programming environment to develop, train and run our networks.

1. Download the `requirements.txt` File
2. Install [Anaconda](https://www.anaconda.com/) for Python v3.7
3. Create the `deepMTJ` virtual environment in your Anaconda terminal and install all necessary libraries (listed in the `requirements.txt` File) using the following code in the terminal window:

```
conda create --name deepMTJ python=3.7 --file requirements.txt
```

4. Install [Pycharm](https://www.jetbrains.com/pycharm/)
5. Clone this GitHub repository onto your local machine using [https://github.com/luuleitner/deepMTJ](https://github.com/luuleitner/deepMTJ)
6. Use `File - Settings - Project Interpreter` to install `#deepMTJ` as your Python project interpreter.
7. Run Python scripts
8. (Running TensorFlow on your GPU might require additional settings. DeepLabCut provids very useful information and additional links on the GPU / Cuda / Python / TensorFlow interfaces. You can find more [information here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md))

# License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](https://github.com/luuleitner/deepMTJ/blob/master/LICENSE) file for details

