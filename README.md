[![GitHub stars](https://img.shields.io/github/stars/luuleitner/deepMTJ?label=Stars&style=social)](https://github.com/luuleitner/deepMTJ)
[![GitHub forks](https://img.shields.io/github/forks/luuleitner/deepMTJ?label=Fork&style=social)](https://github.com/luuleitner/deepMTJ)
<br>
[![License: GPLv3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses)
[![GitHub contributors](https://img.shields.io/badge/Contributions-Welcome-brightgreen)](https://github.com/luuleitner/deepMTJ)
[![Twitter Follow](https://img.shields.io/twitter/follow/luuleitner?label=Follow&style=social)](https://twitter.com/luuleitner)

<p align="center">
<img src="https://github.com/luuleitner/deepMTJ/blob/master/Examples/deepMTJ_Summary.png" width="80%">
</p>

## #deepMTJ: Automatic muscle tendon junction tracking using deep learning

In this repository we provide the `#deepMTJ` Python code for annotation, training and evaluation. If you wish to contribute please `fork` this repository and open a `pull request` or simply get in touch via [E-mail](mailto:christoph.leitner@tugraz.at?subject=[GitHub]#deepMTJ). 

You can download our trained networks (540 MB) via [chriskross.org](https://drive.google.com/file/d/18-tX4SX1xabOf3J77Qt8BTNt412rYiK5/view?usp=sharing).
<p><br /></p>

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

The #deepMTJ repository contains:

### 1. Annotate
- `mtj_tracking/label` folder contains the video annotation tool: start `main.py`

### 2. Train
- `mtj_tracking/train` folder contains the network training and evaluation: start `train.py` for `VGG-Attention-3` model and `train_resnet.py` for the `ResNet50` model.

### 3. Predict
- Trained networks (540 MB) can be downloaded from: [chriskross.org](https://drive.google.com/file/d/18-tX4SX1xabOf3J77Qt8BTNt412rYiK5/view?usp=sharing)

### AddOn
- `examples` folder with all result plots and figures of the IEEE-EMBC 2020 publication in high resolution.

# Getting Started

This is one possible way to run [#deepMTJ](https://github.com/luuleitner/deepMTJ) on your computer. We have used the [Anaconda](https://www.anaconda.com/) package manager and the [Pycharm](https://www.jetbrains.com/pycharm/) programming environment to develop, train and run our networks.

1. Download the `requirements.txt` File
2. Install [Anaconda](https://www.anaconda.com/) (v1.9.12*) for Python v3.7
3. Create the `deepMTJ` virtual environment in your Anaconda terminal and install all necessary libraries (listed in the `requirements.txt` File) using the following code in the terminal window:

```
conda create --name deepMTJ python=3.7 --file requirements.txt
```

4. Install [Pycharm](https://www.jetbrains.com/pycharm/)
5. Clone this GitHub repository onto your local machine using [https://github.com/luuleitner/deepMTJ](https://github.com/luuleitner/deepMTJ)
6. Use `File - Settings - Project Interpreter` to install `deepMTJ` as your Python project interpreter.
7. Run Python scripts
8. (Running TensorFlow code on your GPU might require additional settings. DeepLabCut provids very useful information and additional links on the GPU / Cuda / Python / TensorFlow interfaces. You can find that [information here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md))

**tested version for #deepMTJ*

# License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](https://github.com/luuleitner/deepMTJ/blob/master/LICENSE) file for details

