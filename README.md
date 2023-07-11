# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

** Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Installation:
<a name="installation"></a>

#### 1. Clone the repo

```shell
git clone git@github.com:daniilrobnikov/vits-bengali.git
cd vits-bengali
```

#### 2. Setting up the conda env

This is assuming you have navigated to the `vits-bengali` root after cloning it. 

**NOTE:** This is tested under `python3.6` and `python3.11`. For other python versions, you might encounter version conflicts.


**PyTorch 1.13** 
Please refer [requirements_py6.txt](requirements.txt)

```shell
# install required packages (for pytorch 1.13)
conda create -n py11 python=3.6
conda activate py6
pip install -r requirements_py6.txt
```

**PyTorch 2.0** 
Please refer [requirements_py11.txt](requirements.txt)

```shell
# install required packages (for pytorch 2.0)
conda create -n py11 python=3.11
conda activate py11
pip install -r requirements_py11.txt
```


#### 2. Install espeak (optional)

**NOTE:** This is required for the `preprocess.py` and `inference.ipynb` notebook to work. If you don't need it, you can skip this step.


```shell
# install espeak
sudo apt-get install espeak
```


#### 3. Build Monotonic Alignment Search

```shell
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```


#### 4. Download datasets

**LJ Speech dataset**
1. download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)

```shell
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```

2. rename or create a link to the dataset folder

```shell
ln -s /path/to/LJSpeech-1.1/wavs DUMMY1
```


**VCTK dataset**
1. download and extract the [VCTK dataset](https://www.kaggle.com/datasets/showmik50/vctk-dataset)
2. resample wav files to 22050 Hz. Please refer [preprocess/resample_audio.py](preprocess/resample_audio.py)
2. rename or create a link to the dataset folder
```shell
ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2
```


**Custom dataset**
1. create a folder with wav files
2. resample wav files to 22050 Hz. Please refer [downsample.py](downsample.py)
4. run preprocessing. Please refer [preprocess/phonemizer.py](preprocess/phonemizer.py)
```
3. rename or create a link to the dataset folder
```shell
ln -s /path/to/custom_dataset DUMMY3
```


## Training Examples
```shell
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
See [inference.ipynb](inference.ipynb)


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For multi-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Examples
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
See [inference.ipynb](inference.ipynb)
# vits-bengali
