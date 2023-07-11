# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

\*\* Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" width="100%"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" width="100%"></td>
  </tr>
</table>

## Installation:

<a name="installation"></a>

### 1. Clone the repo

```shell
git clone git@github.com:daniilrobnikov/vits-bengali.git
cd vits-bengali
```

### 2. Setting up the conda env

This is assuming you have navigated to the `vits-bengali` root after cloning it.

**NOTE:** This is tested under `python3.6` and `python3.11`. For other python versions, you might encounter version conflicts.

**PyTorch 1.13**
Please refer [requirements_py6.txt](requirements.txt)

```shell
# install required packages (for pytorch 1.13)
conda create -n py6 python=3.6
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

### 2. Install espeak (optional)

**NOTE:** This is required for the [preprocess.py](preprocess.py) and [inference.ipynb](inference.ipynb) notebook to work. If you don't need it, you can skip this step.

```shell
# install espeak
sudo apt-get install espeak
```

### 3. Build Monotonic Alignment Search

```shell
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

### 4. Download datasets

There are three options you can choose from: LJ Speech, VCTK, and custom dataset.

1. LJ Speech: [LJ Speech dataset](#lj-speech-dataset). Used for single speaker TTS.
2. VCTK: [VCTK dataset](#vctk-dataset). Used for multi-speaker TTS.
3. Custom dataset: You can use your own dataset. Please refer [here](#custom-dataset).

#### LJ Speech dataset

1. download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)

```shell
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```

2. rename or create a link to the dataset folder

```shell
ln -s /path/to/LJSpeech-1.1/wavs DUMMY1
```

#### VCTK dataset

1. download and extract the [VCTK dataset](https://www.kaggle.com/datasets/showmik50/vctk-dataset)
2. resample wav files to 22050 Hz. Please refer [preprocess/resample_audio.py](preprocess/resample_audio.py)
3. rename or create a link to the dataset folder

```shell
ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2
```

#### Custom dataset

1. create a folder with wav files
2. resample wav files to 22050 Hz. Please refer [downsample.py](downsample.py)
3. rename or create a link to the dataset folder

```shell
ln -s /path/to/custom_dataset DUMMY3
```

4. run preprocessing. Please refer [preprocess/phonemizer.py](preprocess/phonemizer.py)
5. create filelists. Please refer [preprocess/split_train_test.py](preprocess/split_train_test.py)
6. modify [config file](configs/) to use your own dataset

```js
{
  "data": {
    "training_files":"filelists/custom_dataset_audio_text_train_filelist.txt.cleaned", // path to training filelist
    "validation_files":"filelists/custom_dataset_audio_text_train_filelist.txt.cleaned", // path to validation filelist
    "text_cleaners":["english_cleaners2"], // text cleaner
    ...
    "sampling_rate": 22050, // sampling rate if you resampled your wav files
    ...
    "n_speakers": 0, // number of speakers in your dataset if you use multi-speaker setting
  }
}
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

## Pretrained Models

See [pretrained_models.md](pretrained_models.md)

## Audio Samples

## Todo

- [ ] text preprocessing
  - [x] add support for Bengali text cleaner
  - [ ] update original text cleaner for multi-language
  - [ ] custom `text/cleaners.py` for multi-language
- [ ] audio preprocessing
  - [x] batch audio resampling. Please refer [preprocess/resample_audio.py](preprocess/resample_audio.py)
  - [x] unit testing for corrupt files with rate assertion. Please refer [preprocess/test_corrupt_files.py](preprocess/test_corrupt_files.py)
  - [x] code snippets to find corruption files in dataset. Please refer [preprocess/find_corrupt_files.py](preprocess/find_corrupt_files.py)
  - [x] code snippets to delete from extension files in dataset. Please refer [preprocess/delete_from_extension.py](preprocess/delete_from_extension.py)
  - [ ] accepting different sample rates. Please refer [vits_chinese](https://github.com/PlayVoice/vits_chinese/blob/master/text/symbols.py)
  - [ ] remove necessity for multispeech speakers indexation
- [ ] filelists preprocessing
  - [x] add filelists preprocessing for multi-speaker. Please refer [preprocess/split_train_test.py](preprocess/split_train_test.py)
  - [x] code snippets for train test split. Please refer [preprocess/split_train_test.py](preprocess/split_train_test.py)
  - [ ] notebook to link filelists with actual wavs. Please refer [preprocess/link_filelists_with_wavs.ipynb](preprocess/link_filelists_with_wavs.ipynb)
- [ ] future work
  - [ ] pre-trained model for Bengali language
  - [ ] update model to naturalspeech. Please refer [naturalspeech](https://arxiv.org/abs/2205.04421)
  - [ ] update naturalspeech to multi-speaker
  - [ ] add support for streaming. Please refer [vits_chinese](https://github.com/PlayVoice/vits_chinese/blob/master/text/symbols.py)

## Acknowledgements

- This repo is based on [VITS](https://github.com/jaywalnut310/vits)
- Text to phones converter for multiple languages is based on [phonemizer](https://github.com/bootphon/phonemizer)
- We also thank GhatGPT for providing writing assistance.

## References

- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103)
- [A TensorFlow implementation of Google's Tacotron speech synthesis with pre-trained model (unofficial)](https://github.com/keithito/tacotron)

# vits-bengali
