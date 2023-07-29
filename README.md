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
    <td><img src="pics/fig_1a.png" alt="VITS at training" width="100%"></td>
    <td><img src="pics/fig_1b.png" alt="VITS at inference" width="100%"></td>
  </tr>
</table>

## Installation:

<a name="installation"></a>

**Clone the repo**

```shell
git clone git@github.com:daniilrobnikov/vits.git
cd vits
```

## Setting up the conda env

This is assuming you have navigated to the `vits` root after cloning it.

**NOTE:** This is tested under `python3.11` with conda env. For other python versions, you might encounter version conflicts.
**NOTE:** This is tested under `python3.11` with conda env. For other python versions, you might encounter version conflicts.

**PyTorch 2.0**
Please refer [requirements.txt](requirements.txt)
Please refer [requirements.txt](requirements.txt)

```shell
# install required packages (for pytorch 2.0)
conda create -n vits python=3.11
conda activate vits
pip install -r requirements.txt
```

## Download datasets

There are three options you can choose from: LJ Speech, VCTK, and custom dataset.

1. LJ Speech: [LJ Speech dataset](#lj-speech-dataset). Used for single speaker TTS.
2. VCTK: [VCTK dataset](#vctk-dataset). Used for multi-speaker TTS.
3. Custom dataset: You can use your own dataset. Please refer [here](#custom-dataset).

### LJ Speech dataset

1. download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)

```shell
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```

2. rename or create a link to the dataset folder

```shell
ln -s /path/to/LJSpeech-1.1/wavs DUMMY1
```

### VCTK dataset

1. download and extract the [VCTK dataset](https://www.kaggle.com/datasets/showmik50/vctk-dataset)

```shell
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip
```

2. (optional): downsample the audio files to 22050 Hz. See [audio_resample.ipynb](preprocess/audio_resample.ipynb)
3. rename or create a link to the dataset folder

```shell
ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2
```

### Custom dataset

1. create a folder with wav files
2. create configuration file in [configs](configs/). Change the following fields in `custom_base.json`:
3. create configuration file in [configs](configs/). Change the following fields in `custom_base.json`:

```js
{
  "data": {
    "training_files": "filelists/custom_audio_text_train_filelist.txt.cleaned", // path to training cleaned filelist
    "validation_files": "filelists/custom_audio_text_val_filelist.txt.cleaned", // path to validation cleaned filelist
    "text_cleaners": ["english_cleaners2"], // text cleaner
    "bits_per_sample": 16, // bit depth of wav files
    "training_files": "filelists/custom_audio_text_train_filelist.txt.cleaned", // path to training cleaned filelist
    "validation_files": "filelists/custom_audio_text_val_filelist.txt.cleaned", // path to validation cleaned filelist
    "text_cleaners": ["english_cleaners2"], // text cleaner
    "bits_per_sample": 16, // bit depth of wav files
    "sampling_rate": 22050, // sampling rate if you resampled your wav files
    ...
    "n_speakers": 0, // number of speakers in your dataset if you use multi-speaker setting
    "cleaned_text": true // if you already cleaned your text (See text_phonemizer.ipynb), set this to true
  },
  ...
    "cleaned_text": true // if you already cleaned your text (See text_phonemizer.ipynb), set this to true
  },
  ...
}
```

3. install espeak-ng (optional)

**NOTE:** This is required for the [preprocess.py](preprocess.py) and [inference.ipynb](inference.ipynb) notebook to work. If you don't need it, you can skip this step. Please refer [espeak-ng](https://github.com/espeak-ng/espeak-ng)

4. preprocess text

You can do this step by step way:

- create a dataset of text files. See [text_dataset.ipynb](preprocess/text_dataset.ipynb)
- phonemize or just clean up the text. Please refer [text_phonemizer.ipynb](preprocess/text_phonemizer.ipynb)
- create filelists and cleaned version with train test split. See [text_split.ipynb](preprocess/text_split.ipynb)
- rename or create a link to the dataset folder. Please refer [text_split.ipynb](preprocess/text_split.ipynb)

```shell
ln -s /path/to/custom_dataset DUMMY3
```

## Training Examples

```shell
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base

# Custom dataset (multi-speaker)
python train_ms.py -c configs/custom_base.json -m custom_base

# Custom dataset (multi-speaker)
python train_ms.py -c configs/custom_base.json -m custom_base
```

## Inference Example

See [inference.ipynb](inference.ipynb)
See [inference_batch.ipynb](inference_batch.ipynb) for multiple sentences inference

## Pretrained Models

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing)

## Audio Samples

## Todo

- [ ] text preprocessing
  - [x] update cleaners for multi-language support with 100+ languages
  - [x] update vocabulary to support all symbols and features from IPA. See [phonemes.md](https://github.com/espeak-ng/espeak-ng/blob/ed9a7bcf5778a188cdec202ac4316461badb28e1/docs/phonemes.md#L5)
  - [x] handling unknown, out of vocabulary symbols. Please refer [vocab.py](text/vocab.py) and [vocab - TorchText](https://pytorch.org/text/stable/vocab.html)
  - [x] remove cleaners from text preprocessing. Most cleaners are already implemented in [phonemizer](https://github.com/bootphon/phonemizer). See [cleaners.py](text/cleaners.py)
  - [ ] remove necessity for speakers indexation. See [vits/issues/58](https://github.com/jaywalnut310/vits/issues/58)
- [ ] audio preprocessing
  - [x] batch audio resampling. Please refer [audio_resample.ipynb](preprocess/audio_resample.ipynb)
  - [x] code snippets to find corrupted files in dataset. Please refer [audio_find_corrupted.ipynb](preprocess/audio_find_corrupted.ipynb)
  - [x] code snippets to delete by extension files in dataset. Please refer [delete_by_ext.ipynb](preprocess/delete_by_ext.ipynb)
  - [x] replace scipy and librosa dependencies with torchaudio. See [load](https://pytorch.org/audio/stable/backend.html#id2) and [MelScale](https://pytorch.org/audio/main/generated/torchaudio.transforms.MelScale.html) docs
  - [x] automatic audio range normalization. Please refer [Loading audio data - Torchaudio docs](https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data)
  - [x] add support for stereo audio (multi-channel). See [Loading audio data - Torchaudio docs](https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data)
  - [x] add support for various audio bit depths (bits per sample). See [load - Torchaudio docs](https://pytorch.org/audio/stable/backend.html#id2)
  - [x] add support for various sample rates. Please refer [load - Torchaudio docs](https://pytorch.org/audio/stable/backend.html#id2)
  - [ ] test stereo audio (multi-channel) training
- [x] filelists preprocessing
  - [x] add filelists preprocessing for multi-speaker. Please refer [text_split.ipynb](preprocess/text_split.ipynb)
  - [x] code snippets for train test split. Please refer [text_split.ipynb](preprocess/text_split.ipynb)
  - [x] notebook to link filelists with actual wavs. Please refer [text_split.ipynb](preprocess/text_split.ipynb)
- [ ] other
  - [x] rewrite code for python 3.11
  - [x] replace Cython Monotonic Alignment Search with numba.jit. See [vits-finetuning](https://github.com/SayaSS/vits-finetuning)
  - [x] updated inference to support batch processing
- [ ] pretrained models
  - [ ] training the model for Bengali language. (For now: 55_000 iterations, ~26 epochs)
  - [ ] add pretrained models for multiple languages
- [ ] future work
  - [ ] update model to naturalspeech. Please refer [naturalspeech](https://arxiv.org/abs/2205.04421)
  - [ ] add support for streaming. Please refer [vits_chinese](https://github.com/PlayVoice/vits_chinese/blob/master/text/symbols.py)
  - [ ] update naturalspeech to multi-speaker
  - [ ] replace speakers with multi-speaker embeddings
  - [ ] replace speakers with multilingual training. Each speaker is a language with thhe same IPA symbols
  - [ ] add support for in-context learning

## Acknowledgements

- This repo is based on [VITS](https://github.com/jaywalnut310/vits)
- Text to phones converter for multiple languages is based on [phonemizer](https://github.com/bootphon/phonemizer)
- We also thank GhatGPT for providing writing assistance.

## References

- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103)
- [A TensorFlow implementation of Google's Tacotron speech synthesis with pre-trained model (unofficial)](https://github.com/keithito/tacotron)

# vits
