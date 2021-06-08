# for downloading the zip data:
# pip install kaggle
# import kaggle
# type on terminal kaggle competitions download -c freesound-audio-tagging

import shutil
import os
import pathlib
import re
import numpy as np
import torch
from glob import glob
from sklearn.utils import shuffle
import pandas as pd
import torchaudio


#extract the zip file
def extract_data():
    filepath = 'freesound-audio-tagging.zip'
    path=pathlib.Path('dataset/audio_test')
    if not path.exists():
        shutil.unpack_archive(filepath, 'dataset')
        os.remove(filepath)


def get_data():
    # get the filenames and split to train and test set
    files = glob('dataset/audio_train/*wav')
    files = shuffle(files)
    split = int(0.8 * len(files))

    train_files = files[:split]
    test_files = files[split:]

    train_df = pd.read_csv('dataset/train.csv')

    return train_files,test_files,train_df

#decode the audio file
def get_audio(filepath):
  decoded_audio=torchaudio.load(filepath)[0]
  return decoded_audio[0]


def get_audio_len(file):
    return len(get_audio(file))




def get_spectrogram(audio,clip_idx):
  if len(audio)<clip_idx:
    padding=torch.zeros(clip_idx-audio.shape[0], dtype=torch.float32)
    audio=torch.cat([audio,padding],0)
  else:
    audio=audio[:clip_idx]
  spectrogram = torch.stft(audio, n_fft=255,win_length=255, hop_length=128)
  spectrogram=torch.norm(spectrogram,dim=2)

  return spectrogram

#plt the spectrogram
def plot_spectrogram(spectrogram,ax,clip_idx,category):
  log_spec = np.log(spectrogram)
  height = log_spec.shape[0]
  X = np.arange(clip_idx, step=height)
  Y = np.arange(height)
  ax.pcolormesh(X, Y, log_spec)
  ax.set_title(category)


