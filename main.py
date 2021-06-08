from utils.utils import extract_data,get_data,get_audio,\
    get_audio_len,get_spectrogram,plot_spectrogram

from models.convolution import CNN
from models.transformer import Transformer_Model

import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from IPython import display
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generator(selector):
    if selector == "train":
        fname = train_files
    else:
        fname = test_files

    n_batchs = len(fname) // batch_size

    while True:
        fname = shuffle(fname)

        for n in range(n_batchs):
            files_batch = fname[n * batch_size:(n + 1) * batch_size]
            audio_batch = map(get_audio, files_batch)
            spectrogram_batch = list(map(get_spectrogram, audio_batch, [clip_idx]*batch_size))
            labels_batch = list(map(get_label,files_batch))

            spectrogram_batch = torch.cat(spectrogram_batch)
            spectrogram_batch = torch.reshape(spectrogram_batch, (batch_size, 1, frequencies_number, -1))
            labels_batch = torch.tensor(labels_batch)

            yield spectrogram_batch, labels_batch

#a function for getting the label of the audio file
def get_label(fname):
  fname=fname.split('\\')[1]
  label=train_df['label'][train_df['fname']==fname]
  return label.values[0]

# define a function to compute the model accuracy
# define a function to compute the model accuracy
def calc_accuracy(t, y):
    preds = torch.argmax(y, 1)
    accuracy = torch.mean((t == preds).to(dtype=torch.float32))
    return accuracy


# the train stage for the models
def train(model_name):
    criterion = torch.nn.CrossEntropyLoss()
    if model_name == 'conv_model':
        model = conv_model
    else:
        model = transformer_model

    optimizer = torch.optim.Adam(model.parameters())

    epochs = 10

    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    n_batchs_train = len(train_files) // batch_size
    n_batchs_test = len(test_files) // batch_size

    for epoch in range(epochs):
        train_epoch_losses = []
        test_epoch_losses = []

        train_epoch_accuracy = []
        test_epoch_accuracy = []

        # inserting the model into train mode
        model.train()

        # since we are using data generator, we need to manually calculate which epoch and batch we are
        batch_count = 0

        start_time = time.time()
        for audio_batch, labels_batch in generator("train"):

            audio_batch = audio_batch.to(device)
            # for the conv model we need extra 1 dimention for the "color" dimention.
            # we don't need it for the transformer model.
            # we also transpose the spectrogram for the transformer model because the sequence is the audio file length
            # and not the frequency axes.
            # It doens't matter for the convolution model so we keep it.
            if model_name == 'transformer_model':
                audio_batch = torch.squeeze(audio_batch, 1)
                audio_batch = audio_batch.permute(0, 2, 1)
            labels_batch = labels_batch.to(device)
            labels_batch=labels_batch.long()
            # the gradient decend step
            optimizer.zero_grad()
            outputs = model(audio_batch)
            train_loss = criterion(outputs, labels_batch)

            train_epoch_losses.append(train_loss.item())

            train_accuracy = calc_accuracy(labels_batch, outputs)
            train_epoch_accuracy.append(train_accuracy.item())

            train_loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count == n_batchs_train:
                break

        # inserting the model into evaluate mode
        model.eval()
        batch_count = 0

        # same for test set
        for audio_batch, labels_batch in generator("test"):
            audio_batch = audio_batch.to(device)
            if model_name == 'transformer_model':
                audio_batch = torch.squeeze(audio_batch, 1)
                audio_batch = audio_batch.permute(0, 2, 1)

            labels_batch = labels_batch.to(device)
            outputs = model(audio_batch)

            test_loss = criterion(outputs, labels_batch)
            test_epoch_losses.append(test_loss.item())

            test_accuracy = calc_accuracy(labels_batch, outputs)
            test_epoch_accuracy.append(test_accuracy.item())

            batch_count += 1
            if batch_count == n_batchs_test:
                end_time = time.time()
                break

        mean_train_epoch_loss = np.mean(train_epoch_losses)
        train_losses.append(mean_train_epoch_loss)

        mean_train_epoch_accuracy = np.mean(train_epoch_accuracy)
        train_accuracies.append(mean_train_epoch_accuracy)

        mean_test_epoch_loss = np.mean(test_epoch_losses)
        test_losses.append(mean_test_epoch_loss)

        mean_test_epoch_accuracy = np.mean(test_epoch_accuracy)
        test_accuracies.append(mean_test_epoch_accuracy)

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {mean_train_epoch_loss:.4f},  Train Accuracy:{mean_train_epoch_accuracy:.4f}, \
      Test Loss: {mean_test_epoch_loss:.4f},  Test Accuracy:{mean_test_epoch_accuracy:.4f},   Epoch time:{(end_time - start_time):.4f}sec')

    return train_losses, test_losses, train_accuracies, test_accuracies

if __name__ == '__main__':
    extract_data()
    train_files, test_files, train_df=get_data()

    # K will be the number of categories, will be used as the model output
    K = len(set(train_df['label']))

    # show random example of audio file and their waveform plot
    for i in range(5):
        random_file = np.random.choice(train_files)
        sample = random_file.split('\\')[1]

        print("Category:", train_df['label'][train_df['fname'] == sample].values[0])

        waveform = get_audio(random_file)
        display.display(display.Audio(waveform, rate=44100))
        plt.plot(waveform)
        plt.show()

    # the distribution of the audio files length. We can see most of the fies length are short
    # There for we will clip the files to a constant length of 200,000 samples

    files_len = list(map(get_audio_len, train_files+test_files))

    plt.hist(files_len, bins=50)
    plt.title("Audio Length Distribution")
    plt.show()

    # the number of samples for each category. major part of the data in unbalanced which can effect the model
    # the samples that weren't manually verified have precision of about 70%. it shuld be taken into considerations.
    label_value_counts = train_df['label'].value_counts()
    manually_verified_labels = train_df['label'][train_df['manually_verified'] == 1]
    manually_verified_labels_count = manually_verified_labels.value_counts()
    plt.figure(figsize=(15, 5))
    label_value_counts.plot.bar(title="Categories Distribution", label="Not Manually Verified", color='r')
    manually_verified_labels_count.plot.bar(label="Manually Verified")
    plt.legend()
    plt.show()

    # we want to convert the waveform(amplitude as function of time) to a 2D representation-
    # Frequencies amplitude as function of frequency and time. This will be the spectrogram

    # We want to have constant length for each sample. we choose length of 200000 frame for all the files,
    # since it covers the length of most of the files, and we won't loose too much information for the
    # files that are longer that that

    # files that are shorted than this length will be padded with zeros
    # files that are longer than that will be clipped
    clip_idx = 100000

    fig, ax = plt.subplots(4, 4, figsize=(14, 9))
    fig.tight_layout(pad=3.0)

    categories = train_df['label'].unique()

    for i in range(4):
        category = np.random.choice(categories, replace=False)

        for j in range(4):
            fnames = train_df['fname'][train_df['label'] == category].values
            fname = np.random.choice(fnames)
            waveform = get_audio("dataset/audio_train\\" + fname)
            spectrogram = get_spectrogram(waveform,clip_idx)
            # plt.title(category)
            plot_spectrogram(spectrogram, ax[j, i],clip_idx,category)

    plt.show()
    frequencies_number, sequence_len = spectrogram.shape

    # from seeing different random spectrograms, we can observ repeating pattern for the different labels.
    # we can also see that usually the pattern can be observ in the beggining of the audio file.
    # therefore i we want faster model computation, we can reduce even more the audio files length. It will hurt the accuracy
    # but not too much

    # transform the labels to numerical value
    le = LabelEncoder()
    le.fit(train_df['label'])
    train_df['label'] = le.transform(train_df['label'])
    train_df.head()

    # we will use data generator because the dataset is too large. The generator output will be
    # the spectrogram and their labels, it will later be sent to the model
    batch_size = 32

    # First we will try the 2D convolution model. We will treat the spectrogram like and image and fit the model
    conv_model = CNN(K).to(device)

    # the second model that we use is the transformer model.
    # since  we added padding to the spectrogram, when we use the convolution network we lose information,
    # because it's more difficult to the network to find the correct feauters as the dimention shirinks.
    # therefore we will try to use the transformer model with a padding mask - it will not take the padded part
    # into consideration and will be ignored using the attention mechanism.

    transformer_model = Transformer_Model(K).to(device)

    train_losses, test_losses, train_accuracies, test_accuracies=train("conv_model")

    # plot the train/test loss and accuracy for the conv2d model
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.legend()
    plt.show()

    train_losses, test_losses, train_accuracies, test_accuracies=train("transformer_model")

    # plot the train/test loss and accuracy for the transformer model
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.legend()
    plt.show()



